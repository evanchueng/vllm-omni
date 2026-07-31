# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iterative Activation Processing hooks (RFC-3).

This module provides hooks that reduce peak activation memory by processing
attention heads and MLP tokens iteratively instead of all at once.

Part 1 — Per-Head Iteration for Attention:
    Implemented directly in ``Attention._run_iterative_local_attention``
    (see ``vllm_omni/diffusion/attention/layer.py``). No hook needed.

Part 2 — Per-Chunk Iteration for MLP:
    ``IterativeMLPHook`` wraps an MLP block's ``forward`` so that tokens are
    processed in configurable chunks, keeping only a small working set of
    intermediate activations in HBM at any time.

    Supports three MLP variants (auto-detected):
    - Sparse MoE: modules with ``gate`` + ``experts`` (routing computed upfront,
      then experts applied per chunk)
    - Gated MLP: modules with ``gate_proj`` + ``up_proj`` + ``down_proj``
      (e.g. Cosmos3GatedMLP)
    - Standard FFN: modules with ``fc1`` + ``fc2`` or ``linear_fc1`` + ``linear_fc2``

Part 3 — Per-Chunk Iteration for Decoder Layer:
    ``IterativeLayerHook`` wraps a transformer decoder layer's ``forward`` so
    that ``hidden_states`` is split along the token dimension into chunks.
    Each chunk's QKV projection, attention, and MLP all operate on only
    ``chunk_size`` tokens, so the full-token-count Q/K/V never materialises
    on GPU simultaneously.  Chunk outputs are streamed to CPU and
    concatenated once at the end.

    This is the most effective strategy for high-resolution / long-video
    generation where QKV projection activations dominate HBM.

    Requirements: the layer's self-attention must be **non-causal** (full
    bidirectional), so token chunks can be processed independently.  This
    holds for diffusion DiT generation layers (e.g. ``Cosmos3GenDecoderLayer``
    cross-attention).  Causal LM layers (``Cosmos3UndDecoderLayer``) must
    not receive this hook.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook

logger = init_logger(__name__)


class IterativeMLPHook(ModelHook):
    """Hook for per-chunk iterative MLP processing (RFC-3 Part 2).

    Instead of passing all tokens through the MLP at once, this hook splits
    tokens into chunks and processes each chunk independently. Only
    ``chunk_size`` tokens' intermediate activations are held in HBM at a time.

    For sparse MoE: routing (gate) is computed for all tokens upfront (cheap),
    then experts are applied per chunk.
    For dense MLP (gated or standard): the full forward is applied per chunk.
    """

    _HOOK_NAME = "iterative_mlp"

    def __init__(self, chunk_size: int = 20480):
        self.chunk_size = chunk_size

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        module = super().initialize_hook(module)
        logger.debug(
            "Iterative MLP enabled on %s: chunk_size=%d",
            module.__class__.__name__,
            self.chunk_size,
        )
        return module

    def new_forward(self, module: nn.Module, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Replace the MLP forward with chunked iteration.

        For large token counts, input and output are streamed to/from CPU
        so peak GPU memory is O(chunk_size * intermediate_size) instead of
        O(num_tokens * hidden_size).
        """
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat_hidden = hidden_states.view(-1, hidden_dim)
        num_tokens = flat_hidden.shape[0]

        if num_tokens <= self.chunk_size:
            return self._mlp_forward(module, flat_hidden, *args, **kwargs).view(orig_shape)

        device = flat_hidden.device
        dtype = flat_hidden.dtype

        # Stream: process chunks, move output to CPU, concatenate at the end.
        # This keeps peak GPU memory at O(chunk_size * intermediate_size)
        # instead of O(num_tokens * hidden_size) for input+output.
        cpu_outputs: list[torch.Tensor] = []
        for start in range(0, num_tokens, self.chunk_size):
            end = min(start + self.chunk_size, num_tokens)
            chunk_in = flat_hidden[start:end].to(device, non_blocking=True)
            chunk_out = self._mlp_forward(module, chunk_in, *args, **kwargs)
            cpu_outputs.append(chunk_out.cpu())
            del chunk_in, chunk_out

        # Concatenate on CPU, then move back to GPU once
        final_hidden = torch.cat(cpu_outputs, dim=0).to(device, non_blocking=True)
        return final_hidden.view(orig_shape)

    @staticmethod
    def _mlp_forward(module: nn.Module, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Run a single MLP forward, auto-detecting MoE vs dense FFN."""
        # Sparse MoE: gate (routing) + experts
        if hasattr(module, "gate") and hasattr(module, "experts"):
            router_logits, _ = module.gate(x)
            return module.experts(hidden_states=x, router_logits=router_logits)

        # Gated MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
        if hasattr(module, "gate_proj"):
            return module.down_proj(F.silu(module.gate_proj(x)) * module.up_proj(x))

        # Standard FFN variant 1: linear_fc2(silu(linear_fc1(x)))
        if hasattr(module, "linear_fc1"):
            return module.linear_fc2(F.silu(module.linear_fc1(x)))

        # Standard FFN variant 2: fc2(silu(fc1(x)))
        if hasattr(module, "fc1"):
            return module.fc2(F.silu(module.fc1(x)))

        # Fallback: call original forward
        return module(x)


def apply_iterative_mlp_hook(
    module: nn.Module,
    chunk_size: int = 20480,
) -> IterativeMLPHook:
    """Register an IterativeMLPHook on *module*.

    Args:
        module: The MLP/MoE module (e.g., ``Cosmos3GatedMLP``, sparse MoE block).
        chunk_size: Number of tokens per chunk.

    Returns:
        The registered hook instance.
    """
    registry = HookRegistry.get_or_create(module)
    hook = IterativeMLPHook(chunk_size=chunk_size)
    registry.register_hook(IterativeMLPHook._HOOK_NAME, hook)
    return hook


def remove_iterative_mlp_hook(module: nn.Module) -> None:
    """Remove the iterative MLP hook from *module*."""
    registry: HookRegistry | None = getattr(module, "_hook_registry", None)
    if registry is not None:
        registry.remove_hook(IterativeMLPHook._HOOK_NAME)
        logger.debug("Removed iterative MLP hook from %s", module.__class__.__name__)


class IterativeLayerHook(ModelHook):
    """Per-chunk iterative decoder layer processing (RFC-3 Part 3).

    Wraps a transformer decoder layer's ``forward`` so that ``hidden_states``
    is split along the token dimension (dim=1) into chunks.  Each chunk's
    QKV projection, attention, and MLP all operate on only ``chunk_size``
    tokens, so the full-token-count Q/K/V never materialises on GPU at once.

    Chunk outputs are streamed to CPU and concatenated once at the end,
    keeping peak GPU memory at O(chunk_size * hidden_size) for the layer's
    activations instead of O(num_tokens * hidden_size).

    The layer's attention must be **non-causal** so chunks are independent.
    """

    _HOOK_NAME = "iterative_layer"

    def __init__(self, chunk_size: int = 20480):
        self.chunk_size = chunk_size

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        module = super().initialize_hook(module)
        logger.debug(
            "Iterative layer enabled on %s: chunk_size=%d",
            module.__class__.__name__,
            self.chunk_size,
        )
        return module

    def new_forward(self, module: nn.Module, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Replace the layer forward with token-chunked iteration."""
        # hidden_states shape: [B, S, H]
        num_tokens = hidden_states.shape[1]
        if num_tokens <= self.chunk_size:
            return self.fn_ref.original_forward(module, hidden_states, *args, **kwargs)

        device = hidden_states.device
        cpu_outputs: list[torch.Tensor] = []

        for start in range(0, num_tokens, self.chunk_size):
            end = min(start + self.chunk_size, num_tokens)
            chunk_hidden = hidden_states[:, start:end, :]
            chunk_out = self.fn_ref.original_forward(module, chunk_hidden, *args, **kwargs)
            # Handle tuple returns (e.g. UND layer returns (hidden, k, v))
            if isinstance(chunk_out, tuple):
                chunk_out = chunk_out[0]
            cpu_outputs.append(chunk_out.cpu())
            del chunk_hidden, chunk_out

        result = torch.cat([o.to(device) for o in cpu_outputs], dim=1)
        return result


def apply_iterative_layer_hook(
    module: nn.Module,
    chunk_size: int = 20480,
) -> IterativeLayerHook:
    """Register an IterativeLayerHook on *module*.

    Args:
        module: The decoder layer (e.g., ``Cosmos3GenDecoderLayer``).
        chunk_size: Number of tokens per chunk.

    Returns:
        The registered hook instance.
    """
    registry = HookRegistry.get_or_create(module)
    hook = IterativeLayerHook(chunk_size=chunk_size)
    registry.register_hook(IterativeLayerHook._HOOK_NAME, hook)
    return hook


def remove_iterative_layer_hook(module: nn.Module) -> None:
    """Remove the iterative layer hook from *module*."""
    registry: HookRegistry | None = getattr(module, "_hook_registry", None)
    if registry is not None:
        registry.remove_hook(IterativeLayerHook._HOOK_NAME)
        logger.debug("Removed iterative layer hook from %s", module.__class__.__name__)


def apply_vae_temporal_chunking(vae: nn.Module, chunk_size: int = 30) -> None:
    """Wrap VAE.decode to process frames in temporal chunks with cache continuity.

    The Wan VAE latent has shape [B, C, T, H, W]. Standard ``_decode`` processes
    all T frames in a single loop, accumulating the output tensor on GPU — peak
    memory is O(T * H_out * W_out) which can be enormous for long videos.

    This wrapper **replicates the ``_decode`` frame loop** (not calling
    ``_decode`` itself) so that:

    - ``feat_cache`` (``_feat_map``) is maintained **across chunk boundaries**,
      preserving causal temporal continuity.  Calling ``_decode`` per chunk
      would ``clear_cache()`` between chunks and break the causal chain.
    - ``first_chunk=True`` is passed **only for the global first frame**
      (``i == 0``), not for each chunk's first frame.  This preserves the
      non-uniform temporal upsampling (1 + (T-1)*factor_t output frames).
    - Completed chunk outputs are moved to CPU immediately, so GPU memory for
      the output tensor is O(chunk_size) instead of O(T).

    Falls back to the original ``decode`` when spatial tiling or distributed VAE
    is active (those paths have their own frame loops + ``clear_cache`` that
    cannot be safely chunked from outside).

    Args:
        vae: The VAE module (must have ``post_quant_conv``, ``decoder``,
            ``_feat_map``, ``_conv_idx``, ``clear_cache`` — i.e. a Wan-style
            causal VAE).
        chunk_size: Number of latent frames per temporal chunk (default: 30).
    """
    required = ("post_quant_conv", "decoder", "clear_cache")
    if not all(hasattr(vae, attr) for attr in required):
        logger.warning(
            "VAE temporal chunking: VAE lacks required attributes %s; skipping",
            [a for a in required if not hasattr(vae, a)],
        )
        return

    orig_decode = vae.decode

    try:
        from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify as _unpatchify
    except ImportError:
        _unpatchify = None
    try:
        from diffusers.models.autoencoders.vae import DecoderOutput
    except ImportError:
        DecoderOutput = None

    def _needs_fallback(z: torch.Tensor) -> bool:
        """True when spatial tiling or distributed VAE would trigger."""
        if hasattr(vae, "is_distributed_enabled") and vae.is_distributed_enabled():
            return True
        if not getattr(vae, "use_tiling", False):
            return False
        ratio = max(getattr(vae, "spatial_compression_ratio", 1), 1)
        tile_min_h = getattr(vae, "tile_sample_min_height", 0) // ratio
        tile_min_w = getattr(vae, "tile_sample_min_width", 0) // ratio
        _, _, _, h, w = z.shape
        return h > tile_min_h or w > tile_min_w

    def _chunked_decode_single(z: torch.Tensor) -> torch.Tensor:
        """Replicate ``_decode`` frame loop with temporal chunking for one batch element.

        ``feat_cache`` persists across chunks (no ``clear_cache`` between them).
        Output is streamed to CPU at chunk boundaries.
        """
        _, _, num_frame, _, _ = z.shape
        vae.clear_cache()
        x = vae.post_quant_conv(z)

        cpu_outputs: list[torch.Tensor] = []
        out: torch.Tensor | None = None

        for start in range(0, num_frame, chunk_size):
            end = min(start + chunk_size, num_frame)
            for j in range(end - start):
                global_i = start + j
                vae._conv_idx = [0]
                frame = x[:, :, global_i : global_i + 1, :, :]
                if global_i == 0:
                    out = vae.decoder(
                        frame,
                        feat_cache=vae._feat_map,
                        feat_idx=vae._conv_idx,
                        first_chunk=True,
                    )
                else:
                    out_ = vae.decoder(
                        frame,
                        feat_cache=vae._feat_map,
                        feat_idx=vae._conv_idx,
                    )
                    if out is None:
                        out = out_
                    else:
                        out = torch.cat([out, out_], dim=2)

            cpu_outputs.append(out.cpu())
            del out
            out = None

        del x
        device = z.device
        result = torch.cat([o.to(device) for o in cpu_outputs], dim=2)

        if getattr(vae.config, "patch_size", None) is not None and _unpatchify is not None:
            result = _unpatchify(result, patch_size=vae.config.patch_size)
        result = torch.clamp(result, min=-1.0, max=1.0)

        vae.clear_cache()
        return result

    def chunked_decode(z, return_dict=True, *args, **kwargs):
        # Fast path: small frame count or non-5D input
        num_frames = z.shape[2] if z.ndim == 5 else 0
        if num_frames <= chunk_size:
            return orig_decode(z, return_dict=return_dict, *args, **kwargs)

        # Fallback for tiled / distributed VAE (their own frame loops + clear_cache)
        if _needs_fallback(z):
            logger.debug(
                "VAE temporal chunking: falling back to full decode "
                "(tiling or distributed VAE active)"
            )
            return orig_decode(z, return_dict=return_dict, *args, **kwargs)

        exec_ctx = getattr(vae, "_execution_context", None)
        ctx = exec_ctx() if callable(exec_ctx) else nullcontext()
        with ctx:
            if getattr(vae, "use_slicing", False) and z.shape[0] > 1:
                decoded = torch.cat([_chunked_decode_single(s) for s in z.split(1)])
            else:
                decoded = _chunked_decode_single(z)

        if not return_dict:
            return (decoded,)
        if DecoderOutput is not None:
            return DecoderOutput(sample=decoded)
        return decoded

    vae.decode = chunked_decode
    logger.info("VAE temporal chunking enabled: chunk_size=%d frames", chunk_size)
