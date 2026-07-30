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
"""

from __future__ import annotations

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
        """Replace the MLP forward with chunked iteration."""
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat_hidden = hidden_states.view(-1, hidden_dim)
        num_tokens = flat_hidden.shape[0]

        if num_tokens <= self.chunk_size:
            return self._mlp_forward(module, flat_hidden, *args, **kwargs).view(orig_shape)

        outputs: list[torch.Tensor] = []
        for start in range(0, num_tokens, self.chunk_size):
            end = min(start + self.chunk_size, num_tokens)
            chunk_hidden = flat_hidden[start:end]
            chunk_out = self._mlp_forward(module, chunk_hidden, *args, **kwargs)
            outputs.append(chunk_out)

        final_hidden = torch.cat(outputs, dim=0)
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
