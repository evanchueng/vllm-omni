# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iterative Activation Processing hooks (RFC-3).

This module provides hooks that reduce peak activation memory by processing
attention heads and MoE tokens iteratively instead of all at once.

Part 1 — Per-Head Iteration for Attention:
    Implemented directly in ``Attention._run_iterative_local_attention``
    (see ``vllm_omni/diffusion/attention/layer.py``). No hook needed.

Part 2 — Per-Chunk Iteration for MoE:
    ``IterativeMoEHook`` wraps a sparse-MoE block's ``forward`` so that
    tokens are processed in configurable chunks, keeping only a small
    working set of expert intermediates in HBM at any time.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook

logger = init_logger(__name__)


class IterativeMoEHook(ModelHook):
    """Hook for per-chunk iterative MoE processing (RFC-3 Part 2).

    Instead of passing all tokens through the MoE experts at once, this hook
    splits tokens into chunks and processes each chunk independently. Only
    ``chunk_size`` tokens' expert intermediates are held in HBM at a time,
    reducing peak activation memory from ``seq_len * num_experts * hidden_dim``
    to ``chunk_size * num_experts * hidden_dim``.

    The routing (gate) is computed for all tokens upfront (cheap), then
    hidden_states and router_logits are chunked together.
    """

    _HOOK_NAME = "iterative_moe"

    def __init__(self, chunk_size: int = 4096):
        self.chunk_size = chunk_size

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        module = super().initialize_hook(module)
        logger.info(
            "Iterative MoE enabled on %s: chunk_size=%d",
            module.__class__.__name__,
            self.chunk_size,
        )
        return module

    def new_forward(self, module: nn.Module, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Replace the MoE block forward with chunked iteration.

        Expected original forward signature:
            forward(hidden_states) -> Tensor

        The module must have:
            - ``module.gate``: router linear (hidden_dim -> num_experts)
            - ``module.experts``: FusedMoE or equivalent callable accepting
              ``hidden_states=`` and ``router_logits=`` keyword arguments.
        """
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat_hidden = hidden_states.view(-1, hidden_dim)
        num_tokens = flat_hidden.shape[0]

        # Compute router logits for ALL tokens upfront (cheap, small tensor)
        router_logits, _ = module.gate(flat_hidden)

        # If sequence is smaller than chunk_size, just run the original path
        if num_tokens <= self.chunk_size:
            out = module.experts(hidden_states=flat_hidden, router_logits=router_logits)
            return out.view(orig_shape)

        # Chunked iteration: split tokens and router_logits
        outputs: list[torch.Tensor] = []
        for start in range(0, num_tokens, self.chunk_size):
            end = min(start + self.chunk_size, num_tokens)
            chunk_hidden = flat_hidden[start:end]
            chunk_logits = router_logits[start:end]

            chunk_out = module.experts(hidden_states=chunk_hidden, router_logits=chunk_logits)
            outputs.append(chunk_out)

        final_hidden = torch.cat(outputs, dim=0)
        return final_hidden.view(orig_shape)


def apply_iterative_moe_hook(
    module: nn.Module,
    chunk_size: int = 4096,
) -> IterativeMoEHook:
    """Register an IterativeMoEHook on *module*.

    Args:
        module: The MoE block module (e.g., ``HunYuanSparseMoeBlock``).
        chunk_size: Number of tokens per chunk.

    Returns:
        The registered hook instance.
    """
    registry = HookRegistry.get_or_create(module)
    hook = IterativeMoEHook(chunk_size=chunk_size)
    registry.register_hook(IterativeMoEHook._HOOK_NAME, hook)
    return hook


def remove_iterative_moe_hook(module: nn.Module) -> None:
    """Remove the iterative MoE hook from *module*."""
    registry: HookRegistry | None = getattr(module, "_hook_registry", None)
    if registry is not None:
        registry.remove_hook(IterativeMoEHook._HOOK_NAME)
        logger.debug("Removed iterative MoE hook from %s", module.__class__.__name__)
