# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for IterativeMoEHook (RFC-3 Part 2)."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.hooks.iterative_activation import (
    apply_iterative_moe_hook,
    remove_iterative_moe_hook,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class _DummyExpert(nn.Module):
    """A simple expert that doubles the input."""

    def forward(self, hidden_states, router_logits=None, **kwargs):
        return hidden_states * 2.0


class _DummyGate(nn.Module):
    """A simple gate that returns uniform router logits."""

    def forward(self, hidden_states):
        num_tokens = hidden_states.shape[0]
        logits = torch.ones(num_tokens, 4)  # 4 experts
        return logits, None


class _DummyMoEBlock(nn.Module):
    """A minimal MoE block with gate + experts for testing."""

    def __init__(self, hidden_dim=32, num_experts=4):
        super().__init__()
        self.gate = _DummyGate()
        self.experts = _DummyExpert()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat = hidden_states.view(-1, hidden_dim)
        router_logits, _ = self.gate(flat)
        out = self.experts(hidden_states=flat, router_logits=router_logits)
        return out.view(orig_shape)


class TestIterativeMoEHook:
    def test_hook_registration_and_removal(self):
        block = _DummyMoEBlock()
        hook = apply_iterative_moe_hook(block, chunk_size=8)
        assert hook.chunk_size == 8

        registry = getattr(block, "_hook_registry", None)
        assert registry is not None
        assert registry.get_hook("iterative_moe") is not None

        remove_iterative_moe_hook(block)
        assert registry.get_hook("iterative_moe") is None

    def test_chunked_output_matches_direct(self):
        """Output of chunked processing must equal direct processing."""
        torch.manual_seed(42)
        hidden_dim = 32
        seq_len = 100
        hidden_states = torch.randn(2, seq_len, hidden_dim)

        # Direct (no hook)
        block_direct = _DummyMoEBlock(hidden_dim=hidden_dim)
        out_direct = block_direct(hidden_states)

        # Chunked (with hook)
        block_chunked = _DummyMoEBlock(hidden_dim=hidden_dim)
        # Copy weights to match
        block_chunked.load_state_dict(block_direct.state_dict())
        apply_iterative_moe_hook(block_chunked, chunk_size=16)
        out_chunked = block_chunked(hidden_states)

        assert out_chunked.shape == out_direct.shape
        assert torch.allclose(out_direct, out_chunked, atol=1e-6)

    def test_small_sequence_uses_direct_path(self):
        """When seq_len <= chunk_size, should behave like original forward."""
        hidden_dim = 16
        hidden_states = torch.randn(4, 10, hidden_dim)  # 40 tokens total

        block_direct = _DummyMoEBlock(hidden_dim=hidden_dim)
        out_direct = block_direct(hidden_states)

        block_hooked = _DummyMoEBlock(hidden_dim=hidden_dim)
        block_hooked.load_state_dict(block_direct.state_dict())
        apply_iterative_moe_hook(block_hooked, chunk_size=100)  # > 40 tokens
        out_hooked = block_hooked(hidden_states)

        assert torch.allclose(out_direct, out_hooked, atol=1e-6)

    def test_chunk_size_splits_correctly(self):
        """Verify the hook processes in chunks of the specified size."""
        hidden_dim = 8
        seq_len = 50
        hidden_states = torch.randn(1, seq_len, hidden_dim)

        block = _DummyMoEBlock(hidden_dim=hidden_dim)
        apply_iterative_moe_hook(block, chunk_size=16)

        # 50 tokens / 16 chunk_size = 4 chunks (16, 16, 16, 2)
        out = block(hidden_states)
        assert out.shape == hidden_states.shape

    def test_preserves_output_shape(self):
        """Output shape must match input shape regardless of chunking."""
        for shape in [(1, 100, 32), (2, 50, 16), (4, 25, 8)]:
            hidden_states = torch.randn(*shape)
            block = _DummyMoEBlock(hidden_dim=shape[-1])
            apply_iterative_moe_hook(block, chunk_size=10)
            out = block(hidden_states)
            assert out.shape == hidden_states.shape, f"Failed for shape {shape}"

    def test_default_chunk_size(self):
        block = _DummyMoEBlock()
        hook = apply_iterative_moe_hook(block)
        assert hook.chunk_size == 4096

    def test_hook_idempotent(self):
        """Re-applying the hook should overwrite, not stack."""
        block = _DummyMoEBlock()
        apply_iterative_moe_hook(block, chunk_size=8)
        apply_iterative_moe_hook(block, chunk_size=16)

        registry = getattr(block, "_hook_registry", None)
        hooks = [k for k in registry._hooks if k == "iterative_moe"]
        assert len(hooks) == 1
        assert registry.get_hook("iterative_moe").chunk_size == 16
