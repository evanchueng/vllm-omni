# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for iterative activation hooks (RFC-3).

Covers:
  Part 1 — Iterative attention (head-group slicing with GQA)
  Part 2 — Iterative MLP (token chunking with CPU streaming)
  Part 3 — VAE temporal chunking (causal frame loop replication)

All tests are self-contained (synthetic modules, no external model).
"""

import gc
import sys

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _empty_cache():
    for _ in range(3):
        gc.collect()
    if torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


# ──────────────────────── Part 1: Iterative attention ──────────────────────


class TestIterativeAttention:
    """Verify head-group-sliced attention == full attention (GQA aware)."""

    @pytest.mark.parametrize("group_size", [1, 2, 4, 8, 16])
    def test_head_group_slicing_matches_full(self, group_size):
        num_heads, num_kv_heads, head_dim = 32, 8, 128
        q_per_kv = num_heads // num_kv_heads
        B, S = 1, 512

        torch.manual_seed(42)
        device = "npu" if torch.npu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device != "cpu" else torch.float32

        query = torch.randn(B, S, num_heads, head_dim, device=device, dtype=dtype)
        key = torch.randn(B, S, num_kv_heads, head_dim, device=device, dtype=dtype)
        value = torch.randn(B, S, num_kv_heads, head_dim, device=device, dtype=dtype)

        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # Reference: all heads at once
        k_expanded = k.repeat_interleave(q_per_kv, dim=1)
        v_expanded = v.repeat_interleave(q_per_kv, dim=1)
        with torch.no_grad():
            ref = F.scaled_dot_product_attention(q, k_expanded, v_expanded)

        # Iterative: head-group slicing with CPU streaming
        if group_size >= num_heads:
            pytest.skip("group_size >= num_heads, no iteration")

        cpu_slices = []
        for start in range(0, num_heads, group_size):
            end = min(start + group_size, num_heads)
            q_slice = q[:, start:end, :, :]
            kv_start = start // q_per_kv
            kv_end = (end + q_per_kv - 1) // q_per_kv
            k_slice = k[:, kv_start:kv_end, :, :].repeat_interleave(
                q_per_kv if (end - start) % q_per_kv == 0 else 1, dim=1
            )[:, : end - start, :, :]
            v_slice = v[:, kv_start:kv_end, :, :].repeat_interleave(
                q_per_kv if (end - start) % q_per_kv == 0 else 1, dim=1
            )[:, : end - start, :, :]
            with torch.no_grad():
                out_slice = F.scaled_dot_product_attention(q_slice, k_slice, v_slice)
            cpu_slices.append(out_slice.cpu())

        out = torch.cat(cpu_slices, dim=1).to(device)

        max_diff = (ref - out).abs().max().item()
        assert max_diff < 1e-5, f"max_diff={max_diff:.2e} for group_size={group_size}"

        del query, key, value, q, k, v, ref, out
        _empty_cache()


# ──────────────────────── Part 2: Iterative MLP ────────────────────────────


class _SimpleGatedMLP(nn.Module):
    """Minimal gated MLP matching Cosmos3GatedMLP's interface."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TestIterativeMLP:
    """Verify chunked-token MLP == full MLP (per-token, no cross-token dep)."""

    @pytest.mark.parametrize(
        "num_tokens,chunk_size",
        [
            (1024, 20480),
            (40960, 20480),
            (40960, 8192),
            (40960, 1024),
            (40960, 1),
        ],
    )
    def test_chunked_matches_direct(self, num_tokens, chunk_size):
        from vllm_omni.diffusion.hooks import apply_iterative_mlp_hook, remove_iterative_mlp_hook

        device = "npu" if torch.npu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device != "cpu" else torch.float32

        hidden_size = 4096
        intermediate_size = 12288
        mlp = _SimpleGatedMLP(hidden_size, intermediate_size).to(device).to(dtype).eval()

        torch.manual_seed(42)
        x = torch.randn(1, num_tokens, hidden_size, device=device, dtype=dtype)

        # Reference: no hook
        remove_iterative_mlp_hook(mlp)
        with torch.no_grad():
            ref = mlp(x)

        # With hook
        apply_iterative_mlp_hook(mlp, chunk_size=chunk_size)
        with torch.no_grad():
            out = mlp(x)

        max_diff = (ref - out).abs().max().item()
        # bf16 matmul tiling differs between full and chunked shapes → small
        # numerical drift (~2e-3) is expected, not a correctness bug.
        tol = 2e-2 if dtype == torch.bfloat16 else 1e-6
        assert max_diff < tol, f"max_diff={max_diff:.2e} for tokens={num_tokens} cs={chunk_size}"

        remove_iterative_mlp_hook(mlp)
        del mlp, x, ref, out
        _empty_cache()

    def test_hook_registration_and_removal(self):
        from vllm_omni.diffusion.hooks import apply_iterative_mlp_hook, remove_iterative_mlp_hook

        mlp = _SimpleGatedMLP(64, 128)
        hook = apply_iterative_mlp_hook(mlp, chunk_size=8)
        assert hook.chunk_size == 8

        registry = getattr(mlp, "_hook_registry", None)
        assert registry is not None
        assert registry.get_hook("iterative_mlp") is not None

        remove_iterative_mlp_hook(mlp)
        assert registry.get_hook("iterative_mlp") is None

    def test_small_sequence_uses_direct_path(self):
        from vllm_omni.diffusion.hooks import apply_iterative_mlp_hook, remove_iterative_mlp_hook

        mlp = _SimpleGatedMLP(32, 64)
        x = torch.randn(4, 10, 32)

        remove_iterative_mlp_hook(mlp)
        ref = mlp(x)

        apply_iterative_mlp_hook(mlp, chunk_size=100)
        out = mlp(x)

        assert torch.allclose(ref, out, atol=1e-6)
        remove_iterative_mlp_hook(mlp)


# ──────────────────────── Part 3: VAE temporal chunking ────────────────────


def _make_small_vae(device: str, dtype: torch.dtype):
    """Create a small synthetic Wan VAE with causal temporal structure."""
    from diffusers.models.autoencoders import AutoencoderKLWan

    z_dim = 4
    vae = AutoencoderKLWan(
        base_dim=8,
        dim_mult=[1, 2],
        z_dim=z_dim,
        latents_mean=[0.0] * z_dim,
        latents_std=[1.0] * z_dim,
        in_channels=4,
        out_channels=4,
        patch_size=2,
        scale_factor_temporal=4,
        scale_factor_spatial=8,
        temperal_downsample=[True],
        num_res_blocks=1,
    )
    vae = vae.to(device).to(dtype).eval()
    vae.use_tiling = False
    vae.use_slicing = False
    return vae


class TestVAETemporalChunking:
    """Verify VAE temporal chunking produces identical output to full decode."""

    @pytest.mark.parametrize(
        "name,T,H,W,chunk_size",
        [
            ("no_chunk", 5, 8, 8, 30),
            ("one_chunk", 10, 8, 8, 30),
            ("two_chunks_even", 60, 8, 8, 30),
            ("two_chunks_uneven", 62, 8, 8, 31),
            ("chunk_size_1", 10, 8, 8, 1),
            ("chunk_size_3", 10, 8, 8, 3),
            ("bigger_spatial", 30, 16, 16, 10),
        ],
    )
    def test_chunked_decode_matches_full(self, name, T, H, W, chunk_size):
        from vllm_omni.diffusion.hooks.iterative_activation import apply_vae_temporal_chunking

        device = "npu" if torch.npu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device != "cpu" else torch.float32

        vae = _make_small_vae(device, dtype)
        num_ch = vae.config.z_dim

        torch.manual_seed(42)
        z = torch.randn(1, num_ch, T, H, W, device=device, dtype=dtype)

        # Ground truth: original decode
        orig_decode = vae.decode
        with torch.no_grad():
            ref = vae.decode(z, return_dict=False)[0]

        # Chunked decode
        apply_vae_temporal_chunking(vae, chunk_size=chunk_size)
        with torch.no_grad():
            chunked = vae.decode(z, return_dict=False)[0]

        assert ref.shape == chunked.shape, f"shape mismatch: ref={ref.shape} chunked={chunked.shape}"

        max_diff = (ref - chunked).abs().max().item()
        assert max_diff < 1e-6, f"max_diff={max_diff:.2e} for {name}"

        vae.decode = orig_decode
        del vae, z, ref, chunked
        _empty_cache()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
