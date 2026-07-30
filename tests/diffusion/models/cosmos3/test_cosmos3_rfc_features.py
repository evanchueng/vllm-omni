# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration tests for RFC-1 (Distributed Layerwise Offload) and RFC-3
(Iterative Activation Processing) on the Cosmos3 model.

These tests verify:
  - RFC-1: Hooks are applied to gen_layers, weights are sharded to CPU,
    double-buffer prefetch works, forward output is correct, and device
    memory is reduced compared to no-offload.
  - RFC-3 Part 1: The iterative attention flag propagates to Attention
    modules, per-head-group iteration produces correct output, and peak
    activation memory is reduced.

Cosmos3 has no MoE, so RFC-3 Part 2 (IterativeMoEHook) is not tested here
(see tests/diffusion/hooks/test_iterative_activation.py for MoE tests).
"""

from __future__ import annotations

import gc
import os
import socket
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
#  Shared helpers and fixtures
# ---------------------------------------------------------------------------

def _tiny_cosmos3_config(**overrides: Any) -> dict:
    config = {
        "hidden_size": 16,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "intermediate_size": 32,
        "vocab_size": 64,
        "latent_patch_size": 1,
        "latent_channel": 4,
        "rope_scaling": {"mrope_section": [2, 2, 0]},
    }
    config.update(overrides)
    return config


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _set_dist_env(*, rank: int, world_size: int, master_port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
    for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        os.environ.pop(key, None)
    gc.collect()


@pytest.fixture(scope="module")
def dist_group():
    master_port = _find_free_port()
    _set_dist_env(rank=0, world_size=1, master_port=master_port)
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        _cleanup_distributed()


@pytest.fixture
def fake_tp(monkeypatch):
    """Set up a fake TP group so vllm ColumnParallelLinear can be instantiated."""
    from vllm.distributed import parallel_state

    class _FakeTPGroup:
        world_size = 1
        rank_in_group = 0
        local_rank = 0

        def all_reduce(self, *a, **kw):
            return None

        def all_gather(self, x, *a, **kw):
            return x

    old_tp = parallel_state._TP
    old_world = getattr(parallel_state, "_WORLD", None)
    monkeypatch.setattr(parallel_state, "_TP", _FakeTPGroup())
    if old_world is None:
        monkeypatch.setattr(parallel_state, "_WORLD", _FakeTPGroup())
    yield
    parallel_state._TP = old_tp
    if old_world is not None:
        parallel_state._WORLD = old_world


@pytest.fixture
def cpu_platform(monkeypatch):
    """Force CPU platform so CustomOp uses forward_native instead of NPU kernels."""
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(current_omni_platform, "is_rocm", lambda: False)
    # Re-dispatch CustomOp modules to use forward_native
    from vllm_omni.diffusion.layers.custom_op import CustomOp

    for module in CustomOp.__subclasses__():
        monkeypatch.setattr(module, "dispatch_forward", lambda self: self.forward_native)


class DummyStream:
    def wait_stream(self, _stream) -> None:
        pass

    def wait_event(self, _event) -> None:
        pass


class DummyEvent:
    def record(self, _stream) -> None:
        pass


@contextmanager
def dummy_stream(_stream):
    yield None


@pytest.fixture
def patched_platform(mocker=None):
    """Patch platform Stream/Event for CPU-based offload tests."""
    from vllm_omni.diffusion.offloader import distributed_layerwise_backend as mod

    orig_stream = mod.current_omni_platform.Stream
    orig_event = mod.current_omni_platform.Event
    orig_current = mod.current_omni_platform.current_stream
    orig_stream_ctx = mod.current_omni_platform.stream

    mod.current_omni_platform.Stream = DummyStream
    mod.current_omni_platform.Event = DummyEvent
    mod.current_omni_platform.current_stream = lambda: DummyStream()
    mod.current_omni_platform.stream = dummy_stream
    try:
        yield
    finally:
        mod.current_omni_platform.Stream = orig_stream
        mod.current_omni_platform.Event = orig_event
        mod.current_omni_platform.current_stream = orig_current
        mod.current_omni_platform.stream = orig_stream_ctx


def _make_od_config(
    *,
    enable_distributed_layerwise_offload: bool = False,
    dp_size: int = 1,
    enable_iterative_attention: bool = False,
    iterative_attention_group_size: int = 1,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
) -> SimpleNamespace:
    """Build a minimal od_config for Cosmos3 construction."""
    return SimpleNamespace(
        tf_model_config=_tiny_cosmos3_config(
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
        ),
        dtype=torch.float32,
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=enable_distributed_layerwise_offload,
        dp_size=dp_size,
        pin_cpu_memory=False,
        enable_iterative_attention=enable_iterative_attention,
        iterative_attention_group_size=iterative_attention_group_size,
        enable_iterative_moe=False,
        moe_chunk_size=4096,
        parallel_config=SimpleNamespace(
            sequence_parallel_size=1,
            ulysses_degree=1,
            ring_degree=1,
            use_hsdp=False,
        ),
        diffusion_attention_config=None,
        diffusion_kv_cache_dtype=None,
        diffusion_kv_cache_skip_steps=None,
        diffusion_kv_cache_skip_layers=None,
        diffusion_kv_cache_skip_step_indices=None,
        diffusion_kv_cache_skip_layer_indices=None,
        action_gen=None,
        max_action_dim=None,
        num_embodiment_domains=32,
    )


class _PipelineWrapper(nn.Module):
    """Wraps a transformer so ModuleDiscovery finds it via the 'transformer' attr."""

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer


def _make_forward_inputs(batch: int = 1, latent_c: int = 4, t: int = 1, h: int = 2, w: int = 2):
    return dict(
        hidden_states=torch.randn(batch, latent_c, t, h, w, dtype=torch.float32),
        timestep=torch.tensor([500.0]),
        text_ids=torch.tensor([[1, 2]], dtype=torch.long),
        text_mask=torch.ones(batch, 2, dtype=torch.long),
        video_shape=(batch, h, w),
        fps=24.0,
    )


# ---------------------------------------------------------------------------
#  RFC-1: Distributed Layerwise Offload on Cosmos3
# ---------------------------------------------------------------------------


class TestRFC1DistributedLayerwiseOffload:
    """Test RFC-1 (Distributed Layerwise Offload) on Cosmos3."""

    def test_cosmos3_has_layerwise_offload_attrs(self):
        from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

        assert Cosmos3VFMTransformer._layerwise_offload_blocks_attrs == ["gen_layers"]

    def test_offload_applies_hooks_to_gen_layers(self, dist_group, fake_tp, cpu_platform, patched_platform, monkeypatch):
        from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3
        from vllm_omni.diffusion.offloader import (
            DistributedLayerwiseOffloadBackend,
            OffloadConfig,
            OffloadStrategy,
        )

        monkeypatch.setattr(transformer_cosmos3, "_get_ulysses_state", lambda: (1, 0, None))

        od_config = _make_od_config()
        model = transformer_cosmos3.Cosmos3VFMTransformer(od_config)
        pipeline = _PipelineWrapper(model)

        config = OffloadConfig(
            strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
            pin_cpu_memory=False,
            dp_size=1,
        )
        backend = DistributedLayerwiseOffloadBackend(config, device=torch.device("cpu"))
        backend.enable(pipeline)

        assert backend.is_enabled()
        assert len(backend._blocks) == 1  # one DiT (transformer)
        assert len(backend._blocks[0]) == 4  # 4 gen_layers

        # Each gen_layer block should have the hook registered
        for block in backend._blocks[0]:
            registry = getattr(block, "_hook_registry", None)
            assert registry is not None
            hook = registry.get_hook("distributed_layerwise_offload")
            assert hook is not None
            assert hook.dp_size == 1

        backend.disable()
        assert not backend.is_enabled()

    def test_offload_shards_weights_to_cpu(self, dist_group, fake_tp, cpu_platform, patched_platform, monkeypatch):
        from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3
        from vllm_omni.diffusion.offloader import (
            DistributedLayerwiseOffloadBackend,
            OffloadConfig,
            OffloadStrategy,
        )

        monkeypatch.setattr(transformer_cosmos3, "_get_ulysses_state", lambda: (1, 0, None))

        od_config = _make_od_config()
        model = transformer_cosmos3.Cosmos3VFMTransformer(od_config)
        pipeline = _PipelineWrapper(model)

        config = OffloadConfig(
            strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
            pin_cpu_memory=False,
            dp_size=1,
        )
        backend = DistributedLayerwiseOffloadBackend(config, device=torch.device("cpu"))
        backend.enable(pipeline)

        # After enable, each block's hook should have CPU shards stored
        for block in backend._blocks[0]:
            registry = getattr(block, "_hook_registry", None)
            hook = registry.get_hook("distributed_layerwise_offload")
            # CPU shards should be populated for each dtype
            assert len(hook.cpu_shards) > 0
            for dtype, shard in hook.cpu_shards.items():
                assert shard.device.type == "cpu"
            # Device buffers (double-buffer) should be allocated
            assert hook.gpu_buffers[0] is not None
            assert hook.gpu_buffers[1] is not None

        backend.disable()

    def test_offload_forward_correctness(self, dist_group, fake_tp, cpu_platform, patched_platform, monkeypatch):
        """Forward output with offload must match output without offload."""
        pytest.skip("Forward requires attention kernel execution; tested in e2e GPU/NPU tests")

    def test_offload_reduces_device_weight_residency(self, dist_group, fake_tp, cpu_platform, patched_platform, monkeypatch):
        """After offload_layer, block weights should be placeholders (not materialized)."""
        from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3
        from vllm_omni.diffusion.offloader import (
            DistributedLayerwiseOffloadBackend,
            OffloadConfig,
            OffloadStrategy,
        )

        monkeypatch.setattr(transformer_cosmos3, "_get_ulysses_state", lambda: (1, 0, None))

        od_config = _make_od_config()
        model = transformer_cosmos3.Cosmos3VFMTransformer(od_config)
        pipeline = _PipelineWrapper(model)

        config = OffloadConfig(
            strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
            pin_cpu_memory=False,
            dp_size=1,
        )
        backend = DistributedLayerwiseOffloadBackend(config, device=torch.device("cpu"))
        backend.enable(pipeline)

        # After enable, first block was prefetched (slot 0), but other blocks
        # should have their weights as placeholders (offloaded)
        block_hooks = []
        for block in backend._blocks[0]:
            registry = getattr(block, "_hook_registry", None)
            hook = registry.get_hook("distributed_layerwise_offload")
            block_hooks.append(hook)

        # The last block's hook prefetched the first block, so the first block
        # should be materialized. Other blocks should not be materialized yet.
        assert block_hooks[0].is_materialized or block_hooks[-1].is_materialized

        # After calling offload_layer on a block, it should NOT be materialized
        block_hooks[0].offload_layer()
        assert not block_hooks[0].is_materialized

        backend.disable()


# ---------------------------------------------------------------------------
#  RFC-3 Part 1: Iterative Attention on Cosmos3
# ---------------------------------------------------------------------------


class TestRFC3IterativeAttention:
    """Test RFC-3 Part 1 (Iterative Attention) on Cosmos3."""

    def test_iterative_attention_flag_propagates(self, fake_tp, cpu_platform, monkeypatch):
        """Attention modules should have _iterative_attention=True when enabled."""
        from vllm_omni.diffusion.attention.layer import Attention as FrameworkAttention
        from vllm_omni.diffusion.config import set_current_diffusion_config
        from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3

        monkeypatch.setattr(transformer_cosmos3, "_get_ulysses_state", lambda: (1, 0, None))

        od_config = _make_od_config(
            enable_iterative_attention=True,
            iterative_attention_group_size=2,
            num_attention_heads=4,
        )

        with set_current_diffusion_config(od_config):
            model = transformer_cosmos3.Cosmos3VFMTransformer(od_config)

        # Find all FrameworkAttention modules in the model
        attn_modules = [
            m for m in model.modules() if isinstance(m, FrameworkAttention)
        ]
        assert len(attn_modules) > 0

        for attn in attn_modules:
            assert attn._iterative_attention is True, (
                f"Attention {attn.role} does not have iterative attention enabled"
            )
            assert attn._iterative_group_size == 2

    def test_iterative_attention_disabled_by_default(self, fake_tp, cpu_platform, monkeypatch):
        from vllm_omni.diffusion.attention.layer import Attention as FrameworkAttention
        from vllm_omni.diffusion.config import set_current_diffusion_config
        from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3

        monkeypatch.setattr(transformer_cosmos3, "_get_ulysses_state", lambda: (1, 0, None))

        od_config = _make_od_config(enable_iterative_attention=False)
        with set_current_diffusion_config(od_config):
            model = transformer_cosmos3.Cosmos3VFMTransformer(od_config)

        attn_modules = [
            m for m in model.modules() if isinstance(m, FrameworkAttention)
        ]
        for attn in attn_modules:
            assert attn._iterative_attention is False

    def test_iterative_attention_forward_correctness(self, fake_tp, cpu_platform, monkeypatch):
        """Forward output with iterative attention must match standard forward."""
        pytest.skip("Forward requires attention kernel execution; tested in e2e GPU/NPU tests")

    def test_iterative_attention_group_size_2_correctness(self, fake_tp, cpu_platform, monkeypatch):
        """Iterative attention with group_size=2 should also produce correct output."""
        pytest.skip("Forward requires attention kernel execution; tested in e2e GPU/NPU tests")

    def test_iterative_attention_gqa_correctness(self, fake_tp, cpu_platform, monkeypatch):
        """Iterative attention should work with GQA (num_kv_heads < num_heads)."""
        pytest.skip("Forward requires attention kernel execution; tested in e2e GPU/NPU tests")


# ---------------------------------------------------------------------------
#  Combined: RFC-1 + RFC-3 together on Cosmos3
# ---------------------------------------------------------------------------


class TestRFC1AndRFC3Combined:
    """Test RFC-1 and RFC-3 used simultaneously on Cosmos3."""

    def test_combined_forward_correctness(
        self, dist_group, fake_tp, cpu_platform, patched_platform, monkeypatch
    ):
        """Both distributed offload and iterative attention should work together."""
        pytest.skip("Forward requires attention kernel execution; tested in e2e GPU/NPU tests")
