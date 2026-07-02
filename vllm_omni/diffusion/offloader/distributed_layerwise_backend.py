# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed Layerwise Offload backend with H2D + AllGather overlap.

This module implements the RFC-1 "Distributed Layerwise Offload" mechanism that:

* Shards model weights across DP ranks and stores only the local shard
  (1/DP_size of full model) in host pinned memory per rank.
* Uses a fixed double-buffer scheme that keeps only two layers' worth of
  weights on each device at any time.
* Asynchronously pipelines both H2D transfers and AllGather communications
  on dedicated streams, fully overlapping them with computation.
* Is hardware-agnostic, supporting both NVIDIA GPU (CUDA) and Ascend NPU
  (CANN) platforms via vLLM-Omni's platform abstraction layer.
"""

from __future__ import annotations

from itertools import chain
from typing import Any

import torch
import torch.distributed
from torch import nn
from torch.distributed.tensor import DTensor
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig
from .module_collector import ModuleDiscovery

logger = init_logger(__name__)


class DistributedLayerwiseOffloadHook(ModelHook):
    """Hook for distributed layerwise offloading with fixed double-buffer.

    Each rank stores only a shard of each block's weights on host memory.
    Two device slots alternate: one holds current weights, one holds next weights.
    H2D and AllGather run asynchronously on dedicated streams, overlapped
    with computation.

    Supports both NVIDIA GPU (CUDA) and Ascend NPU (CANN) platforms.
    """

    _HOOK_NAME = "distributed_layerwise_offload"

    def __init__(
        self,
        next_block: nn.Module,
        device: torch.device,
        dp_group: torch.distributed.ProcessGroup | None,
        dp_size: int,
        rank: int,
        copy_stream: Any | None = None,
        comm_stream: Any | None = None,
        pin_memory: bool = True,
    ):
        assert isinstance(next_block, nn.Module), "transformer block must be type `torch.nn.Module`"

        self.next_block = next_block
        self.device = device
        self.dp_group = dp_group
        self.dp_size = dp_size
        self.rank = rank
        self.pin_memory = pin_memory

        self.copy_stream = copy_stream or current_omni_platform.Stream()
        self.comm_stream = comm_stream or current_omni_platform.Stream()

        # Fixed double buffers: exactly two slots
        self.gpu_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]
        self.ready_events: list[Any | None] = [None, None]

        # Sharded host weights for the next block, keyed by dtype
        self.cpu_shards: dict[torch.dtype, torch.Tensor] = {}
        self.metadata: dict[torch.dtype, list[dict[str, Any]]] = {}

        # Current slot index (0 or 1)
        self.current_slot = 0

        # Backward link to previous hook for fallback (cache-dit skip)
        self._prev_hook: DistributedLayerwiseOffloadHook | None = None

        # Parameters/buffers of the current and next blocks
        self.block_parameters: dict[str, nn.Parameter] = {}
        self.block_buffers: dict[str, torch.Tensor] = {}
        self.next_block_parameters: dict[str, nn.Parameter] = {}
        self.next_block_buffers: dict[str, torch.Tensor] = {}

        # Per-block synchronization primitive: set after H2D copy completes.
        self._prefetch_done: Any | None = None

    # ------------------------------------------------------------------ #
    #  DTensor helpers (shared with LayerwiseOffloadHook)                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_dtensor(t: torch.Tensor) -> bool:
        return isinstance(t, DTensor)

    @staticmethod
    def _set_tensor_storage(target: torch.Tensor, value: torch.Tensor) -> None:
        if DistributedLayerwiseOffloadHook._is_dtensor(target):
            target._local_tensor = value
        else:
            target.data = value

    @staticmethod
    def _make_offload_placeholder(tensor: torch.Tensor) -> torch.Tensor:
        if DistributedLayerwiseOffloadHook._is_dtensor(tensor):
            local_shape = tuple(tensor.to_local().shape)
            return torch.empty(local_shape, device="meta", dtype=tensor.dtype)
        return torch.empty((0,), device=tensor.device, dtype=tensor.dtype)

    @staticmethod
    def _is_materialized_tensor(t: torch.Tensor) -> bool:
        if DistributedLayerwiseOffloadHook._is_dtensor(t):
            local_t = t.to_local()
            return not local_t.is_meta
        return not t.is_meta and t.data.numel() > 0

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        module = super().initialize_hook(module)

        self.block_parameters = dict(module.named_parameters())
        self.block_buffers = dict(module.named_buffers())

        self.next_block_parameters = dict(self.next_block.named_parameters())
        self.next_block_buffers = dict(self.next_block.named_buffers())

        # Shard next block's weights and store local shard in pinned CPU memory
        self.cpu_shards, self.metadata = self._shard_and_pin(
            self.next_block_parameters,
            self.next_block_buffers,
            self.dp_size,
            self.rank,
            self.pin_memory,
        )

        # Pre-allocate two device buffers (double-buffer)
        self._allocate_device_buffers()

        return module

    @staticmethod
    def _shard_and_pin(
        params: dict[str, nn.Parameter],
        bufs: dict[str, torch.Tensor],
        dp_size: int,
        rank: int,
        pin_memory: bool,
    ) -> tuple[dict[torch.dtype, torch.Tensor], dict[torch.dtype, list[dict[str, Any]]]]:
        """Flatten params+buffers by dtype, split into DP shards, store local shard.

        Each rank stores only ``1/dp_size`` of the total weights. The full
        tensor is reconstructed at runtime via AllGather.
        """
        dtype_grouped: dict[torch.dtype, dict[str, torch.Tensor]] = {}
        dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}

        for name, param_or_buf in chain(params.items(), bufs.items()):
            dtype = param_or_buf.dtype
            if dtype not in dtype_grouped:
                dtype_grouped[dtype] = {}
            dtype_grouped[dtype][name] = param_or_buf

        cpu_shards: dict[torch.dtype, torch.Tensor] = {}

        for dtype, name2weights in dtype_grouped.items():
            # Resolve local tensors (handle DTensor via to_local)
            weights_with_local = []
            for name, t in name2weights.items():
                local_t = t.to_local() if hasattr(t, "to_local") else t
                weights_with_local.append((name, t, local_t))

            total_numel = sum(local.numel() for _, _, local in weights_with_local)

            # Build full flattened CPU tensor (temporary), then extract shard
            full_cpu = torch.empty(total_numel, dtype=dtype, device="cpu")

            current_offset = 0
            for name, original_tensor, local_tensor in weights_with_local:
                numel = local_tensor.numel()
                full_cpu[current_offset : current_offset + numel].copy_(local_tensor.flatten())
                if dtype not in dtype_metadata:
                    dtype_metadata[dtype] = []
                dtype_metadata[dtype].append(
                    {
                        "name": name,
                        "offset": current_offset,
                        "numel": numel,
                        "shape": local_tensor.shape,
                    }
                )
                # Replace original tensor with placeholder
                DistributedLayerwiseOffloadHook._set_tensor_storage(
                    original_tensor,
                    DistributedLayerwiseOffloadHook._make_offload_placeholder(original_tensor),
                )
                current_offset += numel

            # Split into shards and keep only this rank's shard
            shard_size = total_numel // dp_size
            remainder = total_numel % dp_size
            # Distribute remainder across first `remainder` ranks
            if remainder > 0 and rank < remainder:
                shard_start = rank * (shard_size + 1)
                shard_end = shard_start + shard_size + 1
            else:
                base_offset = remainder * (shard_size + 1)
                shard_start = base_offset + (rank - remainder) * shard_size
                shard_end = shard_start + shard_size

            shard = full_cpu[shard_start:shard_end].clone()
            if pin_memory:
                shard = shard.pin_memory()
            # Free the full tensor; only the shard survives
            del full_cpu

            cpu_shards[dtype] = shard

        return cpu_shards, dtype_metadata

    def _allocate_device_buffers(self) -> None:
        """Pre-allocate exactly two device buffers (one per slot)."""
        for slot in range(2):
            gpu_weights: dict[torch.dtype, torch.Tensor] = {}
            for dtype, metas in self.metadata.items():
                total_numel = sum(m["numel"] for m in metas)
                gpu_weights[dtype] = torch.empty(total_numel, dtype=dtype, device=self.device)
            self.gpu_buffers[slot] = gpu_weights

    @property
    def is_materialized(self) -> bool:
        """Check whether this block's parameters hold real data on device."""
        for param in self.block_parameters.values():
            return DistributedLayerwiseOffloadHook._is_materialized_tensor(param)
        return True

    # ------------------------------------------------------------------ #
    #  Prefetch: H2D + AllGather (overlapped on dedicated streams)      #
    # ------------------------------------------------------------------ #

    @torch.compiler.disable
    def prefetch_layer(self, slot: int, non_blocking: bool = True) -> None:
        """Prepare next block's weights into the given slot.

        H2D runs on copy_stream, AllGather runs on comm_stream (waits for H2D).
        Both overlap with the compute stream. A ready event is recorded on
        the comm_stream after AllGather completes.
        """
        # --- Stage 1: H2D (local shard: host -> device) on copy_stream ---
        self.copy_stream.wait_stream(current_omni_platform.current_stream())

        gpu_shards: dict[torch.dtype, torch.Tensor] = {}
        evt_h2d = current_omni_platform.Event()

        with current_omni_platform.stream(self.copy_stream):
            for dtype, cpu_shard in self.cpu_shards.items():
                gpu_shard = torch.empty(cpu_shard.shape, dtype=dtype, device=self.device)
                gpu_shard.copy_(cpu_shard, non_blocking=non_blocking)
                gpu_shards[dtype] = gpu_shard
            evt_h2d.record(self.copy_stream)

        # --- Stage 2: AllGather on comm_stream (waits for H2D) ---
        self.comm_stream.wait_stream(self.copy_stream)

        evt = current_omni_platform.Event()

        with current_omni_platform.stream(self.comm_stream):
            for dtype, local_shard in gpu_shards.items():
                full_buffer = self.gpu_buffers[slot][dtype]

                if self.dp_size > 1 and self.dp_group is not None:
                    # AllGather: reconstruct full weights from all ranks
                    gathered = [torch.empty_like(local_shard) for _ in range(self.dp_size)]
                    work = torch.distributed.all_gather(
                        gathered,
                        local_shard,
                        group=self.dp_group,
                        async_op=True,
                    )
                    if work is not None:
                        work.wait()

                    # Concatenate gathered shards into the full buffer
                    offset = 0
                    for shard in gathered:
                        n = shard.numel()
                        full_buffer[offset : offset + n].copy_(shard)
                        offset += n
                else:
                    # Single-rank: just copy the shard (which is the full weight)
                    full_buffer.copy_(local_shard)

            evt.record(self.comm_stream)

        self.ready_events[slot] = evt
        self._prefetch_done = evt

        # Re-point next block's parameters to the device buffer slices
        # (must happen after the AllGather event is recorded, so the
        # compute stream will wait via get_weights before using them)
        for dtype, ordered_metadata in self.metadata.items():
            gpu_weight = self.gpu_buffers[slot][dtype]
            for metadata in ordered_metadata:
                target_name = metadata["name"]
                target = (
                    self.next_block_parameters[target_name]
                    if target_name in self.next_block_parameters
                    else self.next_block_buffers[target_name]
                )
                DistributedLayerwiseOffloadHook._set_tensor_storage(
                    target,
                    gpu_weight[metadata["offset"] : metadata["offset"] + metadata["numel"]].view(
                        metadata["shape"]
                    ),
                )

    def get_weights(self, slot: int) -> dict[torch.dtype, torch.Tensor] | None:
        """Wait for AllGather completion and return full weights for the slot."""
        evt = self.ready_events[slot]
        if evt is not None:
            current_omni_platform.current_stream().wait_event(evt)
        return self.gpu_buffers[slot]

    # ------------------------------------------------------------------ #
    #  Offload: free device memory for current block                     #
    # ------------------------------------------------------------------ #

    @torch.compiler.disable
    def offload_layer(self) -> None:
        """Free GPU memory for current block by replacing tensors with placeholders."""
        evt = self._prefetch_done
        if evt is not None:
            current_omni_platform.current_stream().wait_event(evt)
        self._prefetch_done = None

        for _, param in self.block_parameters.items():
            DistributedLayerwiseOffloadHook._set_tensor_storage(
                param, DistributedLayerwiseOffloadHook._make_offload_placeholder(param)
            )
        for _, buf in self.block_buffers.items():
            DistributedLayerwiseOffloadHook._set_tensor_storage(
                buf, DistributedLayerwiseOffloadHook._make_offload_placeholder(buf)
            )

    # ------------------------------------------------------------------ #
    #  ModelHook interface                                                #
    # ------------------------------------------------------------------ #

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        # If previous hook was skipped (e.g. by cache-dit), sync-prefetch this block
        if not self.is_materialized and self._prev_hook is not None:
            self._prev_hook.prefetch_layer(self.current_slot, non_blocking=False)
            self._prev_hook.get_weights(self.current_slot)

        # Ensure current block's weights are ready (wait on ready event)
        self.get_weights(self.current_slot)

        # Prefetch next layer into the other slot (overlapped with compute)
        next_slot = 1 - self.current_slot
        self.prefetch_layer(next_slot, non_blocking=True)

        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        self.offload_layer()
        # Swap slots: next becomes current
        self.current_slot = 1 - self.current_slot
        return output


# ---------------------------------------------------------------------- #
#  Module-level helpers                                                   #
# ---------------------------------------------------------------------- #


def apply_distributed_block_hook(
    module: nn.Module,
    next_block: nn.Module,
    device: torch.device,
    dp_group: torch.distributed.ProcessGroup | None,
    dp_size: int,
    rank: int,
    copy_stream: Any | None = None,
    comm_stream: Any | None = None,
    pin_memory: bool = True,
) -> DistributedLayerwiseOffloadHook:
    """Register a DistributedLayerwiseOffloadHook on *module*."""
    registry = HookRegistry.get_or_create(module)
    hook = DistributedLayerwiseOffloadHook(
        next_block=next_block,
        device=device,
        dp_group=dp_group,
        dp_size=dp_size,
        rank=rank,
        copy_stream=copy_stream,
        comm_stream=comm_stream,
        pin_memory=pin_memory,
    )
    registry.register_hook(DistributedLayerwiseOffloadHook._HOOK_NAME, hook)
    return hook


def remove_distributed_block_hook(module: nn.Module) -> None:
    """Remove the distributed layerwise offload hook from *module*."""
    registry: HookRegistry | None = getattr(module, "_hook_registry", None)
    if registry is not None:
        registry.remove_hook(DistributedLayerwiseOffloadHook._HOOK_NAME)
        logger.debug("Removed distributed offload hook from %s", module.__class__.__name__)


# ---------------------------------------------------------------------- #
#  Backend                                                                #
# ---------------------------------------------------------------------- #


class DistributedLayerwiseOffloadBackend(OffloadBackend):
    """Distributed layer-wise (block-level) offloading backend.

    Supports both GPU (CUDA) and NPU (CANN) platforms.
    Device type is determined by the device passed to enable().

    Each rank stores only a shard of each block's weights on host memory.
    Two device slots alternate: one holds current weights, one holds next
    weights. H2D and AllGather run asynchronously on dedicated streams,
    overlapped with computation.
    """

    def __init__(self, config: OffloadConfig, device: torch.device):
        super().__init__(config, device)

        self.copy_stream = current_omni_platform.Stream()
        self.comm_stream = current_omni_platform.Stream()
        self.dp_group: torch.distributed.ProcessGroup | None = None
        self.dp_size = config.dp_size
        self.rank = 0
        self._blocks: list[list[nn.Module]] = []

    def _init_dp_group(self) -> None:
        """Create the DP process group with auto-detected backend."""
        if self.dp_size <= 1:
            logger.info("Distributed layerwise offload: dp_size=1, running without AllGather")
            self.dp_group = None
            return

        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "torch.distributed is not initialized. "
                "Distributed layerwise offload with dp_size > 1 requires "
                "a initialized process group."
            )

        # Auto-detect communication backend based on device type
        if self.device.type == "cuda":
            backend = "nccl"
        elif self.device.type == "npu":
            backend = "hccl"
        else:
            backend = "gloo"

        world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()

        if world_size < self.dp_size:
            raise ValueError(
                f"World size ({world_size}) is smaller than dp_size ({self.dp_size}). "
                "Reduce dp_size or increase the number of devices."
            )

        # Create DP group with the first dp_size ranks
        ranks = list(range(self.dp_size))
        self.dp_group = torch.distributed.new_group(ranks=ranks, backend=backend)

        logger.info(
            "Distributed layerwise offload: dp_size=%d, rank=%d, backend=%s",
            self.dp_size,
            self.rank,
            backend,
        )

    def enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("DistributedLayerwiseOffloadBackend already enabled")
            return

        # Initialize DP group
        self._init_dp_group()

        modules = ModuleDiscovery.discover(pipeline)
        if not modules.dits:
            logger.warning("No DiT/transformer modules found, skipping distributed layer-wise offloading")
            return

        # Move encoders to GPU (they stay resident)
        for enc in modules.encoders:
            enc.to(self.device)

        # Move VAE(s) to GPU if available
        for vae in modules.vaes:
            try:
                vae.to(self.device, non_blocking=True)
            except Exception as exc:
                logger.debug("Failed to move VAE to GPU: %s", exc)

        # Move resident modules to GPU (small modules needed every forward)
        for name, module in zip(modules.resident_names, modules.resident_modules):
            try:
                module.to(self.device)
            except Exception as exc:
                logger.debug("Failed to move resident module %s to GPU: %s", name, exc)

        logger.info("Applying distributed layer-wise offloading on %s", modules.dit_names)

        # Apply hooks for each DiT module
        for i, dit_module in enumerate(modules.dits):
            dit_name = modules.dit_names[i]
            logger.info(f"Applying hooks on {dit_name} ({dit_module.__class__.__name__})")

            blocks_attr_names, blocks = DistributedLayerwiseOffloadBackend.get_blocks_from_dit(
                dit_module
            )

            if not blocks:
                logger.warning(
                    "Target layers (blocks) not found. Skipping offloading on %s (%s)",
                    dit_name,
                    dit_module.__class__.__name__,
                )
                dit_module.to(self.device)
                continue

            num_blocks = len(blocks)
            if num_blocks <= 1:
                logger.warning(
                    "#Target layers (blocks) <= 1. Skipping offloading on %s (%s)",
                    dit_name,
                    dit_module.__class__.__name__,
                )
                dit_module.to(self.device)
                continue

            # Move non-block modules to GPU (they stay resident)
            for name, m in dit_module.named_children():
                if name not in blocks_attr_names:
                    m.to(self.device)
                    logger.debug(f"Moved {name} to device {self.device}")
                else:
                    logger.debug(f"Skipped blocks module {name}")

            # Move top-level params/buffers to GPU
            for param in dit_module._parameters.values():
                if param is not None:
                    param.data = param.data.to(self.device, non_blocking=True)
            for buffer in dit_module._buffers.values():
                if buffer is not None:
                    buffer.data = buffer.data.to(self.device, non_blocking=True)

            # Register hooks in a circular sliding window:
            # last block prefetches first block, block i prefetches block (i+1)
            last_block, first_block = blocks[-1], blocks[0]
            last_hook = apply_distributed_block_hook(
                last_block,
                first_block,
                self.device,
                self.dp_group,
                self.dp_size,
                self.rank,
                self.copy_stream,
                self.comm_stream,
                self.config.pin_cpu_memory,
            )
            # Manually prefetch first block (synchronous) so it's ready for the first forward
            last_hook.prefetch_layer(slot=0, non_blocking=False)
            last_hook.get_weights(slot=0)

            block_hooks: list[DistributedLayerwiseOffloadHook] = [last_hook]
            for i, block in enumerate(blocks[:-1]):
                next_block = blocks[(i + 1) % num_blocks]
                hook = apply_distributed_block_hook(
                    block,
                    next_block,
                    self.device,
                    self.dp_group,
                    self.dp_size,
                    self.rank,
                    self.copy_stream,
                    self.comm_stream,
                    self.config.pin_cpu_memory,
                )
                block_hooks.append(hook)

            # Wire backward references for cache-dit fallback
            for i in range(len(block_hooks)):
                block_hooks[i]._prev_hook = block_hooks[i - 1]

            logger.info(
                f"Distributed layer-wise offloading enabled on {num_blocks} layers (blocks), "
                f"dp_size={self.dp_size}"
            )

            self._blocks.append(blocks)

        if len(self._blocks) > 0 and len(self._blocks[0]) > 0:
            self.enabled = True

    def disable(self) -> None:
        if not self.enabled:
            return

        for blocks in self._blocks:
            for block in blocks:
                remove_distributed_block_hook(block)

        self._blocks.clear()
        self.enabled = False
        logger.info("Distributed layer-wise offloading disabled")

    # ------------------------------------------------------------------ #
    #  Block discovery (reuses LayerWiseOffloadBackend logic)            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_blocks_attr_names(model: nn.Module) -> list[str]:
        """Get block attribute names from model class."""
        attrs: list[str] = getattr(model.__class__, "_layerwise_offload_blocks_attrs", [])

        if not attrs:
            old_attr = getattr(model.__class__, "_layerwise_offload_blocks_attr", None)
            if old_attr is not None:
                logger.warning(
                    "'_layerwise_offload_blocks_attr' is deprecated, "
                    "please use '_layerwise_offload_blocks_attrs' instead. "
                    "Example: _layerwise_offload_blocks_attrs = ['blocks']"
                )
                attrs = [old_attr] if isinstance(old_attr, str) else list(old_attr)

        return attrs

    @staticmethod
    def set_blocks_attr_names(model: nn.Module, names: list[str]) -> None:
        if not hasattr(model.__class__, "_layerwise_offload_blocks_attrs"):
            setattr(model.__class__, "_layerwise_offload_blocks_attrs", names)

    @staticmethod
    def get_blocks_from_dit(model: nn.Module) -> tuple[list[str], list[nn.Module]]:
        """Retrieve blocks and attribute names from provided DiT model."""
        blocks_attr_names = DistributedLayerwiseOffloadBackend.get_blocks_attr_names(model)
        if not blocks_attr_names:
            logger.warning(
                f"No _layerwise_offload_blocks_attrs defined for {model.__class__.__name__}, "
                "skipping distributed layerwise offloading"
            )
            return [], []

        blocks: list[nn.Module] = []
        for name in blocks_attr_names:
            attr = getattr(model, name, None)
            if attr is None:
                raise AttributeError(
                    f"Attribute '{name}' declared in _layerwise_offload_blocks_attrs "
                    f"does not exist on model {model.__class__.__name__}"
                )
            try:
                attr_iter = iter(attr)
            except TypeError:
                if isinstance(attr, nn.Module):
                    logger.warning(
                        "Attribute '%s' on %s is not iterable; treating it as one block.",
                        name,
                        model.__class__.__name__,
                    )
                    blocks.append(attr)
                    continue

                logger.warning(
                    "Attribute '%s' on %s is not iterable (got %s); skipping it.",
                    name,
                    model.__class__.__name__,
                    type(attr).__name__,
                )
            else:
                blocks.extend(attr_iter)

        if not blocks:
            logger.warning(
                "No blocks found in %s for %s, skipping distributed layerwise offloading",
                blocks_attr_names,
                model.__class__.__name__,
            )
            return [], []

        return blocks_attr_names, blocks
