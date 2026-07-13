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


def _dtype_size(dtype: torch.dtype) -> int:
    """Return element size in bytes for a torch.dtype."""
    return torch.empty(1, dtype=dtype).element_size()


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
        shared_buffers: list[dict[torch.dtype, torch.Tensor] | None] | None = None,
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

        # Double buffers: either shared (from backend) or self-allocated (lazy)
        if shared_buffers is not None:
            self.gpu_buffers: list[dict[torch.dtype, torch.Tensor] | None] = shared_buffers
            self._owns_buffers = False
        else:
            self.gpu_buffers = [None, None]
            self._owns_buffers = True
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

        # Allocate device buffers only when this hook owns its buffers
        # (standalone usage).  When the backend passes shared_buffers
        # (even [None, None]), _owns_buffers is False and allocation is
        # deferred to the backend's unified _allocate_shared_buffers.
        if self._owns_buffers:
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

            # Equal-sized shards (ceil division) for all_gather_into_tensor
            # compatibility.  Ranks whose shard extends past total_numel are
            # zero-padded.
            shard_size = (total_numel + dp_size - 1) // dp_size  # ceil
            shard_start = rank * shard_size
            shard_end = min(shard_start + shard_size, total_numel)

            # Allocate ONLY the shard (1/dp_size of the block, padded to ceil)
            shard = torch.zeros(shard_size, dtype=dtype, device="cpu")

            current_offset = 0
            for name, original_tensor, local_tensor in weights_with_local:
                numel = local_tensor.numel()
                if dtype not in dtype_metadata:
                    dtype_metadata[dtype] = []
                # Offsets remain relative to the FULL flattened buffer
                # (needed for correct AllGather reconstruction).
                dtype_metadata[dtype].append(
                    {
                        "name": name,
                        "offset": current_offset,
                        "numel": numel,
                        "shape": local_tensor.shape,
                    }
                )

                # Copy ONLY the portion of this param that falls within
                # this rank's shard range [shard_start, shard_end).
                overlap_start = max(current_offset, shard_start)
                overlap_end = min(current_offset + numel, shard_end)
                if overlap_start < overlap_end:
                    src_start = overlap_start - current_offset
                    src_end = overlap_end - current_offset
                    dst_start = overlap_start - shard_start
                    dst_end = overlap_end - shard_start
                    shard[dst_start:dst_end].copy_(
                        local_tensor.flatten()[src_start:src_end]
                    )

                # Replace original tensor with placeholder (frees CPU storage)
                DistributedLayerwiseOffloadHook._set_tensor_storage(
                    original_tensor,
                    DistributedLayerwiseOffloadHook._make_offload_placeholder(original_tensor),
                )
                current_offset += numel

            if pin_memory:
                shard = shard.pin_memory()

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

        evt = current_omni_platform.Event()

        if self.dp_size <= 1 or self.dp_group is None:
            # Fast path for single-rank: H2D directly into shared buffer
            with current_omni_platform.stream(self.copy_stream):
                for dtype, cpu_shard in self.cpu_shards.items():
                    full_buffer = self.gpu_buffers[slot][dtype]
                    full_buffer.copy_(cpu_shard, non_blocking=non_blocking)
                evt.record(self.copy_stream)
        else:
            # Multi-rank: H2D to device shard, then AllGather into shared buffer
            # Key optimization: use all_gather_into_tensor (single collective,
            # no intermediate list, no manual concatenation loop)
            # async_op=True so the Python thread doesn't block, allowing H2D
            # of the next layer to overlap with this AllGather + compute.
            gpu_shards: dict[torch.dtype, torch.Tensor] = {}
            with current_omni_platform.stream(self.copy_stream):
                for dtype, cpu_shard in self.cpu_shards.items():
                    gpu_shard = torch.empty(
                        cpu_shard.shape, dtype=dtype, device=self.device
                    )
                    gpu_shard.copy_(cpu_shard, non_blocking=non_blocking)
                    gpu_shards[dtype] = gpu_shard

            self.comm_stream.wait_stream(self.copy_stream)
            with current_omni_platform.stream(self.comm_stream):
                for dtype, local_shard in gpu_shards.items():
                    full_buffer = self.gpu_buffers[slot][dtype]
                    total_numel = sum(m["numel"] for m in self.metadata[dtype])
                    torch.distributed.all_gather_into_tensor(
                        full_buffer[:total_numel],
                        local_shard,
                        group=self.dp_group,
                        async_op=True,
                    )
                evt.record(self.comm_stream)

        self.ready_events[slot] = evt
        self._prefetch_done = evt

        # Re-point next block's parameters to the device buffer slices
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
    shared_buffers: list[dict[torch.dtype, torch.Tensor] | None] | None = None,
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
        shared_buffers=shared_buffers,
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
        self._all_hook_groups: list[list[DistributedLayerwiseOffloadHook]] = []

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

    def _register_on_demand_hook(self, module: nn.Module, label: str) -> None:
        """Register hooks that move *module* to GPU before forward and back to
        CPU after forward.  Keeps large non-block modules (VAE, encoders) off
        GPU during the DiT compute loop.
        """
        device = self.device
        pre_handles = []
        post_handles = []

        def _pre_forward(mod, args):
            mod.to(device)

        def _post_forward(mod, args, output):
            mod.to("cpu")
            current_omni_platform.synchronize()
            current_omni_platform.empty_cache()

        pre_handles.append(module.register_forward_pre_hook(_pre_forward))
        post_handles.append(module.register_forward_hook(_post_forward))
        if not hasattr(self, "_on_demand_handles"):
            self._on_demand_handles = []
        self._on_demand_handles.extend(pre_handles + post_handles)

    def _try_layerwise_offload_submodule(self, module: nn.Module, name: str) -> bool:
        """Try to apply layerwise offload to a large submodule's blocks.

        Searches for common block-list attributes (layers, blocks, h).
        If found, applies the same distributed layerwise streaming hooks
        used for the DiT, so only 2 layers reside on GPU at a time.
        Returns True if layerwise offload was applied, False otherwise.
        """
        from operator import attrgetter
        blocks = None
        blocks_attr = None
        for attr_name in ("layers", "blocks", "h", "model.layers"):
            try:
                candidate = attrgetter(attr_name)(module)
            except AttributeError:
                continue
            if isinstance(candidate, nn.ModuleList) and len(candidate) > 1:
                blocks = candidate
                blocks_attr = attr_name
                break

        if blocks is None:
            return False

        num_blocks = len(blocks)
        logger.info(
            "Distributed layerwise offload for submodule '%s.%s' (%d blocks, %.0f MB total, dp_size=%d)",
            name, blocks_attr, num_blocks,
            sum(p.nelement() * p.element_size() for p in module.parameters()) / 1048576,
            self.dp_size,
        )

        # Move non-block parts of the submodule to GPU (small: embeddings, norms)
        for child_name, child in module.named_children():
            if child_name != blocks_attr:
                child.to(self.device)

        # Apply distributed hooks with deferred buffer allocation
        last_block, first_block = blocks[-1], blocks[0]
        last_hook = apply_distributed_block_hook(
            last_block, first_block, self.device,
            self.dp_group, self.dp_size, self.rank,
            self.copy_stream, self.comm_stream,
            self.config.pin_cpu_memory,
            shared_buffers=[None, None],
        )
        sub_hooks = [last_hook]
        for i, block in enumerate(blocks[:-1]):
            next_block = blocks[(i + 1) % num_blocks]
            hook = apply_distributed_block_hook(
                block, next_block, self.device,
                self.dp_group, self.dp_size, self.rank,
                self.copy_stream, self.comm_stream,
                self.config.pin_cpu_memory,
                shared_buffers=[None, None],
            )
            sub_hooks.append(hook)

        # Wire backward references + slot alternation
        for i in range(len(sub_hooks)):
            sub_hooks[i]._prev_hook = sub_hooks[i - 1]
        for i, hook in enumerate(sub_hooks):
            hook.current_slot = i % 2

        # Defer buffer allocation and prefetch to enable() unified allocation
        self._all_hook_groups.append(sub_hooks)
        self._blocks.append(blocks)
        return True

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

        # Keep VAE/encoders on CPU; move to GPU on-demand via hooks.
        # This saves several GB HBM per card during the DiT forward pass.
        # They are only needed briefly for text-encoding (before DiT) and
        # VAE-decode (after DiT).
        for enc in modules.encoders:
            self._register_on_demand_hook(enc, "encoder")
        for vae in modules.vaes:
            self._register_on_demand_hook(vae, "vae")

        # Move resident modules to GPU (small modules needed every forward)
        for name, module in zip(modules.resident_names, modules.resident_modules):
            try:
                module.to(self.device)
            except Exception as exc:
                logger.debug("Failed to move resident module %s to GPU: %s", name, exc)

        logger.info("Applying distributed layer-wise offloading on %s", modules.dit_names)

        # Collect ALL hooks across ALL module groups so that a single pair of
        # shared buffers can be allocated at the end (unified sharing).
        all_block_hooks: list[DistributedLayerwiseOffloadHook] = []

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

            # Non-block modules: small ones go to GPU; large ones (> 1 GB)
            # stay on CPU with on-demand hooks (moved to GPU only during their
            # forward pass, then back to CPU).  This includes language_model
            # (~60 GB) and any other large submodule.
            _ON_DEMAND_THRESHOLD = 1024  # MB
            for name, m in dit_module.named_children():
                if name not in blocks_attr_names:
                    _mb = sum(p.nelement() * p.element_size() for p in m.parameters()) / 1048576
                    if _mb > _ON_DEMAND_THRESHOLD:
                        logger.info(
                            "Submodule '%s' (%.0f MB) > %d MB threshold; "
                            "keeping on CPU with on-demand hook",
                            name, _mb, _ON_DEMAND_THRESHOLD,
                        )
                        self._register_on_demand_hook(m, name)
                    else:
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

            # Create hooks with deferred buffer allocation — pass
            # shared_buffers=[None, None] so _owns_buffers=False and
            # initialize_hook skips _allocate_device_buffers().  The
            # backend assigns unified shared buffers after all hooks
            # across all module groups are collected.
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
                shared_buffers=[None, None],
            )

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
                    shared_buffers=[None, None],
                )
                block_hooks.append(hook)

            # Wire backward references for cache-dit fallback
            for i in range(len(block_hooks)):
                block_hooks[i]._prev_hook = block_hooks[i - 1]

            # Initialize alternating slots: even-indexed hooks use slot 0 first,
            # odd-indexed use slot 1. This ensures consecutive blocks never
            # collide on the same shared buffer slot during prefetch.
            for i, hook in enumerate(block_hooks):
                hook.current_slot = i % 2

            all_block_hooks.extend(block_hooks)
            self._blocks.append(blocks)

        if not all_block_hooks:
            return

        # Allocate 2 unified shared buffers sized to the largest block across
        # ALL module groups (not per-group).  This halves peak HBM when there
        # are multiple DiT modules (e.g. 4 buffers → 2).
        shared_buffers = self._allocate_shared_buffers(all_block_hooks)
        for hook in all_block_hooks:
            hook.gpu_buffers = shared_buffers
            hook._owns_buffers = False

        # Prefetch first block of EACH module group (synchronous) so it is
        # ready before the group's first forward pass.
        hook_offset = 0
        for blocks in self._blocks:
            num_blocks = len(blocks)
            group_hooks = all_block_hooks[hook_offset : hook_offset + num_blocks]
            first_slot = group_hooks[0].current_slot
            group_hooks[-1].prefetch_layer(slot=first_slot, non_blocking=False)
            group_hooks[-1].get_weights(first_slot)
            hook_offset += num_blocks

        total_blocks = sum(len(b) for b in self._blocks)
        logger.info(
            f"Distributed layer-wise offloading enabled on {total_blocks} blocks across "
            f"{len(self._blocks)} module group(s), dp_size={self.dp_size}, "
            f"shared_buffers=2 (unified across all groups)"
        )

        self.enabled = True

    def disable(self) -> None:
        if not self.enabled:
            return

        for blocks in self._blocks:
            for block in blocks:
                remove_distributed_block_hook(block)

        for h in getattr(self, "_on_demand_handles", []):
            h.remove()
        self._on_demand_handles = []

        self._blocks.clear()
        self._all_hook_groups.clear()
        self.enabled = False
        logger.info("Distributed layer-wise offloading disabled")

    @staticmethod
    def _allocate_shared_buffers(
        hooks: list[DistributedLayerwiseOffloadHook],
    ) -> list[dict[torch.dtype, torch.Tensor] | None]:
        """Allocate exactly 2 shared device buffers sized to the largest block.

        All hooks share these 2 buffers. At any time, slot 0 holds the current
        layer's weights and slot 1 holds the next layer's weights (or vice
        versa). This ensures only 2 layers' worth of weights reside on device,
        regardless of the total number of blocks.
        """
        max_sizes: dict[torch.dtype, int] = {}
        for hook in hooks:
            for dtype, metas in hook.metadata.items():
                total = sum(m["numel"] for m in metas)
                # AllGather output = dp * ceil(total/dp) (padded for equal shards)
                if hook.dp_size > 1:
                    total = ((total + hook.dp_size - 1) // hook.dp_size) * hook.dp_size
                if dtype not in max_sizes or total > max_sizes[dtype]:
                    max_sizes[dtype] = total

        device = hooks[0].device
        shared_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]
        for slot in range(2):
            gpu_weights: dict[torch.dtype, torch.Tensor] = {}
            for dtype, total_numel in max_sizes.items():
                gpu_weights[dtype] = torch.empty(total_numel, dtype=dtype, device=device)
            shared_buffers[slot] = gpu_weights

        logger.info(
            "Allocated 2 shared device buffers (max block size: %s)",
            {str(k): f"{v * _dtype_size(k) / 1024 / 1024:.1f}MB" for k, v in max_sizes.items()},
        )
        return shared_buffers

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
