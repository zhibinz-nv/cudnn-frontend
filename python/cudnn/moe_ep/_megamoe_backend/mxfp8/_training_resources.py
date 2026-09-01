# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Slotless execution resources and caller-owned MXFP8 training contexts."""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist

from ..._contracts import ForwardConfig
from ..._types import (
    MoeEpBufferLifetime,
    MoeEpTrainingBufferSpec,
    MoeEpTrainingBufferSpecs,
    MoeEpTrainingContext,
    MoeEpTrainingWeights,
)
from .._comm import SymmetricMemoryProvider
from .._plan import PreparedResources
from .._runtime import (
    RuntimeHandle,
    RuntimeManager,
    _RuntimeWatchdog,
    _runtime_debug,
    get_runtime_manager,
)
from .._workspace import (
    BufferRegion,
    LocalMemoryProvider,
    WorkspaceOwner,
    WorkspaceRequirements,
    WorkspaceViews,
)
from ._adapter import _typed_view
from ._backward_compile import PreparedMxfp8BackwardKernel
from ._compile import PreparedMxfp8Kernel
from ._fingerprint import canonical_json_sha256, source_tree_sha256
from ._training_stage import Mxfp8TrainingStager
from ._training_weights import Mxfp8TrainingWeightBindings
from ._training_wgrad import Mxfp8TrainingWgradExporter

_DATA_DTYPE = torch.float8_e4m3fn
_SCALE_DTYPE = torch.float8_e8m0fnu

# These prepared-workspace ports are backed by caller context tensors. Every
# other port is launch-local state and is duplicated only per execution lane.
_FORWARD_EXTERNAL_SYMMETRIC = frozenset({"topk_weights"})
_FORWARD_EXTERNAL_LOCAL = frozenset(
    {"topk_idx", "overflow_flag", "col_quant_data", "col_quant_sf"}
)
_BACKWARD_EXTERNAL_SYMMETRIC = frozenset({"topk_weights"})
_BACKWARD_EXTERNAL_LOCAL = frozenset(
    {
        "topk_idx",
        "overflow_flag",
        "backward_fc1_preact",
        "backward_aux_data",
        "backward_aux_scale",
    }
)


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _align_scale_columns(token_capacity: int) -> int:
    return _round_up((token_capacity + 31) // 32, 4)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    value = 1
    for extent in reversed(shape):
        stride.append(value)
        value *= max(int(extent), 1)
    return tuple(reversed(stride))


def _spec(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    *,
    alignment: int,
    lifetime: MoeEpBufferLifetime | str,
    stride: tuple[int, ...] | None = None,
) -> MoeEpTrainingBufferSpec:
    shape = tuple(int(extent) for extent in shape)
    return MoeEpTrainingBufferSpec(
        shape=shape,
        stride=_contiguous_stride(shape) if stride is None else tuple(stride),
        dtype=dtype,
        device=torch.device(device),
        alignment=int(alignment),
        lifetime=MoeEpBufferLifetime(lifetime),
        capture_pinned=True,
    )


def build_training_buffer_specs(
    config: ForwardConfig,
    forward: PreparedMxfp8Kernel,
    backward: PreparedMxfp8BackwardKernel,
    device: torch.device,
) -> MoeEpTrainingBufferSpecs:
    """Describe every tensor a TE-side fixed-capacity context must provide."""

    if not config.generate_c:
        raise ValueError("slotless training requires generate_c=True")
    if forward.pool_token_capacity != backward.pool_token_capacity:
        raise ValueError(
            "forward/backward pool capacities must match, got "
            f"{forward.pool_token_capacity} and {backward.pool_token_capacity}"
        )
    device = torch.device(device)
    capacity = int(config.max_tokens_per_rank)
    pool_rows = int(forward.pool_token_capacity)
    fwd_shapes = {
        name: tuple(int(extent) for extent in shape)
        for name, shape in forward.kernel.get_aux_output_shapes().items()
    }
    bwd_shapes = {
        name: tuple(int(extent) for extent in shape)
        for name, shape in backward.kernel.get_aux_output_shapes().items()
    }
    scale_columns = _align_scale_columns(pool_rows)
    forward_context = {
        "routing_topk_idx": _spec(
            (capacity, config.top_k),
            torch.int32,
            device,
            alignment=16,
            lifetime="normal_backward",
        ),
        "routing_topk_weights": _spec(
            (capacity, config.top_k),
            torch.float32,
            device,
            alignment=16,
            lifetime="normal_backward",
        ),
        "fc1_preact": _spec(
            fwd_shapes["fc1_c"],
            torch.bfloat16,
            device,
            alignment=128,
            lifetime="normal_backward",
        ),
        # Grouped WGrad consumes x.T. The forward kernel writes the transpose
        # view, whose K mode is unit-stride.
        "fc1_a": _spec(
            (config.hidden_size, pool_rows),
            _DATA_DTYPE,
            device,
            alignment=128,
            lifetime="delayed_wgrad",
        ),
        "fc1_a_scale_compact": _spec(
            fwd_shapes["col_quant_sf"],
            torch.uint8,
            device,
            alignment=16,
            lifetime="normal_backward",
        ),
        "forward_overflow": _spec(
            (1,),
            torch.int32,
            device,
            alignment=16,
            lifetime="overflow_finalize",
        ),
        "backward_overflow": _spec(
            (1,),
            torch.int32,
            device,
            alignment=16,
            lifetime="overflow_finalize",
        ),
    }
    wgrad_gradients = {
        "valid_route_counts": _spec(
            (config.experts_per_rank,),
            torch.int32,
            device,
            alignment=16,
            lifetime="delayed_wgrad",
        ),
        "expert_offsets": _spec(
            (config.experts_per_rank,),
            torch.int32,
            device,
            alignment=16,
            lifetime="delayed_wgrad",
        ),
        "fc1_recompute": _spec(
            bwd_shapes["fc1_recompute"],
            _DATA_DTYPE,
            device,
            alignment=128,
            lifetime="normal_backward",
        ),
        "fc1_recompute_sf": _spec(
            bwd_shapes["fc1_recompute_sf"],
            _SCALE_DTYPE,
            device,
            alignment=128,
            lifetime="normal_backward",
        ),
        "fc1_col_output": _spec(
            bwd_shapes["fc1_col_output"],
            _DATA_DTYPE,
            device,
            alignment=128,
            lifetime="normal_backward",
        ),
        "fc1_col_output_sf": _spec(
            bwd_shapes["fc1_col_output_sf"],
            _SCALE_DTYPE,
            device,
            alignment=128,
            lifetime="normal_backward",
        ),
        "fc2_b": _spec(
            bwd_shapes["grad_y2"],
            _DATA_DTYPE,
            device,
            alignment=128,
            lifetime="delayed_wgrad",
            stride=(1, bwd_shapes["grad_y2"][0]),
        ),
        "fc2_b_scale_compact": _spec(
            bwd_shapes["grad_y2_sf"],
            torch.uint8,
            device,
            alignment=16,
            lifetime="normal_backward",
        ),
        "fc1_sfa": _spec(
            (_round_up(config.hidden_size, 128), scale_columns),
            _SCALE_DTYPE,
            device,
            alignment=128,
            lifetime="delayed_wgrad",
        ),
        "fc1_b": _spec(
            (pool_rows, 2 * config.intermediate_size),
            _DATA_DTYPE,
            device,
            alignment=128,
            lifetime="delayed_wgrad",
            stride=(1, pool_rows),
        ),
        "fc1_sfb": _spec(
            (_round_up(2 * config.intermediate_size, 128), scale_columns),
            _SCALE_DTYPE,
            device,
            alignment=128,
            lifetime="delayed_wgrad",
        ),
        "fc2_a": _spec(
            (config.intermediate_size, pool_rows),
            _DATA_DTYPE,
            device,
            alignment=128,
            lifetime="delayed_wgrad",
        ),
        "fc2_sfa": _spec(
            (_round_up(config.intermediate_size, 128), scale_columns),
            _SCALE_DTYPE,
            device,
            alignment=128,
            lifetime="delayed_wgrad",
        ),
        "fc2_sfb": _spec(
            (_round_up(config.hidden_size, 128), scale_columns),
            _SCALE_DTYPE,
            device,
            alignment=128,
            lifetime="delayed_wgrad",
        ),
    }
    outputs = {
        "forward": _spec(
            (capacity, config.hidden_size),
            torch.bfloat16,
            device,
            alignment=16,
            lifetime=MoeEpBufferLifetime.STREAM_COMPLETION,
        ),
        "grad_activation": _spec(
            (capacity, config.hidden_size),
            torch.float32,
            device,
            alignment=16,
            lifetime=MoeEpBufferLifetime.STREAM_COMPLETION,
        ),
        "dprob": _spec(
            bwd_shapes["dprob"],
            torch.float32,
            device,
            alignment=16,
            lifetime=MoeEpBufferLifetime.STREAM_COMPLETION,
        ),
        "overflow": _spec(
            (1,),
            torch.int32,
            device,
            alignment=16,
            lifetime=MoeEpBufferLifetime.STREAM_COMPLETION,
        ),
    }
    return MoeEpTrainingBufferSpecs(
        forward_context=MappingProxyType(forward_context),
        wgrad_gradients=MappingProxyType(wgrad_gradients),
        outputs=MappingProxyType(outputs),
    )


def _lane_name(lane: int, phase: str, space: str, name: str) -> str:
    return f"lane.{lane}.{phase}.{space}.{name}"


def _clone_region(name: str, region: BufferRegion) -> BufferRegion:
    return BufferRegion(name=name, nbytes=region.nbytes, alignment=region.alignment)


def _add_lane_regions(
    output: list[BufferRegion],
    requirements: WorkspaceRequirements,
    *,
    lane: int,
    phase: str,
    space: str,
    external_names: frozenset[str],
) -> None:
    regions = (
        requirements.symmetric_regions
        if space == "symmetric"
        else requirements.local_regions
    )
    for region in regions:
        if region.name in external_names:
            continue
        output.append(
            _clone_region(_lane_name(lane, phase, space, region.name), region)
        )


def build_training_workspace_requirements(
    config: ForwardConfig,
    forward: PreparedMxfp8Kernel,
    backward: PreparedMxfp8BackwardKernel,
    *,
    lane_count: int,
) -> WorkspaceRequirements:
    """Build the deterministic FE-owned layout containing lane scratch only."""

    if (
        isinstance(lane_count, bool)
        or not isinstance(lane_count, int)
        or lane_count <= 0
    ):
        raise ValueError(f"lane_count must be a positive integer, got {lane_count!r}")
    if not config.generate_c:
        raise ValueError("slotless training requires generate_c=True")
    if forward.pool_token_capacity != backward.pool_token_capacity:
        raise ValueError(
            "forward/backward pool capacities must match, got "
            f"{forward.pool_token_capacity} and {backward.pool_token_capacity}"
        )

    symmetric_regions: list[BufferRegion] = []
    local_regions: list[BufferRegion] = []
    for lane in range(lane_count):
        local_regions.extend(
            (
                BufferRegion(
                    _lane_name(lane, "finalizer", "local", "global_overflow"),
                    torch.int32.itemsize,
                    16,
                ),
                BufferRegion(
                    _lane_name(lane, "finalizer", "local", "overflow_ok"),
                    torch.bool.itemsize,
                    16,
                ),
            )
        )
        _add_lane_regions(
            symmetric_regions,
            forward.workspace_requirements,
            lane=lane,
            phase="forward",
            space="symmetric",
            external_names=_FORWARD_EXTERNAL_SYMMETRIC,
        )
        _add_lane_regions(
            local_regions,
            forward.workspace_requirements,
            lane=lane,
            phase="forward",
            space="local",
            external_names=_FORWARD_EXTERNAL_LOCAL,
        )
        _add_lane_regions(
            symmetric_regions,
            backward.workspace_requirements,
            lane=lane,
            phase="backward",
            space="symmetric",
            external_names=_BACKWARD_EXTERNAL_SYMMETRIC,
        )
        _add_lane_regions(
            local_regions,
            backward.workspace_requirements,
            lane=lane,
            phase="backward",
            space="local",
            external_names=_BACKWARD_EXTERNAL_LOCAL,
        )
    return WorkspaceRequirements(
        max_tokens_per_rank=int(config.max_tokens_per_rank),
        symmetric_regions=tuple(symmetric_regions),
        local_regions=tuple(local_regions),
    )


def _harmonize_symmetric_regions(
    requirements: WorkspaceRequirements,
    runtime: RuntimeHandle,
    device: torch.device,
) -> WorkspaceRequirements:
    """Make every peer-visible lane region identical on all EP ranks."""

    if runtime.world_size <= 1:
        return requirements
    regions = requirements.symmetric_regions
    count = torch.tensor([len(regions)], dtype=torch.int64, device=device)
    minimum_count = count.clone()
    maximum_count = count.clone()
    dist.all_reduce(minimum_count, op=dist.ReduceOp.MIN, group=runtime.group)
    dist.all_reduce(maximum_count, op=dist.ReduceOp.MAX, group=runtime.group)
    if int(minimum_count.item()) != int(maximum_count.item()):
        raise RuntimeError(
            "symmetric workspace region counts differ across EP ranks: "
            f"min={int(minimum_count.item())}, max={int(maximum_count.item())}"
        )

    metadata = "\0".join(
        f"{region.name}:{region.alignment}" for region in regions
    ).encode()
    signature_value = int.from_bytes(
        hashlib.blake2b(metadata, digest_size=8).digest(), "little"
    ) & ((1 << 63) - 1)
    signature = torch.tensor([signature_value], dtype=torch.int64, device=device)
    minimum_signature = signature.clone()
    maximum_signature = signature.clone()
    dist.all_reduce(minimum_signature, op=dist.ReduceOp.MIN, group=runtime.group)
    dist.all_reduce(maximum_signature, op=dist.ReduceOp.MAX, group=runtime.group)
    if int(minimum_signature.item()) != int(maximum_signature.item()):
        raise RuntimeError(
            "symmetric workspace region names, order, or alignments differ "
            "across EP ranks: "
            f"local_signature={signature_value}, "
            f"local_regions={tuple((r.name, r.alignment) for r in regions)}"
        )

    local_sizes = torch.tensor(
        [region.nbytes for region in regions], dtype=torch.int64, device=device
    )
    maximum_sizes = local_sizes.clone()
    dist.all_reduce(maximum_sizes, op=dist.ReduceOp.MAX, group=runtime.group)
    harmonized_sizes = tuple(int(value) for value in maximum_sizes.cpu().tolist())
    changes = tuple(
        f"{region.name}:{region.nbytes}->{size}"
        for region, size in zip(regions, harmonized_sizes)
        if region.nbytes != size
    )
    _runtime_debug(
        "training-plan.symmetric-layout-harmonized",
        region_count=len(regions),
        changed_regions=changes,
    )
    if not changes:
        return requirements
    return WorkspaceRequirements(
        max_tokens_per_rank=requirements.max_tokens_per_rank,
        symmetric_regions=tuple(
            BufferRegion(region.name, size, alignment=region.alignment)
            for region, size in zip(regions, harmonized_sizes)
        ),
        local_regions=requirements.local_regions,
    )


def _block_scaled_tensor_abi(tensor) -> dict[str, object]:
    return {
        "format": tensor.format.value,
        "axis": int(tensor.axis),
        "logical_shape": list(tensor.logical_shape),
        "data": {
            "shape": list(tensor.data.shape),
            "stride": list(tensor.data.stride()),
            "dtype": str(tensor.data.dtype),
        },
        "scale": {
            "shape": list(tensor.scale.shape),
            "stride": list(tensor.scale.stride()),
            "dtype": str(tensor.scale.dtype),
        },
    }


def _workspace_abi(requirements: WorkspaceRequirements) -> dict[str, object]:
    def regions(values) -> list[dict[str, object]]:
        return [
            {
                "name": region.name,
                "nbytes": int(region.nbytes),
                "alignment": int(region.alignment),
            }
            for region in values
        ]

    return {
        "max_tokens_per_rank": requirements.max_tokens_per_rank,
        "symmetric_regions": regions(requirements.symmetric_regions),
        "local_regions": regions(requirements.local_regions),
    }


def _buffer_specs_abi(specs: MoeEpTrainingBufferSpecs) -> dict[str, object]:
    def values(mapping: Mapping[str, MoeEpTrainingBufferSpec]):
        return {
            name: {
                "shape": list(spec.shape),
                "stride": list(spec.stride),
                "dtype": str(spec.dtype),
                "alignment": int(spec.alignment),
                "lifetime": spec.lifetime.value,
                "capture_pinned": spec.capture_pinned,
            }
            for name, spec in mapping.items()
        }

    return {
        "forward_context": values(specs.forward_context),
        "wgrad_gradients": values(specs.wgrad_gradients),
        "outputs": values(specs.outputs),
    }


def _prepared_kernel_abi(prepared) -> dict[str, object]:
    kernel = prepared.kernel
    return {
        "name": str(kernel.name()),
        "architecture": list(prepared.architecture),
        "effective_config": prepared.config.effective_config(
            prepared.launch_cluster_count
        ),
        "launch": {
            "cluster_count": int(prepared.launch_cluster_count),
            "threads_per_cta": int(kernel.threads_per_cta),
            "occupancy": int(getattr(kernel, "occupancy", 1)),
            "smem_capacity": int(getattr(kernel, "smem_capacity", 0)),
        },
        "workspace": _workspace_abi(prepared.workspace_requirements),
        "pool_token_capacity": int(prepared.pool_token_capacity),
    }


def _build_training_abi_facts(
    config: ForwardConfig,
    forward: PreparedMxfp8Kernel,
    backward: PreparedMxfp8BackwardKernel,
    weights: MoeEpTrainingWeights,
    requirements: WorkspaceRequirements,
    buffer_specs: MoeEpTrainingBufferSpecs,
    *,
    lane_count: int,
    source_tree_digest: str | None = None,
) -> dict[str, object]:
    """Return rank-independent JSON-safe facts for the slotless training ABI."""

    if source_tree_digest is None:
        source_root = Path(__file__).resolve().parents[1] / "cutedsl_src"
        source_tree_digest = source_tree_sha256(source_root)
    weight_facts = {
        name: _block_scaled_tensor_abi(getattr(weights, name))
        for name in (
            "forward_fc1",
            "forward_fc2",
            "backward_w2_transpose",
            "backward_w1_transpose",
        )
    }
    return {
        "schema_version": 2,
        "ownership_mode": "caller_context",
        "source_tree_sha256": source_tree_digest,
        "ep": {
            "size": int(config.ep_size),
            "global_ranks": list(config.ep_global_ranks),
        },
        "geometry": {
            "num_experts": int(config.num_experts),
            "experts_per_rank": int(config.experts_per_rank),
            "hidden": int(config.hidden_size),
            "intermediate": int(config.intermediate_size),
            "top_k": int(config.top_k),
            "max_tokens_per_rank": int(config.max_tokens_per_rank),
            "max_recv_size_per_rank": int(forward.config.max_recv_size_per_rank),
        },
        "policy": {
            "drop_on_overflow": bool(config.drop_on_overflow),
            "combine_format": config.combine_format,
            "output_format": config.output_format,
            "apply_topk_in_fc1": bool(config.apply_topk_in_fc1),
            "gate_up_clamp": config.gate_up_clamp,
        },
        "resources": {
            "lane_count": int(lane_count),
            "workspace": _workspace_abi(requirements),
            "context": _buffer_specs_abi(buffer_specs),
        },
        "weights": weight_facts,
        "forward_kernel": _prepared_kernel_abi(forward),
        "backward_kernel": _prepared_kernel_abi(backward),
    }


def _verify_training_abi_across_ranks(
    facts: dict[str, object],
    runtime: RuntimeHandle,
    device: torch.device,
) -> str:
    """Collectively reject rank-divergent training ABI before allocation."""

    digest = canonical_json_sha256(facts)
    if runtime.world_size <= 1:
        return digest
    digest_value = int(digest[:16], 16) & ((1 << 63) - 1)
    minimum = torch.tensor([digest_value], dtype=torch.int64, device=device)
    maximum = minimum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=runtime.group)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=runtime.group)
    if int(minimum.item()) == int(maximum.item()):
        return digest
    rank_digests: list[Any] = [None] * runtime.world_size
    dist.all_gather_object(rank_digests, digest, group=runtime.group)
    raise RuntimeError(
        "MoeEp training ABI differs across expert-parallel ranks before "
        f"workspace allocation: digests={rank_digests}, local_facts={facts}"
    )


def _tensor_span(tensor: torch.Tensor) -> tuple[int, int]:
    if tensor.numel() == 0:
        pointer = int(tensor.data_ptr())
        return pointer, pointer
    last = sum(
        (int(extent) - 1) * int(stride)
        for extent, stride in zip(tensor.shape, tensor.stride())
    )
    start = int(tensor.data_ptr())
    return start, start + (last + 1) * tensor.element_size()


def _validate_tensor(
    name: str,
    tensor: torch.Tensor,
    spec: MoeEpTrainingBufferSpec,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{name} shape must be {spec.shape}, got {tuple(tensor.shape)}"
        )
    if tuple(tensor.stride()) != spec.stride:
        raise ValueError(
            f"{name} stride must be {spec.stride}, got {tuple(tensor.stride())}"
        )
    if tensor.dtype is not spec.dtype:
        raise TypeError(f"{name} dtype must be {spec.dtype}, got {tensor.dtype}")
    if tensor.device != spec.device:
        raise ValueError(f"{name} device must be {spec.device}, got {tensor.device}")
    if tensor.data_ptr() % spec.alignment:
        raise ValueError(f"{name} data pointer must be {spec.alignment}-byte aligned")


def _context_tensors(
    context: MoeEpTrainingContext,
) -> tuple[tuple[str, torch.Tensor], ...]:
    forward = context.forward
    wgrad = context.wgrad
    return (
        ("forward.routing_topk_idx", forward.routing_topk_idx),
        ("forward.routing_topk_weights", forward.routing_topk_weights),
        ("forward.fc1_preact", forward.fc1_preact),
        ("forward.fc1_a", forward.fc1_a),
        ("forward.fc1_a_scale_compact", forward.fc1_a_scale_compact),
        ("forward.forward_overflow", forward.forward_overflow),
        ("forward.backward_overflow", forward.backward_overflow),
        ("wgrad.valid_route_counts", wgrad.valid_route_counts),
        ("wgrad.expert_offsets", wgrad.expert_offsets),
        ("wgrad.fc1_recompute", wgrad.fc1_recompute),
        ("wgrad.fc1_recompute_sf", wgrad.fc1_recompute_sf),
        ("wgrad.fc1_col_output", wgrad.fc1_col_output),
        ("wgrad.fc1_col_output_sf", wgrad.fc1_col_output_sf),
        ("wgrad.fc2_b", wgrad.fc2_b),
        ("wgrad.fc2_b_scale_compact", wgrad.fc2_b_scale_compact),
        ("wgrad.fc1_sfa", wgrad.fc1_sfa),
        ("wgrad.fc1_b", wgrad.fc1_b),
        ("wgrad.fc1_sfb", wgrad.fc1_sfb),
        ("wgrad.fc2_a", wgrad.fc2_a),
        ("wgrad.fc2_sfa", wgrad.fc2_sfa),
        ("wgrad.fc2_sfb", wgrad.fc2_sfb),
    )


@dataclass(frozen=True)
class Mxfp8TrainingExecutionViews:
    """One borrowed TE context bound to one mutable FE execution lane."""

    context: MoeEpTrainingContext
    forward: PreparedResources
    backward: PreparedResources
    forward_expert_size_snapshot: torch.Tensor | None


class Mxfp8TrainingPlanOwner:
    """Own compiled training state and lane scratch, but no microbatch context."""

    def __init__(
        self,
        config: ForwardConfig,
        device: torch.device,
        forward: PreparedMxfp8Kernel,
        backward: PreparedMxfp8BackwardKernel,
        weights: MoeEpTrainingWeights,
        *,
        lane_count: int,
        runtime_manager: Optional[RuntimeManager] = None,
        symmetric_provider: Optional[SymmetricMemoryProvider] = None,
        local_provider: Optional[LocalMemoryProvider] = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.forward_prepared = forward
        self.backward_prepared = backward
        self.weight_bindings = Mxfp8TrainingWeightBindings(weights)
        self.stager = Mxfp8TrainingStager(config.hidden_size, config.top_k)
        self.wgrad_exporter = Mxfp8TrainingWgradExporter(
            experts=config.experts_per_rank,
            hidden=config.hidden_size,
            intermediate=config.intermediate_size,
            sf_padding=backward.config.sf_padding_block,
        )
        self.beta = torch.ones(
            (config.experts_per_rank,), dtype=torch.float32, device=self.device
        )
        self.lane_count = lane_count
        self.buffer_specs = build_training_buffer_specs(
            config, forward, backward, self.device
        )
        self.requirements = build_training_workspace_requirements(
            config, forward, backward, lane_count=lane_count
        )
        self._runtime_manager = runtime_manager or get_runtime_manager()
        self._symmetric_provider = symmetric_provider
        self._local_provider = local_provider
        self._runtime: RuntimeHandle | None = None
        self._workspace: WorkspaceOwner | None = None
        self._abi_fingerprint: str | None = None
        self._closed = False
        self._lock = threading.RLock()

    @property
    def prepared(self) -> bool:
        return (
            not self._closed
            and self._runtime is not None
            and self._workspace is not None
            and self._workspace.allocated
        )

    def prepare(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("training plan is closed")
            if self.prepared:
                return
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "training plan must be prepared before CUDA graph capture"
                )
            _runtime_debug(
                "training-plan.prepare.begin",
                lane_count=self.lane_count,
                local_bytes=sum(r.nbytes for r in self.requirements.local_regions),
                symmetric_bytes=sum(
                    r.nbytes for r in self.requirements.symmetric_regions
                ),
            )
            runtime = self._runtime_manager.acquire(self.config, self.device)
            self._runtime = runtime
            try:
                watchdog = _RuntimeWatchdog("training-plan.symmetric-layout-harmonize")
                watchdog.start()
                try:
                    self.requirements = _harmonize_symmetric_regions(
                        self.requirements, runtime, self.device
                    )
                finally:
                    watchdog.close()
                if runtime.world_size > 1:
                    facts = _build_training_abi_facts(
                        self.config,
                        self.forward_prepared,
                        self.backward_prepared,
                        self.weight_bindings.weights,
                        self.requirements,
                        self.buffer_specs,
                        lane_count=self.lane_count,
                    )
                    self._abi_fingerprint = _verify_training_abi_across_ranks(
                        facts, runtime, self.device
                    )
                workspace = WorkspaceOwner(
                    self.requirements,
                    runtime,
                    symmetric_provider=self._symmetric_provider,
                    local_provider=self._local_provider,
                )
                self._workspace = workspace
                allocation_watchdog = _RuntimeWatchdog(
                    "training-plan.workspace-allocate"
                )
                allocation_watchdog.start()
                try:
                    workspace.ensure_allocated()
                finally:
                    allocation_watchdog.close()
                if runtime.world_size > 1:
                    torch.cuda.current_stream(self.device).synchronize()
                    dist.barrier(group=runtime.group)
            except Exception:
                if self._workspace is not None:
                    self._workspace.close()
                    self._workspace = None
                runtime.close()
                self._runtime = None
                raise
            _runtime_debug("training-plan.prepare.end")

    def _flat_views(self, token_count: int) -> WorkspaceViews:
        self.prepare()
        assert self._workspace is not None
        return self._workspace.views(token_count)

    def _validate_context(self, context: MoeEpTrainingContext) -> None:
        if not isinstance(context, MoeEpTrainingContext):
            raise TypeError("context must be a MoeEpTrainingContext")
        capacity = self.buffer_specs.outputs["forward"].shape[0]
        if context.token_count > capacity:
            raise ValueError(
                f"context token_count ({context.token_count}) exceeds "
                f"capacity ({capacity})"
            )
        forward_specs = self.buffer_specs.forward_context
        wgrad_specs = self.buffer_specs.wgrad_gradients
        for name, tensor in _context_tensors(context):
            group, field = name.split(".", 1)
            specs = forward_specs if group == "forward" else wgrad_specs
            _validate_tensor(name, tensor, specs[field])
        spans = sorted(
            (*_tensor_span(tensor), name)
            for name, tensor in _context_tensors(context)
            if tensor.numel()
        )
        for previous, current in zip(spans, spans[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    "training context buffers must not overlap: "
                    f"{previous[2]} and {current[2]}"
                )

    def validate_output(self, name: str, tensor: torch.Tensor) -> None:
        try:
            spec = self.buffer_specs.outputs[name]
        except KeyError as exc:
            raise ValueError(f"unknown training output {name!r}") from exc
        _validate_tensor(name, tensor, spec)

    def validate_outputs(
        self,
        context: MoeEpTrainingContext,
        outputs: Mapping[str, torch.Tensor],
    ) -> None:
        """Reject malformed outputs and aliases with a borrowed context."""

        for name, tensor in outputs.items():
            self.validate_output(name, tensor)
        tensors = (
            *_context_tensors(context),
            *((f"output.{name}", tensor) for name, tensor in outputs.items()),
        )
        spans = sorted(
            (*_tensor_span(tensor), name) for name, tensor in tensors if tensor.numel()
        )
        for previous, current in zip(spans, spans[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    "training context and output buffers must not overlap: "
                    f"{previous[2]} and {current[2]}"
                )

    @staticmethod
    def _phase_workspace(
        flat: WorkspaceViews,
        requirements: WorkspaceRequirements,
        *,
        context: MoeEpTrainingContext,
        lane: int,
        phase: str,
    ) -> WorkspaceViews:
        symmetric = {}
        local = {}
        for region in requirements.symmetric_regions:
            if region.name == "topk_weights":
                symmetric[region.name] = context.forward.routing_topk_weights
            else:
                symmetric[region.name] = flat.symmetric[
                    _lane_name(lane, phase, "symmetric", region.name)
                ]
        for region in requirements.local_regions:
            if region.name == "topk_idx":
                local[region.name] = context.forward.routing_topk_idx
            elif region.name == "overflow_flag":
                local[region.name] = (
                    context.forward.forward_overflow
                    if phase == "forward"
                    else context.forward.backward_overflow
                )
            elif region.name == "col_quant_data":
                local[region.name] = context.forward.fc1_a.transpose(0, 1)
            elif region.name == "col_quant_sf":
                local[region.name] = context.forward.fc1_a_scale_compact
            elif region.name == "backward_fc1_preact":
                local[region.name] = context.forward.fc1_preact
            elif region.name == "backward_aux_data":
                local[region.name] = context.wgrad.fc1_col_output.view(
                    torch.uint8
                ).reshape(-1)
            elif region.name == "backward_aux_scale":
                local[region.name] = context.wgrad.fc1_col_output_sf.view(
                    torch.uint8
                ).reshape(-1)
            else:
                local[region.name] = flat.local[
                    _lane_name(lane, phase, "local", region.name)
                ]
        return WorkspaceViews(
            token_count=flat.token_count,
            symmetric=MappingProxyType(symmetric),
            local=MappingProxyType(local),
            peer_mapping=flat.peer_mapping,
        )

    def views(
        self,
        *,
        context: MoeEpTrainingContext,
        lane: int,
        token_count: int,
    ) -> Mxfp8TrainingExecutionViews:
        with self._lock:
            if self._closed:
                raise RuntimeError("training plan is closed")
            if lane < 0 or lane >= self.lane_count:
                raise ValueError(f"lane {lane} is outside [0, {self.lane_count})")
            self._validate_context(context)
            if context.token_count != token_count:
                raise ValueError(
                    "context token_count does not match this call: "
                    f"context={context.token_count}, call={token_count}"
                )
            flat = self._flat_views(token_count)
            forward_workspace = self._phase_workspace(
                flat,
                self.forward_prepared.workspace_requirements,
                context=context,
                lane=lane,
                phase="forward",
            )
            backward_workspace = self._phase_workspace(
                flat,
                self.backward_prepared.workspace_requirements,
                context=context,
                lane=lane,
                phase="backward",
            )
            snapshot = None
            if self.forward_prepared.col_quant_sizes_offset is not None:
                snapshot_bytes = forward_workspace.local[
                    "kernel_local_workspace"
                ].narrow(
                    0,
                    self.forward_prepared.col_quant_sizes_offset,
                    self.forward_prepared.col_quant_sizes_bytes,
                )
                snapshot = _typed_view(
                    snapshot_bytes,
                    torch.int32,
                    (self.config.experts_per_rank,),
                )
            assert self._runtime is not None
            return Mxfp8TrainingExecutionViews(
                context=context,
                forward=PreparedResources(
                    runtime=self._runtime, workspace=forward_workspace
                ),
                backward=PreparedResources(
                    runtime=self._runtime, workspace=backward_workspace
                ),
                forward_expert_size_snapshot=snapshot,
            )

    def refresh_weights(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("training plan is closed")
            self.weight_bindings.refresh()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._workspace is not None:
                self._workspace.close()
                self._workspace = None
            if self._runtime is not None:
                self._runtime.close()
                self._runtime = None
            self._closed = True

    def finalize_overflow(
        self,
        contexts: tuple[MoeEpTrainingContext, ...],
        *,
        lane: int,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate caller-context flags and apply the public error policy."""

        if not contexts:
            raise ValueError("finalize_overflow requires at least one context")
        if len({id(context) for context in contexts}) != len(contexts):
            raise ValueError("finalize_overflow contexts must be unique")
        if lane < 0 or lane >= self.lane_count:
            raise ValueError(f"lane {lane} is outside [0, {self.lane_count})")
        for context in contexts:
            self._validate_context(context)
            self.validate_outputs(context, {"overflow": out})
        flat = self._flat_views(0)
        global_overflow = _typed_view(
            flat.local[_lane_name(lane, "finalizer", "local", "global_overflow")],
            torch.int32,
            (1,),
        )
        global_overflow.zero_()
        for context in contexts:
            torch.maximum(
                global_overflow,
                context.forward.forward_overflow,
                out=global_overflow,
            )
            torch.maximum(
                global_overflow,
                context.forward.backward_overflow,
                out=global_overflow,
            )
        assert self._runtime is not None
        if self._runtime.world_size > 1:
            dist.all_reduce(
                global_overflow, op=dist.ReduceOp.MAX, group=self._runtime.group
            )
        out.copy_(global_overflow)
        if not self.config.drop_on_overflow:
            assert_async = getattr(torch, "_assert_async", None)
            if assert_async is None:
                raise RuntimeError(
                    "drop_on_overflow=False training plan requires "
                    "torch._assert_async"
                )
            overflow_ok = _typed_view(
                flat.local[_lane_name(lane, "finalizer", "local", "overflow_ok")],
                torch.bool,
                (1,),
            )
            torch.eq(global_overflow, 0, out=overflow_ok)
            assert_async(
                overflow_ok,
                "Rubin MegaMoE receive route-pool overflow; "
                "the caller outputs are invalid",
            )
        return out


__all__ = [
    "Mxfp8TrainingExecutionViews",
    "Mxfp8TrainingPlanOwner",
    "build_training_buffer_specs",
    "build_training_workspace_requirements",
]
