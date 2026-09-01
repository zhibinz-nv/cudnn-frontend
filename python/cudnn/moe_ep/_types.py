# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lightweight public tensor and format types for :mod:`cudnn.moe_ep`."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Tuple, Union

import torch


class MoeFormat(str, Enum):
    """Data formats supported by the MoE+EP interface."""

    BF16 = "bf16"
    MXFP8 = "mxfp8"
    NVFP4 = "nvfp4"


class MoeEpBufferLifetime(str, Enum):
    """Earliest semantic point at which a caller-owned buffer may retire."""

    NORMAL_BACKWARD = "normal_backward"
    DELAYED_WGRAD = "delayed_wgrad"
    OVERFLOW_FINALIZE = "overflow_finalize"
    STREAM_COMPLETION = "stream_completion"


def parse_format(value: Union[MoeFormat, str]) -> MoeFormat:
    """Normalize a public format value."""

    if isinstance(value, MoeFormat):
        return value
    try:
        return MoeFormat(value.lower())
    except (AttributeError, ValueError) as exc:
        choices = ", ".join(item.value for item in MoeFormat)
        raise ValueError(
            f"unsupported format {value!r}; expected one of: {choices}"
        ) from exc


def _normalize_axis(axis: int, ndim: int) -> int:
    if isinstance(axis, bool):
        raise ValueError(f"axis must be an integer, got {axis!r}")
    try:
        axis = operator.index(axis)
    except TypeError as exc:
        raise ValueError(f"axis must be an integer, got {axis!r}") from exc
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"axis {axis} is out of range for a {ndim}-D tensor")
    return normalized


@dataclass(frozen=True)
class BlockScaledTensor:
    """Data-plus-scale result returned for MXFP8 and NVFP4 outputs."""

    data: torch.Tensor
    scale: torch.Tensor
    format: Union[MoeFormat, str]
    logical_shape: Tuple[int, ...]
    axis: int = -1

    def __post_init__(self) -> None:
        if not isinstance(self.data, torch.Tensor):
            raise ValueError(
                f"data must be a torch.Tensor, got {type(self.data).__name__}"
            )
        if not isinstance(self.scale, torch.Tensor):
            raise ValueError(
                f"scale must be a torch.Tensor, got {type(self.scale).__name__}"
            )
        fmt = parse_format(self.format)
        if fmt is MoeFormat.BF16:
            raise ValueError("BlockScaledTensor only represents mxfp8 or nvfp4")
        if self.data.device != self.scale.device:
            raise ValueError(
                f"data device {self.data.device} does not match "
                f"scale device {self.scale.device}"
            )
        try:
            raw_logical_shape = tuple(self.logical_shape)
        except TypeError as exc:
            raise ValueError("logical_shape must be an iterable of integers") from exc
        logical_shape = []
        for dim in raw_logical_shape:
            if isinstance(dim, bool):
                raise ValueError(
                    f"logical_shape dimensions must be integers, got {dim!r}"
                )
            try:
                dim = operator.index(dim)
            except TypeError as exc:
                raise ValueError(
                    f"logical_shape dimensions must be integers, got {dim!r}"
                ) from exc
            if dim < 0:
                raise ValueError(
                    f"logical_shape dimensions must be non-negative, got {dim}"
                )
            logical_shape.append(dim)
        normalized_shape = tuple(logical_shape)
        axis = _normalize_axis(self.axis, len(normalized_shape))
        logical_extent = normalized_shape[axis]
        block_size = 32 if fmt is MoeFormat.MXFP8 else 16
        payload_extent = (
            logical_extent if fmt is MoeFormat.MXFP8 else (logical_extent + 1) // 2
        )
        scale_extent = (logical_extent + block_size - 1) // block_size
        expected_data_shape = list(normalized_shape)
        expected_data_shape[axis] = payload_extent
        expected_scale_shape = list(normalized_shape)
        expected_scale_shape[axis] = scale_extent
        expected_data_shape = tuple(expected_data_shape)
        expected_scale_shape = tuple(expected_scale_shape)
        if tuple(self.data.shape) != expected_data_shape:
            raise ValueError(
                f"{fmt.value} data shape must be {expected_data_shape}, "
                f"got {tuple(self.data.shape)}"
            )
        if tuple(self.scale.shape) != expected_scale_shape:
            raise ValueError(
                f"{fmt.value} scale shape must be {expected_scale_shape}, "
                f"got {tuple(self.scale.shape)}"
            )
        e4m3_dtype = getattr(torch, "float8_e4m3fn", None)
        if e4m3_dtype is None:
            raise RuntimeError(
                "this PyTorch build does not provide torch.float8_e4m3fn"
            )
        if fmt is MoeFormat.MXFP8:
            expected_data_dtype = e4m3_dtype
            expected_scale_dtype = getattr(torch, "float8_e8m0fnu", None)
            if expected_scale_dtype is None:
                raise RuntimeError(
                    "this PyTorch build does not provide torch.float8_e8m0fnu"
                )
        else:
            expected_data_dtype = torch.uint8
            expected_scale_dtype = e4m3_dtype
        if self.data.dtype is not expected_data_dtype:
            raise ValueError(
                f"{fmt.value} data must have dtype {expected_data_dtype}, "
                f"got {self.data.dtype}"
            )
        if self.scale.dtype is not expected_scale_dtype:
            raise ValueError(
                f"{fmt.value} scale must have dtype {expected_scale_dtype}, "
                f"got {self.scale.dtype}"
            )
        object.__setattr__(self, "format", fmt)
        object.__setattr__(self, "logical_shape", normalized_shape)
        object.__setattr__(self, "axis", axis)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.logical_shape

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def block_size(self) -> int:
        return 32 if self.format is MoeFormat.MXFP8 else 16

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Decode the logical, unswizzled block-scaled representation."""

        logical_extent = self.logical_shape[self.axis]
        scale = self.scale.movedim(self.axis, -1).float()
        expanded_scale = scale.repeat_interleave(
            self.block_size,
            dim=-1,
        )[..., :logical_extent]

        if self.format is MoeFormat.MXFP8:
            values = self.data.movedim(self.axis, -1).float()
        else:
            packed = self.data.movedim(self.axis, -1)
            low = packed & 0x0F
            high = packed >> 4
            codes = torch.stack((low, high), dim=-1).flatten(-2)[..., :logical_extent]
            table = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ],
                dtype=torch.float32,
                device=packed.device,
            )
            values = table[codes.long()]

        return (values * expanded_scale).movedim(-1, self.axis).to(dtype)


@dataclass(frozen=True)
class MoeEpTrainingWeights:
    """Stable MXFP8 bindings for forward and dgrad GEMMs.

    Forward consumes ``forward_fc1`` with logical shape ``(E,H,2I)`` and
    ``forward_fc2`` with ``(E,I,H)``. Backward consumes independently
    quantized transposes: ``backward_w2_transpose=(E,H,I)`` for
    ``dH=dY@W2.T`` and ``backward_w1_transpose=(E,2I,H)`` for
    ``dX=dC@W1.T``. Every tensor is block-scaled along logical axis 1, the
    reduction axis of its corresponding GEMM.
    """

    forward_fc1: BlockScaledTensor
    forward_fc2: BlockScaledTensor
    backward_w2_transpose: BlockScaledTensor
    backward_w1_transpose: BlockScaledTensor


@dataclass(frozen=True)
class MoeEpTrainingWgradOperands:
    """Non-owning views of TE-owned, fixed-capacity MXFP8 WGrad operands."""

    fc1_a: torch.Tensor
    fc1_sfa: torch.Tensor
    fc1_b: torch.Tensor
    fc1_sfb: torch.Tensor
    fc2_a: torch.Tensor
    fc2_sfa: torch.Tensor
    fc2_b: torch.Tensor
    fc2_sfb: torch.Tensor
    expert_offsets: torch.Tensor
    valid_route_counts: torch.Tensor


@dataclass(frozen=True)
class MoeEpTrainingBufferSpec:
    """One caller-owned training buffer requirement.

    ``lifetime`` gives the eager-execution retirement boundary after queued
    stream work completes. ``capture_pinned`` requires the same address to
    remain valid until every graph executable that captured it is destroyed.
    """

    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    alignment: int
    lifetime: MoeEpBufferLifetime
    capture_pinned: bool


@dataclass(frozen=True)
class MoeEpTrainingBufferSpecs:
    """Fixed-capacity buffer ABI used by a TE-side context cache."""

    forward_context: Mapping[str, MoeEpTrainingBufferSpec]
    wgrad_gradients: Mapping[str, MoeEpTrainingBufferSpec]
    outputs: Mapping[str, MoeEpTrainingBufferSpec]


@dataclass(frozen=True)
class MoeEpForwardContextBuffers:
    """TE-owned forward state with field-specific lifetimes from the specs."""

    routing_topk_idx: torch.Tensor
    routing_topk_weights: torch.Tensor
    fc1_preact: torch.Tensor
    fc1_a: torch.Tensor
    fc1_a_scale_compact: torch.Tensor
    forward_overflow: torch.Tensor
    backward_overflow: torch.Tensor


@dataclass(frozen=True)
class MoeEpWgradGradientBuffers:
    """TE-owned destinations populated while preparing delayed WGrad."""

    valid_route_counts: torch.Tensor
    expert_offsets: torch.Tensor
    fc1_recompute: torch.Tensor
    fc1_recompute_sf: torch.Tensor
    fc1_col_output: torch.Tensor
    fc1_col_output_sf: torch.Tensor
    fc2_b: torch.Tensor
    fc2_b_scale_compact: torch.Tensor
    fc1_sfa: torch.Tensor
    fc1_b: torch.Tensor
    fc1_sfb: torch.Tensor
    fc2_a: torch.Tensor
    fc2_sfa: torch.Tensor
    fc2_sfb: torch.Tensor


@dataclass(frozen=True)
class MoeEpTrainingContext:
    """One TE cache entry borrowed by FE for a forward/backward pair.

    ``token_count`` is the exact extent shared by forward and backward, not
    the fixed capacity of the allocated tensors.
    """

    token_count: int
    forward: MoeEpForwardContextBuffers
    wgrad: MoeEpWgradGradientBuffers

    def __post_init__(self) -> None:
        if isinstance(self.token_count, bool):
            raise ValueError("token_count must be a non-negative integer")
        try:
            token_count = operator.index(self.token_count)
        except TypeError as exc:
            raise ValueError("token_count must be a non-negative integer") from exc
        if token_count < 0:
            raise ValueError("token_count must be a non-negative integer")
        object.__setattr__(self, "token_count", token_count)


@dataclass(frozen=True)
class MoeEpExecutionLane:
    """Opaque index of one mutable per-stream execution lane."""

    index: int
    _plan_token: object


class MoeEpTrainingPlan:
    """Slotless FE execution plan borrowing TE-owned contexts per call."""

    def __init__(
        self,
        *,
        owner: Any,
        weights: MoeEpTrainingWeights,
        lane_count: int,
        device: torch.device,
    ) -> None:
        self._owner = owner
        self._plan_token = object()
        self.weights = weights
        self.device = torch.device(device)
        self.lanes = tuple(
            MoeEpExecutionLane(index, self._plan_token) for index in range(lane_count)
        )
        self.buffer_specs = owner.buffer_specs
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _check_lane(self, lane: MoeEpExecutionLane) -> None:
        if self._closed:
            raise RuntimeError("MoeEp training plan is closed")
        if (
            not isinstance(lane, MoeEpExecutionLane)
            or lane._plan_token is not self._plan_token
            or lane not in self.lanes
        ):
            raise ValueError("execution lane does not belong to this training plan")

    def refresh_weights(self) -> None:
        """Enqueue fixed-address weight-layout refreshes on the current stream.

        Call after every in-place data+scale update and before the first
        forward/backward that consumes that version. The caller must establish
        stream/event ordering, must not refresh between a matching forward and
        backward, and must not overlap refresh with any consumer of this plan.
        Replacing source storage requires closing the old operator, creating a
        new ``MoeEp`` instance and plan, and capturing a new
        graph. This method may itself be captured, in which case replay executes
        only the recorded device transforms.
        """

        if self._closed:
            raise RuntimeError("MoeEp training plan is closed")
        self._owner.refresh_weights()

    def forward(
        self,
        context: MoeEpTrainingContext,
        lane: MoeEpExecutionLane,
        activation: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Write one forward into caller output and TE-owned context buffers."""

        self._check_lane(lane)
        execution = self._owner.views(
            context=context,
            lane=lane.index,
            token_count=int(activation.shape[0]),
        )
        from ._megamoe_backend.mxfp8._training_execute import (
            launch_training_forward,
        )

        return launch_training_forward(
            self._owner,
            execution,
            activation,
            topk_idx,
            topk_weights,
            out,
        )

    def backward(
        self,
        context: MoeEpTrainingContext,
        lane: MoeEpExecutionLane,
        grad_output: torch.Tensor,
        *,
        grad_activation_out: torch.Tensor,
        dprob_out: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        MoeEpTrainingWgradOperands,
    ]:
        """Write dgrad/dprob and delayed-WGrad operands into caller buffers."""

        self._check_lane(lane)
        execution = self._owner.views(
            context=context,
            lane=lane.index,
            token_count=int(grad_output.shape[0]),
        )
        from ._megamoe_backend.mxfp8._training_execute import (
            launch_training_backward,
        )

        grad_activation, grad_topk_weights, operands = launch_training_backward(
            self._owner,
            execution,
            grad_output,
            grad_activation_out,
            dprob_out,
        )
        return grad_activation, grad_topk_weights, operands

    def finalize_overflow(
        self,
        contexts: Tuple[MoeEpTrainingContext, ...],
        lane: MoeEpExecutionLane | None = None,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate one computation group's context flags and apply policy."""

        if self._closed:
            raise RuntimeError("MoeEp training plan is closed")
        if lane is None:
            lane = self.lanes[0]
        if (
            not isinstance(lane, MoeEpExecutionLane)
            or lane._plan_token is not self._plan_token
            or lane not in self.lanes
        ):
            raise ValueError(
                "overflow execution lane does not belong to this training plan"
            )
        return self._owner.finalize_overflow(
            contexts,
            lane=lane.index,
            out=out,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._owner.close()
        self._closed = True


MoeTensor = Union[torch.Tensor, BlockScaledTensor]


__all__ = [
    "BlockScaledTensor",
    "MoeEpBufferLifetime",
    "MoeEpExecutionLane",
    "MoeEpForwardContextBuffers",
    "MoeEpTrainingBufferSpec",
    "MoeEpTrainingBufferSpecs",
    "MoeEpTrainingContext",
    "MoeEpTrainingPlan",
    "MoeEpTrainingWeights",
    "MoeEpTrainingWgradOperands",
    "MoeEpWgradGradientBuffers",
    "MoeFormat",
    "MoeTensor",
    "parse_format",
]
