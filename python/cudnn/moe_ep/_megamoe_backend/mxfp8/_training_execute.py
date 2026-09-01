# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Ordinary/capturable launch path over a slotless MXFP8 training plan."""

from __future__ import annotations

import torch

from ..._types import MoeEpTrainingWgradOperands
from .._runtime import _runtime_debug
from .._workspace import padded_mxfp8_scale_columns
from ._adapter import (
    Mxfp8LaunchInputs,
    _typed_view,
)
from ._backward_compile import (
    Mxfp8BackwardLaunchInputs,
    build_backward_runtime_kwargs,
    compile_backward_or_get,
)
from ._compile import compile_or_get
from ._launch import build_runtime_kwargs
from ._training_resources import (
    Mxfp8TrainingExecutionViews,
    Mxfp8TrainingPlanOwner,
)


def _zero_pre_reduced(inputs, prepared) -> None:
    capacity = prepared.config.max_tokens_per_rank
    offset = prepared.pre_reduced_activation_offset
    bytes_per_token = prepared.pre_reduced_activation_bytes_per_token
    if offset is not None and bytes_per_token:
        inputs.shared_workspace.narrow(
            0,
            offset,
            capacity * bytes_per_token,
        ).zero_()
    sf_offset = prepared.pre_reduced_activation_sf_offset
    sf_bytes_per_token = prepared.pre_reduced_activation_sf_bytes_per_token
    if sf_offset is not None and sf_bytes_per_token:
        inputs.shared_workspace.narrow(
            0,
            sf_offset,
            capacity * sf_bytes_per_token,
        ).zero_()


def _activation_views(
    execution: Mxfp8TrainingExecutionViews,
    *,
    backward: bool,
    capacity: int,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    workspace = execution.backward.workspace if backward else execution.forward.workspace
    return (
        _typed_view(
            workspace.symmetric["activation_data"],
            torch.float8_e4m3fn,
            (capacity, hidden),
        ),
        _typed_view(
            workspace.symmetric["activation_scale"],
            torch.float8_e8m0fnu,
            (capacity, padded_mxfp8_scale_columns(hidden)),
        ),
    )


def _write_expert_offsets(
    execution: Mxfp8TrainingExecutionViews,
    padding: int,
) -> None:
    snapshot = execution.forward_expert_size_snapshot
    if snapshot is None:
        raise RuntimeError("training forward requires the persistent expert-size snapshot")
    counts = execution.context.wgrad.valid_route_counts
    offsets = execution.context.wgrad.expert_offsets
    counts.copy_(snapshot)
    torch.add(counts, padding - 1, out=offsets)
    torch.div(offsets, padding, rounding_mode="floor", out=offsets)
    offsets.mul_(padding)
    torch.cumsum(offsets, dim=0, out=offsets)


def launch_training_forward(
    owner: Mxfp8TrainingPlanOwner,
    execution: Mxfp8TrainingExecutionViews,
    activation: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Stage one forward into a borrowed TE context and caller output."""

    prepared = owner.forward_prepared
    config = prepared.config
    capacity = config.max_tokens_per_rank
    context = execution.context
    forward = context.forward
    owner.validate_outputs(context, {"forward": out})
    _runtime_debug(
        "training-forward.begin",
        token_count=int(activation.shape[0]),
    )
    activation_data, activation_sf = _activation_views(
        execution,
        backward=False,
        capacity=capacity,
        hidden=config.hidden,
    )
    _runtime_debug("training-forward.stage.begin")
    owner.stager.stage(
        activation,
        topk_idx,
        topk_weights,
        activation_data,
        activation_sf,
        forward.routing_topk_idx,
        forward.routing_topk_weights,
    )
    _runtime_debug("training-forward.stage.end")

    workspace = execution.forward.workspace
    output_scratch = _typed_view(
        workspace.symmetric["output_data"],
        torch.bfloat16,
        (capacity, config.hidden),
    )
    output_scratch.zero_()
    out.zero_()
    forward.forward_overflow.zero_()
    forward.backward_overflow.zero_()
    forward.fc1_a.zero_()
    forward.fc1_a_scale_compact.zero_()
    _runtime_debug("training-forward.reset.end")
    inputs = Mxfp8LaunchInputs(
        activation=activation_data,
        activation_sf=activation_sf,
        topk_indices=forward.routing_topk_idx,
        topk_scores=forward.routing_topk_weights,
        weights=owner.weight_bindings.forward,
        fc1_c=forward.fc1_preact,
        output_data=output_scratch,
        col_quant_data=forward.fc1_a.transpose(0, 1),
        col_quant_sf=forward.fc1_a_scale_compact,
        overflow_flag=forward.forward_overflow,
        local_workspace=workspace.local["kernel_local_workspace"],
        shared_workspace=workspace.symmetric["kernel_shared_workspace"],
        token_count=int(activation.shape[0]),
    )
    _zero_pre_reduced(inputs, prepared)
    _runtime_debug("training-forward.compile.begin")
    compiled = compile_or_get(
        prepared,
        inputs,
        execution.forward,
    )
    _runtime_debug("training-forward.compile.end")
    _runtime_debug("training-forward.launch.begin")
    compiled.callable(**build_runtime_kwargs(inputs, execution.forward))
    _runtime_debug("training-forward.launch.end")
    out[: inputs.token_count].copy_(output_scratch[: inputs.token_count])
    _runtime_debug("training-forward.offsets.begin")
    _write_expert_offsets(execution, config.token_padding_block)
    _runtime_debug("training-forward.offsets.end")
    _runtime_debug("training-forward.end")
    return out[: inputs.token_count]


def launch_training_backward(
    owner: Mxfp8TrainingPlanOwner,
    execution: Mxfp8TrainingExecutionViews,
    grad_output: torch.Tensor,
    grad_activation_out: torch.Tensor,
    dprob_out: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    MoeEpTrainingWgradOperands,
]:
    """Stage normal backward and materialize operands in a borrowed context."""

    prepared = owner.backward_prepared
    config = prepared.config
    capacity = config.max_tokens_per_rank
    context = execution.context
    forward = context.forward
    wgrad = context.wgrad
    owner.validate_outputs(
        context,
        {
            "grad_activation": grad_activation_out,
            "dprob": dprob_out,
        },
    )
    token_count = int(grad_output.shape[0])
    _runtime_debug(
        "training-backward.begin",
        token_count=token_count,
    )
    activation_data, activation_sf = _activation_views(
        execution,
        backward=True,
        capacity=capacity,
        hidden=config.hidden,
    )
    _runtime_debug("training-backward.stage.begin")
    owner.stager.stage(
        grad_output,
        forward.routing_topk_idx[:token_count],
        forward.routing_topk_weights[:token_count],
        activation_data,
        activation_sf,
        forward.routing_topk_idx,
        forward.routing_topk_weights,
    )
    _runtime_debug("training-backward.stage.end")

    workspace = execution.backward.workspace
    output_scratch = _typed_view(
        workspace.symmetric["output_data"],
        torch.bfloat16,
        (capacity, config.hidden),
    )
    dprob_scratch = _typed_view(
        workspace.symmetric["backward_dprob"],
        torch.float32,
        tuple(dprob_out.shape),
    )
    output_scratch.zero_()
    dprob_scratch.zero_()
    grad_activation_out.zero_()
    dprob_out.zero_()
    forward.backward_overflow.zero_()
    wgrad.fc1_recompute.zero_()
    wgrad.fc1_recompute_sf.view(torch.uint8).fill_(127)
    wgrad.fc1_col_output.zero_()
    wgrad.fc1_col_output_sf.view(torch.uint8).fill_(127)
    wgrad.fc2_b.zero_()
    wgrad.fc2_b_scale_compact.fill_(127)
    _runtime_debug("training-backward.reset.end")

    weights = owner.weight_bindings.backward
    inputs = Mxfp8BackwardLaunchInputs(
        grad_out=activation_data,
        grad_out_sf=activation_sf,
        topk_idx=forward.routing_topk_idx,
        topk_weights=forward.routing_topk_weights,
        fc1_weight=weights.fc1_weight,
        fc1_weight_sf=weights.fc1_weight_sf,
        fc2_weight=weights.fc2_weight,
        fc2_weight_sf=weights.fc2_weight_sf,
        beta=owner.beta,
        fc1_preact=forward.fc1_preact,
        output_activation=output_scratch,
        overflow_flag=forward.backward_overflow,
        dprob=dprob_scratch,
        fc1_recompute=wgrad.fc1_recompute,
        fc1_recompute_sf=wgrad.fc1_recompute_sf,
        fc1_col_output=wgrad.fc1_col_output,
        fc1_col_output_sf=wgrad.fc1_col_output_sf,
        grad_y2=wgrad.fc2_b,
        grad_y2_sf=wgrad.fc2_b_scale_compact,
        local_workspace=workspace.local["kernel_local_workspace"],
        shared_workspace=workspace.symmetric["kernel_shared_workspace"],
        token_count=token_count,
    )
    _zero_pre_reduced(inputs, prepared)
    _runtime_debug("training-backward.compile.begin")
    compiled = compile_backward_or_get(
        prepared,
        inputs,
        execution.backward,
    )
    _runtime_debug("training-backward.compile.end")
    _runtime_debug("training-backward.launch.begin")
    compiled.callable(**build_backward_runtime_kwargs(inputs, execution.backward))
    _runtime_debug("training-backward.launch.end")
    grad_activation_out.copy_(output_scratch)
    dprob_out.copy_(dprob_scratch)
    _runtime_debug("training-backward.wgrad-export.begin")
    operands = owner.wgrad_exporter.export(context)
    _runtime_debug("training-backward.wgrad-export.end")
    _runtime_debug("training-backward.end")
    return (
        grad_activation_out[:token_count],
        dprob_out[:token_count],
        operands,
    )


__all__ = [
    "launch_training_backward",
    "launch_training_forward",
]
