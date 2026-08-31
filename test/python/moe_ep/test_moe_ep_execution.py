# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""MoE EP single-GPU numerical, training, WGrad, and graph execution.

See ``docs/fe-oss-apis/moe_ep.md`` for canonical user examples.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cudnn.moe_ep import MoeEp
from moe_ep.moe_ep_reference import (
    MoeFormat,
    forward_combine_round_trip,
    quantize_blockwise,
)
from moe_ep.moe_ep_test_support import (
    _allocate_dense_grouped_wgrad_outputs,
    _assert_fixed_training_drop_overflow_result,
    _assert_fixed_training_matches_reference,
    _assert_grouped_wgrads_match_reference,
    _assert_matches_reference,
    _assert_training_graph_tails_are_reset,
    _assert_training_weight_sources_changed,
    _capture_fixed_training_batch,
    _copy_training_weight_sources_,
    _dense_wgrads_from_grouped_kernel,
    _dense_wgrads_from_operands,
    _fixed_training_case,
    _fixed_training_drop_overflow_case,
    _fixed_training_drop_overflow_reference,
    _fixed_training_reference,
    _fixed_training_weights,
    _forward_config,
    _grad_output,
    _make_forward_case,
    _output_as_float,
    _prefill_training_graph_sentinels,
    _reference_forward,
    _replay_cuda_graph,
    _run_fixed_training_batch,
    _sm107_device,
    _training_public_pointers,
    _training_source_pointers,
    _training_weight_source_pointers,
    _training_weight_source_values,
    make_forward_inputs,
)

pytestmark = pytest.mark.moe_ep_execution

_L1_EXCLUSIVE = (pytest.mark.L1, pytest.mark.gpu_exclusive)


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            "bf16-fresh-outputs", id="bf16-fresh-outputs", marks=pytest.mark.L0
        ),
        pytest.param("mxfp8-combine", id="mxfp8-combine", marks=_L1_EXCLUSIVE),
        pytest.param("plain-bf16-family", id="plain-bf16-family", marks=_L1_EXCLUSIVE),
        pytest.param("mixed-fp16-fc1", id="mixed-fp16-fc1", marks=_L1_EXCLUSIVE),
        pytest.param("gate-up-clamp", id="gate-up-clamp", marks=pytest.mark.L0),
        pytest.param("tuned-plan-reuse", id="tuned-plan-reuse", marks=pytest.mark.L0),
        pytest.param("topk1", id="topk1", marks=_L1_EXCLUSIVE),
        pytest.param("topk32-boundary", id="topk32-boundary", marks=_L1_EXCLUSIVE),
        pytest.param(
            "weight-family-switch", id="weight-family-switch", marks=_L1_EXCLUSIVE
        ),
        pytest.param(
            "in-kernel-topk-reduce",
            id="in-kernel-topk-reduce",
            marks=_L1_EXCLUSIVE,
        ),
    ],
)
def test_forward_scenarios_match_reference(scenario):
    from cudnn import MoeEpTuningConfig

    device = _sm107_device()
    config = _forward_config()
    args = make_forward_inputs(device)
    tuning = None

    if scenario == "mxfp8-combine":
        config = _forward_config(combine_format="mxfp8")
    elif scenario == "plain-bf16-family":
        args = (
            args[0].dequantize(torch.bfloat16),
            args[1].dequantize(torch.bfloat16),
            args[2].dequantize(torch.bfloat16),
            *args[3:],
        )
    elif scenario == "mixed-fp16-fc1":
        args = (
            args[0],
            args[1].dequantize(torch.float16),
            args[2],
            *args[3:],
        )
    elif scenario == "gate-up-clamp":
        config = _forward_config(gate_up_clamp=0.5)
    elif scenario == "tuned-plan-reuse":
        tuning = MoeEpTuningConfig(
            token_back_mode="standalone_warps",
            epi_flag_batch=(4, 2),
            token_in_flag_batch=4,
            group_hint=64,
        )
    elif scenario == "topk1":
        args = _make_forward_case(
            device,
            experts=2,
            tokens=3,
            hidden=128,
            intermediate=256,
            top_k=1,
            index_dtype=torch.int32,
            weight_dtype=torch.bfloat16,
        )
        config = _forward_config(
            top_k=1,
            max_tokens_per_rank=3,
        )
    elif scenario == "topk32-boundary":
        args = _make_forward_case(
            device,
            experts=32,
            tokens=1,
            hidden=128,
            intermediate=256,
            top_k=32,
            index_dtype=torch.int64,
            weight_dtype=torch.float32,
        )
        config = _forward_config(
            num_experts=32,
            top_k=32,
            max_tokens_per_rank=1,
        )
    elif scenario == "in-kernel-topk-reduce":
        tuning = MoeEpTuningConfig(reduce_topk_in_kernel=True)

    expected = _reference_forward(args, **config)
    with MoeEp(**config, **({"tuning": tuning} if tuning is not None else {})) as op:
        if scenario == "bf16-fresh-outputs":
            first = op(*args)
            snapshot = first.clone()
            second = op(*args)
            args[3].fill_(-1)
            dropped = op(*args)
            torch.cuda.synchronize(device)
            assert isinstance(first, torch.Tensor)
            assert first.shape == second.shape == (5, 128)
            assert first.dtype == second.dtype == torch.bfloat16
            assert first.device == second.device == device
            assert first is not second
            assert first.data_ptr() != second.data_ptr()
            torch.testing.assert_close(first, snapshot, rtol=0, atol=0)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            _assert_matches_reference(first, expected)
            assert dropped.eq(0).all()
            return

        if scenario == "weight-family-switch":
            quantized = op(*args)
            backend = op._forward_backend
            refresh_before = backend._adapter.weight_refresh_count
            plain_args = (
                args[0].dequantize(torch.bfloat16),
                args[1].dequantize(torch.bfloat16),
                args[2].dequantize(torch.bfloat16),
                *args[3:],
            )
            expected_plain = _reference_forward(plain_args)
            plain = op(*plain_args)
            refresh_after = backend._adapter.weight_refresh_count
            torch.cuda.synchronize(device)
            assert refresh_after == refresh_before + 1
            _assert_matches_reference(quantized, expected)
            _assert_matches_reference(plain, expected_plain)
            return

        first = op(*args)
        if scenario == "tuned-plan-reuse":
            backend = op._forward_backend
            compiled = backend._compiled
            workspace = backend._plan._workspace
            second = op(*args)
            assert backend._compiled is compiled
            assert backend._plan._workspace is workspace
            assert backend.kernel_config.tuning_signature(
                backend._prepared_kernel.launch_cluster_count
            ) == ("standalone_warps", (4, 2), 4, 64, False)
            _assert_matches_reference(second, expected)
        if scenario == "in-kernel-topk-reduce":
            assert op._forward_backend.kernel_config.fc2_in_kernel_topk_reduce
        torch.cuda.synchronize(device)

    if scenario == "gate-up-clamp":
        assert not torch.equal(expected, _reference_forward(args))
    _assert_matches_reference(first, expected)


@pytest.mark.L0
def test_mxfp8_combine_oracle_rounds_directly_from_fp32():
    generator = torch.Generator().manual_seed(20260819)
    accumulator = torch.randn(4, 128, generator=generator) * 3.25
    actual = forward_combine_round_trip(accumulator, MoeFormat.MXFP8)
    direct_fp32 = quantize_blockwise(
        accumulator,
        MoeFormat.MXFP8,
    ).dequantize()
    bf16_preround = quantize_blockwise(
        accumulator.to(torch.bfloat16).float(),
        MoeFormat.MXFP8,
    ).dequantize()
    torch.testing.assert_close(actual, direct_fp32, rtol=0, atol=0)
    assert not torch.equal(actual, bf16_preround)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize(
    ("input_kind", "combine_format", "gate_up_clamp", "top_k", "all_dropped"),
    [
        pytest.param("fixed", "bf16", None, 2, False, id="bf16"),
        pytest.param("fixed", "mxfp8", 0.5, 2, False, id="mxfp8-clamp-0.5"),
        pytest.param("routed", "bf16", None, 1, False, id="topk1"),
        pytest.param("routed", "bf16", None, 2, True, id="all-dropped"),
    ],
)
def test_training_scenarios_match_independent_reference(
    input_kind,
    combine_format,
    gate_up_clamp,
    top_k,
    all_dropped,
):
    device = _sm107_device()
    if input_kind == "fixed":
        args, grad_output = _fixed_training_case(device)
        max_recv_size = 1
    else:
        base_args = make_forward_inputs(device)
        args = (
            base_args[0].dequantize(torch.bfloat16),
            base_args[1],
            base_args[2],
            base_args[3][:, :top_k].contiguous(),
            base_args[4][:, :top_k].float().contiguous(),
        )
        if all_dropped:
            args[3].fill_(-1)
            args[4].zero_()
        grad_output = _grad_output(device, args[0].shape[0], seed=20260830)
        max_recv_size = args[0].shape[0] * top_k
    expected = _fixed_training_reference(
        args,
        grad_output,
        combine_format=combine_format,
        gate_up_clamp=gate_up_clamp,
    )

    with MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=top_k,
        max_tokens_per_rank=args[0].shape[0],
        max_recv_size_per_rank=max_recv_size,
        drop_on_overflow=True,
        combine_format=combine_format,
        gate_up_clamp=gate_up_clamp,
    ) as op:
        resources = op.prepare_training_resources(
            _fixed_training_weights(args),
            slot_count=1,
            lane_count=1,
        )
        actual = _run_fixed_training_batch(
            resources,
            resources.lanes[0],
            ((resources.slots[0], args, grad_output),),
        )[0]
        torch.cuda.synchronize(device)
        assert actual.overflow.eq(0).all()
        _assert_fixed_training_matches_reference(
            (actual.y, actual.dx, actual.dprob, actual.wgrads),
            expected,
            args[3],
        )
        if all_dropped:
            actual_dw1, actual_dw2 = _dense_wgrads_from_operands(actual.wgrads)
            expected_dw1, expected_dw2 = expected[3].dense_wgrads()
            assert all(
                tensor.eq(0).all()
                for tensor in (
                    actual.y,
                    expected[0],
                    actual.dx,
                    expected[1],
                    actual.dprob,
                    expected[2],
                    actual_dw1,
                    expected_dw1,
                    actual_dw2,
                    expected_dw2,
                )
            )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_grouped_wgrad_kernel_matches_independent_reference():
    device = _sm107_device()
    base_args = make_forward_inputs(device)
    args = (
        base_args[0].dequantize(torch.bfloat16),
        base_args[1],
        base_args[2],
        base_args[3],
        base_args[4].float(),
    )
    grad_output = _grad_output(device, args[0].shape[0], seed=20260831)
    expected = _fixed_training_reference(
        args,
        grad_output,
        combine_format="bf16",
        gate_up_clamp=None,
    )
    with MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=args[0].shape[0],
        max_recv_size_per_rank=args[0].shape[0] * args[3].shape[1],
        drop_on_overflow=True,
    ) as op:
        resources = op.prepare_training_resources(
            _fixed_training_weights(args),
            slot_count=1,
            lane_count=1,
        )
        actual = _run_fixed_training_batch(
            resources,
            resources.lanes[0],
            ((resources.slots[0], args, grad_output),),
        )[0]
        grouped_wgrads = _dense_wgrads_from_grouped_kernel(actual.wgrads)
        torch.cuda.synchronize(device)
        assert actual.overflow.eq(0).all()
        _assert_fixed_training_matches_reference(
            (actual.y, actual.dx, actual.dprob, actual.wgrads),
            expected,
            args[3],
        )
        torch.testing.assert_close(
            actual.wgrads.valid_route_counts,
            expected[3].valid_route_counts,
            rtol=0,
            atol=0,
        )
        assert actual.wgrads.valid_route_counts.gt(0).all()
        expected_offsets = torch.cumsum(
            torch.div(
                actual.wgrads.valid_route_counts + 127,
                128,
                rounding_mode="floor",
            )
            * 128,
            dim=0,
            dtype=actual.wgrads.expert_offsets.dtype,
        )
        torch.testing.assert_close(
            actual.wgrads.expert_offsets,
            expected_offsets,
            rtol=0,
            atol=0,
        )
        _assert_grouped_wgrads_match_reference(
            grouped_wgrads,
            expected[3].dense_wgrads(),
            reference_name="the independent PyTorch MXFP8 reference",
        )
        _assert_grouped_wgrads_match_reference(
            grouped_wgrads,
            _dense_wgrads_from_operands(actual.wgrads),
            reference_name="the decoded production operand bundle",
            close_kwargs={"rtol": 0.1, "atol": 0.1},
        )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_grouped_wgrad_accumulates_two_microbatches():
    device = _sm107_device()
    base_args = make_forward_inputs(device)
    args0 = (
        base_args[0].dequantize(torch.bfloat16),
        base_args[1],
        base_args[2],
        base_args[3],
        base_args[4].float(),
    )
    args1 = (
        args0[0].mul(-0.5),
        args0[1],
        args0[2],
        args0[3].roll(1, dims=0),
        args0[4].roll(1, dims=0),
    )
    grad_outputs = (
        _grad_output(device, args0[0].shape[0], seed=20260902),
        _grad_output(device, args1[0].shape[0], seed=20260903),
    )
    references = tuple(
        _fixed_training_reference(
            args,
            grad_output,
            combine_format="bf16",
            gate_up_clamp=None,
        )
        for args, grad_output in zip((args0, args1), grad_outputs)
    )
    with MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=args0[0].shape[0],
        max_recv_size_per_rank=args0[0].shape[0] * args0[3].shape[1],
        drop_on_overflow=True,
    ) as op:
        resources = op.prepare_training_resources(
            _fixed_training_weights(args0),
            slot_count=2,
            lane_count=1,
        )
        batch = tuple(
            (slot, args, grad_output)
            for slot, args, grad_output in zip(
                resources.slots,
                (args0, args1),
                grad_outputs,
            )
        )
        actuals = _run_fixed_training_batch(resources, resources.lanes[0], batch)
        accumulated = _allocate_dense_grouped_wgrad_outputs(
            actuals[0].wgrads,
            fill_value=0,
        )
        output_pointers = tuple(output.data_ptr() for output in accumulated)
        for actual in actuals:
            returned = _dense_wgrads_from_grouped_kernel(
                actual.wgrads,
                wgrad_tensors=accumulated,
                accumulate_on_output=True,
            )
            assert tuple(output.data_ptr() for output in returned) == output_pointers
        torch.cuda.synchronize(device)

        for actual, args, reference in zip(actuals, (args0, args1), references):
            assert actual.overflow.eq(0).all()
            _assert_fixed_training_matches_reference(
                (actual.y, actual.dx, actual.dprob, actual.wgrads),
                reference,
                args[3],
            )
        expected_accumulated = tuple(
            reference0.float() + reference1.float()
            for reference0, reference1 in zip(
                references[0][3].dense_wgrads(),
                references[1][3].dense_wgrads(),
            )
        )
        _assert_grouped_wgrads_match_reference(
            accumulated,
            expected_accumulated,
            reference_name="the sum of two independent PyTorch MXFP8 references",
            close_kwargs={"rtol": 0.2, "atol": 0.25},
        )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize(
    "tuned",
    [
        pytest.param(False, id="default"),
        pytest.param(True, id="tuned"),
    ],
)
def test_forward_cuda_graph_replay(tuned):
    from cudnn import MoeEpTuningConfig

    device = _sm107_device()
    args = make_forward_inputs(device)
    original_topk_idx = args[3].clone()
    expected = _reference_forward(args)
    kwargs = {}
    if tuned:
        kwargs["tuning"] = MoeEpTuningConfig(
            token_back_mode="reuse_dispatch_warps",
            epi_flag_batch=(2, 2),
            token_in_flag_batch=2,
            group_hint=128,
        )
    with MoeEp(**_forward_config(), **kwargs) as op:
        _replay_cuda_graph(
            op,
            args,
            original_topk_idx,
            expected,
            device,
        )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            SimpleNamespace(
                combine_format="bf16",
                drop_on_overflow=True,
                max_recv_size=1,
                replay_count=5,
            ),
            id="bf16-drop",
        ),
        pytest.param(
            SimpleNamespace(
                combine_format="mxfp8",
                drop_on_overflow=True,
                max_recv_size=1,
                replay_count=5,
            ),
            id="mxfp8-drop",
        ),
        pytest.param(
            SimpleNamespace(
                combine_format="bf16",
                drop_on_overflow=False,
                max_recv_size=2,
                replay_count=2,
            ),
            id="bf16-error-no-overflow",
        ),
    ],
)
def test_training_cuda_graph_replay(case):
    device = _sm107_device()
    if case.drop_on_overflow:
        args0, grad0 = _fixed_training_case(device)
        topk_idx1 = args0[3].clone()
        topk_idx1[0, 0] = 1
        inputs = (
            (args0, grad0),
            (
                (
                    args0[0].clone(),
                    args0[1],
                    args0[2],
                    topk_idx1,
                    args0[4].clone(),
                ),
                grad0.clone(),
            ),
        )
    else:
        inputs = (_fixed_training_drop_overflow_case(device),)
    references = tuple(
        _fixed_training_reference(
            args,
            grad_output,
            combine_format=case.combine_format,
            gate_up_clamp=None,
        )
        for args, grad_output in inputs
    )
    with MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=inputs[0][0][0].shape[0],
        max_recv_size_per_rank=case.max_recv_size,
        drop_on_overflow=case.drop_on_overflow,
        combine_format=case.combine_format,
        token_padding_size=128,
    ) as op:
        resources = op.prepare_training_resources(
            _fixed_training_weights(inputs[0][0]),
            slot_count=len(inputs),
            lane_count=1,
        )
        lane = resources.lanes[0]
        batch = tuple(
            (slot, args, grad_output)
            for slot, (args, grad_output) in zip(resources.slots, inputs)
        )

        def assert_batch(actuals):
            for actual, (args, _), reference in zip(
                actuals,
                inputs,
                references,
            ):
                assert actual.overflow.shape == (1,)
                assert actual.overflow.dtype == torch.int32
                assert actual.overflow.eq(0).all()
                _assert_fixed_training_matches_reference(
                    (actual.y, actual.dx, actual.dprob, actual.wgrads),
                    reference,
                    args[3],
                )

        eager_actuals = _run_fixed_training_batch(resources, lane, batch)
        torch.cuda.synchronize(device)
        assert_batch(eager_actuals)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        captured = _capture_fixed_training_batch(
            resources,
            lane,
            batch,
            stream,
        )
        for _ in range(case.replay_count):
            captured.graph.replay()
            torch.cuda.synchronize(device)
            assert captured.public_pointers == tuple(
                _training_public_pointers(actual) for actual in captured.actuals
            )
            assert_batch(captured.actuals)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_two_shape_cuda_graph_specialization_addresses_and_refresh():
    device = _sm107_device()
    args, grad_large = _fixed_training_case(device)
    max_tokens = int(args[0].shape[0])
    small_tokens = max_tokens - 2
    assert 0 < small_tokens < max_tokens
    large = SimpleNamespace(
        name="large",
        activation=args[0],
        topk_idx=args[3],
        topk_weights=args[4],
        grad_output=grad_large,
    )
    small = SimpleNamespace(
        name="small",
        activation=args[0][:small_tokens].clone(),
        topk_idx=args[3][:small_tokens].clone(),
        topk_weights=args[4][:small_tokens].clone(),
        grad_output=grad_large[:small_tokens].clone(),
    )
    assert all(
        getattr(large, name).data_ptr() != getattr(small, name).data_ptr()
        for name in ("activation", "topk_idx", "topk_weights", "grad_output")
    )
    weights = _fixed_training_weights(args)
    weight_source_pointers = _training_weight_source_pointers(weights)
    with MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=max_tokens,
        max_recv_size_per_rank=1,
        drop_on_overflow=True,
    ) as op:
        resources = op.prepare_training_resources(
            weights,
            slot_count=1,
            lane_count=1,
        )
        slot = resources.slots[0]
        lane = resources.lanes[0]

        def case_args(case):
            return (
                case.activation,
                weights.forward_fc1,
                weights.forward_fc2,
                case.topk_idx,
                case.topk_weights,
            )

        def independent_reference(case):
            return _fixed_training_reference(
                case_args(case),
                case.grad_output,
                combine_format="bf16",
                gate_up_clamp=None,
            )

        def warmup(case):
            actual = _run_fixed_training_batch(
                resources,
                lane,
                ((slot, case_args(case), case.grad_output),),
            )[0]
            torch.cuda.synchronize(device)
            assert actual.overflow.eq(0).all()
            _assert_fixed_training_matches_reference(
                (actual.y, actual.dx, actual.dprob, actual.wgrads),
                independent_reference(case),
                case.topk_idx,
            )

        warmup(large)
        warmup(small)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))

        def capture(case):
            captured = _capture_fixed_training_batch(
                resources,
                lane,
                ((slot, case_args(case), case.grad_output),),
                capture_stream,
            )
            return SimpleNamespace(
                case=case,
                graph=captured.graph,
                actual=captured.actuals[0],
                public_pointers=captured.public_pointers[0],
                source_pointers=_training_source_pointers(case),
            )

        large_graph = capture(large)
        small_graph = capture(small)
        slot_views = resources._owner.views(
            slot=slot.index,
            lane=lane.index,
            token_count=max_tokens,
        ).slot

        def replay_and_check(captured):
            _prefill_training_graph_sentinels(slot_views, captured.actual)
            captured.graph.replay()
            torch.cuda.synchronize(device)
            assert captured.actual.overflow.eq(0).all()
            assert (
                _training_public_pointers(captured.actual) == captured.public_pointers
            )
            assert _training_source_pointers(captured.case) == captured.source_pointers
            assert _training_weight_source_pointers(weights) == weight_source_pointers
            _assert_fixed_training_matches_reference(
                (
                    captured.actual.y,
                    captured.actual.dx,
                    captured.actual.dprob,
                    captured.actual.wgrads,
                ),
                independent_reference(captured.case),
                captured.case.topk_idx,
            )
            _assert_training_graph_tails_are_reset(
                slot_views,
                captured.actual,
                token_count=int(captured.case.activation.shape[0]),
                capacity=max_tokens,
            )
            return captured.actual.y.clone()

        for captured in (large_graph, small_graph, large_graph):
            replay_and_check(captured)

        small_source_pointers = _training_source_pointers(small)
        small.activation.mul_(-0.5)
        small.topk_idx.fill_(-1)
        small.topk_idx[0, 0] = 1
        small.topk_weights.zero_()
        small.topk_weights[0, 0] = 0.625
        small.grad_output.mul_(-0.75)
        assert _training_source_pointers(small) == small_source_pointers
        for captured in (small_graph, large_graph, small_graph):
            replay_and_check(captured)

        old_large_y = replay_and_check(large_graph)
        old_weight_values = _training_weight_source_values(weights)
        generator = torch.Generator(device=device).manual_seed(20260829)
        new_fc1 = (
            torch.randn(
                weights.forward_fc1.logical_shape,
                generator=generator,
                device=device,
            )
            / 16
        )
        new_fc2 = (
            torch.randn(
                weights.forward_fc2.logical_shape,
                generator=generator,
                device=device,
            )
            / 16
        )
        replacement = _fixed_training_weights(
            (
                large.activation,
                new_fc1,
                new_fc2,
                large.topk_idx,
                large.topk_weights,
            )
        )
        _copy_training_weight_sources_(weights, replacement)
        assert _training_weight_source_pointers(weights) == weight_source_pointers
        _assert_training_weight_sources_changed(weights, old_weight_values)
        new_large_y = replay_and_check(large_graph)
        assert not torch.equal(new_large_y, old_large_y)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_overflow_boundary_and_cuda_graph_transitions():
    device = _sm107_device()
    args, grad_output = _fixed_training_drop_overflow_case(device)
    assert args[0].shape[0] == 1
    assert args[3].detach().cpu().tolist() == [[0, 1]]
    references = {
        expected_overflow: _fixed_training_drop_overflow_reference(
            args,
            grad_output,
            drop_expert1=bool(expected_overflow),
        )
        for expected_overflow in (0, 1)
    }

    def assert_result(actual, expected_overflow):
        expected, reference_topk_idx = references[expected_overflow]
        _assert_fixed_training_drop_overflow_result(
            actual,
            expected,
            reference_topk_idx,
            expected_overflow=expected_overflow,
        )

    with MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=1,
        max_recv_size_per_rank=2,
        drop_on_overflow=True,
        token_padding_size=128,
    ) as op:
        resources = op.prepare_training_resources(
            _fixed_training_weights(args),
            slot_count=1,
            lane_count=1,
        )
        actual = _run_fixed_training_batch(
            resources,
            resources.lanes[0],
            ((resources.slots[0], args, grad_output),),
        )[0]
        torch.cuda.synchronize(device)
        assert_result(actual, 0)
        assert args[3][0, 1].eq(1)
        assert actual.wgrads.valid_route_counts.detach().cpu().tolist() == [1, 1]
        assert actual.wgrads.expert_offsets.detach().cpu().tolist() == [128, 256]

    overflow_routing = args[3].clone()
    expert0_only_routing = overflow_routing.clone()
    expert0_only_routing[0, 1] = -1
    routing_pointer = args[3].data_ptr()
    with MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=1,
        max_recv_size_per_rank=1,
        drop_on_overflow=True,
        token_padding_size=128,
    ) as op:
        resources = op.prepare_training_resources(
            _fixed_training_weights(args),
            slot_count=1,
            lane_count=1,
        )
        slot = resources.slots[0]
        lane = resources.lanes[0]
        batch = ((slot, args, grad_output),)
        warmup = _run_fixed_training_batch(resources, lane, batch)[0]
        grouped_outputs = _allocate_dense_grouped_wgrad_outputs(warmup.wgrads)
        _dense_wgrads_from_grouped_kernel(
            warmup.wgrads,
            wgrad_tensors=grouped_outputs,
        )
        torch.cuda.synchronize(device)
        assert_result(warmup, 1)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        captured = _capture_fixed_training_batch(
            resources,
            lane,
            batch,
            capture_stream,
            grouped_wgrad_outputs=(grouped_outputs,),
        )
        graph_actual = captured.actuals[0]
        graph_grouped_wgrads = captured.grouped_wgrads[0]
        grouped_output_pointers = tuple(output.data_ptr() for output in grouped_outputs)
        for routing, expected_overflow in (
            (overflow_routing, 1),
            (expert0_only_routing, 0),
            (overflow_routing, 1),
        ):
            args[3].copy_(routing)
            for output in grouped_outputs:
                output.fill_(float("nan"))
            assert args[3].data_ptr() == routing_pointer
            expected = _fixed_training_drop_overflow_reference(
                args,
                grad_output,
                drop_expert1=bool(expected_overflow),
            )
            captured.graph.replay()
            torch.cuda.synchronize(device)
            _assert_fixed_training_drop_overflow_result(
                graph_actual,
                *expected,
                expected_overflow=expected_overflow,
            )
            assert (
                tuple(output.data_ptr() for output in graph_grouped_wgrads)
                == grouped_output_pointers
            )
            _assert_grouped_wgrads_match_reference(
                graph_grouped_wgrads,
                expected[0][3].dense_wgrads(),
                reference_name="the independent PyTorch MXFP8 graph reference",
            )
            _assert_grouped_wgrads_match_reference(
                graph_grouped_wgrads,
                _dense_wgrads_from_operands(graph_actual.wgrads),
                reference_name="the decoded captured production operand bundle",
                close_kwargs={"rtol": 0.1, "atol": 0.1},
            )
            assert all(torch.isfinite(output).all() for output in graph_grouped_wgrads)
            if expected_overflow:
                assert graph_grouped_wgrads[0][1].eq(0).all()
                assert graph_grouped_wgrads[1][1].eq(0).all()
            assert captured.public_pointers[0] == _training_public_pointers(
                graph_actual
            )
