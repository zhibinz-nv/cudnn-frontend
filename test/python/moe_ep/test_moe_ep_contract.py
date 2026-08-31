# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""MoE EP host and critical internal contracts.

See ``docs/fe-oss-apis/moe_ep.md`` for canonical user examples.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import fields, replace
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import cudnn
import numpy as np
import pytest
import torch
import torch.distributed as dist

from cudnn.moe_ep import (
    MoeEp,
    MoeEpExecutionLane,
    MoeEpTrainingResources,
    MoeEpTrainingSlot,
    MoeEpTrainingWgradOperands,
)
from cudnn.moe_ep._megamoe_backend._workspace import (
    BufferRegion,
    WorkspaceRequirements,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._fingerprint import (
    canonical_json_sha256,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._training_resources import (
    Mxfp8TrainingResourceOwner,
    _build_training_abi_facts,
    _harmonize_symmetric_regions,
    _verify_training_abi_across_ranks,
    build_training_workspace_requirements,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._training_stage import (
    Mxfp8TrainingStager,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._training_weights import (
    Mxfp8TrainingWeightBindings,
)
from cudnn.moe_ep._validation import validate_training_weights
from moe_ep.moe_ep_test_support import (
    _forward_config,
    _training_abi_prepared,
    _training_config,
    _training_contract_resources,
    _training_inputs,
    _training_prepared_pair,
    _training_staging_tensors,
    _training_weight_defect,
    _training_weights,
)

pytestmark = pytest.mark.moe_ep_contract


def _public_nvfp4(data, scale, logical_shape):
    from cudnn import BlockScaledTensor

    return BlockScaledTensor(
        data=data,
        scale=scale,
        format="nvfp4",
        logical_shape=logical_shape,
        axis=1,
    )


def _request(activation, fc1, fc2):
    return SimpleNamespace(
        activation=activation,
        fc1_weight=fc1,
        fc2_weight=fc2,
    )


def _operator(**overrides) -> MoeEp:
    values = {
        "num_experts": 2,
        "hidden_size": 128,
        "intermediate_size": 256,
        "top_k": 2,
        "max_tokens_per_rank": 4,
    }
    values.update(overrides)
    return MoeEp(**values)


def _install_contract_backend(
    monkeypatch,
    *,
    weights=None,
    slot_count=1,
    lane_count=1,
):
    import cudnn.moe_ep._backend as backend_seam
    import cudnn.moe_ep.api as api_module

    weights = weights or SimpleNamespace(mock_training_weights=True)
    state = SimpleNamespace(
        backends=[],
        validate=Mock(return_value=torch.device("cpu")),
    )

    def create_backend(config, device):
        del config, device
        owner = _ContractOwner(
            slot_count=slot_count,
            lane_count=lane_count,
        )
        backend = SimpleNamespace(
            owner=owner,
            prepare_training_resources=Mock(return_value=owner),
            close=Mock(),
        )
        state.backends.append(backend)
        return backend

    monkeypatch.setattr(api_module, "validate_training_weights", state.validate)
    monkeypatch.setattr(backend_seam, "validate_config", lambda config: None)
    monkeypatch.setattr(backend_seam, "create_backend", create_backend)
    return weights, state


class _ContractOwner:
    def __init__(self, *, slot_count: int = 2, lane_count: int = 1) -> None:
        self.slot_count = slot_count
        self.lane_count = lane_count
        self.close_calls = 0
        self.refresh_calls = 0
        self.views_calls = 0

    def refresh_weights(self) -> None:
        self.refresh_calls += 1

    def views(self, **kwargs):
        del kwargs
        self.views_calls += 1
        raise AssertionError("binding rejection must happen before owner views")

    def _flat_views(self, token_count: int):
        del token_count
        raise AssertionError("invalid finalization must fail before workspace access")

    def finalize_overflow(self, slots, *, lane):
        return Mxfp8TrainingResourceOwner.finalize_overflow(
            self,
            slots,
            lane=lane,
        )

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def runtime_module():
    from cudnn.moe_ep._megamoe_backend import _runtime

    with _runtime._PROCESS_RUNTIME_REGISTRY.lock:
        _runtime._PROCESS_RUNTIME_REGISTRY.active = None
    yield _runtime
    with _runtime._PROCESS_RUNTIME_REGISTRY.lock:
        _runtime._PROCESS_RUNTIME_REGISTRY.active = None


class _FakeRuntimeProvider:
    def __init__(self, runtime_module, state=None):
        self._runtime_module = runtime_module
        self._state = state or runtime_module.RuntimeInitState.NOT_INITIALIZED
        self._world = None
        self.initialize_count = 0
        self.finalize_count = 0

    def initialization_state(self):
        return self._state

    def initialize(self, device, world):
        del device
        self.initialize_count += 1
        self._world = world
        self._state = self._runtime_module.RuntimeInitState.INITIALIZED

    def rank(self):
        return self._world.rank

    def world_size(self):
        return self._world.size

    def device(self):
        return torch.device("cuda", 0)

    def finalize(self):
        self.finalize_count += 1
        self._state = self._runtime_module.RuntimeInitState.NOT_INITIALIZED


@pytest.mark.L0
@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("padding", id="padding"),
        pytest.param("untyped-tuning", id="untyped-tuning"),
        pytest.param(
            "incompatible-in-kernel-reduce", id="incompatible-in-kernel-reduce"
        ),
        pytest.param("training-nvfp4-combine", id="training-nvfp4-combine"),
        pytest.param("training-mxfp8-output", id="training-mxfp8-output"),
        pytest.param("training-post-fc2-topk", id="training-post-fc2-topk"),
        pytest.param("nvfp4-request-before-cuda", id="nvfp4-request-before-cuda"),
    ],
)
def test_constructor_and_capability_validation(monkeypatch, scenario):
    from cudnn import MoeEpTuningConfig
    from cudnn.moe_ep._megamoe_backend._capability import (
        validate_config,
        validate_request,
    )

    if scenario == "padding":
        invalid = (
            {"token_padding_size": True},
            {"token_padding_size": 0},
            {"sf_padding_size": 64},
            {"sf_padding_size": 128.0},
        )
        for kwargs in invalid:
            with pytest.raises(ValueError):
                MoeEp(**_forward_config(), **kwargs)
        return

    if scenario == "untyped-tuning":
        with pytest.raises(TypeError, match="MoeEpTuningConfig"):
            MoeEp(**_forward_config(), tuning={"group_hint": 768})
        return

    if scenario == "incompatible-in-kernel-reduce":
        with pytest.raises(ValueError, match="reduce_topk_in_kernel requires"):
            MoeEp(
                **_forward_config(combine_format="mxfp8"),
                tuning=MoeEpTuningConfig(reduce_topk_in_kernel=True),
            )
        return

    unsupported = {
        "training-nvfp4-combine": {"combine_format": "nvfp4"},
        "training-mxfp8-output": {"output_format": "mxfp8"},
        "training-post-fc2-topk": {"apply_topk_in_fc1": False},
    }
    if scenario in unsupported:
        with MoeEp(**_forward_config(**unsupported[scenario])) as operator:
            with pytest.raises(NotImplementedError, match="training MegaMoE"):
                validate_config(operator._forward_config)
        return

    operand = _public_nvfp4(
        torch.zeros(2, 64, dtype=torch.uint8),
        torch.ones(2, 8).to(torch.float8_e4m3fn),
        (2, 128),
    )
    request = _request(operand, operand, operand)
    request.device = torch.device("cuda", 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: pytest.fail("CUDA capability queried too early"),
    )
    with pytest.raises(NotImplementedError, match="only MXFP8"):
        validate_request(request)


@pytest.mark.L1
@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("happy-path", id="happy-path"),
        pytest.param("logical-shape", id="logical-shape"),
        pytest.param("plain-representation", id="plain-representation"),
        pytest.param("axis", id="axis"),
        pytest.param("format", id="format"),
        pytest.param("data-and-scale-contiguity", id="data-and-scale-contiguity"),
        pytest.param("cross-field-device", id="cross-field-device"),
    ],
)
def test_training_weight_validation(scenario):
    weights = _training_weights()
    if scenario == "happy-path":
        assert validate_training_weights(_training_config(), weights) == torch.device(
            "cpu"
        )
        return

    defects = {
        "logical-shape": ("forward_fc2", "logical_shape"),
        "plain-representation": ("forward_fc1", "plain_tensor"),
        "axis": ("forward_fc1", "axis"),
        "format": ("forward_fc1", "format"),
        "cross-field-device": ("backward_w1_transpose", "device"),
    }
    selected = (
        (("forward_fc1", "data_noncontiguous"), ("forward_fc1", "scale_noncontiguous"))
        if scenario == "data-and-scale-contiguity"
        else (defects[scenario],)
    )
    for field, defect in selected:
        invalid, error_type, message = _training_weight_defect(
            weights,
            field,
            defect,
        )
        with pytest.raises(error_type) as exc_info:
            validate_training_weights(_training_config(), invalid)
        assert str(exc_info.value) == message


@pytest.mark.L1
@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("shape", id="shape"),
        pytest.param("dtype-and-contiguity", id="dtype-and-contiguity"),
        pytest.param("capacity", id="capacity"),
        pytest.param("device", id="device"),
    ],
)
def test_training_stager_validation(scenario):
    checks = {
        "shape": (
            (
                lambda tensors: tensors.update(
                    source=tensors["source"][:, :-1].contiguous()
                ),
                ValueError,
                r"source must have shape \(T, 128\)",
            ),
            (
                lambda tensors: tensors.update(
                    topk_idx=tensors["topk_idx"][:, :-1].contiguous()
                ),
                ValueError,
                "topk_idx shape mismatch",
            ),
        ),
        "dtype-and-contiguity": (
            (
                lambda tensors: tensors.update(
                    topk_idx=tensors["topk_idx"].to(torch.int64)
                ),
                TypeError,
                "contiguous Int32",
            ),
            (
                lambda tensors: tensors.update(
                    topk_weights=tensors["topk_weights"].t().contiguous().t()
                ),
                TypeError,
                "contiguous FP32",
            ),
        ),
        "capacity": (
            (
                lambda tensors: tensors.update(
                    **{
                        name: value[:4]
                        for name, value in tensors.items()
                        if name.startswith("output")
                    }
                ),
                ValueError,
                "token count 5 exceeds capacity 4",
            ),
        ),
        "device": (
            (
                lambda tensors: tensors.update(
                    source=torch.empty_like(tensors["source"], device="meta")
                ),
                ValueError,
                "must share one device",
            ),
        ),
    }
    for mutator, error_type, message in checks[scenario]:
        tensors = _training_staging_tensors()
        mutator(tensors)
        with pytest.raises(error_type, match=message):
            Mxfp8TrainingStager(hidden=128, top_k=2)._validate(**tensors)


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("prepare-and-close", id="prepare-and-close", marks=pytest.mark.L0),
        pytest.param(
            "duplicate-close-reopen",
            id="duplicate-close-reopen",
            marks=pytest.mark.L1,
        ),
        pytest.param(
            "binding-finalization-close-errors",
            id="binding-finalization-close-errors",
            marks=pytest.mark.L1,
        ),
        pytest.param(
            "capture-rejections",
            id="capture-rejections",
            marks=pytest.mark.L1,
        ),
    ],
)
def test_training_resource_lifecycle(monkeypatch, scenario):
    if scenario == "prepare-and-close":
        weights = _training_weights()
        _, state = _install_contract_backend(
            monkeypatch,
            weights=weights,
            slot_count=2,
        )
        operator = _operator()
        resources = operator.prepare_training_resources(
            weights,
            slot_count=2,
            lane_count=1,
        )
        assert isinstance(resources, MoeEpTrainingResources)
        assert all(isinstance(slot, MoeEpTrainingSlot) for slot in resources.slots)
        assert isinstance(resources.lanes[0], MoeEpExecutionLane)
        resources.refresh_weights()
        owner = state.backends[0].owner
        assert owner.refresh_calls == 1
        operator.close()
        assert resources.closed
        assert owner.close_calls == 1
        return

    if scenario == "duplicate-close-reopen":
        weights, state = _install_contract_backend(monkeypatch)
        old_operator = _operator()
        old_resources = old_operator.prepare_training_resources(weights)
        with pytest.raises(RuntimeError, match="already exist"):
            old_operator.prepare_training_resources(weights)
        old_resources.close()
        with pytest.raises(RuntimeError, match="create a new MoeEp instance"):
            old_operator.prepare_training_resources(weights)
        old_operator.close()
        with _operator() as new_operator:
            new_resources = new_operator.prepare_training_resources(weights)
            assert not new_resources.closed
        assert len(state.backends) == 2
        assert state.validate.call_count == 2
        return

    if scenario == "capture-rejections":
        from cudnn.moe_ep._megamoe_backend.mxfp8._backend import Mxfp8Backend

        monkeypatch.setattr(
            torch.cuda,
            "is_current_stream_capturing",
            lambda: True,
        )
        owner = object.__new__(Mxfp8TrainingResourceOwner)
        owner._lock = threading.RLock()
        owner._closed = False
        owner._runtime = None
        owner._workspace = None
        with pytest.raises(
            RuntimeError,
            match="must be prepared before CUDA graph capture",
        ):
            owner.prepare()

        backend = object.__new__(Mxfp8Backend)
        backend._lock = threading.RLock()
        backend._closed = False
        backend.device = torch.device("cuda")
        monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
        with pytest.raises(RuntimeError, match="cannot be closed during"):
            backend.close()
        return

    resources, owner = _training_contract_resources(
        owner=_ContractOwner(slot_count=2, lane_count=1)
    )
    foreign, _ = _training_contract_resources(
        owner=_ContractOwner(slot_count=2, lane_count=1)
    )
    slot = resources.slots[0]
    lane = resources.lanes[0]
    activation = torch.empty((0, 128), dtype=torch.bfloat16)
    routing = (
        torch.empty((0, 2), dtype=torch.int32),
        torch.empty((0, 2), dtype=torch.float32),
    )
    binding_checks = (
        (
            "training slot does not belong",
            resources.forward,
            (foreign.slots[0], lane, activation, *routing),
        ),
        (
            "execution lane does not belong",
            resources.forward,
            (slot, foreign.lanes[0], activation, *routing),
        ),
        (
            "training slot does not belong",
            resources.backward,
            (MoeEpTrainingSlot(99, slot._resource_token), lane, activation.float()),
        ),
        (
            "execution lane does not belong",
            resources.backward,
            (slot, MoeEpExecutionLane(99, lane._resource_token), activation.float()),
        ),
    )
    for message, call, args in binding_checks:
        with pytest.raises(ValueError, match=message):
            call(*args)

    finalization_checks = (
        ("at least one slot", (), lane),
        ("slots must be unique", (slot, slot), lane),
        ("overflow slot does not belong", (foreign.slots[0],), lane),
        ("overflow execution lane does not belong", (slot,), foreign.lanes[0]),
    )
    for message, slots, selected_lane in finalization_checks:
        with pytest.raises(ValueError, match=message):
            resources.finalize_overflow(slots, selected_lane)

    resources.close()
    resources.close()
    assert resources.closed
    assert owner.close_calls == 1
    closed_calls = (
        resources.refresh_weights,
        lambda: resources.forward(slot, lane, activation, *routing),
        lambda: resources.backward(slot, lane, activation.float()),
        lambda: resources.finalize_overflow((slot,), lane),
    )
    for call in closed_calls:
        with pytest.raises(RuntimeError, match="resources are closed"):
            call()
    assert owner.refresh_calls == 0
    assert owner.views_calls == 0


@pytest.mark.L0
def test_public_training_surface_and_overflow_requirements(monkeypatch):
    from cudnn.moe_ep.api import _validate_training_assert_capability

    expected = [
        f"fc{layer}_{part}" for layer in (1, 2) for part in ("a", "sfa", "b", "sfb")
    ]
    expected += ["expert_offsets", "valid_route_counts"]
    assert [field.name for field in fields(MoeEpTrainingWgradOperands)] == expected
    assert not hasattr(cudnn, "MoeEpWgradForwardStash")
    assert not hasattr(cudnn, "MoeEpWgradOperands")

    monkeypatch.setattr(torch, "_assert_async", None)
    with pytest.raises(RuntimeError, match="callable torch._assert_async"):
        _validate_training_assert_capability(
            SimpleNamespace(drop_on_overflow=False, ep_size=1)
        )
    _validate_training_assert_capability(
        SimpleNamespace(drop_on_overflow=True, ep_size=1)
    )

    monkeypatch.setattr(torch, "_assert_async", lambda *args, **kwargs: None)
    monkeypatch.setattr(dist, "get_backend", lambda group: "gloo")
    with pytest.raises(NotImplementedError, match="NCCL"):
        _validate_training_assert_capability(
            SimpleNamespace(
                drop_on_overflow=False,
                ep_size=2,
                ep_group=object(),
            )
        )


@pytest.mark.L0
def test_training_weight_and_workspace_address_stability():
    weights = _training_weights()
    bindings = Mxfp8TrainingWeightBindings(weights)
    bindings.refresh()
    tensors = (
        bindings.forward.fc1_weight,
        bindings.forward.fc1_weight_sf,
        bindings.forward.fc2_weight,
        bindings.forward.fc2_weight_sf,
        bindings.backward.fc1_weight,
        bindings.backward.fc1_weight_sf,
        bindings.backward.fc2_weight,
        bindings.backward.fc2_weight_sf,
    )
    pointers = tuple(tensor.data_ptr() for tensor in tensors)
    first_snapshot = tensors[0].clone()
    weights.forward_fc1.data.view(torch.uint8).bitwise_xor_(1)
    bindings.refresh()
    assert tuple(tensor.data_ptr() for tensor in tensors) == pointers
    assert not torch.equal(bindings.forward.fc1_weight, first_snapshot)

    config = _training_config()
    forward, backward = _training_prepared_pair(config)

    class Runtime:
        device = torch.device("cpu")
        rank = 0
        world_size = 1
        nvshmem_enabled = False
        closed = False

        def ensure_open(self):
            assert not self.closed

        def close(self):
            self.closed = True

    runtime = Runtime()
    owner = Mxfp8TrainingResourceOwner(
        config,
        torch.device("cpu"),
        forward,
        backward,
        _training_weights(),
        slot_count=2,
        lane_count=1,
        runtime_manager=SimpleNamespace(
            acquire=lambda actual_config, actual_device: runtime
        ),
    )
    try:
        first = owner.views(slot=0, lane=0, token_count=4)
        second = owner.views(slot=1, lane=0, token_count=4)
        assert (
            first.forward.workspace.local["kernel_local_workspace"].data_ptr()
            == second.forward.workspace.local["kernel_local_workspace"].data_ptr()
        )
        assert first.slot.fc1_preact.data_ptr() != second.slot.fc1_preact.data_ptr()
        assert first.slot.dprob.data_ptr() != second.slot.dprob.data_ptr()
        assert (
            first.forward_expert_size_snapshot.data_ptr()
            == second.forward_expert_size_snapshot.data_ptr()
        )
    finally:
        owner.close()
    assert runtime.closed


@pytest.mark.L0
def test_training_workspace_layout_and_abi_contracts(monkeypatch):
    base = WorkspaceRequirements.for_mxfp8(
        _training_config(),
        kernel_local_workspace_bytes=64,
        kernel_shared_workspace_bytes=128,
        backward_fc1_preact_bytes=1024,
        backward_dprob_bytes=32,
        backward_aux_data_bytes=512,
        backward_aux_scale_bytes=256,
    )
    base_regions = {
        "symmetric": {region.name: region for region in base.symmetric_regions},
        "local": {region.name: region for region in base.local_regions},
    }
    assert base_regions["symmetric"]["backward_dprob"].nbytes == 32
    assert base_regions["local"]["backward_fc1_preact"].nbytes == 1024
    assert base_regions["local"]["backward_fc1_preact"].alignment == 128

    config = _training_config()
    forward, backward = _training_prepared_pair(config)
    requirements = build_training_workspace_requirements(
        config,
        forward,
        backward,
        slot_count=2,
        lane_count=1,
    )
    symmetric_names = tuple(region.name for region in requirements.symmetric_regions)
    local_names = tuple(region.name for region in requirements.local_regions)
    assert "lane.0.forward.symmetric.kernel_shared_workspace" in symmetric_names
    assert "lane.0.backward.symmetric.kernel_shared_workspace" in symmetric_names
    assert "slot.0.backward.symmetric.backward_dprob" in symmetric_names
    assert "slot.1.backward.symmetric.backward_dprob" in symmetric_names
    assert "slot.0.persistent.local.fc1_preact" in local_names
    assert "slot.1.persistent.local.fc1_preact" in local_names

    uneven = WorkspaceRequirements(
        max_tokens_per_rank=1,
        symmetric_regions=(
            BufferRegion("first", 1),
            BufferRegion("second", 257),
        ),
        local_regions=(BufferRegion("local", 1),),
    )
    runtime = SimpleNamespace(world_size=2, group=object())

    def harmonize_reduce(tensor, *, op, group):
        assert group is runtime.group
        if tensor.numel() == 2 and op == dist.ReduceOp.MAX:
            tensor.copy_(torch.tensor([257, 257], dtype=torch.int64))

    monkeypatch.setattr(dist, "all_reduce", harmonize_reduce)
    harmonized = _harmonize_symmetric_regions(
        uneven,
        runtime,
        torch.device("cpu"),
    )
    assert tuple(region.nbytes for region in harmonized.symmetric_regions) == (
        257,
        257,
    )

    abi_requirements = WorkspaceRequirements(
        max_tokens_per_rank=4,
        symmetric_regions=(BufferRegion("symmetric", 256),),
        local_regions=(BufferRegion("local", 128),),
    )
    facts = _build_training_abi_facts(
        _training_config(ep_size=2, ep_global_ranks=(0, 1)),
        _training_abi_prepared("forward"),
        _training_abi_prepared("backward"),
        _training_weights(),
        abi_requirements,
        slot_count=2,
        lane_count=1,
        source_tree_digest="source",
    )
    changed = _build_training_abi_facts(
        _training_config(ep_size=2, ep_global_ranks=(0, 1)),
        _training_abi_prepared("forward"),
        _training_abi_prepared("backward"),
        _training_weights(),
        abi_requirements,
        slot_count=2,
        lane_count=2,
        source_tree_digest="source",
    )
    assert canonical_json_sha256(facts) == canonical_json_sha256(facts)
    assert canonical_json_sha256(facts) != canonical_json_sha256(changed)

    def mismatch_reduce(tensor, *, op, group):
        assert group is runtime.group
        if op == dist.ReduceOp.MAX:
            tensor.add_(1)

    monkeypatch.setattr(dist, "all_reduce", mismatch_reduce)
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda output, value, *, group: output.__setitem__(
            slice(None), [value, "different"]
        ),
    )
    with pytest.raises(RuntimeError, match="ABI differs"):
        _verify_training_abi_across_ranks(
            {"schema_version": 1},
            runtime,
            torch.device("cpu"),
        )


@pytest.mark.L1
def test_symmetric_region_harmonization_rejects_metadata_mismatch(monkeypatch):
    requirements = WorkspaceRequirements(
        max_tokens_per_rank=1,
        symmetric_regions=(
            BufferRegion("first", 64, alignment=128),
            BufferRegion("second", 128, alignment=256),
        ),
        local_regions=(),
    )
    runtime = SimpleNamespace(world_size=2, group=object())
    for mismatch_reduce, message in (
        (2, "region counts differ"),
        (4, "names, order, or alignments differ"),
    ):
        reduce_calls = []

        def all_reduce(tensor, *, op, group):
            assert group is runtime.group
            reduce_calls.append(op)
            if len(reduce_calls) == mismatch_reduce:
                tensor.add_(1)

        monkeypatch.setattr(dist, "all_reduce", all_reduce)
        with pytest.raises(RuntimeError, match=message):
            _harmonize_symmetric_regions(
                requirements,
                runtime,
                torch.device("cpu"),
            )
        assert len(reduce_calls) == mismatch_reduce


@pytest.mark.L0
def test_ep_topology_order_nonmember_and_runtime_mismatch(monkeypatch):
    from cudnn.moe_ep._megamoe_backend._runtime import _resolve_world
    from cudnn.moe_ep.api import _resolve_ep_topology

    group = object()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda selected=None: 2)
    monkeypatch.setattr(dist, "get_rank", lambda selected=None: 1)
    monkeypatch.setattr(
        dist,
        "get_global_rank",
        lambda selected, group_rank: (3, 1)[group_rank],
    )
    assert _resolve_ep_topology(group) == (2, 1, (3, 1))

    monkeypatch.setattr(dist, "get_rank", lambda selected=None: -1)
    with pytest.raises(ValueError, match="must be a member"):
        _resolve_ep_topology(group)

    monkeypatch.setattr(dist, "get_rank", lambda selected=None: 1)
    config = SimpleNamespace(
        ep_group=group,
        ep_size=2,
        ep_rank=1,
        ep_global_ranks=(3, 1),
    )
    assert _resolve_world(config).identity == (1, 2, (3, 1))
    config.ep_global_ranks = (1, 3)
    with pytest.raises(RuntimeError, match="membership does not match"):
        _resolve_world(config)


@pytest.mark.L0
def test_runtime_manager_subgroup_lifecycle(runtime_module, monkeypatch):
    world = runtime_module.RuntimeWorld(
        rank=1,
        size=2,
        group=object(),
        global_ranks=(1, 3),
    )
    provider = _FakeRuntimeProvider(runtime_module)
    manager = runtime_module.RuntimeManager(
        provider_factory=lambda: provider,
        world_resolver=lambda config: world,
    )
    first = manager.acquire(object(), torch.device("cuda", 0))
    second = manager.acquire(object(), torch.device("cuda", 0))
    assert manager.ref_count == 2
    assert second.global_ranks == (1, 3)
    second.close()
    first.close()
    assert manager.ref_count == 0
    assert provider.finalize_count == 1

    keep_alive_provider = _FakeRuntimeProvider(runtime_module)
    keep_alive = runtime_module.RuntimeManager(
        provider_factory=lambda: keep_alive_provider,
        world_resolver=lambda config: world,
        keep_alive=True,
    )
    handle = keep_alive.acquire(object(), torch.device("cuda", 0))
    handle.close()
    reused = keep_alive.acquire(object(), torch.device("cuda", 0))
    reused.close()
    assert keep_alive_provider.initialize_count == 1
    assert keep_alive_provider.finalize_count == 0
    keep_alive.shutdown()
    assert keep_alive_provider.finalize_count == 1

    first_world = runtime_module.RuntimeWorld(
        rank=0,
        size=2,
        group=object(),
        global_ranks=(0, 2),
    )
    second_world = runtime_module.RuntimeWorld(
        rank=0,
        size=2,
        group=object(),
        global_ranks=(0, 3),
    )
    shared_provider = _FakeRuntimeProvider(runtime_module)
    first_manager = runtime_module.RuntimeManager(
        provider_factory=lambda: shared_provider,
        world_resolver=lambda config: first_world,
    )
    second_manager = runtime_module.RuntimeManager(
        provider_factory=lambda: shared_provider,
        world_resolver=lambda config: second_world,
    )
    handle = first_manager.acquire(object(), torch.device("cuda", 0))
    with pytest.raises(RuntimeError, match="different EP subgroup"):
        second_manager.acquire(object(), torch.device("cuda", 0))
    handle.close()

    external_provider = _FakeRuntimeProvider(
        runtime_module,
        runtime_module.RuntimeInitState.INITIALIZED,
    )
    external_provider._world = world
    external = runtime_module.RuntimeManager(
        provider_factory=lambda: external_provider,
        world_resolver=lambda config: world,
    )
    monkeypatch.setattr(
        runtime_module,
        "_spans_default_distributed_world",
        lambda selected: False,
    )
    with pytest.raises(RuntimeError, match="cannot safely attach"):
        external.acquire(object(), torch.device("cuda", 0))


@pytest.mark.L0
def test_nvshmem_uid_broadcast_uses_subgroup_root(runtime_module, monkeypatch):
    class _FakeDevice:
        def __init__(self, index):
            self.index = index

        def set_current(self):
            return None

    cuda_module = ModuleType("cuda")
    cuda_core_module = ModuleType("cuda.core")
    cuda_experimental_module = ModuleType("cuda.core.experimental")
    cuda_experimental_module.Device = _FakeDevice
    cuda_core_module.experimental = cuda_experimental_module
    cuda_module.core = cuda_core_module
    monkeypatch.setitem(sys.modules, "cuda", cuda_module)
    monkeypatch.setitem(sys.modules, "cuda.core", cuda_core_module)
    monkeypatch.setitem(
        sys.modules,
        "cuda.core.experimental",
        cuda_experimental_module,
    )

    init_args = {}

    class _FakeUid:
        def __init__(self):
            self._data = np.arange(16, dtype=np.uint8)

    core = SimpleNamespace(
        get_unique_id=lambda empty: _FakeUid(),
        init=lambda **kwargs: init_args.update(kwargs),
    )
    monkeypatch.setattr(runtime_module, "_load_nvshmem_core", lambda: core)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    group = object()
    broadcast_args = {}
    monkeypatch.setattr(dist, "get_backend", lambda selected: "gloo")
    monkeypatch.setattr(
        dist,
        "get_global_rank",
        lambda selected, group_rank: (1, 3)[group_rank],
    )
    monkeypatch.setattr(
        dist,
        "broadcast",
        lambda tensor, *, src, group: broadcast_args.update(
            tensor=tensor,
            src=src,
            group=group,
        ),
    )
    monkeypatch.setattr(dist, "barrier", lambda *, group: None)
    runtime_module._DefaultNvshmemRuntimeProvider().initialize(
        torch.device("cuda", 0),
        runtime_module.RuntimeWorld(
            rank=0,
            size=2,
            group=group,
            global_ranks=(1, 3),
        ),
    )
    assert broadcast_args["src"] == 1
    assert broadcast_args["group"] is group
    assert broadcast_args["tensor"].device.type == "cpu"
    assert init_args["rank"] == 0
    assert init_args["nranks"] == 2


@pytest.mark.L0
def test_ep32_capability_and_peer_mapping():
    from cutlass._mlir import ir

    from cudnn.moe_ep._megamoe_backend._capability import validate_config
    from cudnn.moe_ep._megamoe_backend._comm import PeerMapping
    from cudnn.moe_ep._megamoe_backend.cutedsl_src.communication.nvlink_domain.symmetric_buffer import (
        SymmetricBufferDevice,
    )
    from cudnn.moe_ep._megamoe_backend.mxfp8._config import Mxfp8KernelConfig

    with MoeEp(**_forward_config()) as operator:
        ep32_config = replace(
            operator._forward_config,
            num_experts=32,
            experts_per_rank=1,
            ep_size=32,
            ep_rank=31,
            ep_group=object(),
            ep_global_ranks=tuple(range(32)),
        )
    validate_config(ep32_config)
    kernel_config = Mxfp8KernelConfig.from_forward_config(ep32_config)
    assert kernel_config.world_size == 32
    assert kernel_config.local_rank == 31

    offsets = tuple(index * 4096 for index in range(32))
    host = PeerMapping(
        base_address=0x1000,
        offsets=offsets,
        rank=0,
    ).to_sym_buffer_host()
    with ir.Context():
        device_type = SymmetricBufferDevice(
            None,
            host.max_ranks,
        ).__get_mlir_types__()[0]
    assert host.offsets == offsets
    assert int(host.max_ranks) == 32
    assert str(device_type) == "vector<32xi64>"


@pytest.mark.L0
def test_tuning_mapping_cache_and_rank_handshake(monkeypatch):
    from cudnn import MoeEpTuningConfig
    from cudnn.moe_ep import MoeEpTuningConfig as PackageMoeEpTuningConfig
    from cudnn.moe_ep._megamoe_backend.mxfp8._backend import Mxfp8Backend
    from cudnn.moe_ep._megamoe_backend.mxfp8._config import Mxfp8KernelConfig

    assert PackageMoeEpTuningConfig is MoeEpTuningConfig
    tuning = MoeEpTuningConfig(
        token_back_mode="standalone_warps",
        epi_flag_batch=(4, 2),
        token_in_flag_batch=4,
        group_hint=768,
    )
    with MoeEp(**_forward_config(), tuning=tuning) as operator:
        assert operator.tuning is tuning
        tuned = Mxfp8KernelConfig.from_forward_config(operator._forward_config)
    assert tuned.tuning_signature(123) == (
        "standalone_warps",
        (4, 2),
        4,
        768,
        False,
    )
    with MoeEp(**_forward_config()) as operator:
        default = Mxfp8KernelConfig.from_forward_config(operator._forward_config)
    key_args = (torch.device("cuda", 0), (10, 7), 123, ())
    assert tuned.compile_key(*key_args) != default.compile_key(*key_args)

    with MoeEp(**_forward_config()) as operator:
        backend = Mxfp8Backend(
            operator._forward_config,
            torch.device("cuda", 0),
        )
    backend._ep_launch_ready = False
    stream = SimpleNamespace(synchronize=lambda: None)
    resources = SimpleNamespace(runtime=SimpleNamespace(group=object(), world_size=2))
    monkeypatch.setattr(
        backend,
        "_ensure_prepared_kernel",
        lambda: SimpleNamespace(launch_cluster_count=123),
    )
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda output, signature, *, group: output.__setitem__(
            slice(None),
            [signature, ("standalone_warps", (1, 1), 1, 123)],
        ),
    )
    barrier_called = False

    def unexpected_barrier(*, group):
        nonlocal barrier_called
        barrier_called = True

    monkeypatch.setattr(dist, "barrier", unexpected_barrier)
    with pytest.raises(RuntimeError, match="MoeEp tuning must match"):
        backend._ensure_ep_launch_ready(resources, stream)
    assert not barrier_called
    assert not backend._ep_launch_ready


@pytest.mark.L0
def test_forward_host_capacity_overflow_and_finalization(monkeypatch):
    import cudnn.moe_ep.api as api_module
    from cudnn.moe_ep._megamoe_backend._capability import validate_config
    from cudnn.moe_ep._megamoe_backend.mxfp8._config import Mxfp8KernelConfig
    from cudnn.moe_ep._megamoe_backend.mxfp8._launch import _check_overflow

    with MoeEp(
        **_forward_config(),
        max_recv_size_per_rank=7,
        drop_on_overflow=False,
    ) as operator:
        config = Mxfp8KernelConfig.from_forward_config(operator._forward_config)
    assert config.max_recv_size_per_rank == 7
    assert config.drop_on_overflow is False
    with pytest.raises(ValueError, match="max_recv_size_per_rank"):
        MoeEp(**_forward_config(), max_recv_size_per_rank=0)
    with pytest.raises(ValueError, match="drop_on_overflow"):
        MoeEp(**_forward_config(), drop_on_overflow=1)

    with MoeEp(**_forward_config(num_experts=4, top_k=3)) as operator:
        distributed = replace(
            operator._forward_config,
            experts_per_rank=2,
            ep_size=2,
            ep_global_ranks=(0, 1),
        )
    validate_config(distributed)

    calls = []
    monkeypatch.setattr(
        torch,
        "_assert_async",
        lambda condition, message: calls.append((condition.clone(), message)),
    )
    _check_overflow(torch.zeros(1, dtype=torch.int32))
    assert len(calls) == 1
    assert bool(calls[0][0])
    assert "route-pool overflow" in calls[0][1]
    monkeypatch.setattr(torch, "_assert_async", None)
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    with pytest.raises(RuntimeError, match="receive route-pool overflow"):
        _check_overflow(torch.ones(1, dtype=torch.int32))

    class Backend:
        close_calls = 0

        def close(self):
            self.close_calls += 1
            raise RuntimeError("cleanup failed")

    operator = MoeEp(**_forward_config())
    backend = Backend()
    operator._forward_backend = backend
    with pytest.warns(ResourceWarning, match="cleanup failed"):
        operator.__del__()
    assert backend.close_calls == 1
    assert not hasattr(api_module, "_FAILED_FINALIZER_BACKENDS")
    operator._forward_backend = None
    operator._closed = True

    args = (
        torch.empty((3, 128), dtype=torch.bfloat16),
        torch.empty((2, 128, 256), dtype=torch.bfloat16),
        torch.empty((2, 128, 128), dtype=torch.bfloat16),
        torch.zeros((3, 2), dtype=torch.int32),
        torch.ones((3, 2), dtype=torch.float32),
    )
    with MoeEp(
        **_forward_config(intermediate_size=128, max_tokens_per_rank=3)
    ) as operator:
        with pytest.raises(
            NotImplementedError,
            match=r"intermediate_size .*divisible by 256",
        ):
            operator(*args)
        assert operator._forward_backend is None
