# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""CUTLASS DSL version-gate tests for Rubin MegaMoE."""

import pytest


@pytest.mark.L0
@pytest.mark.parametrize(
    ("version", "accepted"),
    [
        pytest.param("4.5.0", False, id="old"),
        pytest.param("4.8.0rc1", True, id="prerelease"),
        pytest.param("4.8.0", True, id="minimum"),
        pytest.param("4.9.0", True, id="newer"),
    ],
)
def test_rubin_cutedsl_version_gate(monkeypatch, version, accepted):
    from cudnn.moe_ep._megamoe_backend.mxfp8 import _cutedsl

    monkeypatch.setattr(_cutedsl, "_public_cutedsl_version", lambda: version)

    if accepted:
        _cutedsl.require_rubin_cutedsl()
    else:
        with pytest.raises(
            RuntimeError,
            match=r"nvidia-cutlass-dsl>=4\.8\.0",
        ):
            _cutedsl.require_rubin_cutedsl()


@pytest.mark.L0
@pytest.mark.parametrize(
    "module_name, function_name",
    [
        (
            "cudnn.moe_ep._megamoe_backend.mxfp8._compile",
            "prepare_kernel",
        ),
        (
            "cudnn.moe_ep._megamoe_backend.mxfp8._backward_compile",
            "prepare_backward_kernel",
        ),
    ],
)
def test_rubin_prepare_gates_before_cuda_initialization(
    monkeypatch,
    module_name,
    function_name,
):
    module = __import__(module_name, fromlist=[function_name])

    class GateReached(RuntimeError):
        pass

    def reject():
        raise GateReached

    monkeypatch.setattr(module, "require_rubin_cutedsl", reject)

    with pytest.raises(GateReached):
        getattr(module, function_name)(None, None, None)
