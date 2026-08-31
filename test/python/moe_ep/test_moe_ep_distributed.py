# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""MoE EP single-node multiprocessing execution.

See ``docs/fe-oss-apis/moe_ep.md`` for canonical user examples.
"""

from __future__ import annotations

import os

import pytest
import torch.multiprocessing as mp

from moe_ep.moe_ep_distributed_workers import (
    _distributed_backward_reference_worker,
    _distributed_output_worker,
    _distributed_subgroup_backward_reference_worker,
    _distributed_subgroup_output_worker,
)
from moe_ep.moe_ep_test_support import _require_distributed_sm107

pytestmark = [
    pytest.mark.L1,
    pytest.mark.gpu_exclusive,
    pytest.mark.moe_ep_distributed,
]


@pytest.mark.parametrize(
    ("world_size", "combine_format"),
    [
        pytest.param(2, "bf16", id="ep2-bf16"),
        pytest.param(3, "mxfp8", id="ep3-mxfp8"),
        pytest.param(4, "bf16", id="ep4-bf16"),
    ],
)
def test_forward_multi_gpu_matches_reference(
    world_size,
    combine_format,
    tmp_path,
):
    _require_distributed_sm107(world_size)
    os.environ.setdefault("NVIDIA_IMEX_CHANNELS", "0")
    init_file = tmp_path / f"{combine_format}_combine_ep{world_size}.init"
    mp.spawn(
        _distributed_output_worker,
        args=(world_size, str(init_file), combine_format),
        nprocs=world_size,
        join=True,
    )


def test_forward_noncontiguous_ep2_subgroups(tmp_path):
    global_world_size = 4
    _require_distributed_sm107(global_world_size)
    os.environ.setdefault("NVIDIA_IMEX_CHANNELS", "0")
    init_file = tmp_path / "two_noncontiguous_ep2.init"
    mp.spawn(
        _distributed_subgroup_output_worker,
        args=(global_world_size, str(init_file)),
        nprocs=global_world_size,
        join=True,
    )


@pytest.mark.parametrize(
    ("world_size", "combine_format", "gate_up_clamp"),
    [
        pytest.param(2, "bf16", None, id="ep2-bf16"),
        pytest.param(2, "mxfp8", None, id="ep2-mxfp8"),
        pytest.param(4, "bf16", None, id="ep4-bf16"),
        pytest.param(2, "bf16", 0.5, id="ep2-bf16-clamp-0.5"),
    ],
)
def test_grouped_wgrad_multi_gpu_matches_reference(
    world_size,
    combine_format,
    gate_up_clamp,
    tmp_path,
):
    _require_distributed_sm107(world_size)
    os.environ.setdefault("NVIDIA_IMEX_CHANNELS", "0")
    clamp_id = "none" if gate_up_clamp is None else str(gate_up_clamp)
    init_file = (
        tmp_path / f"backward_ep{world_size}_{combine_format}_clamp_{clamp_id}.init"
    )
    mp.spawn(
        _distributed_backward_reference_worker,
        args=(
            world_size,
            str(init_file),
            combine_format,
            gate_up_clamp,
        ),
        nprocs=world_size,
        join=True,
    )


def test_grouped_wgrad_noncontiguous_ep2_subgroups(tmp_path):
    global_world_size = 4
    _require_distributed_sm107(global_world_size)
    os.environ.setdefault("NVIDIA_IMEX_CHANNELS", "0")
    init_file = tmp_path / "backward_two_noncontiguous_ep2.init"
    mp.spawn(
        _distributed_subgroup_backward_reference_worker,
        args=(global_world_size, str(init_file)),
        nprocs=global_world_size,
        join=True,
    )
