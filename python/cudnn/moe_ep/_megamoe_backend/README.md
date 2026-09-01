# MegaMoE backend

The private MegaMoE backend provides Rubin SM107 MXFP8 execution for
`cudnn.moe_ep`.

## Executable capability

- CUDA Rubin SM107 (compute capability 10.7)
- BF16 output with BF16 or MXFP8 combine
- `hidden_size % 128 == 0`
- `intermediate_size % 256 == 0`
- `top_k <= min(32, num_experts)`
- explicit positive `max_tokens_per_rank`
- `apply_topk_in_fc1=True`

Inference accepts plain BF16/FP16/FP32 or MXFP8 operands and stages plain
operands to MXFP8. Slotless training accepts contiguous BF16/FP32
activation and grad-output tensors, contiguous Int32 routing indices,
contiguous FP32 routing weights, and four contiguous MXFP8 training-weight
packs. NVFP4 operands, non-BF16 output, and `apply_topk_in_fc1=False` are not
executable.

## Public execution paths

`MoeEp.__call__` is the inference-forward surface. It returns only the fused
BF16 `(T, H)` output and does not expose a compact training stash or backward.
Inference CUDA Graph capture requires `MoeEp.warmup` with the exact capture
bindings before capture. EP ranks must align after warmup and replay in the
same cross-rank order.

Training uses a slotless plan and caller-owned context:

```python
plan = op.prepare_training(weights, lane_count=1)
context0 = te_context_cache.acquire(
    plan.buffer_specs, token_count=activation0.shape[0]
)
lane0 = plan.lanes[0]

plan.refresh_weights()
y0 = plan.forward(
    context0, lane0, x0, topk_idx0, topk_weights0, out=output0
)
dx0, dprob0, wgrad0 = plan.backward(
    context0,
    lane0,
    grad0,
    grad_activation_out=grad_activation0,
    dprob_out=dprob0_buffer,
)
overflow = plan.finalize_overflow(
    (context0,), lane0, out=overflow0
)
```

`prepare_training` is collective over the EP group, executes outside
capture, and fixes the training kernel to FC1-preactivation generation,
fixed-capacity WGrad operands, and token/scale-factor padding 128.

The same methods execute ordinarily during warmup and enqueue identical nodes
inside a caller-owned outer CUDA Graph. MoeEP does not own or wrap graph
replay. The ordinary warmup must cover
`refresh_weights -> forward -> backward -> finalize_overflow` so staging,
forward, backward, and WGrad-export kernels are compiled before capture.

## Slotless ownership model

- A TE-owned context holds one microbatch's routing snapshot, pool-native FC1
  preactivation, overflow flags, backward auxiliaries, and fixed-capacity
  WGrad operands.
- An execution lane owns mutable router, barrier, and kernel scratch.
- Forward, grad-activation, dprob, and overflow destinations are ordinary
  caller-owned output buffers.
- Each context records the exact shared forward/backward token extent.
  Buffer-spec lifetimes apply after queued eager work completes; every
  `capture_pinned` address remains valid until its graph executable is
  destroyed.
- Every symmetric region is built in deterministic order and its size is
  normalized by name across EP ranks before allocation.
- Multiple streams require distinct lanes. Distributed MegaMoE kernels must be
  ordered consistently on every rank with captured CUDA events; independent
  lane storage does not permit unordered communication overlap.
- `max_recv_size_per_rank` bounds allocation. Capacity never grows during
  capture; changing it requires a new plan, contexts, and graph capture.

## Weights and WGrad outputs

`MoeEpTrainingWeights` contains four address-stable MXFP8 block-scaled tensors:
forward W1/W2 and independently quantized backward W2-transpose/W1-transpose.
Their public layout differs from the K-major, gate/up-interleaved, and
blocked-scale kernel bindings. After every in-place data+scale update, the
caller must enqueue `plan.refresh_weights()` before the first consumer,
with explicit stream/event ordering. A matching forward/backward pair must use
one version; refresh cannot overlap any consumer on another context/lane. Replacing
source storage requires closing the old operator, creating a new `MoeEp`
instance and plan, and capturing a new graph. Closed plans are
terminal and cannot be replaced on the same operator. Capturing the refresh
turns these transforms into fixed-address graph nodes, so replay does not call
Python.

Backward returns kernel dprob directly. It follows the MXFP8-staged numerical
contract and relaxed atomic accumulation order.

`MoeEpTrainingWgradOperands` is a non-owning view of the TE context's
fixed-capacity producer ABI. The normal-backward kernel retains
`dfc2_recompute` and writes the FC2 activation operand into that context.
Device
`expert_offsets` and `valid_route_counts` describe the current valid K extent;
padding is zeroed. No specific downstream grouped-WGrad consumer is guaranteed
by this milestone.

## Overflow policy

`max_recv_size_per_rank` bounds the fixed receive pool. When omitted, it uses
the worst-case `ep_size * max_tokens_per_rank * top_k`; an explicit value is
capped at that count.

The slotless transport truncates deterministically so every rank
completes its communication protocol. `finalize_overflow` aggregates the
selected contexts and performs a scalar MAX all-reduce for EP2+. With
`drop_on_overflow=True`, it returns a one-element Int32 status tensor and
dropped routes contribute zero. With `drop_on_overflow=False`, the graph tail
uses `torch._assert_async`; EP2+ error mode requires NCCL.

## Distributed support

Hardware acceptance covers EP1, EP2/4, EP8, EP16, and EP32 on one MNNVL
peer-access domain. The Python capability layer has no hard EP-size ceiling;
the listed sizes are validated scope rather than cross-MNNVL support.

Current tests additionally cover single-node EP3 inference, noncontiguous EP2
subgroups, multi-node EP4/6/12/16 forward, multi-node EP8/16/32 backward, and
EP8/16/32 slotless graph launchers. EP2+ probes perform collective
warmup, independent capture, capture alignment, diagnostic replay, lockstep
production-like replay bursts, overflow/recovery, ordered multi-lane
execution, and collective teardown.

The kernels use direct peer pointers obtained from NVSHMEM symmetric tensors.
`NVSHMEM_REMOTE_TRANSPORT=none` is valid only when every EP rank is directly
P2P-accessible (`NVSHMEM_TEAM_SHARED` spans the EP world). IBRC initialization
alone does not make non-P2P peers directly addressable by these kernels.
