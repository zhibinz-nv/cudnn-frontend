# Mixture of Experts with Expert Parallelism

The MoeEP operation fuses token routing, expert SwiGLU computation, and
expert-parallel communication. Global experts are sharded contiguously across
the ranks of an expert-parallel process group.

## Operation

Let $x_t \in \mathbb{R}^{H}$ be token $t$, $e_{t,k}$ its $k$-th
selected global expert, and $p_{t,k}$ the corresponding routing weight. For
each valid route, split the FC1 result into gate and up projections:

$$
\left[g_{t,k}, u_{t,k}\right]
    = x_t W^{\mathrm{fc1}}_{e_{t,k}},
\qquad
h_{t,k}
    = p_{t,k}\left(\mathrm{SiLU}(g_{t,k}) \odot u_{t,k}\right),
\qquad
z_{t,k}
    = h_{t,k} W^{\mathrm{fc2}}_{e_{t,k}}.
$$

The final token output is the sum over its selected experts:

$$
y_t = \sum_{\substack{0 \le k < K \\ e_{t,k} \ne -1}} z_{t,k}.
$$

When `gate_up_clamp=C`, the operation uses
$\min(g_{t,k}, C)$ for the gate and
$\mathrm{clip}(u_{t,k}, -C, C)$ for the up projection. A route whose
expert ID is `-1` contributes zero. Because the executable backend requires
`apply_topk_in_fc1=True`, it applies $p_{t,k}$ to the SwiGLU result before
FC2. The backend also stages plain inputs to MXFP8 and requantizes the routed
intermediate before FC2, so the equations describe the mathematical operation
rather than its finite-precision rounding.

With $E$ global experts and an expert-parallel group of size $P$, each rank
stores $E_{\mathrm{local}}=E/P$ consecutive experts. Global expert $e$ is
owned by group-relative rank

$$
\mathrm{owner}(e)
    = \left\lfloor \frac{e}{E_{\mathrm{local}}} \right\rfloor.
$$

## Python API

The operation is exposed by the frontend-only `cudnn.MoeEp` object API. Static
model, parallelism, capacity, and format choices are set in the constructor:

```python
from cudnn import MoeEp

op = MoeEp(
    num_experts=E,
    hidden_size=H,
    intermediate_size=I,
    top_k=K,
    ep_group=ep_group,                 # None for EP1
    max_tokens_per_rank=max_tokens,
    max_recv_size_per_rank=None,       # Defaults to P * max_tokens * K
    drop_on_overflow=False,
    output_format="bf16",
    combine_format="bf16",             # "bf16" or "mxfp8"
    apply_topk_in_fc1=True,
    gate_up_clamp=None,
)
```

`topk_idx` contains global expert IDs. Each rank passes its local tokens and
its contiguous shard of both expert-weight tensors:

```python
output = op(
    activation,       # (T, H)
    fc1_weight,       # (E_local, H, 2I)
    fc2_weight,       # (E_local, I, H)
    topk_idx,         # (T, K), global expert IDs or -1
    topk_weights,     # (T, K)
)                     # (T, H), BF16
```

For inference CUDA Graph capture, call `op.warmup(...)` with the exact
bindings before capture. `MoeEp` supports `close()` and context-manager use.

Slotless training uses the same operator with graph-stable weights,
caller-owned contexts/outputs, and FE-owned execution lanes:

```python
from cudnn import MoeEpTrainingWeights

weights = MoeEpTrainingWeights(
    forward_fc1=forward_fc1_mxfp8,
    forward_fc2=forward_fc2_mxfp8,
    backward_w2_transpose=backward_w2t_mxfp8,
    backward_w1_transpose=backward_w1t_mxfp8,
)
plan = op.prepare_training(weights, lane_count=1)
context = te_context_cache.acquire(
    plan.buffer_specs, token_count=activation.shape[0]
)
lane = plan.lanes[0]

plan.refresh_weights()
output = plan.forward(
    context, lane, activation, topk_idx, topk_weights, out=output_buffer
)
grad_activation, dprob, wgrad_operands = plan.backward(
    context,
    lane,
    grad_output,
    grad_activation_out=grad_activation_buffer,
    dprob_out=dprob_buffer,
)
overflow = plan.finalize_overflow(
    (context,), lane, out=overflow_buffer
)
```

The WGrad result is a fixed-capacity grouped-GEMM operand bundle, not dense
optimizer-ready weight gradients. See the detailed
[MoE + Expert Parallel API](../fe-oss-apis/moe_ep.md) reference for
installation, all constructor arguments, tensor formats, training context
lifecycle, tuning, overflow handling, and CUDA Graph requirements. MoeEP is
distinct from the cuDNN graph [MoE Grouped Matmul](MoeGroupedMatmul.md)
operation.

## Execution support

- NVIDIA Rubin SM107 GPUs (compute capability 10.7).
- CUDA and PyTorch execution.
- `nvidia-cutlass-dsl>=4.8.0` for the Rubin kernels. The package-wide
  `cutedsl` extra retains its 4.5.0 installation floor so other cuDNN Frontend
  operations remain usable with older compatible DSL versions.
- Fused SwiGLU with contiguous expert sharding.
- `apply_topk_in_fc1=True`.
- `hidden_size` divisible by 128.
- `intermediate_size` divisible by 256.
- `top_k <= min(32, num_experts)`.
- `num_experts` divisible by the expert-parallel group size.
- An explicit positive `max_tokens_per_rank`.

The slotless CUDA Graph path has hardware acceptance through EP32 when
all ranks are in one direct-P2P MNNVL peer-access domain. The Python capability
layer does not impose an EP-size ceiling; cross-MNNVL execution is not part of
the validated support surface.

## Data formats

Inference activation and expert weights accept:

- BF16, FP16, or FP32 plain tensors, staged internally to MXFP8; or
- MXFP8 `BlockScaledTensor` values with logical block axis 1.

The current executable output format is BF16. The expert-combine path accepts
BF16 or MXFP8. NVFP4 types are represented by the public API but native NVFP4
operands, combine, and output are not executable by this backend.

Slotless training narrows dynamic activation and gradient inputs to
contiguous BF16 or FP32 tensors. Training weights are contiguous MXFP8
block-scaled tensors.

## Tensor contracts

Let:

- $T$ be the local token count;
- $H$ be `hidden_size`;
- $I$ be `intermediate_size`;
- $K$ be `top_k`;
- $E_{\mathrm{local}}$ be the local expert count.

Inference uses:

- `activation`: `(T, H)`;
- `topk_idx`: `(T, K)`, Int32 or Int64, containing `-1` or a valid global
  expert ID;
- `topk_weights`: `(T, K)`, floating point;
- FC1 weights: `(E_local, H, 2I)`;
- FC2 weights: `(E_local, I, H)`;
- output: `(T, H)`, BF16.

All inference tensors must reside on one device, and the local token count must
satisfy `T <= max_tokens_per_rank`.

Slotless training uses a narrower graph-stable contract:

- `activation` and `grad_output`: contiguous `(T, H)`, BF16 or FP32;
- `topk_idx`: contiguous `(T, K)`, Int32;
- `topk_weights`: contiguous `(T, K)`, FP32;
- forward FC1 and FC2 weights: contiguous MXFP8 block-scaled tensors with
  shapes `(E_local, H, 2I)` and `(E_local, I, H)`;
- transposed backward weights: contiguous MXFP8 block-scaled tensors with
  shapes `(E_local, H, I)` and `(E_local, 2I, H)`;
- forward output: `(T, H)`, BF16;
- `grad_activation`: caller-provided `(T, H)`, FP32;
- `dprob`: source-order `(T, K)`, FP32;
- `wgrad_operands`: a fixed-capacity `MoeEpTrainingWgradOperands` bundle.

All dynamic training tensors must reside on one device and satisfy
`T <= max_tokens_per_rank`.

## Expert-parallel communication

EP2+ execution requires:

- an initialized NCCL process group;
- `nvshmem4py` and usable NVSHMEM libraries;
- direct peer access among every pair of participating ranks; and
- consistent rank ordering, plan/context schemas, tuning, context selection, and lane
  ordering across the group.

`max_recv_size_per_rank` bounds receive capacity. When omitted, it defaults to
the worst-case route count:

```text
ep_size * max_tokens_per_rank * top_k
```

Plans and contexts cannot grow during CUDA Graph replay. Capacity or storage
changes require plan preparation, new contexts, and graph capture again.
