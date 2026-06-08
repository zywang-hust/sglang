"""InfLLM-V2 sparse FlashAttention public API.

Ported (drop-in) from ``3rdparty/infllmv2_cuda_impl/infllm_v2/infllmv2_sparse_attention.py``.
The CUDA backend now lives in the standalone ``infllm_ops`` extension and the
bitmask helpers come from ``sgl_kernel.infllm_v2``.
"""

from typing import Optional, Tuple, Union

import torch
from sgl_kernel.infllm_v2._loader import load_infllm_ops
from sgl_kernel.infllm_v2.bitmask import blockmask_to_uint64 as cuda_blockmask_to_uint64
from sgl_kernel.infllm_v2.bitmask import topk_to_uint64 as cuda_topk_to_uint64
from sgl_kernel.infllm_v2.bitmask import uint64_to_bool as cuda_uint64_to_bool


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def round_multiple(x, m):
    return (x + m - 1) // m * m


# torch.compile() support is only enabled for pytorch >= 2.4
if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:

    def noop_custom_op_wrapper(
        name, fn=None, /, *, mutates_args, device_types=None, schema=None
    ):
        def wrap(func):
            return func

        if fn is None:
            return wrap
        return fn

    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func

        if fn is None:
            return wrap
        return fn

    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper


@_torch_custom_op_wrapper(
    "sgl_infllmv2_attn::_infllmv2_attn_varlen_forward",
    mutates_args=(),
    device_types="cuda",
)
def _infllmv2_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
    block_table: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    topk_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    infllm_ops = load_infllm_ops()
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    if topk_idx is not None:
        nheads_q = q.shape[1]
        nheads_k = k.shape[1]
        group_size = nheads_q // nheads_k
        head_dim = q.shape[-1]

        if nheads_k == 1:
            q = q.reshape(-1, 1, head_dim).contiguous()
            cu_seqlens_q = cu_seqlens_q * nheads_q
            max_seqlen_q = max_seqlen_q * nheads_q
        else:
            q = (
                q.reshape(-1, nheads_k, group_size, head_dim)
                .transpose(1, 2)
                .reshape(-1, nheads_k, head_dim)
                .contiguous()
            )
            cu_seqlens_q = cu_seqlens_q * group_size
            max_seqlen_q = max_seqlen_q * group_size

        assert topk_idx.dtype == torch.int32
        fwd_blockmask_uint64, _ = cuda_topk_to_uint64(
            topk_idx, max_seqlen_k, 64
        )  # N_BLOCK_DIM=64
    else:
        fwd_blockmask_uint64 = None

    out, softmax_lse, S_dmask, rng_state = infllm_ops.varlen_fwd(
        q,
        k,
        v,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        None,
        fwd_blockmask_uint64,
    )
    if topk_idx is not None:
        if nheads_k == 1:
            out = out.reshape(-1, nheads_q, head_dim).contiguous()
        else:
            out = (
                out.reshape(-1, group_size, nheads_k, head_dim)
                .transpose(1, 2)
                .reshape(-1, nheads_q, head_dim)
                .contiguous()
            )

    return out, softmax_lse, S_dmask, fwd_blockmask_uint64, rng_state


_wrapped_infllmv2_attn_varlen_forward = _infllmv2_attn_varlen_forward


@_torch_custom_op_wrapper(
    "sgl_infllmv2_attn::_infllmv2_attn_varlen_backward",
    mutates_args=("dq", "dk", "dv"),
    device_types="cuda",
)
def _infllmv2_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    bwd_blockmask_uint64: Optional[torch.Tensor] = None,
    rng_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    infllm_ops = load_infllm_ops()
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    group_size = q.shape[-2] // k.shape[-2]
    total_q, nheads_q, dim = q.shape
    nheads_k = k.shape[-2]

    if nheads_k == 1:
        q_final = q.reshape(total_q * nheads_q, 1, dim).contiguous()
        dout_final = dout.reshape(total_q * nheads_q, 1, dim).contiguous()
        out_final = out.reshape(total_q * nheads_q, 1, dim).contiguous()
    else:
        q_final = q.reshape(total_q, nheads_k, group_size, dim)
        q_final = q_final.permute(0, 2, 1, 3)
        q_final = q_final.reshape(total_q * group_size, nheads_k, dim).contiguous()

        dout_final = dout.reshape(total_q, nheads_k, group_size, dim)
        dout_final = dout_final.permute(0, 2, 1, 3)
        dout_final = dout_final.reshape(
            total_q * group_size, nheads_k, dim
        ).contiguous()

        out_final = out.reshape(total_q, nheads_k, group_size, dim)
        out_final = out_final.permute(0, 2, 1, 3)
        out_final = out_final.reshape(total_q * group_size, nheads_k, dim).contiguous()

    if nheads_k == 1:
        cu_seqlens_q_expanded = torch.zeros(
            cu_seqlens_q.shape[0], device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype
        )
        for i in range(cu_seqlens_q.shape[0] - 1):
            cu_seqlens_q_expanded[i + 1] = (
                cu_seqlens_q[i + 1] - cu_seqlens_q[i]
            ) * nheads_q + cu_seqlens_q_expanded[i]
        max_seqlen_q_expanded = max_seqlen_q * nheads_q
    else:
        cu_seqlens_q_expanded = torch.zeros(
            cu_seqlens_q.shape[0], device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype
        )
        for i in range(cu_seqlens_q.shape[0] - 1):
            cu_seqlens_q_expanded[i + 1] = (
                cu_seqlens_q[i + 1] - cu_seqlens_q[i]
            ) * group_size + cu_seqlens_q_expanded[i]
        max_seqlen_q_expanded = max_seqlen_q * group_size

    dq_temp = torch.empty_like(q_final)

    _, _, _, softmax_d = infllm_ops.varlen_bwd(
        dout_final,
        q_final,
        k,
        v,
        out_final,
        softmax_lse,
        dq_temp,
        dk,
        dv,
        cu_seqlens_q_expanded,
        cu_seqlens_k,
        alibi_slopes,
        max_seqlen_q_expanded,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        bwd_blockmask_uint64,
        None,
        rng_state,
    )
    dout_final = out_final = None
    torch.cuda.empty_cache()

    if nheads_k == 1:
        dq_temp = dq_temp.reshape(total_q, nheads_q, dim)
    else:
        dq_temp = dq_temp.reshape(total_q, group_size, nheads_k, dim)
        dq_temp = dq_temp.permute(0, 2, 1, 3)
        dq_temp = dq_temp.reshape(total_q, nheads_q, dim)

    dq.copy_(dq_temp)

    q_final = dq_temp = None
    torch.cuda.empty_cache()
    return softmax_d


if torch.__version__ >= "2.4.0":
    _wrapped_infllmv2_attn_varlen_backward = (
        torch.ops.sgl_infllmv2_attn._infllmv2_attn_varlen_backward
    )
else:
    _wrapped_infllmv2_attn_varlen_backward = _infllmv2_attn_varlen_backward


class Infllmv2AttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        block_table,
        topk_idx,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        (
            out_padded,
            softmax_lse,
            S_dmask,
            fwd_blockmask_uint64,
            rng_state,
        ) = _wrapped_infllmv2_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=block_table,
            topk_idx=topk_idx,
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            out_padded,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            fwd_blockmask_uint64,
            rng_state,
        )
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.topk_idx = topk_idx
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            fwd_blockmask_uint64,
            rng_state,
        ) = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        bwd_blockmask_uint64 = None
        if fwd_blockmask_uint64 is not None:
            fwd_blockmask_bool = cuda_uint64_to_bool(
                fwd_blockmask_uint64, (ctx.max_seqlen_k + 64 - 1) // 64
            )
            transposed_blockmask = fwd_blockmask_bool.transpose(1, 2).contiguous()
            torch.cuda.synchronize()
            bwd_blockmask_uint64, _ = cuda_blockmask_to_uint64(transposed_blockmask)

        head_size_og = dout.size(2)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
        _wrapped_infllmv2_attn_varlen_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            bwd_blockmask_uint64,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def infllmv2_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    topk_idx=None,
):
    """Variable-length InfLLM-V2 sparse attention (supports autograd).

    Drop-in replacement for ``infllm_v2.infllmv2_attn_varlen_func``. Supports
    MQA/GQA. See the original docstring for full argument semantics.
    """
    return Infllmv2AttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_table,
        topk_idx,
    )


def infllmv2_attn_stage1(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    cu_seqlens_v,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=True,
    block_table=None,
):
    """Neighborhood Sparse Attention (NSA) Stage 1 with varlen support.

    Drop-in replacement for ``infllm_v2.infllmv2_attn_stage1``. Returns the
    attention-score matrix with the NSA sparsity pattern, shape
    ``(num_heads_k, total_q, max_seqlen_k)``.
    """
    infllm_ops = load_infllm_ops()
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    total_q, nheads, head_dim = q.shape
    nheads_k = k.shape[1]
    nheads_per_group = nheads // nheads_k

    q = q.reshape(total_q, nheads_k, nheads_per_group, head_dim)
    q = (
        q.transpose(1, 2)
        .reshape(total_q * nheads_per_group, nheads_k, head_dim)
        .contiguous()
    )

    result = infllm_ops.varlen_fwd_stage1(
        q,
        k,
        v,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_v,
        None,
        None,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        True,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        True,
        None,
    )

    return result[0]


def infllmv2_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
    topk_idx=None,
):
    """InfLLM-V2 attention with a (paged) KV cache.

    Drop-in replacement for ``infllm_v2.infllmv2_attn_with_kvcache``. Does not
    support backward. See the original docstring for full argument semantics.
    """
    infllm_ops = load_infllm_ops()
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
    if topk_idx is not None:
        assert topk_idx.dtype == torch.int32
        blockmask, _ = cuda_topk_to_uint64(
            topk_idx,
            (
                k_cache.shape[1]
                if block_table is None
                else block_table.shape[1] * k_cache.shape[1]
            ),
            64,
        )  # N_BLOCK_DIM=64
    else:
        blockmask = None
    out, softmax_lse = infllm_ops.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        None,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        num_splits,
        blockmask,
    )
    return (out, softmax_lse) if return_softmax_lse else out
