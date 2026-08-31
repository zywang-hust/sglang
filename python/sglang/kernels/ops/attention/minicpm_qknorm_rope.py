"""Fused per-head Q/K RMSNorm + NeoX RoPE, in place on a fused QKV tensor,
for MiniCPM lightning (SALA) mixers.

The generic fused_qk_norm_rope op is not reused:
it recomputes cos/sin in kernel from base/factor under fast-math --
not bit-parity with the eager rope path.
This kernel instead reads the fp32 cos_sin_cache the eager rope reads,
and rounds the normed values to the storage dtype before rotation,
so wherever the layout applies it runs without a numerical-safety flag."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# BLOCK_M=16, 2 warps: amortizes the weight load over tokens,
# with little tail waste at decode-sized T.
_BLOCK_M = 16


@triton.jit
def _minicpm_qknorm_rope_kernel(
    QKV,
    Q_W,
    K_W,
    COS_SIN,
    POSITIONS,
    T,
    NH_Q: tl.constexpr,
    NH_QKV: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    block_idx = tl.program_id(0)
    head_lin = tl.program_id(1)
    is_q = head_lin < NH_Q

    rows = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < T
    rows64 = rows.to(tl.int64)
    cols = tl.arange(0, D)
    offs = rows64[:, None] * (NH_QKV * D) + head_lin * D + cols[None, :]
    x = tl.load(QKV + offs, mask=rmask[:, None], other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=1) / D
    rstd = 1.0 / tl.sqrt(var + EPS)
    if is_q:
        w = tl.load(Q_W + cols).to(tl.float32)
    else:
        w = tl.load(K_W + cols).to(tl.float32)
    # Round the normed value to the storage dtype before RoPE --
    # same rounding point as the eager rmsnorm-kernel -> rope-kernel chain.
    xn = (x * rstd[:, None] * w[None, :]).to(QKV.dtype.element_ty).to(tl.float32)
    x1n, x2n = tl.split(tl.permute(tl.reshape(xn, (BLOCK_M, 2, HALF)), (0, 2, 1)))

    half_cols = tl.arange(0, HALF)
    pos = tl.load(POSITIONS + rows, mask=rmask, other=0).to(tl.int64)
    cos_offs = pos[:, None] * D + half_cols[None, :]
    cos = tl.load(COS_SIN + cos_offs)
    sin = tl.load(COS_SIN + cos_offs + HALF)

    y1 = x1n * cos - x2n * sin
    y2 = x2n * cos + x1n * sin
    offs1 = rows64[:, None] * (NH_QKV * D) + head_lin * D + half_cols[None, :]
    tl.store(QKV + offs1, y1.to(QKV.dtype.element_ty), mask=rmask[:, None])
    tl.store(QKV + offs1 + HALF, y2.to(QKV.dtype.element_ty), mask=rmask[:, None])


def minicpm_qknorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused q/k per-head rmsnorm + neox rope,
    in place on the Q/K segments of ``qkv`` (V untouched).

    qkv: ``(T, (NH_Q + NH_K + NH_V) * D)`` contiguous bf16/fp16;
    q_weight/k_weight: ``(D,)`` rmsnorm weights;
    cos_sin_cache: ``(max_pos, D)`` fp32, ``[cos | sin]`` per row;
    positions: ``(T,)`` int32/int64.
    Returns q/k/v ``(T, NH, D)`` strided views into ``qkv``.
    """
    T = qkv.shape[0]
    num_qkv_heads = num_q_heads + num_k_heads + num_v_heads
    grid = (triton.cdiv(T, _BLOCK_M), num_q_heads + num_k_heads)
    _minicpm_qknorm_rope_kernel[grid](
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        T,
        NH_Q=num_q_heads,
        NH_QKV=num_qkv_heads,
        D=head_dim,
        HALF=head_dim // 2,
        EPS=eps,
        BLOCK_M=_BLOCK_M,
        num_warps=2,
        num_stages=2,
    )
    heads = qkv.view(T, num_qkv_heads, head_dim)
    return (
        heads[:, :num_q_heads],
        heads[:, num_q_heads : num_q_heads + num_k_heads],
        heads[:, num_q_heads + num_k_heads :],
    )
