import logging
import math
from typing import Optional, Union

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
    fused_mamba_state_scatter_with_mask,
)

# Import Simple GLA from fla if available
try:
    from fla.ops.simple_gla import chunk_simple_gla
    from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
    from fla.ops.utils.op import exp as _fla_exp

    SIMPLE_GLA_AVAILABLE = True
except ImportError:
    SIMPLE_GLA_AVAILABLE = False
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "USE_INITIAL_STATE_INDICES": lambda args: args["h0_indices"] is not None,
        "STORE_HT": lambda args: args["ht_all"] is not None,
    }
)
@triton.jit
def _fused_recurrent_gla_verify_tree_paths_kernel(
    q,
    k,
    v,
    g_gamma,
    o,
    h0,
    h0_indices,
    ht_all,
    cu_seqlens,
    scale,
    retrieve_parent_token_ptr,
    q_stride_t,
    q_stride_h,
    k_stride_t,
    k_stride_h,
    v_stride_t,
    v_stride_h,
    o_stride_nk,
    o_stride_t,
    o_stride_h,
    h0_stride_n,
    h0_stride_h,
    ht_stride_n,
    ht_stride_t,
    ht_stride_h,
    stride_retrieve_parent_token_seq,
    stride_retrieve_parent_token_token,
    T: tl.constexpr,
    NP2_T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_STATE_INDICES: tl.constexpr,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr,
    STORE_HT: tl.constexpr,
):
    # One program per (seq, token, head, state tile). Each token's state is
    # recomputed from the live initial state by replaying its root-to-token
    # path in registers, so no state ever transits memory between tokens.
    # The replay applies the same fp32 decay/accumulation order as the fla
    # reference fused_recurrent_simple_gla, keeping every state and output
    # bitwise identical to it.
    i_v = tl.program_id(0).to(tl.int64)
    i_k = tl.program_id(1).to(tl.int64)
    i_nth = tl.program_id(2).to(tl.int64)
    i_h = i_nth % H
    i_nt = i_nth // H
    i_t = i_nt % T
    i_n = i_nt // T

    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    # CUDA-graph padding rows carry an empty token range; they replay and
    # store nothing.
    valid_token = i_t < eos - bos

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]
    b_g_gamma = tl.load(g_gamma + i_h)

    # Walk token -> root (same pattern as the commit kernel), then replay the
    # collected path in root-first order. Chain mode (topk=1) shares the walk
    # with an implicit parent of node - 1, so both modes feed the replay loop
    # the same dataflow and compile to the same floating-point schedule.
    token_offsets = tl.arange(0, NP2_T)
    path_nodes = tl.zeros((NP2_T,), dtype=tl.int64)
    node = i_t
    done = ~valid_token
    depth = tl.full((), 0, dtype=tl.int64)
    for i in tl.static_range(0, T):
        active = ~done
        path_nodes = tl.where(
            token_offsets == i,
            tl.where(active, node, 0),
            path_nodes,
        )
        depth += active.to(tl.int64)
        valid_parent_load = active & (node >= 0) & (node < T)
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            parent_node = tl.load(
                retrieve_parent_token_ptr
                + i_n * stride_retrieve_parent_token_seq
                + node * stride_retrieve_parent_token_token,
                mask=valid_parent_load,
                other=0,
            ).to(tl.int64)
        else:
            parent_node = tl.maximum(node - 1, 0)
        done = done | (node == 0) | ~valid_parent_load
        node = parent_node

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        h0_n = i_n
        valid_h0 = True
        if USE_INITIAL_STATE_INDICES:
            h0_n = tl.load(h0_indices + i_n).to(tl.int64)
            valid_h0 = h0_n >= 0
        p_h0 = (
            h0
            + h0_n * h0_stride_n
            + i_h * h0_stride_h
            + o_k[:, None] * V
            + o_v[None, :]
        )
        b_h += tl.load(p_h0, mask=m_h & valid_h0, other=0).to(tl.float32)

    for step in tl.static_range(0, T):
        valid_step = step < depth
        reverse_idx = depth - 1 - step
        path_node = tl.sum(tl.where(token_offsets == reverse_idx, path_nodes, 0))
        p_k = k + (bos + path_node) * k_stride_t + i_h * k_stride_h + o_k
        p_v = v + (bos + path_node) * v_stride_t + i_h * v_stride_h + o_v
        b_k = tl.load(p_k, mask=m_k & valid_step, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v & valid_step, other=0).to(tl.float32)
        next_h = b_h * _fla_exp(b_g_gamma) + b_k[:, None] * b_v[None, :]
        b_h = tl.where(valid_step, next_h, b_h)

    p_q = q + (bos + i_t) * q_stride_t + i_h * q_stride_h + o_k
    b_q = tl.load(p_q, mask=m_k & valid_token, other=0).to(tl.float32) * scale
    b_o = b_h * b_q[:, None]
    b_o = tl.sum(b_o, axis=0)
    p_o = o + i_k * o_stride_nk + (bos + i_t) * o_stride_t + i_h * o_stride_h + o_v
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v & valid_token)

    if STORE_HT:
        p_ht = (
            ht_all
            + i_n * ht_stride_n
            + i_t * ht_stride_t
            + i_h * ht_stride_h
            + o_k[:, None] * V
            + o_v[None, :]
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=m_h & valid_token)


def _fused_recurrent_gla_verify_tree_paths(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_gamma: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    cu_seqlens: torch.Tensor,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    ht_buf: Optional[torch.Tensor] = None,
    tokens_per_seq: Optional[int] = None,
) -> torch.Tensor:
    """Target-verify GLA forward with per-token tree-path recompute.

    ``ht_buf`` is only needed when an external consumer reads the per-token
    states afterwards (the ``intermediate_ssm`` scatter-commit fallback); the
    GLA recompute-commit path passes ``None`` and keeps states in registers.

    ``tokens_per_seq`` defaults to the uniform verify layout. Callers packing
    variable-length sequences must pass the maximum sequence length so every
    token gets a program; shorter rows are masked off by ``cu_seqlens``.
    """
    _, total_t, num_heads, head_dim = q.shape
    value_dim = v.shape[-1]
    num_seqs = len(cu_seqlens) - 1

    block_k = min(triton.next_power_of_2(head_dim), 64)
    block_v = min(triton.next_power_of_2(value_dim), 64)
    num_k_blocks = triton.cdiv(head_dim, block_k)
    num_v_blocks = triton.cdiv(value_dim, block_v)
    if tokens_per_seq is None:
        tokens_per_seq = total_t // num_seqs

    o_buf = q.new_empty(num_k_blocks, *v.shape, dtype=torch.float32)
    if retrieve_parent_token is not None:
        stride_parent_seq = retrieve_parent_token.stride(0)
        stride_parent_token = retrieve_parent_token.stride(1)
    else:
        stride_parent_seq = 0
        stride_parent_token = 0

    grid = (num_v_blocks, num_k_blocks, num_seqs * tokens_per_seq * num_heads)
    _fused_recurrent_gla_verify_tree_paths_kernel[grid](
        q=q,
        k=k,
        v=v,
        g_gamma=g_gamma,
        o=o_buf,
        h0=initial_state,
        h0_indices=initial_state_indices,
        ht_all=ht_buf,
        cu_seqlens=cu_seqlens,
        scale=scale,
        retrieve_parent_token_ptr=retrieve_parent_token,
        q_stride_t=q.stride(1),
        q_stride_h=q.stride(2),
        k_stride_t=k.stride(1),
        k_stride_h=k.stride(2),
        v_stride_t=v.stride(1),
        v_stride_h=v.stride(2),
        o_stride_nk=o_buf.stride(0),
        o_stride_t=o_buf.stride(2),
        o_stride_h=o_buf.stride(3),
        h0_stride_n=initial_state.stride(0) if initial_state is not None else 0,
        h0_stride_h=initial_state.stride(1) if initial_state is not None else 0,
        ht_stride_n=ht_buf.stride(0) if ht_buf is not None else 0,
        ht_stride_t=ht_buf.stride(1) if ht_buf is not None else 0,
        ht_stride_h=ht_buf.stride(2) if ht_buf is not None else 0,
        stride_retrieve_parent_token_seq=stride_parent_seq,
        stride_retrieve_parent_token_token=stride_parent_token,
        T=tokens_per_seq,
        NP2_T=triton.next_power_of_2(tokens_per_seq),
        H=num_heads,
        K=head_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None,
    )
    return o_buf.sum(0).to(q.dtype)


@triton.jit
def _recompute_and_scatter_gla_state_from_tree_paths_kernel(
    k,
    v,
    g_gamma,
    states,
    retrieve_parent_token,
    selected_req_indices,
    initial_state_indices,
    commit_nodes,
    commit_dst_indices,
    track_nodes,
    track_dst_indices,
    k_stride_l,
    k_stride_b,
    k_stride_t,
    k_stride_h,
    v_stride_l,
    v_stride_b,
    v_stride_t,
    v_stride_h,
    state_stride_l,
    state_stride_n,
    state_stride_h,
    parent_stride_b,
    parent_stride_t,
    REQS: tl.constexpr,
    DST_SIZE: tl.constexpr,
    DRAFT_TOKEN_NUM: tl.constexpr,
    NP2_T: tl.constexpr,
    PATHS: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_TRACK: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_k = tl.program_id(1).to(tl.int64)
    i_layer_path_h = tl.program_id(2).to(tl.int64)
    i_h = i_layer_path_h % H
    i_layer_path = i_layer_path_h // H
    i_path = i_layer_path % PATHS
    i_layer = i_layer_path // PATHS

    req_idx = tl.load(selected_req_indices + i_path).to(tl.int64)
    h0_idx = tl.load(initial_state_indices + i_path).to(tl.int64)
    valid_req = (req_idx >= 0) & (req_idx < REQS)
    valid_h0 = (h0_idx >= 0) & (h0_idx < DST_SIZE)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]
    token_offsets = tl.arange(0, NP2_T)
    b_g_gamma = tl.load(g_gamma + i_h)

    # Track buffers are prefix-cache slots. Store them before live commit so the
    # tracked state is recomputed from the pre-verify live state.
    for which in tl.static_range(0 if HAS_TRACK else 1, 2):
        if which == 0:
            selected_node = tl.load(track_nodes + i_path).to(tl.int64)
            dst_idx = tl.load(track_dst_indices + i_path).to(tl.int64)
        else:
            selected_node = tl.load(commit_nodes + i_path).to(tl.int64)
            dst_idx = tl.load(commit_dst_indices + i_path).to(tl.int64)

        valid_dst = (dst_idx >= 0) & (dst_idx < DST_SIZE)
        valid_selected = (selected_node >= 0) & (selected_node < DRAFT_TOKEN_NUM)
        valid_target = valid_req & valid_h0 & valid_dst & valid_selected

        path_nodes = tl.zeros((NP2_T,), dtype=tl.int64)
        node = selected_node
        done = ~valid_target
        depth = tl.full((), 0, dtype=tl.int64)
        for i in tl.static_range(0, DRAFT_TOKEN_NUM):
            active = ~done
            path_nodes = tl.where(
                token_offsets == i,
                tl.where(active, node, 0),
                path_nodes,
            )
            depth += active.to(tl.int64)
            valid_parent_load = active & (node >= 0) & (node < DRAFT_TOKEN_NUM)
            parent_node = tl.load(
                retrieve_parent_token
                + req_idx * parent_stride_b
                + node * parent_stride_t,
                mask=valid_parent_load,
                other=0,
            ).to(tl.int64)
            done = done | (node == 0) | ~valid_parent_load
            node = parent_node

        p_h0 = (
            states
            + i_layer * state_stride_l
            + h0_idx * state_stride_n
            + i_h * state_stride_h
            + o_k[:, None] * V
            + o_v[None, :]
        )
        b_h = tl.load(p_h0, mask=m_h & valid_target, other=0).to(tl.float32)

        for step in tl.static_range(0, DRAFT_TOKEN_NUM):
            valid_step = step < depth
            reverse_idx = depth - 1 - step
            path_node = tl.sum(tl.where(token_offsets == reverse_idx, path_nodes, 0))
            p_k = (
                k
                + i_layer * k_stride_l
                + req_idx * k_stride_b
                + path_node * k_stride_t
                + i_h * k_stride_h
                + o_k
            )
            p_v = (
                v
                + i_layer * v_stride_l
                + req_idx * v_stride_b
                + path_node * v_stride_t
                + i_h * v_stride_h
                + o_v
            )
            b_k = tl.load(p_k, mask=m_k & valid_step & valid_target, other=0).to(
                tl.float32
            )
            b_v = tl.load(p_v, mask=m_v & valid_step & valid_target, other=0).to(
                tl.float32
            )
            next_h = b_h * _fla_exp(b_g_gamma) + b_k[:, None] * b_v[None, :]
            b_h = tl.where(valid_step & valid_target, next_h, b_h)

        p_dst = (
            states
            + i_layer * state_stride_l
            + dst_idx * state_stride_n
            + i_h * state_stride_h
            + o_k[:, None] * V
            + o_v[None, :]
        )
        tl.store(p_dst, b_h.to(p_dst.dtype.element_ty), mask=m_h & valid_target)


def _recompute_and_scatter_gla_state_from_tree_paths(
    k: torch.Tensor,
    v: torch.Tensor,
    g_gamma: torch.Tensor,
    states: torch.Tensor,
    retrieve_parent_token: torch.Tensor,
    selected_req_indices: torch.Tensor,
    initial_state_indices: torch.Tensor,
    commit_nodes: torch.Tensor,
    commit_dst_indices: torch.Tensor,
    track_nodes: Optional[torch.Tensor] = None,
    track_dst_indices: Optional[torch.Tensor] = None,
) -> None:
    layers, num_reqs, draft_token_num, num_heads, head_dim = k.shape
    value_dim = v.shape[-1]
    num_paths = selected_req_indices.shape[0]

    if (track_nodes is None) != (track_dst_indices is None):
        raise ValueError("track_nodes and track_dst_indices must be provided together")
    has_track = track_nodes is not None

    block_k = min(triton.next_power_of_2(head_dim), 64)
    block_v = min(triton.next_power_of_2(value_dim), 64)
    num_k_blocks = triton.cdiv(head_dim, block_k)
    num_v_blocks = triton.cdiv(value_dim, block_v)

    grid = (num_v_blocks, num_k_blocks, layers * num_paths * num_heads)
    _recompute_and_scatter_gla_state_from_tree_paths_kernel[grid](
        k=k,
        v=v,
        g_gamma=g_gamma,
        states=states,
        retrieve_parent_token=retrieve_parent_token,
        selected_req_indices=selected_req_indices,
        initial_state_indices=initial_state_indices,
        commit_nodes=commit_nodes,
        commit_dst_indices=commit_dst_indices,
        track_nodes=track_nodes,
        track_dst_indices=track_dst_indices,
        k_stride_l=k.stride(0),
        k_stride_b=k.stride(1),
        k_stride_t=k.stride(2),
        k_stride_h=k.stride(3),
        v_stride_l=v.stride(0),
        v_stride_b=v.stride(1),
        v_stride_t=v.stride(2),
        v_stride_h=v.stride(3),
        state_stride_l=states.stride(0),
        state_stride_n=states.stride(1),
        state_stride_h=states.stride(2),
        parent_stride_b=retrieve_parent_token.stride(0),
        parent_stride_t=retrieve_parent_token.stride(1),
        REQS=num_reqs,
        DST_SIZE=states.shape[1],
        DRAFT_TOKEN_NUM=draft_token_num,
        NP2_T=triton.next_power_of_2(draft_token_num),
        PATHS=num_paths,
        H=num_heads,
        K=head_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        HAS_TRACK=has_track,
    )


@triton.jit
def _build_retrieve_parent_token_kernel(
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    out_ptr,
    active_bs,
    stride_next_seq: tl.constexpr,
    stride_next_token: tl.constexpr,
    stride_sibling_seq: tl.constexpr,
    stride_sibling_token: tl.constexpr,
    stride_out_seq: tl.constexpr,
    stride_out_token: tl.constexpr,
    DRAFT_TOKEN_NUM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    token_idx = tl.arange(0, BLOCK)
    token_mask = token_idx < DRAFT_TOKEN_NUM
    active = batch_idx < active_bs

    next_token = tl.load(
        retrieve_next_token_ptr
        + batch_idx * stride_next_seq
        + token_idx * stride_next_token,
        mask=active & token_mask,
        other=-1,
    )
    next_sibling = tl.load(
        retrieve_next_sibling_ptr
        + batch_idx * stride_sibling_seq
        + token_idx * stride_sibling_token,
        mask=active & token_mask,
        other=-1,
    )
    parent = tl.zeros((BLOCK,), dtype=tl.int32)

    # EAGLE tree nodes are emitted in topological order: a parent appears before
    # its children, and siblings share the current node's parent.
    for curr in tl.static_range(0, DRAFT_TOKEN_NUM):
        child = tl.sum(tl.where(token_idx == curr, next_token, 0))
        if child != -1:
            parent = tl.where(token_idx == child, curr, parent)

        sibling = tl.sum(tl.where(token_idx == curr, next_sibling, 0))
        if sibling != -1:
            curr_parent = tl.sum(tl.where(token_idx == curr, parent, 0))
            parent = tl.where(token_idx == sibling, curr_parent, parent)

    tl.store(
        out_ptr + batch_idx * stride_out_seq + token_idx * stride_out_token,
        parent,
        mask=token_mask,
    )


def _build_retrieve_parent_token_ref(
    retrieve_next_token: Optional[torch.Tensor],
    retrieve_next_sibling: Optional[torch.Tensor],
    out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Pure-Python reference for ``_build_retrieve_parent_token`` (tests only)."""
    if retrieve_next_token is None or retrieve_next_sibling is None:
        return out

    batch_size, draft_token_num = retrieve_next_token.shape
    next_cpu = retrieve_next_token.cpu().tolist()
    sibling_cpu = retrieve_next_sibling.cpu().tolist()
    parent_cpu = [[0] * draft_token_num for _ in range(batch_size)]

    for batch_idx in range(batch_size):
        queue = [0]
        seen = {0}
        for node in queue:
            child = next_cpu[batch_idx][node]
            while child != -1:
                if not 0 <= child < draft_token_num:
                    raise ValueError(
                        f"Invalid EAGLE child index {child} for "
                        f"draft_token_num={draft_token_num}."
                    )
                if child not in seen:
                    parent_cpu[batch_idx][child] = node
                    queue.append(child)
                    seen.add(child)
                sibling = sibling_cpu[batch_idx][child]
                if sibling != -1 and not 0 <= sibling < draft_token_num:
                    raise ValueError(
                        f"Invalid EAGLE sibling index {sibling} for "
                        f"draft_token_num={draft_token_num}."
                    )
                child = sibling

    parent = torch.tensor(
        parent_cpu,
        dtype=retrieve_next_token.dtype,
        device=retrieve_next_token.device,
    )
    if out is None:
        return parent

    out[:batch_size].copy_(parent)
    if out.shape[0] > batch_size:
        out[batch_size:].zero_()
    return out


def _build_retrieve_parent_token(
    retrieve_next_token: Optional[torch.Tensor],
    retrieve_next_sibling: Optional[torch.Tensor],
    out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if retrieve_next_token is None or retrieve_next_sibling is None:
        return out

    assert retrieve_next_token.is_cuda and retrieve_next_sibling.is_cuda

    batch_size, draft_token_num = retrieve_next_token.shape
    if out is None:
        out = torch.empty_like(retrieve_next_token)

    block = triton.next_power_of_2(draft_token_num)
    _build_retrieve_parent_token_kernel[(out.shape[0],)](
        retrieve_next_token,
        retrieve_next_sibling,
        out,
        batch_size,
        retrieve_next_token.stride(0),
        retrieve_next_token.stride(1),
        retrieve_next_sibling.stride(0),
        retrieve_next_sibling.stride(1),
        out.stride(0),
        out.stride(1),
        DRAFT_TOKEN_NUM=draft_token_num,
        BLOCK=block,
    )
    return out


def _build_slope_tensor(nheads: int) -> torch.Tensor:
    """Build ALiBi slope tensor - matches MiniCPM implementation.

    This function computes the Attention with Linear Biases (ALiBi) slopes
    used in Simple GLA for decay calculation. The slopes are computed using
    a geometric progression with power-of-2 optimization.

    Args:
        nheads: Number of attention heads

    Returns:
        slopes: Tensor of shape (nheads,) containing ALiBi slopes
    """

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.tensor(get_slopes(nheads))
    return slopes


# Kernel to track mamba states if needed based on track mask
@triton.jit
def track_mamba_state_if_needed_kernel(
    conv_states_ptr,
    ssm_states_ptr,
    cache_indices_ptr,
    mamba_track_mask_ptr,
    mamba_track_indices_ptr,
    conv_state_stride_0,  # stride for first dimension (batch/pool index)
    ssm_state_stride_0,  # stride for first dimension (batch/pool index)
    conv_state_numel_per_row: tl.constexpr,  # total elements per row
    ssm_state_numel_per_row: tl.constexpr,  # total elements per row
    BLOCK_SIZE: tl.constexpr,
):
    """
    Track conv_states and ssm_states rows based on track mask.

    This kernel replaces a Python loop that copies state tensors for mamba attention.
    For each batch element, if the track mask is True, it copies the entire row from
    the source index (cache_indices[i]) to the destination index (mamba_track_indices[i]).

    Grid: (batch_size,)
    Each block handles one batch element, using multiple threads to copy data in parallel.
    """
    batch_idx = tl.program_id(0)

    # Load the copy mask for this batch element
    track_mask = tl.load(mamba_track_mask_ptr + batch_idx)

    # Early exit if we don't need to track
    if not track_mask:
        return

    # Load source and destination indices
    src_idx = tl.load(cache_indices_ptr + batch_idx)
    dst_idx = tl.load(mamba_track_indices_ptr + batch_idx)

    # Copy conv_states
    # Each thread handles BLOCK_SIZE elements
    for offset in range(0, conv_state_numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < conv_state_numel_per_row

        src_ptr = conv_states_ptr + src_idx * conv_state_stride_0 + element_indices
        dst_ptr = conv_states_ptr + dst_idx * conv_state_stride_0 + element_indices

        data = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, data, mask=mask)

    # Copy ssm_states
    for offset in range(0, ssm_state_numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < ssm_state_numel_per_row

        src_ptr = ssm_states_ptr + src_idx * ssm_state_stride_0 + element_indices
        dst_ptr = ssm_states_ptr + dst_idx * ssm_state_stride_0 + element_indices

        data = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, data, mask=mask)


def track_mamba_states_if_needed(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    mamba_track_mask: torch.Tensor,
    mamba_track_indices: torch.Tensor,
    batch_size: int,
):
    """
    Track mamba states using Triton kernel for better performance.

    Args:
        conv_states: Convolution states tensor [pool_size, ...]
        ssm_states: SSM states tensor [pool_size, ...]
        cache_indices: Source indices for each batch element [batch_size]
        mamba_track_mask: Boolean mask indicating which elements to track [batch_size]
        mamba_track_indices: Indices to track for each batch element [batch_size]
        batch_size: Number of batch elements
    """
    conv_state_numel_per_row = conv_states[0].numel()
    ssm_state_numel_per_row = ssm_states[0].numel()

    # Choose BLOCK_SIZE based on the size of the data
    BLOCK_SIZE = 1024

    # Launch kernel with batch_size blocks
    grid = (batch_size,)
    track_mamba_state_if_needed_kernel[grid](
        conv_states,
        ssm_states,
        cache_indices,
        mamba_track_mask,
        mamba_track_indices,
        conv_states.stride(0),
        ssm_states.stride(0),
        conv_state_numel_per_row,
        ssm_state_numel_per_row,
        BLOCK_SIZE,
    )


class MambaAttnBackendBase(AttentionBackend):
    # Conv backends (Mamba2/GDN) get retrieve_parent_token computed as a fused
    # side output of causal_conv1d_update, so metadata only needs a scratch
    # buffer. Backends without a conv1d (SimpleGLA) override this to True and
    # build the parent-token map during metadata prep instead.
    builds_retrieve_parent_in_metadata: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.pad_slot_id = PAD_SLOT_ID
        self.device = model_runner.device
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.is_draft_worker = model_runner.is_draft_worker
        self.req_to_token_pool: HybridReqToTokenPool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.forward_metadata: ForwardMetadata = None
        self.state_indices_list = []
        self.query_start_loc_list = []
        self.retrieve_next_token_list = []
        self.retrieve_next_sibling_list = []
        self.retrieve_parent_token_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None
        self.conv_states_shape: tuple[int, int] = None

    def _execute_deferred_mamba_cow_and_clear(self, forward_batch: ForwardBatch):
        """Run deferred clear/COW ops on the forward stream to avoid races."""
        if not forward_batch.forward_mode.is_extend() or self.is_draft_worker:
            return
        if (
            forward_batch.mamba_clear_indices is not None
            and len(forward_batch.mamba_clear_indices) > 0
        ):
            self.req_to_token_pool.mamba_pool.clear_slots(
                forward_batch.mamba_clear_indices
            )
        if (
            forward_batch.mamba_cow_src_indices is not None
            and len(forward_batch.mamba_cow_src_indices) > 0
        ):
            self.req_to_token_pool.mamba_pool.copy_from(
                forward_batch.mamba_cow_src_indices, forward_batch.mamba_cow_dst_indices
            )
        forward_batch.mamba_clear_indices = None
        forward_batch.mamba_cow_src_indices = None
        forward_batch.mamba_cow_dst_indices = None

    def _forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size

        retrieve_next_token = None
        retrieve_next_sibling = None
        retrieve_parent_token = None
        track_conv_indices = None
        track_ssm_h_src = None
        track_ssm_h_dst = None
        track_ssm_final_src = None
        track_ssm_final_dst = None

        mamba_cache_indices = self.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        )

        if forward_batch.forward_mode.is_decode_or_idle():
            query_start_loc = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
        elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            if forward_batch.forward_mode.is_draft_extend_v2():
                # HybridLinearAttnBackend.init_forward_metadata calls all sub-backends
                # unconditionally, but DRAFT_EXTEND_V2 only runs full-attn layers in
                # the draft model, so mamba metadata can be skipped.
                query_start_loc = None
            elif forward_batch.forward_mode.is_target_verify():
                query_start_loc = torch.arange(
                    0,
                    forward_batch.input_ids.shape[0] + 1,
                    step=forward_batch.spec_info.draft_token_num,
                    dtype=torch.int32,
                    device=forward_batch.input_ids.device,
                )

                if self.topk > 1:
                    retrieve_next_token = forward_batch.spec_info.retrieve_next_token
                    retrieve_next_sibling = (
                        forward_batch.spec_info.retrieve_next_sibling
                    )
                    # retrieve_next_token is None during dummy run, where the
                    # builder returns None and skips tensor creation.
                    if self.builds_retrieve_parent_in_metadata:
                        # SimpleGLA has no conv1d to fuse the parent-token map,
                        # so build it explicitly here.
                        retrieve_parent_token = _build_retrieve_parent_token(
                            retrieve_next_token, retrieve_next_sibling
                        )
                    elif retrieve_next_token is not None:
                        # Conv backends get retrieve_parent_token fused inside
                        # causal_conv1d_update; this is the scratch buffer the
                        # conv kernel writes into.
                        retrieve_parent_token = torch.empty_like(retrieve_next_token)
            else:
                query_start_loc = torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                )
                query_start_loc[:bs] = forward_batch.extend_start_loc
                query_start_loc[bs] = (
                    forward_batch.extend_start_loc[-1]
                    + forward_batch.extend_seq_lens[-1]
                )
                if (
                    forward_batch.mamba_track_mask is not None
                    and forward_batch.mamba_track_mask.any()
                ):
                    track_conv_indices = self._init_track_conv_indices(
                        query_start_loc, forward_batch
                    )

                    (
                        track_ssm_h_src,
                        track_ssm_h_dst,
                        track_ssm_final_src,
                        track_ssm_final_dst,
                    ) = self._init_track_ssm_indices(mamba_cache_indices, forward_batch)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")

        has_mamba_track_mask = bool(
            forward_batch.mamba_track_mask is not None
            and forward_batch.mamba_track_mask.any()
        )

        return ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
            track_conv_indices=track_conv_indices,
            track_ssm_h_src=track_ssm_h_src,
            track_ssm_h_dst=track_ssm_h_dst,
            track_ssm_final_src=track_ssm_final_src,
            track_ssm_final_dst=track_ssm_final_dst,
            has_mamba_track_mask=has_mamba_track_mask,
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        # seq_lens_cpu is unused by _replay_metadata for the non-target-verify
        # case but kept in the contract for compatibility. forward_batch is
        # forwarded so _replay_metadata can read the replay-view side channel
        # (_source_forward_batch) for the exact padding count.
        self.forward_metadata = self._replay_metadata(
            forward_batch.batch_size,
            forward_batch.req_pool_indices,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            forward_batch.seq_lens_cpu if not in_capture else None,
            forward_batch,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._execute_deferred_mamba_cow_and_clear(forward_batch)
        self.forward_metadata = self._forward_metadata(forward_batch)

    def _init_track_conv_indices(
        self, query_start_loc: torch.Tensor, forward_batch: ForwardBatch
    ):
        """
        Compute indices for extracting conv states from the input sequence during extend.

        In Mamba models, the conv layer maintains a sliding window of recent inputs.
        After processing a prefill chunk, we need to save the last `conv_state_len` tokens
        of the processed region for prefix caching.

        The key insight is that FLA (Flash Linear Attention) and Mamba2 processes sequences in chunks
        of the chunk size (FLA_CHUNK_SIZE=64 for FLA, mamba_chunk_size for Mamba2).
        We only track the conv state up to the last complete chunk boundary (aligned_len).

        start_indices is the starting token index of the conv state to track in this extend batch.
        indices include all pos to track in this extend batch, conv_state_len for each req that
        needs to be tracked (i.e. mamba_track_mask is True)

        Returns:
            indices: Tensor of shape [num_tracked_requests, conv_state_len] containing
                     flattened positions into the packed input tensor.
        """
        conv_state_len = self.conv_states_shape[-1]

        # Calculate the end position of the last aligned chunk
        lens_to_track = (
            forward_batch.mamba_track_seqlens - forward_batch.extend_prefix_lens
        )
        mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
        aligned_len = (lens_to_track // mamba_cache_chunk_size) * mamba_cache_chunk_size
        start_indices = query_start_loc[:-1] + aligned_len - conv_state_len
        start_indices = start_indices[forward_batch.mamba_track_mask]

        # Create indices: [batch_size, conv_state_len]
        indices = start_indices.unsqueeze(-1) + torch.arange(
            conv_state_len,
            device=self.device,
            dtype=start_indices.dtype,
        )

        return indices.clamp(0, query_start_loc[-1] - 1)

    def _init_track_ssm_indices(
        self, mamba_cache_indices: torch.Tensor, forward_batch: ForwardBatch
    ):
        """
        Compute source and destination indices for tracking SSM states for prefix caching.

        After processing a prefill, we need to save the SSM recurrent state for prefix caching.
        The kernel outputs intermediate hidden states `h` at each chunk boundary,
        plus a `last_recurrent_state` at the end of the chunked prefill size.

        The chunk size varies by model type:
        - FLA models: FLA_CHUNK_SIZE (64)
        - Mamba2 models: mamba_chunk_size (256)

        The challenge is that sequences may or may not end on a chunk boundary:
          - Aligned case (len % chunk_size == 0): The to-cache state is stored in
            the last_recurrent_state.
          - Unaligned case (len % chunk_size != 0): The last_recurrent_state includes the
            unaligned position, but we only want state up to the last chunk boundary.
            We must extract from the intermediate `h` tensor at the appropriate chunk index.

        We compute the src and dst indices for all requests that need to be cached
        (i.e. mamba_track_mask is True) based on the rule above.

        For example (assuming chunk_size=64):
        1. If chunked prefill length is < chunk_size, then only final state has value.
           In this case we cache `final` state.
        2. If chunked prefill length == chunk_size, then only final state has value.
           In this case we cache pos chunk_size, from `final` state.
        3. If chunked prefill length > chunk_size and < 2 * chunk_size, then both h and
           final state have value. We cache pos chunk_size from `h` state.
        4. If chunked prefill length == 2 * chunk_size, then both h and final state have
           value. We cache pos 2 * chunk_size from `final` state. Note `h` doesn't include
           the final position.

        Returns:
            track_ssm_h_src: Source indices into the packed `h` tensor (for unaligned seqs)
            track_ssm_h_dst: Destination cache slot indices (for unaligned seqs)
            track_ssm_final_src: Source indices into last_recurrent_state buffer (for aligned seqs)
            track_ssm_final_dst: Destination cache slot indices (for aligned seqs)
        """
        mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
        # Move to CPU to avoid kernel launches for masking operations
        mamba_track_mask = forward_batch.mamba_track_mask.cpu()
        extend_seq_lens = forward_batch.extend_seq_lens.cpu()
        mamba_track_indices = forward_batch.mamba_track_indices.cpu()
        mamba_cache_indices = mamba_cache_indices.cpu()
        mamba_track_seqlens = forward_batch.mamba_track_seqlens.cpu()
        prefix_lens = forward_batch.extend_prefix_lens.cpu()

        # Calculate the number of hidden states per request
        if isinstance(self, Mamba2AttnBackend):
            num_h_states = extend_seq_lens // mamba_cache_chunk_size
        else:
            num_h_states = (extend_seq_lens - 1) // mamba_cache_chunk_size + 1

        # Calculate the starting offset for each sequence in the packed batch
        track_ssm_src_offset = torch.zeros_like(num_h_states)
        track_ssm_src_offset[1:] = torch.cumsum(num_h_states[:-1], dim=0)

        # Filter variables by track mask
        lens_to_track = mamba_track_seqlens - prefix_lens
        lens_masked = lens_to_track[mamba_track_mask]
        offset_masked = track_ssm_src_offset[mamba_track_mask]
        dst_masked = mamba_track_indices[mamba_track_mask]

        # Determine if the sequence ends at a chunk boundary
        is_aligned = (lens_masked % mamba_cache_chunk_size) == 0

        # Case 1: Aligned. Use last_recurrent_state from ssm_states.
        track_ssm_final_src = mamba_cache_indices[mamba_track_mask][is_aligned]
        track_ssm_final_dst = dst_masked[is_aligned]

        # Case 2: Unaligned. Use intermediate state from h.
        # TODO: if support mamba_cache_chunk_size % page size != 0, then need to modify this
        not_aligned = ~is_aligned
        track_ssm_h_src = offset_masked[not_aligned] + (
            lens_masked[not_aligned] // mamba_cache_chunk_size
        )
        track_ssm_h_dst = dst_masked[not_aligned]

        # Move back to GPU
        return (
            track_ssm_h_src.to(self.device, non_blocking=True),
            track_ssm_h_dst.to(self.device, non_blocking=True),
            track_ssm_final_src.to(self.device, non_blocking=True),
            track_ssm_final_dst.to(self.device, non_blocking=True),
        )

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        self.forward_metadata = self._capture_metadata(
            bs, req_pool_indices, forward_mode, spec_info
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        assert (
            max_num_tokens % max_bs == 0
        ), f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        draft_token_num = max_num_tokens // max_bs
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.query_start_loc_list.append(
                torch.zeros((i + 2,), dtype=torch.int32, device=self.device)
            )
            self.retrieve_next_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_next_sibling_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_parent_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.cached_cuda_graph_verify_query_start_loc = torch.arange(
            0,
            max_bs * draft_token_num + 1,
            step=draft_token_num,
            dtype=torch.int32,
            device=self.device,
        )

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        assert (
            max_num_tokens % max_bs == 0
        ), f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.query_start_loc_list.append(
                torch.empty((i + 2,), dtype=torch.int32, device=self.device)
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )

    def _capture_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif forward_mode.is_target_verify():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and self.topk > 1:
            # They are None during cuda graph capture so skip the copy_...
            # self.retrieve_next_token_list[bs - 1].copy_(spec_info.retrieve_next_token)
            # self.retrieve_next_sibling_list[bs - 1].copy_(spec_info.retrieve_next_sibling)
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
            )

    def _replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        forward_batch: Optional[ForwardBatch] = None,
    ):
        # CUDA-graph replay pads the batch up to a captured size; padding reqs
        # occupy the tail [raw_bs:bs]. Derive the count from the live (un-padded)
        # request count off the replay-view side channel rather than from seq_len
        # values: the mamba fill value (1) collides with a genuine 1-token verify
        # prefix. ``_source_forward_batch`` is set by build_replay_fb_view; absent
        # it (e.g. capture, where seq_lens_cpu is None) the count is 0.
        source_fb = (
            getattr(forward_batch, "_source_forward_batch", None)
            if forward_batch is not None
            else None
        )
        if source_fb is not None:
            num_padding = bs - source_fb.batch_size
        elif seq_lens_cpu is None:
            num_padding = 0
        else:
            num_padding = int(
                torch.count_nonzero(
                    seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
                )
            )
        # Make sure forward metadata is correctly handled for padding reqs
        req_pool_indices[bs - num_padding :] = 0
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        mamba_indices[bs - num_padding :] = -1
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        if forward_mode.is_decode_or_idle():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    bs - num_padding
                )
        elif forward_mode.is_target_verify():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    (bs - num_padding) * spec_info.draft_token_num
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and self.topk > 1:
            if (
                spec_info is not None
                and getattr(spec_info, "retrieve_next_token", None) is not None
            ):
                bs_without_pad = spec_info.retrieve_next_token.shape[0]
                self.retrieve_next_token_list[bs - 1][:bs_without_pad].copy_(
                    spec_info.retrieve_next_token
                )
                self.retrieve_next_sibling_list[bs - 1][:bs_without_pad].copy_(
                    spec_info.retrieve_next_sibling
                )
                # Conv backends get retrieve_parent_token fused inside
                # causal_conv1d_update; SimpleGLA has no conv1d, so build the
                # parent-token map into the captured buffer here.
                if self.builds_retrieve_parent_in_metadata:
                    _build_retrieve_parent_token(
                        spec_info.retrieve_next_token,
                        spec_info.retrieve_next_sibling,
                        out=self.retrieve_parent_token_list[bs - 1],
                    )
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1  # Mamba attn does not use seq lens to index kv cache

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def _track_mamba_state_decode(
        self,
        forward_batch: ForwardBatch,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ):
        """
        Track and copy Mamba conv/SSM states during decode for prefix caching.

        During decode, each token update modifies conv_states and ssm_states in-place
        at positions indexed by cache_indices (the working slots). For prefix caching,
        we need to copy these updated states to persistent cache slots (mamba_track_indices)
        so they can be prefix cached.

        This delegates to `track_mamba_states_if_needed`, which performs:
            conv_states[mamba_track_indices[i]] = conv_states[cache_indices[i]]
            ssm_states[mamba_track_indices[i]] = ssm_states[cache_indices[i]]
        for all requests where mamba_track_mask[i] is True.
        """
        if forward_batch.mamba_track_mask is not None:
            track_mamba_states_if_needed(
                conv_states,
                ssm_states,
                cache_indices,
                forward_batch.mamba_track_mask,
                forward_batch.mamba_track_indices,
                forward_batch.batch_size,
            )

    def _track_mamba_state_extend(
        self,
        forward_batch: ForwardBatch,
        h: torch.Tensor,
        ssm_states: torch.Tensor,
        forward_metadata: ForwardMetadata,
    ):
        """
        Track and copy SSM states during extend for prefix caching.

        After the chunked prefill kernel runs, we need to save the SSM recurrent
        state at the last chunk boundary so it can be reused for prefix caching.
        The source of the state depends on whether the sequence length is aligned
        to the chunk size. See `_init_track_ssm_indices` for more details on how
        the source and destination indices are computed.

        Note: Conv state tracking for extend is handled separately via gather operations
        using indices computed by `_init_track_conv_indices`.
        """
        if forward_metadata.has_mamba_track_mask:
            h = h.squeeze(0)

            if forward_metadata.track_ssm_h_src.numel() > 0:
                ssm_states[forward_metadata.track_ssm_h_dst] = h[
                    forward_metadata.track_ssm_h_src
                ].to(ssm_states.dtype, copy=False)
            if forward_metadata.track_ssm_final_src.numel() > 0:
                ssm_states[forward_metadata.track_ssm_final_dst] = ssm_states[
                    forward_metadata.track_ssm_final_src
                ]


class Mamba2AttnBackend(MambaAttnBackendBase):
    """Attention backend wrapper for Mamba2Mixer kernels."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        config = model_runner.mamba2_config
        assert config is not None
        self.mamba_chunk_size = config.mamba_chunk_size
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )

        if model_runner.server_args.enable_mamba_extra_buffer():
            assert (
                self.conv_states_shape[-1] < self.mamba_chunk_size
            ), f"{self.conv_states_shape[-1]=} should be less than {self.mamba_chunk_size}"
            assert (
                model_runner.server_args.mamba_track_interval >= self.mamba_chunk_size
            ), f"mamba_track_interval ({model_runner.server_args.mamba_track_interval}) must be >= mamba_chunk_size ({self.mamba_chunk_size})"

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        metadata = self._replay_metadata(
            forward_batch.batch_size,
            forward_batch.req_pool_indices,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            forward_batch.seq_lens_cpu if not in_capture else None,
            forward_batch,
        )
        spec_info = forward_batch.spec_info
        draft_token_num = spec_info.draft_token_num if spec_info is not None else 1
        self.forward_metadata = Mamba2Metadata.prepare_decode(
            metadata,
            forward_batch.seq_lens,
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._execute_deferred_mamba_cow_and_clear(forward_batch)
        metadata = self._forward_metadata(forward_batch)
        self.forward_metadata = Mamba2Metadata.prepare_mixed(
            metadata,
            self.mamba_chunk_size,
            forward_batch,
        )

    def forward(
        self,
        mixer: MambaMixer2,
        hidden_states: torch.Tensor,
        output: Optional[torch.Tensor],
        layer_id: int,
        forward_batch: ForwardBatch,
        mup_vector: Optional[torch.Tensor] = None,
        use_triton_causal_conv: bool = False,
    ):
        assert isinstance(self.forward_metadata, Mamba2Metadata)
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        mixer_out, intermediate_states = mixer.forward(
            hidden_states=hidden_states,
            output=output,
            layer_cache=layer_cache,
            metadata=self.forward_metadata,
            forward_batch=forward_batch,
            mup_vector=mup_vector,
            use_triton_causal_conv=use_triton_causal_conv,
        )

        if forward_batch.mamba_track_mask is not None:
            if (
                intermediate_states is not None
                and forward_batch.mamba_track_mask is not None
                and forward_batch.mamba_track_mask.any()
            ):
                self._track_mamba_state_extend(
                    forward_batch,
                    intermediate_states,
                    layer_cache.temporal,
                    self.forward_metadata,
                )

            if self.forward_metadata.num_decodes > 0:
                num_decodes = self.forward_metadata.num_decodes
                track_mamba_states_if_needed(
                    layer_cache.conv[0],
                    layer_cache.temporal,
                    self.forward_metadata.mamba_cache_indices[-num_decodes:],
                    forward_batch.mamba_track_mask[-num_decodes:],
                    forward_batch.mamba_track_indices[-num_decodes:],
                    num_decodes,
                )

        return mixer_out

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )


class HybridLinearAttnBackend(AttentionBackend):
    """Manages a full and linear attention backend"""

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: MambaAttnBackendBase,
        full_attn_layers: list[int],
    ):
        self.full_attn_layers = full_attn_layers
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend
        self.attn_backend_list = [full_attn_backend, linear_attn_backend]
        # Dispatcher aliases the full-attn backend's pool refs.
        self.token_to_kv_pool = full_attn_backend.token_to_kv_pool
        self.req_to_token_pool = full_attn_backend.req_to_token_pool

    def _is_full_attn(
        self, layer: Optional[RadixAttention], layer_id: Optional[int] = None
    ) -> bool:
        if layer is not None:
            layer_id = layer.layer_id
        assert layer_id is not None, "either layer or layer_id must be provided"
        return layer_id in self.full_attn_layers

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_draft_extend_v2():
            # DRAFT_EXTEND_V2 only runs full-attn layers in the draft model,
            # so skip linear/mamba backend metadata which requires query_start_loc.
            self.full_attn_backend.init_forward_metadata(forward_batch)
            return
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata(forward_batch)

    def init_mha_chunk_metadata(
        self, forward_batch: ForwardBatch, disable_flashinfer_ragged: bool = False
    ):
        # Hybrid MLA models (Ring/Ling, Kimi-Linear) resolve this via
        # get_attn_backend(), which returns this wrapper; delegate to the
        # full-attn backend so its chunked/one-shot prefill metadata is planned.
        init = getattr(self.full_attn_backend, "init_mha_chunk_metadata", None)
        if init is not None:
            init(forward_batch, disable_flashinfer_ragged)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cpu_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_capture_cpu_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.full_attn_backend.get_cuda_graph_seq_len_fill_value()

    def get_cpu_graph_seq_len_fill_value(self):
        return self.full_attn_backend.get_cpu_graph_seq_len_fill_value()

    def forward_decode(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For GDN linear attention
        b: Optional[torch.Tensor] = None,  # For GDN linear attention
        **kwargs,
    ):
        if self._is_full_attn(layer, kwargs.get("layer_id")):
            return self.full_attn_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        # Linear attention backend
        return self.linear_attn_backend.forward_decode(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )

    def forward_extend(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For GDN linear attention
        b: Optional[torch.Tensor] = None,  # For GDN linear attention
        **kwargs,
    ):
        if self._is_full_attn(layer, kwargs.get("layer_id")):
            return self.full_attn_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        # Linear attention backend
        return self.linear_attn_backend.forward_extend(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )

    def forward(
        self,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        layer: RadixAttention = None,
        forward_batch: ForwardBatch = None,
        save_kv_cache: bool = True,
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For linear attention
        b: Optional[torch.Tensor] = None,  # For linear attention
        **kwargs,
    ):
        is_linear_attn = not self._is_full_attn(layer, kwargs.get("layer_id"))

        if forward_batch.forward_mode.is_idle():
            if is_linear_attn:
                return mixed_qkv.new_empty(
                    mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim
                )
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                layer,
                forward_batch,
                save_kv_cache,
                q,
                k,
                v,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )
        else:
            return self.forward_extend(
                layer,
                forward_batch,
                save_kv_cache,
                q,
                k,
                v,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )

    def update_mamba_state_after_mtp_verify(
        self,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        model,
    ):
        """
        Update mamba states after MTP verify using fully fused Triton kernel.

        This replaces the original advanced indexing operations with a single fused
        gather-scatter kernel that also handles masking internally, avoiding:
        - index_elementwise_kernel from tensor[bool_mask]
        - index_select kernel launches
        - nonzero kernel launches
        """
        request_number = last_correct_step_indices.shape[0]

        state_indices_tensor = (
            self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                :request_number
            ]
        )

        mamba_caches = (
            self.linear_attn_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        )

        if mamba_caches.gla_verify_k is not None:
            self._update_simple_gla_state_after_mtp_verify_with_recompute(
                last_correct_step_indices=last_correct_step_indices,
                mamba_track_indices=mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                state_indices_tensor=state_indices_tensor,
                mamba_caches=mamba_caches,
                request_number=request_number,
            )
            return

        has_conv = len(mamba_caches.conv) > 0
        conv_states = mamba_caches.conv[0] if has_conv else None
        ssm_states = mamba_caches.temporal
        intermediate_state_cache = mamba_caches.intermediate_ssm
        intermediate_conv_window_cache = (
            mamba_caches.intermediate_conv_window[0] if has_conv else None
        )

        # Use fully fused kernel that handles masking internally
        # This avoids separate nonzero() and index_select() calls
        fused_mamba_state_scatter_with_mask(
            ssm_states,
            intermediate_state_cache,
            state_indices_tensor,
            last_correct_step_indices,
        )
        if has_conv:
            fused_mamba_state_scatter_with_mask(
                conv_states,
                intermediate_conv_window_cache,
                state_indices_tensor,
                last_correct_step_indices,
            )

        # Track indices used for tracking mamba states for prefix cache
        if mamba_track_indices is not None:
            assert mamba_steps_to_track is not None
            # Use fully fused kernel for track scatter operations
            fused_mamba_state_scatter_with_mask(
                ssm_states,
                intermediate_state_cache,
                mamba_track_indices,
                mamba_steps_to_track,
            )
            if has_conv:
                fused_mamba_state_scatter_with_mask(
                    conv_states,
                    intermediate_conv_window_cache,
                    mamba_track_indices,
                    mamba_steps_to_track,
                )

    def _update_simple_gla_state_after_mtp_verify_with_recompute(
        self,
        *,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        state_indices_tensor: torch.Tensor,
        mamba_caches,
        request_number: int,
    ) -> None:
        ssm_states = mamba_caches.temporal
        saved_k = mamba_caches.gla_verify_k
        saved_v = mamba_caches.gla_verify_v
        assert saved_k is not None
        assert saved_v is not None

        device = last_correct_step_indices.device
        state_indices = state_indices_tensor.to(torch.int64)
        selected_req_indices = torch.arange(
            request_number, dtype=torch.int64, device=device
        )
        commit_steps = last_correct_step_indices.to(torch.int64).contiguous()
        retrieve_parent_token = (
            self.linear_attn_backend.forward_metadata.retrieve_parent_token
        )
        if retrieve_parent_token is None:
            draft_token_num = saved_k.shape[2]
            retrieve_parent_token = torch.empty(
                (request_number, draft_token_num), dtype=torch.int64, device=device
            )
            retrieve_parent_token[:, 0] = 0
            if draft_token_num > 1:
                retrieve_parent_token[:, 1:] = torch.arange(
                    0, draft_token_num - 1, dtype=torch.int64, device=device
                )
        else:
            retrieve_parent_token = retrieve_parent_token[:request_number]

        has_track = mamba_track_indices is not None
        if has_track:
            assert mamba_steps_to_track is not None
            track_indices = mamba_track_indices[:request_number].to(torch.int64)
            track_steps = (
                mamba_steps_to_track[:request_number].to(torch.int64).contiguous()
            )
        else:
            track_indices = None
            track_steps = None

        _recompute_and_scatter_gla_state_from_tree_paths(
            k=saved_k,
            v=saved_v,
            g_gamma=self.linear_attn_backend.g_gamma,
            states=ssm_states,
            retrieve_parent_token=retrieve_parent_token,
            selected_req_indices=selected_req_indices,
            initial_state_indices=state_indices,
            commit_nodes=commit_steps,
            commit_dst_indices=state_indices,
            track_nodes=track_steps,
            track_dst_indices=track_indices,
        )


class SimpleGLAAttnBackend(MambaAttnBackendBase):
    """Attention backend for MiniCPM hybrid models using the SimpleGLA CUDA kernels.

    This backend assumes the model's ``mixer_types`` includes ``"lightning-attn"`` and
    that the optional ``fla`` package is installed. It does **not** perform any
    convolution or Mamba‑style processing; instead it forwards the query, key and
    value tensors directly to ``parallel_simple_gla``.

    If the ``fla`` package is missing an ``ImportError`` is raised during
    construction, allowing the caller to fall back to an alternative backend such
    as ``Mamba2AttnBackend``.
    """

    # No conv1d to fuse the parent-token map into, so build it in metadata prep.
    builds_retrieve_parent_in_metadata: bool = True

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        minicpm_config = getattr(model_runner, "minicpm_hybrid_config", None)
        assert (
            minicpm_config is not None
        ), "minicpm_hybrid_config is required for SimpleGLA backend"

        self.conv_states_shape = None

        tp_size = get_tensor_model_parallel_world_size()
        total_num_heads = minicpm_config.lightning_nkv or 16
        # Must divide by tp_size, same as OLD implementation
        assert (
            total_num_heads % tp_size == 0
        ), f"lightning_nkv ({total_num_heads}) must be divisible by tp_size ({tp_size})"
        num_heads = total_num_heads // tp_size
        self.num_heads = num_heads

        self.g_gamma = _build_slope_tensor(num_heads).to(
            dtype=torch.float32, device=self.device
        ) * (
            -1.0
        )  # (h)

        head_dim = getattr(minicpm_config, "lightning_head_dim", 128)
        scale_config = getattr(minicpm_config, "lightning_scale", "1/sqrt(d)")
        if scale_config == "1/sqrt(d)":
            self.scale = head_dim ** (-0.5)
        elif scale_config == "1/d":
            self.scale = head_dim ** (-1.0)
        else:
            self.scale = 1.0

        if not SIMPLE_GLA_AVAILABLE:
            raise ImportError(
                "Simple GLA backend requested but the 'fla' package is not installed. "
                "Install it or configure the model to use a supported attention backend (e.g., Mamba2)."
            )

    def _get_mamba_indices(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Get mamba cache indices with fallback logic.

        First tries to get from forward_metadata.mamba_cache_indices.
        If not available, falls back to req_to_token_pool.get_mamba_indices().

        Args:
            forward_batch: Forward batch containing forward metadata

        Returns:
            mamba_indices: Tensor of cache indices for each batch element

        Raises:
            RuntimeError: If both sources fail to provide indices
        """
        if (
            hasattr(self, "forward_metadata")
            and self.forward_metadata is not None
            and hasattr(self.forward_metadata, "mamba_cache_indices")
            and self.forward_metadata.mamba_cache_indices is not None
        ):
            return self.forward_metadata.mamba_cache_indices
        else:
            return self.req_to_token_pool.get_mamba_indices(
                forward_batch.req_pool_indices
            )

    def _mamba_layer_cache(self, layer_id: int):
        """Resolve the mamba cache slot and per-layer cache for ``layer_id``.

        Raises:
            RuntimeError: If the layer is not registered in mamba_map.
        """
        cache_idx = self.req_to_token_pool.mamba_map.get(layer_id)
        if cache_idx is None:
            raise RuntimeError(
                f"SimpleGLAAttnBackend layer {layer_id} is missing from mamba_map. "
                f"This indicates a misconfiguration - linear-attention layers must be "
                f"registered in the pool's mamba_map. "
                f"Available layers: {list(self.req_to_token_pool.mamba_map.keys())}"
            )
        layer_cache = self.req_to_token_pool.mamba_pool.mamba2_layer_cache(cache_idx)
        return cache_idx, layer_cache

    def _init_track_conv_indices(
        self, query_start_loc: torch.Tensor, forward_batch: ForwardBatch
    ):
        """
        Simple GLA doesn't use convolution states, so return None.
        This overrides the parent class method that would fail without conv_states_shape.
        """
        return None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        metadata = self._forward_metadata(forward_batch)
        self.forward_metadata = metadata

    # CUDA graph capture/replay is inherited from
    # MambaAttnBackendBase.init_forward_metadata_out_graph, which routes both
    # capture and replay through the inherited _replay_metadata.

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        output_attentions: bool = False,
    ) -> torch.Tensor:

        num_heads = q.shape[2]
        head_dim = q.shape[3]
        is_target_verify = forward_batch.forward_mode.is_target_verify()

        mamba_indices = self._get_mamba_indices(forward_batch)
        initial_state = None
        initial_state_indices = None
        layer_cache = None
        if (
            forward_batch.forward_mode.is_decode()
            or is_target_verify
            or (
                forward_batch.extend_prefix_lens is not None
                and (forward_batch.extend_prefix_lens > 0).any()
            )
        ):
            cache_idx, layer_cache = self._mamba_layer_cache(layer_id)
            if is_target_verify:
                initial_state = layer_cache.temporal
                initial_state_indices = mamba_indices
            else:
                initial_state = layer_cache.temporal[mamba_indices, :].contiguous()

        scale = self.scale

        g_gamma = self.g_gamma

        if is_target_verify:
            draft_token_num = forward_batch.spec_info.draft_token_num
            batch_size = forward_batch.batch_size
            total_tokens = batch_size * draft_token_num
            spec_cache = (
                self.req_to_token_pool.get_speculative_mamba2_params_all_layers()
            )
            # SimpleGLA always allocates gla_verify_k under speculative decode,
            # so the commit kernel replays the accepted path from the saved k/v
            # and no per-token state is materialized here.
            ht_buf = None
            # gla_verify_k/v are fp32; k/v are bf16 model tensors. The implicit
            # bf16->fp32 upcast here is lossless and required so the recompute
            # kernel runs at the same precision as the fp32 forward accumulator
            # (token-exact rollback).
            spec_cache.gla_verify_k[cache_idx, :batch_size, :draft_token_num].copy_(
                k.reshape(batch_size, draft_token_num, num_heads, head_dim)
            )
            spec_cache.gla_verify_v[cache_idx, :batch_size, :draft_token_num].copy_(
                v.reshape(batch_size, draft_token_num, num_heads, v.shape[-1])
            )
            cu_seqlens = torch.arange(
                0,
                total_tokens + 1,
                step=draft_token_num,
                dtype=torch.int64,
                device=q.device,
            )
            o = _fused_recurrent_gla_verify_tree_paths(
                q=q,
                k=k,
                v=v,
                g_gamma=g_gamma,
                scale=scale,
                initial_state=initial_state,
                initial_state_indices=initial_state_indices,
                cu_seqlens=cu_seqlens,
                retrieve_parent_token=self.forward_metadata.retrieve_parent_token,
                ht_buf=ht_buf,
            )
            final_state = None
        else:
            seq_len = (
                1
                if forward_batch.forward_mode.is_decode()
                else torch.max(forward_batch.extend_seq_lens)
            )
            mode = "fused_recurrent" if seq_len < 64 else "chunk"
            if forward_batch.forward_mode.is_decode() or mode == "fused_recurrent":
                o, final_state = fused_recurrent_simple_gla(
                    q=q,
                    k=k,
                    v=v,
                    g_gamma=g_gamma,
                    scale=scale,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=self.forward_metadata.query_start_loc,
                )
            else:
                o, final_state = chunk_simple_gla(
                    q=q,
                    k=k,
                    v=v,
                    g_gamma=g_gamma,
                    initial_state=initial_state,
                    output_final_state=True,
                    scale=scale,
                    cu_seqlens=self.forward_metadata.query_start_loc,
                )

        if final_state is not None:
            if layer_cache is None:
                _, layer_cache = self._mamba_layer_cache(layer_id)
            layer_cache.temporal[mamba_indices, :] = final_state

        o = o.reshape(-1, num_heads * head_dim)

        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        return self.forward(q, k, v, forward_batch, layer_id, output_attentions)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        return self.forward(q, k, v, forward_batch, layer_id, output_attentions)
