import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.linear.seg_la import (
    _load_tree_links,
    seg_la_sum_kernel,
)


@triton.jit
def _build_parents(Next, Sibling, Parents, T: tl.constexpr, N: tl.constexpr):
    b = tl.program_id(0)
    tokens, children, siblings = _load_tree_links(Next, Sibling, b, T, 1, T, 1, T, N)
    parents = tl.zeros((N,), tl.int32)
    for i in range(T):
        parent = tl.sum(tl.where(tokens == i, parents, 0))
        child = tl.sum(tl.where(tokens == i, children, 0))
        sibling = tl.sum(tl.where(tokens == i, siblings, 0))
        parents = tl.where(tokens == child, i, parents)
        parents = tl.where(tokens == sibling, parent, parents)
    tl.store(Parents + b * T + tokens, parents, tokens < T)


def build_verify_parents(*, next_token, next_sibling, parents):
    steps = next_token.shape[1]
    _build_parents[(next_token.shape[0],)](
        next_token, next_sibling, parents, steps, triton.next_power_of_2(steps)
    )


@triton.jit
def _replay_path(
    initial,
    K,
    V,
    decay,
    parents,
    tokens,
    end,
    stride_k,
    stride_v,
    TREE: tl.constexpr,
    STATE_DTYPE: tl.constexpr,
):
    state = initial
    path_len = end + 1
    if TREE:
        path = tl.zeros(tokens.shape, tl.int32)
        path_len = 0
        ancestor = end
        while ancestor >= 0:
            path = tl.where(tokens == path_len, ancestor, path)
            path_len += 1
            parent = tl.sum(tl.where(tokens == ancestor, parents, 0))
            ancestor = tl.where(ancestor > 0, parent, -1)
    for depth in range(path_len):
        node = depth
        if TREE:
            node = tl.sum(tl.where(tokens == path_len - depth - 1, path, 0))
            # Tree edges observe the persistent state's storage dtype.
            state = state.to(STATE_DTYPE).to(tl.float32)
        k = tl.load(K + node * stride_k).to(tl.float32)
        v = tl.load(V + node * stride_v).to(tl.float32)
        state = state * decay + k[:, None] * v[None, :]

    return state


@triton.jit
def _verify(
    Q,
    K,
    V,
    S,
    KV,
    Out,
    Slots,
    Slopes,
    Parents,
    scale,
    sq,
    sk,
    sv,
    ss,
    sc,
    so,
    H: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    N: tl.constexpr,
    TREE: tl.constexpr,
):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(Slots + b)
    if slot < 0:
        return
    ki, vi = tile // (D // BV), tile % (D // BV)
    kk = ki * BK + tl.arange(0, BK)
    vv = vi * BV + tl.arange(0, BV)
    state_ptr = S + slot * ss + (h * D + kk[:, None]) * D + vv[None, :]
    initial = tl.load(state_ptr).to(tl.float32)
    state = initial
    decay = tl.exp(-tl.load(Slopes + h))
    kp = K + b * T * sk + h * D + kk
    vp = V + b * T * sv + h * D + vv
    tokens = tl.arange(0, N)
    parents = tl.zeros((N,), tl.int32)
    if TREE:
        # Eager links are int64; the local path stack uses int32 node indices.
        parents = tl.load(Parents + b * T + tokens, tokens < T, 0).to(tl.int32)
    for i in range(T):
        if TREE:
            parent = tl.sum(tl.where(tokens == i, parents, 0))
            if i > 0 and parent != i - 1:
                state = _replay_path(
                    initial,
                    kp,
                    vp,
                    decay,
                    parents,
                    tokens,
                    parent,
                    sk,
                    sv,
                    True,
                    S.dtype.element_ty,
                )
            state = state.to(S.dtype.element_ty).to(tl.float32)
        q = tl.load(Q + (b * T + i) * sq + h * D + kk).to(tl.float32) * scale
        k = tl.load(kp + i * sk).to(tl.float32)
        v = tl.load(vp + i * sv).to(tl.float32)
        state = state * decay + k[:, None] * v[None, :]
        out = tl.sum(q[:, None] * state, axis=0)
        tl.store(Out + ki * so + ((b * T + i) * H + h) * D + vv, out)
        cache = KV + b * sc + i * 2 * H * D + h * D
        if vi == 0:
            tl.store(cache + kk, k)
        if ki == 0:
            tl.store(cache + H * D + vv, v)


def verify_with_kv_cache(*, q, k, v, state, kv_cache, slots, slopes, parents, scale):
    bs = slots.numel()
    length, heads, dim = q.shape
    steps = length // bs
    block_k, block_v = 32, 32 if bs <= 2 else 64
    partial = torch.empty(
        (dim // block_k, length, heads, dim), device=q.device, dtype=q.dtype
    )
    out = torch.empty_like(q)
    _verify[(bs, heads, (dim // block_k) * (dim // block_v))](
        q,
        k,
        v,
        state,
        kv_cache,
        partial,
        slots,
        slopes,
        parents,
        scale,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        state.stride(0),
        kv_cache.stride(0),
        partial.stride(0),
        heads,
        dim,
        steps,
        block_k,
        block_v,
        triton.next_power_of_2(steps),
        parents is not None,
        num_warps=2,
        num_stages=3,
    )
    seg_la_sum_kernel[(length,)](
        partial, out, DIM=heads * dim, NUM_BLOCK=dim // block_k
    )
    return out


@triton.jit
def _commit(
    S,
    KV,
    Slots,
    Last,
    Slopes,
    Parents,
    TrackSlots,
    TrackSteps,
    sl,
    ss,
    kl,
    kb,
    H: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    N: tl.constexpr,
    TREE: tl.constexpr,
    TRACK: tl.constexpr,
):
    b, lh, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    layer, h = lh // H, lh % H
    slot = tl.load(Slots + b)
    if slot < 0:
        return
    end = tl.load(Last + b).to(tl.int32)
    ki, vi = tile // (D // BV), tile % (D // BV)
    kk = ki * BK + tl.arange(0, BK)
    vv = vi * BV + tl.arange(0, BV)
    offset = layer * sl + (h * D + kk[:, None]) * D + vv[None, :]
    initial = tl.load(S + offset + slot * ss).to(tl.float32)
    decay = tl.exp(-tl.load(Slopes + lh))
    kp = KV + layer * kl + b * kb + h * D + kk
    vp = KV + layer * kl + b * kb + H * D + h * D + vv
    tokens = tl.arange(0, N)
    parents = tl.zeros((N,), tl.int32)
    if TREE:
        parents = tl.load(Parents + b * T + tokens, tokens < T, 0).to(tl.int32)
    state = _replay_path(
        initial,
        kp,
        vp,
        decay,
        parents,
        tokens,
        end,
        2 * H * D,
        2 * H * D,
        TREE,
        S.dtype.element_ty,
    )
    if TRACK:
        track_slot = tl.load(TrackSlots + b)
        track_end = tl.load(TrackSteps + b).to(tl.int32)
        if track_slot >= 0 and track_end >= 0:
            track_state = _replay_path(
                initial,
                kp,
                vp,
                decay,
                parents,
                tokens,
                track_end,
                2 * H * D,
                2 * H * D,
                TREE,
                S.dtype.element_ty,
            )
            tl.store(S + offset + track_slot * ss, track_state)
    if end >= 0:
        tl.store(S + offset + slot * ss, state)


def commit_verify_kv(
    *,
    state,
    kv_cache,
    slots,
    last_steps,
    slopes,
    parents,
    track_slots=None,
    track_steps=None,
):
    layers, _, heads, dim, _ = state.shape
    steps = kv_cache.shape[2] if parents is None else parents.shape[1]
    bs = slots.numel()
    block_k, block_v = 32, 32 if bs <= 2 else 64
    _commit[(bs, layers * heads, (dim // block_k) * (dim // block_v))](
        state,
        kv_cache,
        slots,
        last_steps,
        slopes,
        parents,
        track_slots,
        track_steps,
        state.stride(0),
        state.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        heads,
        dim,
        steps,
        block_k,
        block_v,
        triton.next_power_of_2(steps),
        parents is not None,
        track_slots is not None,
        num_warps=2,
        num_stages=3,
    )
