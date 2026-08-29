"""Block-native sparse prefill attention over per-token top-k KV block lists.

A query tile unions its tokens' block lists and keeps a per-query bitmask on
each union entry, so one loaded K/V block serves the whole tile."""

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634
# Finite softmax-max sentinel: with -inf, when a row has no valid column,
# alpha = exp2(-inf - (-inf)) = NaN.
_M_INIT = tl.constexpr(-1e30)
# Query-tile width cap: keeps QUERY_TILE * HEADS_PER_GROUP within _QUERY_ROWS
# at 16 query heads per KV head; not tuned beyond that.
_MAX_QUERY_TILE = 8
# tl.dot M-dim cap for the tile kernel.
_QUERY_ROWS = 128


@triton.jit
def _build_query_tile_masks(
    topk_ptr,
    masks_ptr,
    num_tokens,
    num_blocks,
    # topk_idx is freshly allocated per chunk,
    # so its head stride varies per chunk and must stay a runtime arg.
    topk_stride_h,
    topk_stride_t,
    topk_stride_k,
    TOPK: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    token_head = tl.program_id(0)
    head = token_head // num_tokens
    token = token_head - head * num_tokens
    num_tiles = tl.cdiv(num_tokens, QUERY_TILE)
    tile = token // QUERY_TILE
    query = token - tile * QUERY_TILE
    offsets = tl.arange(0, BLOCK_TOPK)
    blocks = tl.load(
        topk_ptr
        + head * topk_stride_h
        + token * topk_stride_t
        + offsets * topk_stride_k,
        mask=offsets < TOPK,
        other=-1,
    )
    tl.atomic_or(
        masks_ptr + (head * num_tiles + tile) * num_blocks + blocks,
        1 << query,
        mask=(blocks >= 0) & (blocks < num_blocks),
    )


@triton.jit
def _compact_query_tile_masks(
    masks_ptr,
    union_ids_ptr,
    union_masks_ptr,
    union_counts_ptr,
    num_blocks,
    UNION_CAPACITY: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    tile_head = tl.program_id(0)
    offsets = tl.arange(0, BLOCKS)
    masks = tl.load(
        masks_ptr + tile_head * num_blocks + offsets,
        mask=offsets < num_blocks,
        other=0,
    )
    valid = masks != 0
    ranks = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    output_offsets = tile_head * UNION_CAPACITY + ranks
    tl.store(union_ids_ptr + output_offsets, offsets, mask=valid)
    tl.store(union_masks_ptr + output_offsets, masks, mask=valid)
    tl.store(union_counts_ptr + tile_head, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _split_query_tile_metadata(
    union_ids_ptr,
    union_masks_ptr,
    union_counts_ptr,
    pair_ids_ptr,
    pair_masks_ptr,
    pair_counts_ptr,
    num_tokens,
    PARENT_CAPACITY: tl.constexpr,
    PAIR_CAPACITY: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    UNION_THRESHOLD: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    pair_head = tl.program_id(0)
    num_pair_tiles = tl.cdiv(num_tokens, 2)
    kv_head = pair_head // num_pair_tiles
    pair_tile = pair_head - kv_head * num_pair_tiles
    pairs_per_parent = QUERY_TILE // 2
    parent_tile = pair_tile // pairs_per_parent
    pair_in_parent = pair_tile - parent_tile * pairs_per_parent
    num_parent_tiles = tl.cdiv(num_tokens, QUERY_TILE)
    parent_head = kv_head * num_parent_tiles + parent_tile
    parent_count = tl.load(union_counts_ptr + parent_head)
    if parent_count <= UNION_THRESHOLD:
        tl.store(pair_counts_ptr + pair_head, 0)
        return

    offsets = tl.arange(0, BLOCKS)
    parent_offsets = parent_head * PARENT_CAPACITY + offsets
    masks = tl.load(
        union_masks_ptr + parent_offsets,
        mask=offsets < parent_count,
        other=0,
    )
    pair_masks = (masks >> (pair_in_parent * 2)) & 3
    valid = pair_masks != 0
    ranks = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    output_offsets = pair_head * PAIR_CAPACITY + ranks
    union_ids = tl.load(
        union_ids_ptr + parent_offsets,
        mask=offsets < parent_count,
        other=0,
    )
    tl.store(pair_ids_ptr + output_offsets, union_ids, mask=valid)
    tl.store(pair_masks_ptr + output_offsets, pair_masks, mask=valid)
    tl.store(pair_counts_ptr + pair_head, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _attention_step(q, k, v, valid, m, denom, acc, scale):
    scores = tl.dot(q, k) * scale
    scores = tl.where(valid, scores, float("-inf"))
    m_new = tl.maximum(m, tl.max(scores, axis=1))
    alpha = tl.exp2(m - m_new)
    p = tl.exp2(scores - m_new[:, None])
    denom = denom * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return m_new, denom, acc


@triton.jit
def _load_kv(
    k_ptr,
    v_ptr,
    slots,
    kv_head,
    dims,
    dim_mask,
    pos_mask,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_t,
    stride_v_h,
    stride_v_d,
):
    k = tl.load(
        k_ptr
        + slots[None, :] * stride_k_t
        + kv_head * stride_k_h
        + dims[:, None] * stride_k_d,
        mask=dim_mask[:, None] & pos_mask[None, :],
        other=0.0,
    )
    v = tl.load(
        v_ptr
        + slots[:, None] * stride_v_t
        + kv_head * stride_v_h
        + dims[None, :] * stride_v_d,
        mask=pos_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    return k, v


@triton.jit
def _block_sparse_attention_kernel(
    q_ptr,  # bf16 [num_tokens, num_q_heads, head_dim]
    k_ptr,  # bf16 [pool_size, num_kv_heads, head_dim]
    v_ptr,  # bf16 [pool_size, num_kv_heads, head_dim]
    page_table_ptr,  # int [seq_len] token position -> pool slot
    union_ids_ptr,  # int32 [num_kv_heads, num_tiles, union_capacity] block ids
    # int32 [num_kv_heads, num_tiles, union_capacity] per-query bitmask
    union_masks_ptr,
    union_counts_ptr,  # int32 [num_kv_heads, num_tiles] live union entries
    out_ptr,  # bf16 [num_tokens, num_q_heads, head_dim]
    num_tokens,
    prefix_len,
    seq_len,
    softmax_scale_log2e,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_k_t: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_k_d: tl.constexpr,
    stride_v_t: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_d: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_d: tl.constexpr,
    UNION_CAPACITY: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNION_THRESHOLD: tl.constexpr,
):
    tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    num_tiles = tl.cdiv(num_tokens, QUERY_TILE)
    tile_head = kv_head * num_tiles + tile
    union_count = tl.load(union_counts_ptr + tile_head)
    if union_count > UNION_THRESHOLD:
        return

    rows = tl.arange(0, QUERY_TILE * HEADS_PER_GROUP)
    dims = tl.arange(0, BLOCK_D)
    cols = tl.arange(0, BLOCK_N)
    query_in_tile = rows // HEADS_PER_GROUP
    query_head = rows - query_in_tile * HEADS_PER_GROUP
    query_token = tile * QUERY_TILE + query_in_tile
    row_mask = query_token < num_tokens
    dim_mask = dims < HEAD_DIM
    q = tl.load(
        q_ptr
        + query_token[:, None] * stride_q_t
        + (kv_head * HEADS_PER_GROUP + query_head)[:, None] * stride_q_h
        + dims[None, :] * stride_q_d,
        mask=row_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    m_i = tl.full((QUERY_TILE * HEADS_PER_GROUP,), _M_INIT, tl.float32)
    l_i = tl.zeros((QUERY_TILE * HEADS_PER_GROUP,), tl.float32)
    acc = tl.zeros((QUERY_TILE * HEADS_PER_GROUP, BLOCK_D), tl.float32)
    union_base = tile_head * UNION_CAPACITY

    for union_idx in tl.range(0, union_count):
        block = tl.load(union_ids_ptr + union_base + union_idx)
        query_mask = tl.load(union_masks_ptr + union_base + union_idx)
        positions = block * BLOCK_N + cols
        pos_mask = positions < seq_len
        slots = tl.load(
            page_table_ptr + positions,
            mask=pos_mask,
            other=0,
        ).to(tl.int64)
        k, v = _load_kv(
            k_ptr=k_ptr,
            v_ptr=v_ptr,
            slots=slots,
            kv_head=kv_head,
            dims=dims,
            dim_mask=dim_mask,
            pos_mask=pos_mask,
            stride_k_t=stride_k_t,
            stride_k_h=stride_k_h,
            stride_k_d=stride_k_d,
            stride_v_t=stride_v_t,
            stride_v_h=stride_v_h,
            stride_v_d=stride_v_d,
        )
        selected = (query_mask & (1 << query_in_tile)) != 0
        valid = (
            row_mask[:, None]
            & selected[:, None]
            & (positions[None, :] < prefix_len + query_token[:, None] + 1)
        )
        m_i, l_i, acc = _attention_step(
            q=q,
            k=k,
            v=v,
            valid=valid,
            m=m_i,
            denom=l_i,
            acc=acc,
            scale=softmax_scale_log2e,
        )

    out = acc / l_i[:, None]
    tl.store(
        out_ptr
        + query_token[:, None] * stride_o_t
        + (kv_head * HEADS_PER_GROUP + query_head)[:, None] * stride_o_h
        + dims[None, :] * stride_o_d,
        out,
        mask=row_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def _block_sparse_attention_pair_kernel(
    q_ptr,  # bf16 [num_tokens, num_q_heads, head_dim]
    k_ptr,  # bf16 [pool_size, num_kv_heads, head_dim]
    v_ptr,  # bf16 [pool_size, num_kv_heads, head_dim]
    page_table_ptr,  # int [seq_len] token position -> pool slot
    pair_ids_ptr,  # int32 [num_kv_heads, num_pair_tiles, pair_capacity] block ids
    # int32 [num_kv_heads, num_pair_tiles, pair_capacity] per-query bitmask
    pair_masks_ptr,
    pair_counts_ptr,  # int32 [num_kv_heads, num_pair_tiles] live pair entries
    out_ptr,  # bf16 [num_tokens, num_q_heads, head_dim]
    num_tokens,
    prefix_len,
    seq_len,
    softmax_scale_log2e,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_k_t: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_k_d: tl.constexpr,
    stride_v_t: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_d: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_d: tl.constexpr,
    PAIR_CAPACITY: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    num_pair_tiles = tl.cdiv(num_tokens, 2)
    tile_head = kv_head * num_pair_tiles + tile
    pair_count = tl.load(pair_counts_ptr + tile_head)
    if pair_count == 0:
        return

    heads = tl.arange(0, HEADS_PER_GROUP)
    dims = tl.arange(0, BLOCK_D)
    cols = tl.arange(0, BLOCK_N)
    dim_mask = dims < HEAD_DIM
    token0 = tile * 2
    token1 = token0 + 1
    token1_valid = token1 < num_tokens
    q_base = (
        q_ptr
        + (kv_head * HEADS_PER_GROUP + heads)[:, None] * stride_q_h
        + dims[None, :] * stride_q_d
    )
    q0 = tl.load(q_base + token0 * stride_q_t, mask=dim_mask[None, :], other=0.0)
    q1 = tl.load(
        q_base + token1 * stride_q_t,
        mask=token1_valid & dim_mask[None, :],
        other=0.0,
    )
    m0 = tl.full((HEADS_PER_GROUP,), _M_INIT, tl.float32)
    l0 = tl.zeros((HEADS_PER_GROUP,), tl.float32)
    acc0 = tl.zeros((HEADS_PER_GROUP, BLOCK_D), tl.float32)
    m1 = tl.full((HEADS_PER_GROUP,), _M_INIT, tl.float32)
    l1 = tl.zeros((HEADS_PER_GROUP,), tl.float32)
    acc1 = tl.zeros((HEADS_PER_GROUP, BLOCK_D), tl.float32)
    pair_base = tile_head * PAIR_CAPACITY

    for pair_idx in tl.range(0, pair_count):
        block = tl.load(pair_ids_ptr + pair_base + pair_idx)
        query_mask = tl.load(pair_masks_ptr + pair_base + pair_idx)
        positions = block * BLOCK_N + cols
        pos_mask = positions < seq_len
        slots = tl.load(page_table_ptr + positions, mask=pos_mask, other=0).to(tl.int64)
        k, v = _load_kv(
            k_ptr=k_ptr,
            v_ptr=v_ptr,
            slots=slots,
            kv_head=kv_head,
            dims=dims,
            dim_mask=dim_mask,
            pos_mask=pos_mask,
            stride_k_t=stride_k_t,
            stride_k_h=stride_k_h,
            stride_k_d=stride_k_d,
            stride_v_t=stride_v_t,
            stride_v_h=stride_v_h,
            stride_v_d=stride_v_d,
        )
        if query_mask & 1:
            m0, l0, acc0 = _attention_step(
                q=q0,
                k=k,
                v=v,
                valid=(positions < prefix_len + token0 + 1)[None, :],
                m=m0,
                denom=l0,
                acc=acc0,
                scale=softmax_scale_log2e,
            )
        if query_mask & 2:
            m1, l1, acc1 = _attention_step(
                q=q1,
                k=k,
                v=v,
                valid=(positions < prefix_len + token1 + 1)[None, :],
                m=m1,
                denom=l1,
                acc=acc1,
                scale=softmax_scale_log2e,
            )

    out_base = (
        out_ptr
        + (kv_head * HEADS_PER_GROUP + heads)[:, None] * stride_o_h
        + dims[None, :] * stride_o_d
    )
    tl.store(
        out_base + token0 * stride_o_t,
        acc0 / l0[:, None],
        mask=dim_mask[None, :],
    )
    tl.store(
        out_base + token1 * stride_o_t,
        acc1 / l1[:, None],
        mask=token1_valid & dim_mask[None, :],
    )


def _build_query_tile_metadata(
    topk_idx: torch.Tensor,
    *,
    num_blocks: int,
    query_tile: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_kv_heads, num_tokens, topk = topk_idx.shape
    num_tiles = triton.cdiv(num_tokens, query_tile)
    # Tight bound: pairwise-disjoint per-token top-k sets.
    union_capacity = query_tile * topk
    query_tile_masks = torch.zeros(
        num_kv_heads,
        num_tiles,
        num_blocks,
        dtype=torch.int32,
        device=topk_idx.device,
    )
    union_ids = torch.empty(
        num_kv_heads,
        num_tiles,
        union_capacity,
        dtype=torch.int32,
        device=topk_idx.device,
    )
    union_masks = torch.empty_like(union_ids)
    union_counts = torch.empty(
        num_kv_heads,
        num_tiles,
        dtype=torch.int32,
        device=topk_idx.device,
    )

    _build_query_tile_masks[(num_tokens * num_kv_heads,)](
        topk_idx,
        query_tile_masks,
        num_tokens,
        num_blocks,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        TOPK=topk,
        QUERY_TILE=query_tile,
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=4,
    )
    _compact_query_tile_masks[(num_kv_heads * num_tiles,)](
        query_tile_masks,
        union_ids,
        union_masks,
        union_counts,
        num_blocks,
        UNION_CAPACITY=union_capacity,
        BLOCKS=triton.next_power_of_2(num_blocks),
        num_warps=8,
    )
    return union_ids, union_masks, union_counts


def _build_query_pair_metadata(
    union_ids: torch.Tensor,
    union_masks: torch.Tensor,
    union_counts: torch.Tensor,
    *,
    num_tokens: int,
    query_tile: int,
    topk: int,
    union_threshold: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_kv_heads = union_ids.shape[0]
    num_pair_tiles = triton.cdiv(num_tokens, 2)
    # Exact worst case for a pair: two disjoint top-k sets.
    pair_capacity = 2 * topk
    pair_ids = torch.empty(
        num_kv_heads,
        num_pair_tiles,
        pair_capacity,
        dtype=torch.int32,
        device=union_ids.device,
    )
    pair_masks = torch.empty_like(pair_ids)
    pair_counts = torch.empty(
        num_kv_heads,
        num_pair_tiles,
        dtype=torch.int32,
        device=union_ids.device,
    )

    _split_query_tile_metadata[(num_kv_heads * num_pair_tiles,)](
        union_ids,
        union_masks,
        union_counts,
        pair_ids,
        pair_masks,
        pair_counts,
        num_tokens,
        PARENT_CAPACITY=union_ids.shape[-1],
        PAIR_CAPACITY=pair_capacity,
        QUERY_TILE=query_tile,
        UNION_THRESHOLD=union_threshold,
        BLOCKS=triton.next_power_of_2(union_ids.shape[-1]),
        num_warps=8,
    )
    return pair_ids, pair_masks, pair_counts


def _common_launch_args(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    num_tokens: int,
    prefix_len: int,
    seq_len: int,
    softmax_scale_log2e: float,
    heads_per_group: int,
    head_dim: int,
    block_size: int,
) -> dict:
    return {
        "q_ptr": q,
        "k_ptr": k_cache,
        "v_ptr": v_cache,
        "page_table_ptr": page_table,
        "out_ptr": out,
        "num_tokens": num_tokens,
        "prefix_len": prefix_len,
        "seq_len": seq_len,
        "softmax_scale_log2e": softmax_scale_log2e,
        "stride_q_t": q.stride(0),
        "stride_q_h": q.stride(1),
        "stride_q_d": q.stride(2),
        "stride_k_t": k_cache.stride(0),
        "stride_k_h": k_cache.stride(1),
        "stride_k_d": k_cache.stride(2),
        "stride_v_t": v_cache.stride(0),
        "stride_v_h": v_cache.stride(1),
        "stride_v_d": v_cache.stride(2),
        "stride_o_t": out.stride(0),
        "stride_o_h": out.stride(1),
        "stride_o_d": out.stride(2),
        "HEADS_PER_GROUP": heads_per_group,
        "HEAD_DIM": head_dim,
        "BLOCK_D": triton.next_power_of_2(head_dim),
        "BLOCK_N": block_size,
    }


@torch.no_grad()
def block_sparse_attention(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    topk_idx: torch.Tensor,
    prefix_len: int,
    seq_len: int,
    block_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse prefill for one contiguous request; tiles whose union exceeds
    union_threshold fall back to the pair kernel."""
    num_tokens, num_q_heads, head_dim = q.shape
    # seq_len bounds the KV and page_table loads; the causal mask derives each
    # query's position from prefix_len, so the keys it admits must be loaded.
    assert seq_len >= prefix_len + num_tokens, (seq_len, prefix_len, num_tokens)
    # No top-k row is all padding: the selection keeps each token's own block.
    num_kv_heads = k_cache.shape[1]
    heads_per_group = num_q_heads // num_kv_heads
    # The pair kernel splits a tile into two-query pairs, so the tile is even.
    query_tile = min(_MAX_QUERY_TILE, _QUERY_ROWS // heads_per_group) // 2 * 2
    assert query_tile >= 2, heads_per_group
    num_tiles = triton.cdiv(num_tokens, query_tile)
    num_blocks = triton.cdiv(seq_len, block_size)
    topk = topk_idx.shape[-1]
    # The tile kernel wastes (union - topk) block scores per query, the pair
    # kernel at most topk; the split point between them is arbitrary, not measured.
    union_threshold = topk + 2 * topk // 3
    union_ids, union_masks, union_counts = _build_query_tile_metadata(
        topk_idx,
        num_blocks=num_blocks,
        query_tile=query_tile,
    )
    pair_ids, pair_masks, pair_counts = _build_query_pair_metadata(
        union_ids,
        union_masks,
        union_counts,
        num_tokens=num_tokens,
        query_tile=query_tile,
        topk=topk,
        union_threshold=union_threshold,
    )

    out = torch.empty_like(q)
    common_args = _common_launch_args(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        out=out,
        num_tokens=num_tokens,
        prefix_len=prefix_len,
        seq_len=seq_len,
        softmax_scale_log2e=softmax_scale * _LOG2E,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        block_size=block_size,
    )
    _block_sparse_attention_kernel[(num_tiles, num_kv_heads)](
        **common_args,
        union_ids_ptr=union_ids,
        union_masks_ptr=union_masks,
        union_counts_ptr=union_counts,
        UNION_CAPACITY=union_ids.shape[-1],
        QUERY_TILE=query_tile,
        UNION_THRESHOLD=union_threshold,
        num_warps=8,
        num_stages=4,
    )
    num_pair_tiles = triton.cdiv(num_tokens, 2)
    _block_sparse_attention_pair_kernel[(num_pair_tiles, num_kv_heads)](
        **common_args,
        pair_ids_ptr=pair_ids,
        pair_masks_ptr=pair_masks,
        pair_counts_ptr=pair_counts,
        PAIR_CAPACITY=pair_ids.shape[-1],
        num_warps=4,
        num_stages=3,
    )
    return out
