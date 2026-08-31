import torch
import triton
import triton.language as tl


# TODO. Now only page size == 1 is supported. Consider extend to page size > 1
@triton.jit
def compress_k_complete_kernel_new(
    key_cache_ptr,
    token_table_ptr,
    cu_new_k_token_nums_ptr,
    history_compress_k_token_nums_ptr,
    compressed_k_table_ptr,
    cu_total_compress_k_token_nums_ptr,
    full_compressed_k_ptr,
    batch_size,
    max_chunks_per_seq,
    token_table_cols,
    compressed_k_table_cols,
    head_num_k: tl.constexpr,
    head_dim: tl.constexpr,
    kernel_size: tl.constexpr,
    kernel_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    max_grid_chunks: tl.constexpr,
):
    """
    Single-kernel implementation that fuses k computation, key compression,
    key_cache write, and full_compressed_k read for ALL chunks (history + new).

    Grid: (batch_size, min(max_total_chunks, max_grid_chunks), head_num_k)
    where max_total_chunks = max_chunks_per_seq + max_history_chunks
    - chunk_in_seq in [0, history_chunks_in_seq): process HISTORY chunks
    - chunk_in_seq in [history_chunks_in_seq, total_chunks_in_seq): process NEW chunks

    If total_chunks > max_grid_chunks, each thread block loops to handle multiple chunks.

    Each program processes one (batch, chunk_in_seq, head) combination.
    """
    batch_idx = tl.program_id(0)
    grid_chunk_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    # Total number of chunks this thread block needs to process
    chunk_stride = max_grid_chunks

    if batch_idx >= batch_size or head_idx >= head_num_k:
        return

    # ====================================================================
    # PHASE 0: Determine chunk type and boundaries
    # ====================================================================

    history_compress = tl.load(history_compress_k_token_nums_ptr + batch_idx)

    # Compute how many NEW chunks this sequence actually has
    cu_new_k_start = tl.load(cu_new_k_token_nums_ptr + batch_idx)
    cu_new_k_end = tl.load(cu_new_k_token_nums_ptr + batch_idx + 1)
    new_k_count = cu_new_k_end - cu_new_k_start
    new_chunks_in_seq = tl.where(
        new_k_count >= kernel_size, (new_k_count - kernel_size) // kernel_stride + 1, 0
    )

    # Total chunks = history + new
    history_chunks_in_seq = history_compress
    total_chunks_in_seq = history_chunks_in_seq + new_chunks_in_seq

    output_start = tl.load(cu_total_compress_k_token_nums_ptr + batch_idx)

    # ====================================================================
    # LOOP: Handle multiple chunks per thread block if needed
    # ====================================================================

    # Iterate over all chunks assigned to this thread block
    chunk_in_seq = grid_chunk_idx

    while chunk_in_seq < total_chunks_in_seq:
        # Determine if processing history or new chunks
        is_history_chunk = chunk_in_seq < history_chunks_in_seq

        if is_history_chunk:
            # ====================================================================
            # PHASE 1: Process HISTORY chunks
            # ====================================================================

            # chunk_in_seq in [0, history_compress) -> history chunk index
            history_chunk_idx = chunk_in_seq

            global_full_idx = output_start + history_chunk_idx

            # Read from compressed_k_table: indices at y = history_chunk_idx
            full_compressed_idx = tl.load(
                compressed_k_table_ptr
                + batch_idx * compressed_k_table_cols
                + history_chunk_idx
            ).to(tl.int32)

            head_offset = (
                full_compressed_idx * head_num_k * head_dim + head_idx * head_dim
            )
            x = tl.load(
                key_cache_ptr + head_offset + tl.arange(0, BLOCK_SIZE),
                mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                other=0.0,
            )
            out_offset = global_full_idx * head_num_k * head_dim + head_idx * head_dim
            tl.store(
                full_compressed_k_ptr + out_offset + tl.arange(0, BLOCK_SIZE),
                x,
                mask=tl.arange(0, BLOCK_SIZE) < head_dim,
            )

        else:
            # ====================================================================
            # PHASE 2: Process NEW chunks
            # ====================================================================

            # chunk_in_seq in [history_compress, total_chunks_in_seq) -> new chunk index
            new_chunk_idx = chunk_in_seq - history_chunks_in_seq

            # Compute y index in token_table for this new chunk
            # y = new_chunk_idx * kernel_stride + history_compress * kernel_stride
            y = (new_chunk_idx + history_compress) * kernel_stride

            # Use nested if instead of continue (Triton doesn't support continue)
            if y < token_table_cols:
                # Compute y index in compressed_k_table for new_compressed_k_indices
                # y = new_chunk_idx + history_compress
                compressed_table_y = new_chunk_idx + history_compress

                if compressed_table_y < compressed_k_table_cols:
                    # Read new_compressed_k_indices from compressed_k_table
                    new_compressed_k_indices = tl.load(
                        compressed_k_table_ptr
                        + batch_idx * compressed_k_table_cols
                        + compressed_table_y
                    ).to(tl.int32)

                    # ====================================================================
                    # PHASE 3: Perform mean pooling compression on k
                    # ====================================================================

                    # Accumulate over all tokens in this chunk
                    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

                    for token_offset in range(kernel_size):
                        # Compute k_indices for this token
                        token_y = (
                            new_chunk_idx * kernel_stride + token_offset
                        ) + history_compress * kernel_stride

                        # Read k_indices from token_table
                        if token_y < token_table_cols:
                            token_k_indices = tl.load(
                                token_table_ptr + batch_idx * token_table_cols + token_y
                            ).to(tl.int32)
                        else:
                            token_k_indices = 0

                        # Load k from key_cache: key_cache[token_k_indices, head_idx, :]
                        key_base_offset = (
                            token_k_indices * head_num_k * head_dim
                            + head_idx * head_dim
                        )

                        # Vectorized load of head_dim values
                        x = tl.load(
                            key_cache_ptr + key_base_offset + tl.arange(0, BLOCK_SIZE),
                            mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                            other=0.0,
                        ).to(tl.float32)

                        acc += x

                    # Compute mean over the chunk
                    acc = acc / kernel_size

                    head_offset = (
                        new_compressed_k_indices * head_num_k * head_dim
                        + head_idx * head_dim
                    )
                    tl.store(
                        key_cache_ptr + head_offset + tl.arange(0, BLOCK_SIZE),
                        acc,
                        mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                    )

                    global_full_idx = output_start + history_compress + new_chunk_idx
                    out_offset = (
                        global_full_idx * head_num_k * head_dim + head_idx * head_dim
                    )
                    tl.store(
                        full_compressed_k_ptr + out_offset + tl.arange(0, BLOCK_SIZE),
                        acc,
                        mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                    )

        # Move to next chunk for this thread block
        chunk_in_seq += chunk_stride


@triton.jit
def _eagle_draft_tree_mask_kernel(
    out_ptr,  # bool [padded_bs*num_draft_tokens*num_draft_tokens]
    num_visible_out_ptr,  # int32 [padded_bs*num_draft_tokens]
    custom_mask_ptr,  # bool packed [num_draft_tokens, seq_len+num_draft_tokens] rows
    seq_lens_ptr,
    num_draft_tokens,
    bs,
    NP2_BS: tl.constexpr,
    NP2_DT: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    q = tl.arange(0, NP2_DT)[:, None].to(tl.int64)
    k = tl.arange(0, NP2_DT)[None, :].to(tl.int64)
    square = (q < num_draft_tokens) & (k < num_draft_tokens)

    i = tl.arange(0, NP2_BS)
    prev = tl.load(seq_lens_ptr + i, mask=(i < b) & (b < bs), other=0).to(tl.int64)
    base = (
        tl.sum(tl.where(i < b, prev + num_draft_tokens, 0), axis=0) * num_draft_tokens
    )
    seq_len = tl.load(seq_lens_ptr + b, mask=b < bs, other=0).to(tl.int64)
    kv_len = seq_len + num_draft_tokens

    vals = tl.load(
        custom_mask_ptr + base + q * kv_len + seq_len + k,
        mask=square & (b < bs),
        other=1,
    )
    tl.store(
        out_ptr + (b * num_draft_tokens + q) * num_draft_tokens + k,
        vals,
        mask=square,
    )
    num_visible = tl.sum(tl.where(k <= q, vals.to(tl.int32), 0), axis=1)
    tl.store(
        num_visible_out_ptr + b * num_draft_tokens + tl.arange(0, NP2_DT),
        num_visible,
        mask=tl.arange(0, NP2_DT) < num_draft_tokens,
    )


def copy_eagle_draft_tree_mask(
    out: torch.Tensor,
    *,
    num_visible_out: torch.Tensor,
    custom_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    num_draft_tokens: int,
    bs: int,
    padded_bs: int,
) -> None:
    """Copy into ``out`` (flat, padded_bs squares)
    each request's trailing draft x draft square of EAGLE's packed custom_mask;
    copy each row's causal-prefix visible-key popcount,
    matching the compaction's ``key_pos < token_pos`` gate, into ``num_visible_out``;
    rows in [bs, padded_bs) come out all-True and popcount to the 1-based offsets.
    """
    _eagle_draft_tree_mask_kernel[(padded_bs,)](
        out,
        num_visible_out,
        custom_mask,
        seq_lens,
        num_draft_tokens,
        bs,
        NP2_BS=triton.next_power_of_2(bs),
        NP2_DT=triton.next_power_of_2(num_draft_tokens),
    )


@triton.jit
def _compact_sparse_tree_page_table_kernel(
    topk_rows_ptr,
    topk_idx_ptr,
    token_to_bs_ptr,
    token_pos_in_bs_ptr,
    prefix_lens_ptr,
    draft_tree_mask_ptr,
    source_row_lens_ptr,
    out_page_table_ptr,
    max_sparse_tokens: tl.constexpr,
    sparse_topk: tl.constexpr,
    page_stride_0,
    page_stride_1,
    topk_stride_0,
    topk_stride_1,
    topk_stride_2,
    out_stride_0,
    out_stride_1,
    head_group_num: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    in_range = offs < max_sparse_tokens

    token = row // head_group_num
    head_group = row - token * head_group_num
    batch = tl.load(token_to_bs_ptr + token)
    token_pos = tl.load(token_pos_in_bs_ptr + token)
    prefix = tl.load(prefix_lens_ptr + batch)
    query_idx = token_pos - prefix - 1

    topk_block = offs // block_size
    block_off = offs - topk_block * block_size
    topk_values = tl.load(
        topk_idx_ptr
        + head_group * topk_stride_0
        + token * topk_stride_1
        + topk_block * topk_stride_2,
        mask=in_range & (topk_block < sparse_topk),
        other=-1,
    )
    key_pos = topk_values * block_size + block_off

    source_len = tl.load(source_row_lens_ptr + row)
    valid = in_range & (offs < source_len) & (topk_values >= 0) & (key_pos < token_pos)
    key_draft_idx = key_pos - prefix
    draft_key_valid = (key_draft_idx >= 0) & (key_draft_idx < num_draft_tokens)
    draft_offsets = (
        batch * num_draft_tokens * num_draft_tokens
        + query_idx * num_draft_tokens
        + key_draft_idx
    )
    draft_visible = tl.load(
        draft_tree_mask_ptr + draft_offsets, mask=valid & draft_key_valid, other=0
    ).to(tl.int1)
    keep = valid & ((key_pos < prefix) | draft_visible)

    keep_i32 = keep.to(tl.int32)
    ranks = tl.cumsum(keep_i32, 0) - 1
    count = tl.sum(keep_i32, axis=0)

    values = tl.load(
        topk_rows_ptr + row * page_stride_0 + offs * page_stride_1,
        mask=in_range,
        other=0,
    )
    # Consecutive tl.store have no barrier:
    # the tail zero and the scatter must write disjoint ranges.
    tl.store(
        out_page_table_ptr + row * out_stride_0 + offs * out_stride_1,
        0,
        mask=in_range & (offs >= count),
    )
    tl.store(
        out_page_table_ptr + row * out_stride_0 + ranks * out_stride_1,
        values,
        mask=keep,
    )


def compact_sparse_tree_page_table(
    topk_rows: torch.Tensor,
    *,
    topk_idx: torch.Tensor,
    token_to_bs: torch.Tensor,
    token_pos_in_bs: torch.Tensor,
    prefix_lens: torch.Tensor,
    draft_tree_mask: torch.Tensor,
    source_row_lens: torch.Tensor,
    out_page_table: torch.Tensor,
    num_draft_tokens: int,
    block_size: int,
    head_group_num: int,
) -> None:
    """Compact the staged verify rows into ``out_page_table`` by tree visibility;
    ``draft_tree_mask`` is the flat square from copy_eagle_draft_tree_mask."""
    rows, max_sparse_tokens = topk_rows.shape
    sparse_topk = topk_idx.shape[2]
    _compact_sparse_tree_page_table_kernel[(rows,)](
        topk_rows,
        topk_idx,
        token_to_bs,
        token_pos_in_bs,
        prefix_lens,
        draft_tree_mask,
        source_row_lens,
        out_page_table,
        max_sparse_tokens=max_sparse_tokens,
        sparse_topk=sparse_topk,
        page_stride_0=topk_rows.stride(0),
        page_stride_1=topk_rows.stride(1),
        topk_stride_0=topk_idx.stride(0),
        topk_stride_1=topk_idx.stride(1),
        topk_stride_2=topk_idx.stride(2),
        out_stride_0=out_page_table.stride(0),
        out_stride_1=out_page_table.stride(1),
        head_group_num=head_group_num,
        num_draft_tokens=num_draft_tokens,
        block_size=block_size,
        BLOCK_N=triton.next_power_of_2(max_sparse_tokens),
        num_warps=8,
    )


@triton.jit
def _fill_dense_page_table_rows_kernel(
    page_table_ptr,
    token_to_bs_ptr,
    token_pos_in_bs_ptr,
    prefix_lens_ptr,
    draft_tree_mask_ptr,
    out_page_table_ptr,
    dense_len,
    out_width,
    page_stride_0,
    page_stride_1,
    out_stride_0,
    out_stride_1,
    head_group_num: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // head_group_num
    group = row - token * head_group_num
    batch = tl.load(token_to_bs_ptr + token)
    token_pos = tl.load(token_pos_in_bs_ptr + token)
    if token_pos >= dense_len:
        return
    prefix = tl.load(prefix_lens_ptr + batch)

    query_idx = token_pos - prefix - 1
    d = tl.arange(0, BLOCK_D)
    # Same causal gate as _eagle_draft_tree_mask_kernel's popcount:
    # the staged square is all-True on capture seeds and padded rows.
    bits = tl.load(
        draft_tree_mask_ptr
        + batch * num_draft_tokens * num_draft_tokens
        + query_idx * num_draft_tokens
        + d,
        mask=(d < num_draft_tokens) & (d <= query_idx),
        other=0,
    ).to(tl.int32)
    num_visible = tl.sum(bits, axis=0)
    ranks = tl.cumsum(bits, axis=0) - 1
    draft_locs = tl.load(
        page_table_ptr + batch * page_stride_0 + (prefix + d) * page_stride_1,
        mask=bits > 0,
        other=0,
    )
    # Head-group encoded as loc * head_group_num + group, matching get_block_table.
    tl.store(
        out_page_table_ptr + row * out_stride_0 + (prefix + ranks) * out_stride_1,
        draft_locs * head_group_num + group,
        mask=bits > 0,
    )

    count = prefix + num_visible
    for start in range(0, out_width, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        is_prefix = offs < prefix
        locs = tl.load(
            page_table_ptr + batch * page_stride_0 + offs * page_stride_1,
            mask=is_prefix,
            other=0,
        )
        tl.store(
            out_page_table_ptr + row * out_stride_0 + offs * out_stride_1,
            tl.where(is_prefix, locs * head_group_num + group, 0),
            mask=(offs < out_width) & (is_prefix | (offs >= count)),
        )


def fill_dense_page_table_rows(
    page_table: torch.Tensor,
    *,
    token_to_bs: torch.Tensor,
    token_pos_in_bs: torch.Tensor,
    prefix_lens: torch.Tensor,
    draft_tree_mask: torch.Tensor,
    out_page_table: torch.Tensor,
    dense_len: int,
    num_draft_tokens: int,
    head_group_num: int,
) -> None:
    """Overwrite dense rows (token_pos_in_bs < dense_len,
    the dense side of the dense_len dispatch threshold) of ``out_page_table`` in place,
    with prefix + visible-draft slots and a zeroed tail."""
    rows, out_width = out_page_table.shape
    _fill_dense_page_table_rows_kernel[(rows,)](
        page_table,
        token_to_bs,
        token_pos_in_bs,
        prefix_lens,
        draft_tree_mask,
        out_page_table,
        dense_len,
        out_width,
        page_stride_0=page_table.stride(0),
        page_stride_1=page_table.stride(1),
        out_stride_0=out_page_table.stride(0),
        out_stride_1=out_page_table.stride(1),
        head_group_num=head_group_num,
        num_draft_tokens=num_draft_tokens,
        # Arbitrary power-of-2 tile for the tail-zero loop;
        # only its trip count changes.
        BLOCK_N=1024,
        BLOCK_D=triton.next_power_of_2(num_draft_tokens),
        num_warps=8,
    )
