from typing import Optional, Tuple

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
    k_stride,
    compressed_k_table_ptr,
    cu_new_compress_k_token_nums_ptr,
    cu_total_compress_k_token_nums_ptr,
    total_compress_k_token_nums_ptr,
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

    Each thread processes one (batch, chunk_in_seq, head) combination.
    Each program computes and writes its own head slice.

    Args:
        key_cache_ptr: Input key cache tensor [total_tokens, head_num_k, head_dim]
        token_table_ptr: Token table [batch_size, token_table_cols]
        cu_new_k_token_nums_ptr: Cumulative new token nums [batch_size + 1]
        history_compress_k_token_nums_ptr: History compressed token nums [batch_size]
        k_stride: Stride for k computation
        compressed_k_table_ptr: Compressed k table [batch_size, compressed_k_table_cols]
        cu_new_compress_k_token_nums_ptr: Cumulative new compressed token nums [batch_size + 1]
        cu_total_compress_k_token_nums_ptr: Cumulative total compressed token nums [batch_size + 1]
        total_compress_k_token_nums_ptr: Total compressed token nums per batch [batch_size]
        full_compressed_k_ptr: Output buffer [total_compressed_tokens, head_num_k, head_dim]
        batch_size: Number of sequences in batch
        max_chunks_per_seq: Maximum possible NEW chunks per sequence
        token_table_cols: Number of columns in token_table
        compressed_k_table_cols: Number of columns in compressed_k_table
        head_num_k: Number of attention heads
        head_dim: Dimension per head
        kernel_size: Tokens per chunk for compression
        kernel_stride: Stride between chunk starts
        BLOCK_SIZE: Vectorized load/store width
        max_grid_chunks: Maximum grid dimension for chunks (kernel loops if more chunks needed)
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

    # Get cumulative positions for this batch
    cu_total_start = tl.load(cu_total_compress_k_token_nums_ptr + batch_idx)

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

            # Gather this program's head slice from the compressed cache slot
            # into the contiguous output (pure copy, one head per program).
            history_chunk_idx = chunk_in_seq
            global_full_idx = cu_total_start + history_chunk_idx
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
            ).to(tl.float32)
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

            new_chunk_idx = chunk_in_seq - history_chunks_in_seq
            y = new_chunk_idx * kernel_stride + history_compress * k_stride

            # Use nested if instead of continue (Triton doesn't support continue)
            if y < token_table_cols:
                compressed_table_y = new_chunk_idx + history_compress

                if compressed_table_y < compressed_k_table_cols:
                    new_compressed_k_indices = tl.load(
                        compressed_k_table_ptr
                        + batch_idx * compressed_k_table_cols
                        + compressed_table_y
                    ).to(tl.int32)

                    # Mean-pool this program's head over the chunk. The loop is
                    # unrolled so the per-token table loads pipeline instead of
                    # forming a serial load dependency chain; the accumulation
                    # order matches the original two-phase kernel, keeping the
                    # mean bitwise identical.
                    acc = tl.zeros([head_dim], dtype=tl.float32)
                    for token_offset in tl.static_range(kernel_size):
                        token_y = (
                            new_chunk_idx * kernel_stride + token_offset
                        ) + history_compress * k_stride
                        token_k_indices = tl.load(
                            token_table_ptr + batch_idx * token_table_cols + token_y,
                            mask=token_y < token_table_cols,
                            other=0,
                        ).to(tl.int32)
                        key_base_offset = (
                            token_k_indices * head_num_k * head_dim
                            + head_idx * head_dim
                        )
                        x = tl.load(
                            key_cache_ptr + key_base_offset + tl.arange(0, BLOCK_SIZE),
                            mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                            other=0.0,
                        ).to(tl.float32)
                        acc += x
                    acc = acc / kernel_size

                    # Store this head to its compressed cache slot, then read it
                    # back for the contiguous output so the stored value keeps
                    # the exact cache-dtype roundtrip of the two-phase original.
                    slot_offset = (
                        new_compressed_k_indices * head_num_k * head_dim
                        + head_idx * head_dim
                    )
                    tl.store(
                        key_cache_ptr + slot_offset + tl.arange(0, BLOCK_SIZE),
                        acc,
                        mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                    )
                    global_full_idx = cu_total_start + history_compress + new_chunk_idx
                    rt = tl.load(
                        key_cache_ptr + slot_offset + tl.arange(0, BLOCK_SIZE),
                        mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                        other=0.0,
                    ).to(tl.float32)
                    out_offset = (
                        global_full_idx * head_num_k * head_dim + head_idx * head_dim
                    )
                    tl.store(
                        full_compressed_k_ptr + out_offset + tl.arange(0, BLOCK_SIZE),
                        rt,
                        mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                    )

        # Move to next chunk for this thread block
        chunk_in_seq += chunk_stride


@triton.jit
def compress_k_complete_kernel_new_padded(
    key_cache_ptr,
    token_table_ptr,
    cu_new_k_token_nums_ptr,
    history_compress_k_token_nums_ptr,
    k_stride,
    compressed_k_table_ptr,
    cu_new_compress_k_token_nums_ptr,
    cu_total_compress_k_token_nums_ptr,
    total_compress_k_token_nums_ptr,
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
    Padded layout version: stores compressed keys in batch-major order.

    Output layout: full_compressed_k[batch_idx * max_chunks_per_seq + chunk_idx]
    This allows using reshape() to view per-batch data for debugging.

    Grid: (batch_size, min(max_total_chunks, max_grid_chunks), head_num_k)
    where max_total_chunks = max_chunks_per_seq + max_history_chunks

    If total_chunks > max_grid_chunks, each thread block loops to handle multiple chunks.
    """
    batch_idx = tl.program_id(0)
    grid_chunk_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    # Total number of chunks this thread block needs to process
    # Each thread block handles: grid_chunk_idx, grid_chunk_idx + max_grid_chunks, grid_chunk_idx + 2*max_grid_chunks, ...
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

    # ====================================================================
    # LOOP: Handle multiple chunks per thread block if needed
    # ====================================================================

    # Iterate over all chunks assigned to this thread block
    # chunk_in_seq = grid_chunk_idx, grid_chunk_idx + chunk_stride, grid_chunk_idx + 2*chunk_stride, ...
    chunk_in_seq = grid_chunk_idx

    while chunk_in_seq < total_chunks_in_seq:
        # Skip if this chunk_in_seq doesn't exist
        # (This check is now inside the loop)

        # Determine if processing history or new chunks
        is_history_chunk = chunk_in_seq < history_chunks_in_seq

        if is_history_chunk:
            # ====================================================================
            # PHASE 1: Process HISTORY chunks (PADDED LAYOUT)
            # ====================================================================

            # Gather this program's head slice from the compressed cache slot
            # into the batch-major output (pure copy, one head per program).
            history_chunk_idx = chunk_in_seq
            global_full_idx = batch_idx * max_chunks_per_seq + history_chunk_idx
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
            ).to(tl.float32)
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

            new_chunk_idx = chunk_in_seq - history_chunks_in_seq
            y = new_chunk_idx * kernel_stride + history_compress * k_stride

            # Use nested if instead of continue (Triton doesn't support continue)
            if y < token_table_cols:
                compressed_table_y = new_chunk_idx + history_compress

                if compressed_table_y < compressed_k_table_cols:
                    new_compressed_k_indices = tl.load(
                        compressed_k_table_ptr
                        + batch_idx * compressed_k_table_cols
                        + compressed_table_y
                    ).to(tl.int32)

                    # Mean-pool this program's head over the chunk. The loop is
                    # unrolled so the per-token table loads pipeline instead of
                    # forming a serial load dependency chain; the accumulation
                    # order matches the original two-phase kernel, keeping the
                    # mean bitwise identical.
                    acc = tl.zeros([head_dim], dtype=tl.float32)
                    for token_offset in tl.static_range(kernel_size):
                        token_y = (
                            new_chunk_idx * kernel_stride + token_offset
                        ) + history_compress * k_stride
                        token_k_indices = tl.load(
                            token_table_ptr + batch_idx * token_table_cols + token_y,
                            mask=token_y < token_table_cols,
                            other=0,
                        ).to(tl.int32)
                        key_base_offset = (
                            token_k_indices * head_num_k * head_dim
                            + head_idx * head_dim
                        )
                        x = tl.load(
                            key_cache_ptr + key_base_offset + tl.arange(0, BLOCK_SIZE),
                            mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                            other=0.0,
                        ).to(tl.float32)
                        acc += x
                    acc = acc / kernel_size

                    # Store this head to its compressed cache slot, then read it
                    # back for the batch-major output so the stored value keeps
                    # the exact cache-dtype roundtrip of the two-phase original.
                    slot_offset = (
                        new_compressed_k_indices * head_num_k * head_dim
                        + head_idx * head_dim
                    )
                    tl.store(
                        key_cache_ptr + slot_offset + tl.arange(0, BLOCK_SIZE),
                        acc,
                        mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                    )
                    global_full_idx = (
                        batch_idx * max_chunks_per_seq
                        + history_compress
                        + new_chunk_idx
                    )
                    rt = tl.load(
                        key_cache_ptr + slot_offset + tl.arange(0, BLOCK_SIZE),
                        mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                        other=0.0,
                    ).to(tl.float32)
                    out_offset = (
                        global_full_idx * head_num_k * head_dim + head_idx * head_dim
                    )
                    tl.store(
                        full_compressed_k_ptr + out_offset + tl.arange(0, BLOCK_SIZE),
                        rt,
                        mask=tl.arange(0, BLOCK_SIZE) < head_dim,
                    )

        # Move to next chunk for this thread block
        chunk_in_seq += chunk_stride


@triton.jit
def _compact_sparse_tree_page_table_kernel(
    page_table,
    topk_idx,
    token_to_bs,
    token_pos_in_bs,
    prefix_lens,
    mask_batch_offsets,
    custom_mask,
    draft_tree_mask,
    source_cache_seqlens,
    out_page_table,
    out_cache_seqlens,
    max_sparse_tokens: tl.constexpr,
    sparse_topk: tl.constexpr,
    page_stride_0: tl.constexpr,
    page_stride_1: tl.constexpr,
    topk_stride_0: tl.constexpr,
    topk_stride_1: tl.constexpr,
    topk_stride_2: tl.constexpr,
    out_stride_0: tl.constexpr,
    out_stride_1: tl.constexpr,
    head_group_num: tl.constexpr,
    draft_token_num: tl.constexpr,
    block_size: tl.constexpr,
    use_draft_tree_mask: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    in_range = offs < max_sparse_tokens

    token = row // head_group_num
    head_group = row - token * head_group_num
    batch = tl.load(token_to_bs + token)
    token_pos = tl.load(token_pos_in_bs + token)
    prefix = tl.load(prefix_lens + batch)
    kv_len = prefix + draft_token_num
    query_idx = token_pos - prefix - 1

    topk_block = offs // block_size
    block_off = offs - topk_block * block_size
    topk_values = tl.load(
        topk_idx
        + head_group * topk_stride_0
        + token * topk_stride_1
        + topk_block * topk_stride_2,
        mask=in_range & (topk_block < sparse_topk),
        other=-1,
    )
    key_pos = topk_values * block_size + block_off

    source_len = tl.load(source_cache_seqlens + row)
    valid = (
        in_range
        & (offs < source_len)
        & (topk_values >= 0)
        & (key_pos < kv_len)
        & (key_pos < token_pos)
        & (query_idx >= 0)
        & (query_idx < draft_token_num)
    )
    if use_draft_tree_mask:
        key_draft_idx = key_pos - prefix
        draft_key_valid = (key_draft_idx >= 0) & (key_draft_idx < draft_token_num)
        draft_offsets = (
            batch * draft_token_num * draft_token_num
            + query_idx * draft_token_num
            + key_draft_idx
        )
        draft_visible = tl.load(
            draft_tree_mask + draft_offsets,
            mask=valid & draft_key_valid,
            other=0,
        ).to(tl.int1)
        visible = (key_pos < prefix) | draft_visible
    else:
        mask_offsets = (
            tl.load(mask_batch_offsets + batch) + query_idx * kv_len + key_pos
        )
        visible = tl.load(custom_mask + mask_offsets, mask=valid, other=0).to(tl.int1)
    keep = valid & visible
    keep_i32 = keep.to(tl.int32)
    ranks = tl.cumsum(keep_i32, 0) - 1
    count = tl.sum(keep_i32, axis=0)

    values = tl.load(
        page_table + row * page_stride_0 + offs * page_stride_1,
        mask=in_range,
        other=0,
    )
    # The scatter below fills exactly [0, count); zero only the tail so the two
    # stores write disjoint ranges. A full-row zero fill would race with the
    # scatter across warps (no barrier between consecutive tl.store calls).
    tl.store(
        out_page_table + row * out_stride_0 + offs * out_stride_1,
        0,
        mask=in_range & (offs >= count),
    )
    tl.store(
        out_page_table + row * out_stride_0 + ranks * out_stride_1,
        values,
        mask=keep,
    )
    tl.store(out_cache_seqlens + row, count)


def compact_sparse_tree_page_table(
    page_table: torch.Tensor,
    topk_idx: torch.Tensor,
    token_to_bs: torch.Tensor,
    token_pos_in_bs: torch.Tensor,
    prefix_lens: torch.Tensor,
    mask_batch_offsets: Optional[torch.Tensor],
    custom_mask: torch.Tensor,
    source_cache_seqlens: torch.Tensor,
    draft_token_num: int,
    block_size: int,
    draft_tree_mask: Optional[torch.Tensor] = None,
    out_page_table: Optional[torch.Tensor] = None,
    out_cache_seqlens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter sparse slots by EAGLE tree visibility and compact each row.

    CUDA only; the pure-torch reference lives next to the unit tests
    (test_minicpm_sparse_target_verify.py).
    """
    if draft_tree_mask is None and mask_batch_offsets is None:
        raise ValueError(
            "MiniCPM sparse tree compaction needs mask_batch_offsets when "
            "no draft_tree_mask is given."
        )
    if not page_table.is_cuda:
        raise ValueError("MiniCPM sparse tree compaction expects CUDA tensors.")

    rows, max_sparse_tokens = page_table.shape
    head_group_num, token_num, sparse_topk = topk_idx.shape
    if rows != token_num * head_group_num:
        raise ValueError(
            "MiniCPM sparse tree compaction expects token-major rows: "
            f"rows={rows}, token_num={token_num}, head_group_num={head_group_num}."
        )
    if out_page_table is None:
        out_page_table = torch.empty_like(page_table)
    if out_cache_seqlens is None:
        out_cache_seqlens = torch.empty(
            (rows,), dtype=torch.int32, device=page_table.device
        )

    block_n = triton.next_power_of_2(max_sparse_tokens)
    _compact_sparse_tree_page_table_kernel[(rows,)](
        page_table,
        topk_idx,
        token_to_bs,
        token_pos_in_bs,
        prefix_lens,
        # Pointer placeholders: each is only loaded on the branch where the
        # corresponding mask is in use, so any valid tensor works for the other.
        mask_batch_offsets if mask_batch_offsets is not None else source_cache_seqlens,
        custom_mask if custom_mask is not None else draft_tree_mask,
        draft_tree_mask if draft_tree_mask is not None else custom_mask,
        source_cache_seqlens,
        out_page_table,
        out_cache_seqlens,
        max_sparse_tokens=max_sparse_tokens,
        sparse_topk=sparse_topk,
        page_stride_0=page_table.stride(0),
        page_stride_1=page_table.stride(1),
        topk_stride_0=topk_idx.stride(0),
        topk_stride_1=topk_idx.stride(1),
        topk_stride_2=topk_idx.stride(2),
        out_stride_0=out_page_table.stride(0),
        out_stride_1=out_page_table.stride(1),
        head_group_num=head_group_num,
        draft_token_num=int(draft_token_num),
        block_size=int(block_size),
        use_draft_tree_mask=draft_tree_mask is not None,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return out_page_table, out_cache_seqlens


# Sparse page_table -> FlashInfer CSR conversion: torch.cumsum builds the row
# offsets (kv_indptr) as a parallel scan, then flatten_and_fill_kernel scatters
# each row's valid kv indices into the CSR layout.


@triton.jit
def flatten_and_fill_kernel(
    sparse_page_table_ptr,
    cache_seqlens_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    kv_last_page_len_ptr,
    max_sparse_tokens: tl.constexpr,
    sparse_bs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 256,
):
    """Flatten sparse_page_table and fill kv_last_page_len."""
    pid = tl.program_id(axis=0)

    if pid >= sparse_bs:
        return

    # Get offset and num_valid
    offset = tl.load(kv_indptr_ptr + pid)
    num_valid = tl.load(cache_seqlens_ptr + pid)

    # Copy valid entries
    num_loops = tl.cdiv(num_valid, BLOCK_SIZE)
    for i in range(num_loops):
        idx = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < num_valid

        src_idx = pid * max_sparse_tokens + idx
        data = tl.load(sparse_page_table_ptr + src_idx, mask=mask, other=0)

        dst_idx = offset + idx
        tl.store(kv_indices_ptr + dst_idx, data, mask=mask)

    # Fill kv_last_page_len
    tl.store(kv_last_page_len_ptr + pid, 1)


def convert_sparse_page_table_to_flashinfer(
    sparse_page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert sparse_page_table to FlashInfer format.

    Args:
        sparse_page_table: [sparse_bs, max_sparse_tokens] - Valid entries at start
        cache_seqlens: [sparse_bs] - Number of valid entries per row
        kv_indptr: Pre-allocated [sparse_bs + 1] buffer for output
        kv_indices: Pre-allocated [sparse_bs * max_sparse_tokens] buffer for output
        kv_last_page_len: Pre-allocated [sparse_bs] buffer for output

    Returns:
        Tuple of (kv_indptr, kv_indices, kv_last_page_len) - modified in-place
    """
    sparse_bs = cache_seqlens.shape[0]
    max_sparse_tokens = sparse_page_table.shape[1]

    # Row offsets = exclusive prefix sum of the per-row valid counts. torch.cumsum
    # is a parallel scan; the old cumsum_kernel walked all sparse_bs rows on a
    # single thread-block (sparse_bs = query_tokens * head_group = 16384 for an 8K
    # chunk, ~1ms/call). Integer prefix sum is bit-identical. The leading 0 is
    # zeroed on-device via zero_(): under CUDA-graph capture kv_indptr is a
    # persistent buffer, and a scalar ``kv_indptr[0] = 0`` would be an illegal
    # unpinned CPU->CUDA copy.
    kv_indptr[:1].zero_()
    kv_indptr[1:] = torch.cumsum(cache_seqlens, dim=0, dtype=kv_indptr.dtype)

    # Flatten and fill
    BLOCK_SIZE = 256
    flatten_and_fill_kernel[(sparse_bs,)](
        sparse_page_table,
        cache_seqlens,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        max_sparse_tokens=max_sparse_tokens,
        sparse_bs=sparse_bs,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return kv_indptr, kv_indices, kv_last_page_len
