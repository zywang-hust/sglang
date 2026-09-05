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
def _select_topk_blocks_kernel(
    scores_ptr,
    out_ptr,
    num_blocks,
    topk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    valid = offs < num_blocks
    score = tl.load(
        scores_ptr + row * num_blocks + offs, mask=valid, other=float("-inf")
    )
    finite = score > float("-inf")
    # Order-preserving unsigned image of the float bits, held in int64.
    bits = score.to(tl.int32, bitcast=True)
    ordered = tl.where(bits < 0, ~bits, bits ^ (-2147483648))
    key = tl.where(finite, ordered.to(tl.int64) & 0xFFFFFFFF, 0)

    bins = tl.arange(0, 256)
    k_rem = tl.minimum(topk, tl.sum(finite.to(tl.int32), 0))
    prefix = tl.zeros([1], dtype=tl.int64)
    prefix = tl.sum(prefix, 0)
    for shift in tl.static_range(24, -1, -8):
        cand = finite & ((key >> (shift + 8)) == prefix)
        digit = ((key >> shift) & 0xFF).to(tl.int32)
        hist = tl.histogram(tl.where(cand, digit, 0), 256)
        not_cand = tl.sum((~cand).to(tl.int32), 0)
        hist = hist - tl.where(bins == 0, not_cand, 0)
        cum_ge = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist
        cum_gt = cum_ge - hist
        hit = (cum_ge >= k_rem) & (cum_gt < k_rem)
        k_rem = k_rem - tl.sum(tl.where(hit, cum_gt, 0), 0)
        prefix = (prefix << 8) | tl.sum(tl.where(hit, bins, 0), 0).to(tl.int64)

    above = finite & (key > prefix)
    at = finite & (key == prefix)
    # Resolve cutoff ties by block index; the prefix sum also preserves index order.
    selected = above | (at & (tl.cumsum(at.to(tl.int32), 0) <= k_rem))
    slot = tl.cumsum(selected.to(tl.int32), 0) - 1
    tl.store(out_ptr + row * topk + slot, offs, mask=selected)
    num_selected = tl.sum(selected.to(tl.int32), 0)
    slots = tl.arange(0, BLOCK_K)
    tl.store(
        out_ptr + row * topk + slots,
        -1,
        mask=(slots >= num_selected) & (slots < topk),
    )


def select_topk_blocks(scores: torch.Tensor, *, topk: int) -> torch.Tensor:
    """Top-k block indices of every row of ``scores`` ([..., num_blocks]
    fp32), ascending, -1 padded; rows with -inf scores select fewer."""
    num_blocks = scores.shape[-1]
    rows = scores.numel() // num_blocks
    out = torch.empty(
        (*scores.shape[:-1], topk), dtype=torch.int32, device=scores.device
    )
    _select_topk_blocks_kernel[(rows,)](
        scores,
        out,
        num_blocks,
        topk,
        BLOCK_N=triton.next_power_of_2(num_blocks),
        BLOCK_K=triton.next_power_of_2(topk),
    )
    return out
