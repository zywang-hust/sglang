"""Sparse attention utilities for MiniCPM models.

This module provides sparse attention helpers and utilities for MiniCPM models,
combining both backend-agnostic sparse attention components and kernel utilities.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionMetadata,
    )
    from sglang.srt.layers.attention.minicpm_backend import MiniCPMBackendMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

import math

import tilelang
import tilelang.math
import triton
import triton.language as tl
from sgl_kernel import infllmv2_attn_stage1, max_pooling_1d_varlen

from sglang.srt.layers.attention.minicpm_fuse_kernel import _bucket_size
from sglang.srt.layers.attention.minicpm_sparse_kernels import (
    compress_k_complete_kernel_new,
    compress_k_complete_kernel_new_padded,
)
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

logger = logging.getLogger(__name__)


def sparse_block_quantized_count(
    positions: torch.Tensor, block_size: int, sparse_topk: int
) -> torch.Tensor:
    """Number of sparse KV tokens visible to a query at each causal position.

    This is the single source of truth for MiniCPM sparse "cache seqlens". Below
    the sparse budget (``block_size * sparse_topk``) every key is visible, so the
    count equals the position itself. Above the budget the sparse attention reads
    ``sparse_topk`` blocks of ``block_size`` keys each; because the local window
    forces the frontier (most recent) block to be selected, that frontier block
    is the only partial one, giving ``(sparse_topk - 1)`` full blocks plus
    ``position % block_size``.

    Both the decode count precompute (``ScheduleBatch``, one count per request at
    ``seq_len``) and the tree-verify CUDA-graph plan (``MiniCPMSparseBackend``, one per
    draft token at its causal ``token_pos``) build on this; routing them through
    one helper keeps their FlashInfer split-KV plans byte-identical to the data
    the in-graph kernels write. Note this encodes the local-window-always-selected
    invariant: if the sparse block-selection policy changes (window eviction,
    dynamic window, non-flat draft layout), revisit here.

    ``positions`` may be any integer tensor (CPU or CUDA); the result keeps its
    dtype/device.
    """
    budget = block_size * sparse_topk
    mod = positions % block_size
    capped = torch.where(
        mod == 0,
        torch.full_like(positions, budget),
        (sparse_topk - 1) * block_size + mod,
    )
    return torch.where(positions <= budget, positions, capped)


@triton.jit
def _compress_level_metadata_kernel(
    prefix_ptr,  # int32 [bs]   prefix (history) seq lens
    history_ptr,  # int32 [bs]   out: history_compress_token_nums
    new_token_ptr,  # int32 [bs]   out: new_token_nums (stored unclamped)
    new_compress_ptr,  # int32 [bs]   out: new_compress_token_nums
    total_ptr,  # int32 [bs]   out: total_compress_token_nums
    cu_seqlens_ptr,  # int32 [bs+1] out: cumsum(seq_lens_k) into [1:bs+1]
    cu_new_token_ptr,  # int32 [bs+1] out: cumsum(new_token_nums) into [1:bs+1]
    cu_new_compress_ptr,  # int32 [bs+1] out: cumsum(new_compress) into [1:bs+1]
    cu_total_ptr,  # int32 [bs+1] out: cumsum(total_compress) into [1:bs+1]
    dtn,  # int   draft_token_num
    ksize,  # int   compressor kernel_size
    kstride,  # int   compressor kernel_stride
    bs,  # int   (padded) batch size
    BLOCK: tl.constexpr,
):
    """One launch builds a whole compression level's verify-replay metadata.

    Replaces ~27 eager ops per level (the integer ``(x - ksize)//kstride + 1``
    chains, four ``torch.cumsum`` calls, four device copies) that each cost a
    kernel launch + Python dispatch on tiny ``bs``-element vectors in the
    host-bound verify-prepare window. The arithmetic is the device mirror of
    ``_compute_single_compression_metadata``; ``cu_*[0]`` stays the pre-zeroed 0.

    Bit-exact note: ``((x - ksize)//kstride + 1).clamp(min=0)`` is positive only
    when ``x >= ksize`` (there the numerator is non-negative, so trunc == floor),
    and is exactly 0 otherwise, so ``where(t >= 0, t // kstride + 1, 0)`` matches
    torch's floor-div+clamp without the negative-numerator floor subtlety.
    ``new_token_nums`` is stored unclamped, matching the eager builder.
    """
    offs = tl.arange(0, BLOCK)
    mask = offs < bs
    prefix = tl.load(prefix_ptr + offs, mask=mask, other=0)
    full = prefix + dtn

    t = full - ksize
    seq_lens_k = tl.where(t >= 0, t // kstride + 1, 0)
    t = prefix - ksize
    history = tl.where(t >= 0, t // kstride + 1, 0)
    new_token = full - history * kstride
    t = new_token - ksize
    new_compress = tl.where(t >= 0, t // kstride + 1, 0)
    total = history + new_compress

    tl.store(history_ptr + offs, history, mask=mask)
    tl.store(new_token_ptr + offs, new_token, mask=mask)
    tl.store(new_compress_ptr + offs, new_compress, mask=mask)
    tl.store(total_ptr + offs, total, mask=mask)

    # Inclusive prefix sums over the valid lanes only; padding lanes are zeroed so
    # they neither contribute to a valid prefix nor get stored.
    seq_lens_k = tl.where(mask, seq_lens_k, 0)
    new_token = tl.where(mask, new_token, 0)
    new_compress = tl.where(mask, new_compress, 0)
    total = tl.where(mask, total, 0)
    tl.store(cu_seqlens_ptr + 1 + offs, tl.cumsum(seq_lens_k, axis=0), mask=mask)
    tl.store(cu_new_token_ptr + 1 + offs, tl.cumsum(new_token, axis=0), mask=mask)
    tl.store(cu_new_compress_ptr + 1 + offs, tl.cumsum(new_compress, axis=0), mask=mask)
    tl.store(cu_total_ptr + 1 + offs, tl.cumsum(total, axis=0), mask=mask)


def fill_compress_level_metadata(
    level: "CompressionLevelMetadata",
    prefix_seq_lens_i32: torch.Tensor,
    draft_token_num: int,
    kernel_size: int,
    kernel_stride: int,
    bs: int,
):
    """Fill one compression level's count/cumsum buffers in a single launch.

    ``prefix_seq_lens_i32`` is the int32 prefix (history) length per request.
    The eight target buffers are the CUDA-graph-captured slices on ``level``;
    they are written in place so graph replay sees the refreshed metadata.
    """
    _compress_level_metadata_kernel[(1,)](
        prefix_seq_lens_i32,
        level.history_compress_token_nums,
        level.new_token_nums,
        level.new_compress_token_nums,
        level.total_compress_token_nums,
        level.cu_seqlens,
        level.cu_new_token_nums,
        level.cu_new_compress_token_nums,
        level.cu_total_compress_token_nums,
        int(draft_token_num),
        int(kernel_size),
        int(kernel_stride),
        int(bs),
        BLOCK=triton.next_power_of_2(int(bs)),
    )


@triton.jit
def _verify_plan_counts_kernel(
    seq_lens_ptr,  # int32 [bs]            prefix seq lens
    offsets_ptr,  # int32 [token_num]     verify_token_offsets (draft flat index)
    mask_ptr,  # bool  [token_num*dtn]  sparse_tree_draft_mask (flat)
    out_ptr,  # int32 [token_num*hg]   num_kv_per_row
    dtn,
    block_size,
    sparse_topk,
    head_group_num,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One launch builds the FlashInfer split-KV per-row KV counts.

    Replaces ~18 eager ops (a mask row-sum reduce, two repeat_interleaves, the
    ``sparse_block_quantized_count`` chain, the subtract/add, the head-group
    broadcast) with one program per draft token. The per-row count is
    ``sc(token_pos) - off + popcount`` (see the invariant in
    ``_refresh_target_verify_graph_metadata``); ``sparse_block_quantized_count``
    is inlined. All ints fit int32, so this is bit-exact with the int64 eager
    path (integer ops are exact and values stay well under 2**31).
    """
    t = tl.program_id(0)
    # popcount: visible drafts = sum of this token's dtn tree-mask bits
    d = tl.arange(0, BLOCK_D)
    bits = tl.load(mask_ptr + t * dtn + d, mask=d < dtn, other=0).to(tl.int32)
    num_visible = tl.sum(bits, axis=0)

    seq_len = tl.load(seq_lens_ptr + t // dtn)
    off = tl.load(offsets_ptr + t)
    token_pos = seq_len + off

    # inline sparse_block_quantized_count(token_pos)
    budget = block_size * sparse_topk
    mod = token_pos % block_size
    capped = tl.where(mod == 0, budget, (sparse_topk - 1) * block_size + mod)
    sc = tl.where(token_pos <= budget, token_pos, capped)

    num_kv = sc - off + num_visible

    h = tl.arange(0, BLOCK_H)
    tl.store(out_ptr + t * head_group_num + h, num_kv, mask=h < head_group_num)


def build_verify_plan_counts(
    seq_lens_i32: torch.Tensor,
    verify_token_offsets: torch.Tensor,
    sparse_tree_draft_mask: torch.Tensor,
    draft_token_num: int,
    block_size: int,
    sparse_topk: int,
    head_group_num: int,
    token_num: int,
):
    """Device ``num_kv_per_row`` for the verify-replay FlashInfer plan, one launch.

    Returns an int32 tensor of length ``token_num * head_group_num`` ready for the
    synchronous D2H copy into the pinned plan buffer.
    """
    out = torch.empty(
        token_num * head_group_num, dtype=torch.int32, device=seq_lens_i32.device
    )
    _verify_plan_counts_kernel[(token_num,)](
        seq_lens_i32,
        verify_token_offsets,
        sparse_tree_draft_mask,
        out,
        int(draft_token_num),
        int(block_size),
        int(sparse_topk),
        int(head_group_num),
        BLOCK_D=triton.next_power_of_2(int(draft_token_num)),
        BLOCK_H=triton.next_power_of_2(int(head_group_num)),
    )
    return out


def batched_gather(a, cu_seqlen_q, select):
    #
    select_bs = len(select)
    select = torch.tensor(select, device="cpu")
    starts = cu_seqlen_q[select]
    ends = cu_seqlen_q[select + 1]
    lengths = ends - starts

    max_len = lengths.max()
    local_offsets = torch.arange(max_len, device=a.device)[None, :]
    mask = local_offsets < lengths[:, None]

    local_offsets = local_offsets.expand(select_bs, -1)[mask]

    starts_expanded = starts.repeat_interleave(lengths)
    index = starts_expanded + local_offsets

    return a[index]


def effective_seq_lens_cpu(
    forward_batch: "ForwardBatch", base_metadata: "MiniCPMBackendMetadata"
) -> torch.Tensor:
    """Seq lens for sparse decisions; target verify overrides them with
    prefix + draft lengths via ``seq_lens_cpu_for_sparse``."""
    if base_metadata.seq_lens_cpu_for_sparse is not None:
        return base_metadata.seq_lens_cpu_for_sparse
    return forward_batch.seq_lens_cpu


def compress_k_core_new(
    full_compressed_k,  # output
    layer,
    batch,
    k_stride,
    key_cache,
    token_table,
    compressed_k_table,
    new_k_token_nums,
    cu_new_k_token_nums,
    history_compress_k_token_nums,
    cu_new_compress_k_token_nums,
    new_compress_k_token_nums,
    total_compress_k_token_nums,
    cu_total_compress_k_token_nums,
    kernel_size,
    kernel_stride,
    max_context_length,
):

    head_num_k = key_cache.shape[1]
    head_dim = key_cache.shape[2]

    # ==============================================================================
    # BUFFER ALLOCATION
    # ==============================================================================

    # Use provided explicit parameters for buffer allocation
    # max_chunks_per_seq is already the maximum possible chunks for any sequence
    # given max_context_length, kernel_size, and kernel_stride
    max_chunks_per_seq = max(0, (max_context_length - kernel_size) // kernel_stride + 1)

    # ==============================================================================
    # Launch kernel for ALL chunks (history + new)
    # ==============================================================================
    # Grid: (batch, max_chunks_per_seq, head_num_k)
    # - chunk_in_seq in [0, history_compress): process HISTORY chunks
    # - chunk_in_seq in [history_compress, total_chunks_in_seq): process NEW chunks
    #
    # max_chunks_per_seq is already the maximum possible chunks for any sequence,
    # so it's sufficient for both history and new chunks.
    #
    # All operations are in a single kernel, CUDA graph compatible.

    # Limit grid size to avoid too many thread blocks
    # If max_chunks_per_seq > max_grid_chunks, kernel will loop to handle remaining chunks
    MAX_GRID_CHUNKS = 1024  # Adjustable limit for grid dimension
    max_grid_chunks = min(max_chunks_per_seq, MAX_GRID_CHUNKS)

    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    # Grid size is now limited, kernel uses loop to handle all chunks
    grid = (batch, max_grid_chunks, head_num_k)

    compress_k_complete_kernel_new[grid](
        key_cache,
        token_table,
        cu_new_k_token_nums,
        history_compress_k_token_nums,
        k_stride,
        compressed_k_table,
        cu_new_compress_k_token_nums,
        cu_total_compress_k_token_nums,
        total_compress_k_token_nums,
        full_compressed_k,
        batch,
        max_chunks_per_seq,
        token_table.shape[1],
        compressed_k_table.shape[1],
        head_num_k,
        head_dim,
        kernel_size,
        kernel_stride,
        BLOCK_SIZE,
        max_grid_chunks,  # Pass the limit to kernel for loop control
    )

    return


def get_compress_k_v2(
    layer,
    forward_batch,
    metadata: "FlashAttentionMetadata",
    full_compressed_k1,
    full_compressed_k2,
    max_context_length,
):
    batch = len(forward_batch.req_pool_indices)

    # k1 stride is 16, window is 32
    # k2 stride is 64, windiw is 128
    k1_stride = 16
    k1_l = 32
    k2_stride = 64
    k2_l = 128

    #################### prepare arguments ##############################
    # TODO in summary, the arguments needed are:
    # key_cache [-1, head_num, head_size]
    # metadata.cu_seqlens_q [batch_size + 1]
    # metadata.cu_seqlens_k [batch_size + 1]
    # token_table [batch_size, token_num]: the pre-allocated locs of normal tokens
    # k1_table [batch_size, total_compress_k1_token_num]: the pre-allocated locs of compress k1 tokens
    # k2_table [batch_size, total_compress_k2_token_num]: the pre-allocated locs of compress k2 tokens
    # these arguments should be directly passed in

    # get key cache ptr, zero over head
    key_cache = get_token_to_kv_pool().get_key_buffer(layer.layer_id)
    key_cache = key_cache.view(-1, layer.tp_k_head_num, layer.head_dim)

    ##################### prepare of computation ######################

    # deal with k1
    compress_k_core_new(
        full_compressed_k1,
        layer,
        batch,
        k1_stride,
        key_cache,
        metadata.page_table,
        metadata.k1.table,
        metadata.k1.new_token_nums,
        metadata.k1.cu_new_token_nums,
        metadata.k1.history_compress_token_nums,
        metadata.k1.cu_new_compress_token_nums,
        metadata.k1.new_compress_token_nums,
        metadata.k1.total_compress_token_nums,
        metadata.k1.cu_total_compress_token_nums,
        k1_l,
        k1_stride,
        max_context_length,
    )

    # deal with k2
    compress_k_core_new(
        full_compressed_k2,
        layer,
        batch,
        k2_stride,
        key_cache,
        metadata.page_table,
        metadata.k2.table,
        metadata.k2.new_token_nums,
        metadata.k2.cu_new_token_nums,
        metadata.k2.history_compress_token_nums,
        metadata.k2.cu_new_compress_token_nums,
        metadata.k2.new_compress_token_nums,
        metadata.k2.total_compress_token_nums,
        metadata.k2.cu_total_compress_token_nums,
        k2_l,
        k2_stride,
        max_context_length,
    )

    return


def compress_k_core_new_padded(
    full_compressed_k,  # output
    layer,
    batch,
    k_stride,
    key_cache,
    token_table,
    compressed_k_table,
    new_k_token_nums,
    cu_new_k_token_nums,
    history_compress_k_token_nums,
    cu_new_compress_k_token_nums,
    new_compress_k_token_nums,
    total_compress_k_token_nums,
    cu_total_compress_k_token_nums,
    kernel_size,
    kernel_stride,
    max_context_length,
):
    """Padded layout version: stores data in batch-major order for reshape compatibility."""
    head_num_k = key_cache.shape[1]
    head_dim = key_cache.shape[2]

    # Compute max_chunks_per_seq for padded layout
    # Must match: batch_size * max_context_length // kernel_stride
    max_chunks_per_seq = max_context_length // kernel_stride

    MAX_GRID_CHUNKS = 1024  # Adjustable limit for grid dimension
    max_grid_chunks = min(max_chunks_per_seq, MAX_GRID_CHUNKS)

    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    grid = (batch, max_grid_chunks, head_num_k)

    compress_k_complete_kernel_new_padded[grid](
        key_cache,
        token_table,
        cu_new_k_token_nums,
        history_compress_k_token_nums,
        k_stride,
        compressed_k_table,
        cu_new_compress_k_token_nums,
        cu_total_compress_k_token_nums,
        total_compress_k_token_nums,
        full_compressed_k,
        batch,
        max_chunks_per_seq,
        token_table.shape[1],
        compressed_k_table.shape[1],
        head_num_k,
        head_dim,
        kernel_size,
        kernel_stride,
        BLOCK_SIZE,
        max_grid_chunks,
    )
    return


def get_compress_k_v2_padded(
    layer,
    forward_batch,
    metadata: "FlashAttentionMetadata",
    full_compressed_k1,
    full_compressed_k2,
    max_context_length,
):
    """Padded layout version for debugging with reshape()."""
    batch = len(forward_batch.req_pool_indices)

    k1_stride = 16
    k1_l = 32
    k2_stride = 64
    k2_l = 128

    key_cache = get_token_to_kv_pool().get_key_buffer(layer.layer_id)
    key_cache = key_cache.view(-1, layer.tp_k_head_num, layer.head_dim)

    # deal with k1
    compress_k_core_new_padded(
        full_compressed_k1,
        layer,
        batch,
        k1_stride,
        key_cache,
        metadata.page_table,
        metadata.k1.table,
        metadata.k1.new_token_nums,
        metadata.k1.cu_new_token_nums,
        metadata.k1.history_compress_token_nums,
        metadata.k1.cu_new_compress_token_nums,
        metadata.k1.new_compress_token_nums,
        metadata.k1.total_compress_token_nums,
        metadata.k1.cu_total_compress_token_nums,
        k1_l,
        k1_stride,
        max_context_length,
    )

    # deal with k2
    compress_k_core_new_padded(
        full_compressed_k2,
        layer,
        batch,
        k2_stride,
        key_cache,
        metadata.page_table,
        metadata.k2.table,
        metadata.k2.new_token_nums,
        metadata.k2.cu_new_token_nums,
        metadata.k2.history_compress_token_nums,
        metadata.k2.cu_new_compress_token_nums,
        metadata.k2.new_compress_token_nums,
        metadata.k2.total_compress_token_nums,
        metadata.k2.cu_total_compress_token_nums,
        k2_l,
        k2_stride,
        max_context_length,
    )

    return


def allocate_and_compress_keys(
    layer,
    forward_batch,
    metadata: "FlashAttentionMetadata",
    k1_token_nums: int,
    k2_token_nums: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = None,
    max_context_length: int = 32768,
    split_stage1: bool = False,
):
    """Allocate compressed key tensors and run compression.

    Args:
        layer: Model layer with head configuration
        forward_batch: Forward batch info
        metadata: FlashAttention metadata
        k1_token_nums: Number of k1 tokens to allocate
        k2_token_nums: Number of k2 tokens to allocate
        dtype: Tensor data type (default: bfloat16)
        device: Tensor device (default: layer device)
        max_context_length: Maximum context length for the model (default: 32768)
        split_stage1: If True, use padded kernel

    Returns:
        Tuple of (full_compressed_k1, full_compressed_k2)
    """
    if device is None:
        device = (
            layer.device if hasattr(layer, "device") else torch.cuda.current_device()
        )

    full_compressed_k1 = torch.full(
        (k1_token_nums, layer.tp_k_head_num, layer.head_dim),
        dtype=dtype,
        device=device,
        fill_value=float("-inf"),
    )
    full_compressed_k2 = torch.full(
        (k2_token_nums, layer.tp_k_head_num, layer.head_dim),
        dtype=dtype,
        device=device,
        fill_value=float("-inf"),
    )

    if split_stage1:
        get_compress_k_v2_padded(
            layer,
            forward_batch,
            metadata,
            full_compressed_k1,
            full_compressed_k2,
            max_context_length=max_context_length,
        )
    else:
        get_compress_k_v2(
            layer,
            forward_batch,
            metadata,
            full_compressed_k1,
            full_compressed_k2,
            max_context_length=max_context_length,
        )

    return full_compressed_k1, full_compressed_k2


def compressed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    max_seqlen_q: int,
    # max_seqlen_k: int,
    max_context_len: int,
    sm_scale: Optional[float] = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
    cache_lens: Optional[torch.Tensor] = None,
    total_q: int = -1,
    cu_seqlens_q_adjusted: Optional[torch.Tensor] = None,
    max_seqlen_q_adjusted: Optional[int] = None,
    split_stage1: bool = False,
) -> torch.Tensor:
    """Compressed attention computation for sparse attention.

    Computes attention scores between query and compressed keys (k and k2),
    then performs max pooling and selects top-k blocks.

    Args:
        q: Query tensor, shape (total_q_len, num_heads, head_dim)
        k: Compressed key tensor k1, shape (total_k_len, num_heads, head_dim)
        k2: Compressed key tensor k2, shape (total_k_len, num_heads, head_dim)
        kernel_size: Size of compression kernel
        kernel_stride: Stride of compression kernel
        block_size: Size of attention blocks
        topk: Number of top blocks to select
        cu_seqlens_q: Cumulative sequence lengths for query, shape (batch_size + 1)
        cu_seqlens_k: Cumulative sequence lengths for k, shape (batch_size + 1)
        cu_seqlens_k2: Cumulative sequence lengths for k2, shape (batch_size + 1)
        max_seqlen_q: Maximum sequence length in query
        max_seqlen_k: Maximum sequence length in key
        sm_scale: Softmax scaling factor (unused, kept for compatibility)
        init_blocks: Number of initial blocks to always attend to
        local_blocks: Number of local blocks to consider
        cache_lens: Cache lengths for each batch (optional)
        total_q: Total number of queries (used for pooling)
        cu_seqlens_q_adjusted: Adjusted cumulative sequence lengths for query (for stage1 optimization)
        max_seqlen_q_adjusted: Adjusted maximum sequence length for query (for stage1 optimization)

    Returns:
        Top-k block indices, shape (num_heads, total_q_len, topk)
    """
    with torch.no_grad():
        batch_size = cu_seqlens_q.shape[0] - 1

        current_ratio = q.shape[-2] // k.shape[-2]
        required_ratio = 16
        if current_ratio < required_ratio:
            repeat_times = required_ratio // current_ratio
            q = q.repeat_interleave(repeat_times, dim=-2)

        is_prefilling = max_seqlen_q > 1

        # Stage1 optimization: q_idx computation is no longer needed
        if is_prefilling:
            if cache_lens is None:
                cache_lens = torch.zeros(batch_size, dtype=torch.int32, device=q.device)
        #     q_idx = torch.cat(
        #         [
        #             (
        #                 torch.arange(
        #                     cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=q.device
        #                 )
        #                 + cache_lens[i]
        #             )
        #             // block_size
        #             for i in range(batch_size)
        #         ],
        #         dim=0,
        #     )
        # else:
        #     q_idx = cache_lens // block_size

        # split-stage1 -> bmm+softmax+reduce_sum
        if not is_prefilling and split_stage1:
            batch_size = q.shape[0]
            k1_len = k.shape[0]
            q_head = q.shape[1]
            kv_head = k.shape[1]
            group_size = q_head // kv_head
            head_dim = k.shape[2]
            q_reshape = (
                q.reshape(batch_size, 1, q_head, head_dim)
                .transpose(1, 2)
                .reshape(batch_size, kv_head, group_size, head_dim)
                .transpose(0, 1)
                .reshape(-1, group_size, head_dim)
            )
            k_reshape = (
                k.reshape(batch_size, k1_len // batch_size, kv_head, head_dim)
                .transpose(1, 2)
                .transpose(-2, -1)
                .transpose(0, 1)
                .reshape(-1, head_dim, k1_len // batch_size)
            )

            scale = 1.0 / math.sqrt(head_dim)
            score = torch.bmm(q_reshape, k_reshape).mul_(scale)
            torch.nan_to_num(score, nan=float("-inf"), posinf=float("-inf"), out=score)
            torch.softmax(score, dim=-1, out=score)
            score = score.reshape(
                kv_head, batch_size, group_size, k1_len // batch_size
            ).sum(dim=2)
        else:
            score = infllmv2_attn_stage1(
                q.contiguous(),
                k.contiguous(),
                k2.contiguous(),
                cu_seqlens_q=cu_seqlens_q_adjusted,
                cu_seqlens_k=cu_seqlens_k,
                cu_seqlens_v=cu_seqlens_k2,
                max_seqlen_q=max_seqlen_q_adjusted,
                max_seqlen_k=max_context_len // kernel_stride,
                causal=is_prefilling,
            )

        block_score = max_pooling_1d_varlen(
            score.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens,
            max_seqlen_q,
            # max_seqlen_k,
            max_context_len,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=kernel_stride,
            total_q=total_q,
        )

        topk_idx = block_score.topk(topk, dim=-1).indices.sort(-1).values
        # Stage1 optimization: skip q_idx filtering
        # topk_idx[topk_idx > q_idx[None, :, None]] = -1
        topk_idx = topk_idx.to(torch.int32)

    return topk_idx


def compressed_attention_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
    cache_lens=None,
    fused_kernel=None,
    max_cache_len=-1,
) -> torch.Tensor:
    """
    使用 tilelang online topk kernel 计算 compressed attention topk indices
    """
    with torch.no_grad():
        batch_size = cu_seqlens_q.shape[0] - 1

        # Check if it's prefilling stage
        # Use max_seqlen_q > 1 to avoid .item() call for CUDA Graph compatibility
        is_prefilling = cache_lens is None or max_seqlen_q > 1

        # Fixed max_cache_len for CUDA Graph compatibility (avoid .item() calls)
        # max_cache_len = 525312  # 512k

        # Get tensor dimensions
        # q shape: [total_q_len, num_kv_heads, groups, head_dim] or [total_q_len, num_heads, head_dim]
        # k shape: [total_k_len, num_kv_heads, head_dim]
        total_q_len = q.shape[0]
        total_k_len = k.shape[0]
        num_kv_heads = k.shape[1]
        head_dim = k.shape[2]

        # Determine num_heads and groups
        # if q.dim() == 4:
        #     groups = q.shape[2]
        #     num_heads = num_kv_heads * groups
        #     # Reshape q from [total_q_len, num_kv_heads, groups, head_dim] to [total_q_len * groups, num_kv_heads, head_dim]
        #     q_kernel = q.transpose(1, 2).reshape(total_q_len * groups, num_kv_heads, head_dim).contiguous()
        # else:
        num_heads = q.shape[1]
        groups = num_heads // num_kv_heads
        # q shape: [total_q_len, num_heads, head_dim]
        # Need to reshape to [total_q_len * groups, num_kv_heads, head_dim]
        q_kernel = q.view(total_q_len, num_kv_heads, groups, head_dim)
        q_kernel = (
            q_kernel.transpose(1, 2)
            .reshape(total_q_len * groups, num_kv_heads, head_dim)
            .contiguous()
        )

        k_kernel = k.contiguous()

        # Compute pooled_k_len using infllmv2 formula (based on block count):
        # total_len = max_seqlen_q + cache_len
        # out_len = (total_len + block_size - 1) // block_size
        if is_prefilling:
            # Prefill: use the actual max_seqlen_q
            total_len = max_seqlen_q
            pooled_k_len = (max_cache_len + block_size - 1) // block_size
        else:
            # Decode: use fixed max_cache_len for CUDA Graph compatibility
            # Kernel uses actual_pooled_k_len internally for dynamic bounds checking
            pooled_k_len = (max_cache_len + block_size - 1) // block_size
            # assert decode_fused_kernel is not None, "decode_fused_kernel is not initialized"

        assert fused_kernel is not None, "fused_kernel is not initialized"

        # Pooling parameters - aligned with infllmv2_cuda_impl:
        # block_stride = block_size // kernel_stride = 64 // 16 = 4
        # pad_len = kernel_size // kernel_stride - 1 = 32 // 16 - 1 = 1
        # num_offs = kernel_size // kernel_stride + block_size // kernel_stride - 1 = 2 + 4 - 1 = 5
        pooling_block_stride = block_size // kernel_stride  # = 64 // 16 = 4
        pooling_pad_len = kernel_size // kernel_stride - 1  # = 32 // 16 - 1 = 1
        pooling_num_offs = (
            kernel_size // kernel_stride + block_size // kernel_stride - 1
        )  # = 2 + 4 - 1 = 5

        # Compute actual output topk (same as original: min(topk, num_blocks))
        output_topk = min(topk, pooled_k_len)

        # For the kernel, we need power of 2 topk
        topk_power2 = tilelang.math.next_power_of_2(output_topk)
        kernel_topk = min(topk_power2, pooled_k_len)
        # Make sure it's still power of 2
        if kernel_topk != tilelang.math.next_power_of_2(kernel_topk):
            kernel_topk = tilelang.math.next_power_of_2(kernel_topk) // 2
        kernel_topk = max(8, kernel_topk)  # Minimum topk for kernel

        # Determine dtype string
        if q.dtype == torch.float16:
            dtype_str = "float16"
        elif q.dtype == torch.bfloat16:
            dtype_str = "bfloat16"
        else:
            dtype_str = "bfloat16"

        # Allocate output tensors
        topk_indices = torch.full(
            (num_kv_heads, total_q_len, kernel_topk),
            -1,
            dtype=torch.int32,
            device=q.device,
        )
        topk_values = torch.full(
            (num_kv_heads, total_q_len, kernel_topk),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )

        if is_prefilling:
            # =================================================================
            # PREFILL: Use bucketed max_seqlen_q and pooled_k_len
            # Compiles once per unique bucket combination
            # Supports chunk prefill with cache_lens tensor
            # =================================================================
            # bucketed_max_seqlen_q = _bucket_size(max_seqlen_q)
            bucketed_pooled_k_len = _bucket_size(pooled_k_len)
            # Also bucket actual_max_seqlen_q/k to reduce kernel recompilation
            # bucketed_actual_max_seqlen_q = _bucket_size(max_seqlen_q)
            # bucketed_actual_max_seqlen_k = _bucket_size(max_seqlen_k)

            # Prepare cache_lens tensor for chunk prefill support
            # For standard prefill: cache_lens is None -> use zeros
            # For chunk prefill: cache_lens has values -> use as-is
            if cache_lens is None:
                cache_lens_tensor = torch.zeros(
                    batch_size, dtype=torch.int32, device=q.device
                )
            else:
                cache_lens_tensor = cache_lens.to(torch.int32)

            # Run prefill kernel with cache_lens for chunk prefill support
            fused_kernel(
                q_kernel,
                k_kernel,
                cu_seqlens_q,
                cu_seqlens_k,
                cache_lens_tensor,
                topk_indices,
                topk_values,
            )
        else:
            # =================================================================
            # DECODE: max_seqlen_q=1 (fixed), cache_lens passed as tensor
            # Compiles ONCE and reuses for all decode steps!
            # =================================================================
            bucketed_pooled_k_len = _bucket_size(pooled_k_len)

            # Prepare cache_lens as tensor (runtime value, not compile-time constant!)
            cache_lens_tensor = cache_lens.to(torch.int32)

            # kernel = fused_attn_pooling_online_topk_decode(
            #     batch_size=batch_size,
            #     groups=groups,
            #     heads=num_heads,
            #     dim=head_dim,
            #     topk=kernel_topk,
            #     pooled_k_len=bucketed_pooled_k_len,
            #     m_block_dim=16,
            #     block_stride=pooling_block_stride,
            #     pad_len=pooling_pad_len,
            #     num_offs=pooling_num_offs,
            #     block_size=block_size,
            #     init_blocks=init_blocks,
            #     local_blocks=local_blocks,
            #     dtype_str=dtype_str
            # )

            # Run decode kernel with cache_lens as tensor
            fused_kernel(
                q_kernel,
                k_kernel,
                cu_seqlens_q,
                cu_seqlens_k,
                cache_lens_tensor,
                topk_indices,
                topk_values,
            )

        # Note: q_idx masking is handled inside the kernel via causal_mask
        # which sets scores to -1e9 for K blocks beyond the causal boundary.
        # These blocks won't be selected in topk due to their low scores.

        # Sort with -1 values at the end (match original behavior)
        # Replace -1 with large value, sort, then replace back
        large_val = pooled_k_len + 1000  # Any value larger than max valid index
        topk_for_sort = topk_indices.clone()
        topk_for_sort[topk_for_sort == -1] = large_val
        topk_idx = topk_for_sort.sort(-1).values
        topk_idx[topk_idx == large_val] = -1

        # Truncate to output_topk (same as original: min(topk, num_blocks))
        topk_idx = topk_idx[:, :, :output_topk]

        return topk_idx


@dataclass
class CompressionLevelMetadata:
    """Metadata for a single compression level (k1 or k2).

    This dataclass groups all metadata fields for one compression level,
    reducing duplication and making the code more maintainable.
    """

    # Cumulative sequence lengths for compressed cache
    cu_seqlens: Optional[torch.Tensor] = None
    max_seq_len: int = 0

    # Token mapping table (request pool indices -> compressed cache tokens)
    table: Optional[torch.Tensor] = None

    # Compressed cache metadata
    history_compress_token_nums: Optional[torch.Tensor] = None
    new_token_nums: Optional[torch.Tensor] = None
    cu_new_token_nums: Optional[torch.Tensor] = None
    new_compress_token_nums: Optional[torch.Tensor] = None
    cu_new_compress_token_nums: Optional[torch.Tensor] = None
    total_compress_token_nums: Optional[torch.Tensor] = None
    cu_total_compress_token_nums: Optional[torch.Tensor] = None


@dataclass
class SparseConfig:
    """Configuration for sparse attention in MiniCPM models.

    This dataclass stores all sparse attention configuration parameters,
    including kernel sizes, block sizes, top-K selection, and model
    architecture parameters.

    The configuration is derived from the model's hf_config and
    model_config, and is used by SparseMetadataBuilder to compute
    metadata for sparse attention batches.
    """

    # Basic sparse config parameters (from hf_config)
    sparse_len: int  # Threshold for activating sparse attention
    sparse_topk: int  # Number of top-K global blocks to select
    kernel_size: int  # Kernel size for compressed key computation
    kernel_stride: int  # Kernel stride for compressed key computation
    block_size: int  # Block size for top-K selection
    window_size: int  # Window size for sliding window attention
    dense_len: int  # Length of dense attention at beginning

    # Model architecture parameters
    head_dim: int  # Dimension of each attention head
    num_kv_heads: int  # Number of key-value heads
    head_group_num: int  # Number of head groups for sparse attention

    # Compressor kernel sizes
    k1_kernel_size: int
    k1_kernel_stride: int
    k2_kernel_size: int
    k2_kernel_stride: int

    @property
    def local_blocks(self) -> int:
        """Number of local blocks (window_size // block_size)."""
        return self.window_size // self.block_size

    @property
    def sparse_topk_total(self) -> int:
        """Total top-K including local blocks (topk + local_blocks)."""
        return self.sparse_topk + self.local_blocks

    @property
    def num_sparse_topk_tokens(self) -> int:
        """Total number of tokens in sparse top-K selection."""
        return self.block_size * self.sparse_topk_total

    @classmethod
    def from_model_config(cls, hf_config, model_config) -> "SparseConfig":
        """Create SparseConfig from model configuration.

        Args:
            hf_config: The HuggingFace model config (MiniCPMHybridConfig)
            model_config: The SGLang model config

        Returns:
            SparseConfig instance with all parameters set
        """
        sparse_topk = hf_config.sparse_topk
        kernel_size = hf_config.sparse_kernel_size
        kernel_stride = hf_config.sparse_kernel_stride
        block_size = hf_config.sparse_block_size
        window_size = hf_config.sparse_window_size
        dense_len = hf_config.sparse_dense_len

        head_dim = model_config.head_dim
        num_kv_heads = model_config.num_key_value_heads
        head_group_num = model_config.num_key_value_heads

        k1_kernel_size = kernel_size
        k1_kernel_stride = kernel_stride
        k2_kernel_size = kernel_size * 4
        k2_kernel_stride = kernel_stride * 4

        return cls(
            sparse_len=dense_len,
            sparse_topk=sparse_topk,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            window_size=window_size,
            dense_len=dense_len,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            head_group_num=head_group_num,
            k1_kernel_size=k1_kernel_size,
            k1_kernel_stride=k1_kernel_stride,
            k2_kernel_size=k2_kernel_size,
            k2_kernel_stride=k2_kernel_stride,
        )


class SparseMetadataBuilder:
    """Builder for constructing sparse attention metadata.

    This helper class builds the metadata required for sparse attention processing,
    including sequence lengths, token mappings, and page tables.

    The metadata is computed from the forward batch and the sparse batch
    indices selected by the backend.
    """

    def __init__(
        self, config: SparseConfig, num_kv_heads: int, max_context_len: int = 32768
    ):
        """Initialize SparseMetadataBuilder with sparse configuration.

        Args:
            config: SparseConfig containing kernel parameters, head_dim, etc.
            num_kv_heads: Number of key-value heads.
            max_context_len: Maximum context length for the model. Default is 32768.
        """
        self.config = config
        self.num_kv_heads = num_kv_heads

    def build_sequence_lengths(
        self,
        cu_seqlens_q: torch.Tensor,
        sparse_bs_list: list[int],
    ) -> list[int]:
        """Per-sparse-request query lengths for the sparse token mapping.

        Only the query lengths are returned. The key lengths the block-table
        kernel needs are full-batch (indexed by the original request id carried
        in token_to_bs, not the sparse subset) and equal the full cache seqlens,
        so the caller aliases cache_seqlens_int32 for them instead.

        Args:
            cu_seqlens_q: Cumulative query sequence lengths for all batches
            sparse_bs_list: List of batch indices that are sparse

        Returns:
            seqlen_q_sparse_bs: Python list of per-sparse-request query lengths
        """
        return cu_seqlens_q.diff()[sparse_bs_list].tolist()

    def build_token_mappings(
        self,
        extend_prefix_lens_sparse: torch.Tensor,
        seqlen_q_sparse_bs: list[int],
        sparse_bs_list: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token maps for the flattened sparse query stream (all on CPU).

        Each sparse request contributes ``seqlen_q_sparse_bs[i]`` consecutive
        tokens. ``token_to_bs`` tags each token with its original request id
        (``sparse_bs_list[i]``); ``token_pos_in_bs`` is the token's 1-based
        absolute position in that request (``local_index + 1 + prefix_len``).

        Args:
            extend_prefix_lens_sparse: prefix length per sparse request
                (size len(sparse_bs_list)).
            seqlen_q_sparse_bs: query length per sparse request.
            sparse_bs_list: original request ids; defaults to 0..n-1.

        Returns:
            (token_to_bs, token_pos_in_bs), both int32 CPU tensors.
        """
        if sparse_bs_list is None:
            sparse_bs_list = list(range(len(seqlen_q_sparse_bs)))
        elif len(sparse_bs_list) != len(seqlen_q_sparse_bs):
            raise ValueError(
                "sparse_bs_list must have the same length as seqlen_q_sparse_bs; "
                f"got {len(sparse_bs_list)} and {len(seqlen_q_sparse_bs)}."
            )

        seqlens = torch.tensor(seqlen_q_sparse_bs, dtype=torch.int32)
        token_to_bs = torch.tensor(sparse_bs_list, dtype=torch.int32).repeat_interleave(
            seqlens
        )

        # token_pos = global_index - segment_start + 1 + prefix, folded into one
        # per-segment offset broadcast over the segment's tokens.
        prefixes = extend_prefix_lens_sparse.to(dtype=torch.int32, device="cpu")
        segment_starts = seqlens.cumsum(0, dtype=torch.int32) - seqlens
        offset = 1 + prefixes - segment_starts
        token_pos_in_bs = torch.arange(
            int(seqlens.sum()), dtype=torch.int32
        ) + offset.repeat_interleave(seqlens)

        return token_to_bs, token_pos_in_bs

    def _compute_single_compression_metadata(
        self,
        forward_batch: ForwardBatch,
        base_metadata: FlashAttentionMetadata,
        req_to_sparse_token: torch.Tensor,
        kernel_size: int,
        kernel_stride: int,
        cu_seqlens_q: torch.Tensor,
    ) -> dict:
        """Compute compression metadata for a single compression level (k1 or k2).

        Args:
            forward_batch: The forward batch to analyze
            base_metadata: Base metadata with cu_seqlens_q, cu_seqlens_k
            req_to_sparse_token: Mapping from request pool to sparse tokens
            kernel_size: Kernel size for compression
            kernel_stride: Kernel stride for compression
            cu_seqlens_q: Cumulative query sequence lengths

        Returns:
            Dictionary with compression metadata fields for this level
        """
        bs = forward_batch.batch_size
        seq_lens_cpu_for_sparse = effective_seq_lens_cpu(forward_batch, base_metadata)
        seqlen_cpu = torch.zeros(
            (bs,), dtype=base_metadata.cu_seqlens_q.dtype, device="cpu"
        )

        for i in range(bs):
            # if forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            #     seqlen_cpu[i] = max(
            #         0,
            #         (forward_batch.extend_seq_lens_cpu[i] - kernel_size)
            #         // kernel_stride
            #         + 1,
            #     )
            # else:
            seq_len_i = int(seq_lens_cpu_for_sparse[i])
            seqlen_cpu[i] = max(
                0,
                (seq_len_i - kernel_size) // kernel_stride + 1,
            )

        max_seq_len = seqlen_cpu.max().item()
        cu_seqlens = F.pad(
            torch.cumsum(
                seqlen_cpu.to(device=cu_seqlens_q.device), dim=0, dtype=torch.int32
            ),
            (1, 0),
        )
        token_table = req_to_sparse_token[forward_batch.req_pool_indices]

        token_nums = base_metadata.cu_seqlens_k[1:] - base_metadata.cu_seqlens_k[:-1]
        input_lens = base_metadata.cu_seqlens_q[1:] - base_metadata.cu_seqlens_q[:-1]
        history_lens = token_nums - input_lens

        history_compress_token_nums = torch.maximum(
            (history_lens - kernel_size) // kernel_stride + 1,
            torch.zeros(1, device=history_lens.device, dtype=torch.int32),
        )

        new_token_nums = token_nums - history_compress_token_nums * kernel_stride

        cu_new_token_nums = F.pad(
            torch.cumsum(new_token_nums, dim=0, dtype=torch.int32), (1, 0)
        )

        new_compress_token_nums = torch.maximum(
            (new_token_nums - kernel_size) // kernel_stride + 1,
            torch.zeros(1, device=new_token_nums.device, dtype=torch.int32),
        )

        cu_new_compress_token_nums = F.pad(
            torch.cumsum(new_compress_token_nums, dim=0, dtype=torch.int32), (1, 0)
        )

        total_compress_token_nums = (
            history_compress_token_nums + new_compress_token_nums
        )

        cu_total_compress_token_nums = F.pad(
            torch.cumsum(total_compress_token_nums, dim=0, dtype=torch.int32), (1, 0)
        )

        return {
            "cu_seqlens": cu_seqlens,
            "max_seq_len": max_seq_len,
            "token_table": token_table,
            "history_compress_token_nums": history_compress_token_nums,
            "new_token_nums": new_token_nums,
            "cu_new_token_nums": cu_new_token_nums,
            "new_compress_token_nums": new_compress_token_nums,
            "cu_new_compress_token_nums": cu_new_compress_token_nums,
            "total_compress_token_nums": total_compress_token_nums,
            "cu_total_compress_token_nums": cu_total_compress_token_nums,
        }

    def build_k1_k2_compression_metadata(
        self,
        forward_batch: ForwardBatch,
        base_metadata: FlashAttentionMetadata,
        req_to_sparse_k1_token: torch.Tensor,
        req_to_sparse_k2_token: torch.Tensor,
        k1_kernel_size: int,
        k1_kernel_stride: int,
        k2_kernel_size: int,
        k2_kernel_stride: int,
        cu_seqlens_q: torch.Tensor,
    ) -> dict[str, CompressionLevelMetadata]:
        """Build k1/k2 compression metadata.

        This method computes all k1/k2 compressed cache metadata needed for
        sparse attention by calling _compute_single_compression_metadata for each level.

        Args:
            forward_batch: The forward batch to analyze
            base_metadata: Base metadata with cu_seqlens_q, cu_seqlens_k
            req_to_sparse_k1_token: Mapping from request pool to sparse k1 tokens
            req_to_sparse_k2_token: Mapping from request pool to sparse k2 tokens
            k1_kernel_size: Kernel size for k1 compression
            k1_kernel_stride: Kernel stride for k1 compression
            k2_kernel_size: Kernel size for k2 compression
            k2_kernel_stride: Kernel stride for k2 compression
            cu_seqlens_q: Cumulative query sequence lengths

        Returns:
            Dictionary with 'k1' and 'k2' keys containing CompressionLevelMetadata
        """
        # Compute k1 metadata
        k1_dict = self._compute_single_compression_metadata(
            forward_batch,
            base_metadata,
            req_to_sparse_k1_token,
            k1_kernel_size,
            k1_kernel_stride,
            cu_seqlens_q,
        )

        # Compute k2 metadata
        k2_dict = self._compute_single_compression_metadata(
            forward_batch,
            base_metadata,
            req_to_sparse_k2_token,
            k2_kernel_size,
            k2_kernel_stride,
            cu_seqlens_q,
        )

        return {
            "k1": CompressionLevelMetadata(
                cu_seqlens=k1_dict["cu_seqlens"],
                max_seq_len=k1_dict["max_seq_len"],
                table=k1_dict["token_table"],
                history_compress_token_nums=k1_dict["history_compress_token_nums"],
                new_token_nums=k1_dict["new_token_nums"],
                cu_new_token_nums=k1_dict["cu_new_token_nums"],
                new_compress_token_nums=k1_dict["new_compress_token_nums"],
                cu_new_compress_token_nums=k1_dict["cu_new_compress_token_nums"],
                total_compress_token_nums=k1_dict["total_compress_token_nums"],
                cu_total_compress_token_nums=k1_dict["cu_total_compress_token_nums"],
            ),
            "k2": CompressionLevelMetadata(
                cu_seqlens=k2_dict["cu_seqlens"],
                max_seq_len=k2_dict["max_seq_len"],
                table=k2_dict["token_table"],
                history_compress_token_nums=k2_dict["history_compress_token_nums"],
                new_token_nums=k2_dict["new_token_nums"],
                cu_new_token_nums=k2_dict["cu_new_token_nums"],
                new_compress_token_nums=k2_dict["new_compress_token_nums"],
                cu_new_compress_token_nums=k2_dict["cu_new_compress_token_nums"],
                total_compress_token_nums=k2_dict["total_compress_token_nums"],
                cu_total_compress_token_nums=k2_dict["cu_total_compress_token_nums"],
            ),
        }

    def build_sparse_prefill_metadata(
        self,
        forward_batch: ForwardBatch,
        base_metadata: FlashAttentionMetadata,
        sparse_bs_list: list[int],
        head_group_num: int,
        dense_len: int,
        sparse_topk: int,
        block_size: int,
        cu_seqlens_q: torch.Tensor,
        sparse_page_table_dtype: torch.dtype,
        sparse_page_table_device: torch.device,
        tree_mode: bool = False,
    ) -> dict:
        """Build sparse prefill metadata.

        This method handles the complex page table and batch mapping logic
        for sparse prefill mode.

        Args:
            forward_batch: The forward batch to analyze
            base_metadata: Base metadata with cu_seqlens_q
            sparse_bs_list: List of sparse batch indices
            head_group_num: Number of head groups
            dense_len: Dense length threshold for sparse activation
            sparse_topk: Top-K value for sparse attention
            block_size: Block size for sparse attention
            cu_seqlens_q: Cumulative query sequence lengths
            sparse_page_table_dtype: Data type for sparse page table
            sparse_page_table_device: Device for sparse page table
            tree_mode: When True (topk>1 target verify), dense requests also use
                the token-major row layout so the whole batch is uniform and the
                tree compaction kernel can run over every row. Their compacted
                rows are filled separately (full prefix + tree-visible drafts);
                the page table is only widened to hold each dense prefix.

        Returns:
            Dictionary with prefill metadata
        """
        bs = forward_batch.batch_size
        seq_lens_cpu_for_sparse = effective_seq_lens_cpu(forward_batch, base_metadata)

        # One pass over the batch decides each request's page-table rows and the
        # query-position step every row contributes. Sparse requests (and dense
        # ones under tree verify) are token-major: extend*head_group rows, each a
        # single query position (step 1). Plain dense requests are head-major:
        # head_group rows, each spanning the whole extend (step = extend). The
        # page table is widened to the cache a row can reach -- the pruned topk
        # for sparse, the full prefix otherwise.
        max_sparse_cache_len = -1
        sparse_max_seq_len_q = 1
        old_bs_to_new_bs_range = [0] * (bs + 1)
        row_steps = []
        for i in range(bs):
            seq_len_i = int(seq_lens_cpu_for_sparse[i])
            extend_i = int(forward_batch.extend_seq_lens_cpu[i])
            if seq_len_i >= dense_len:
                rows, step = extend_i * head_group_num, 1
                max_sparse_cache_len = max(
                    max_sparse_cache_len, sparse_topk * block_size
                )
            elif tree_mode:
                rows, step = extend_i * head_group_num, 1
                max_sparse_cache_len = max(max_sparse_cache_len, seq_len_i)
            else:
                rows, step = head_group_num, extend_i
                max_sparse_cache_len = max(max_sparse_cache_len, seq_len_i)
                sparse_max_seq_len_q = max(sparse_max_seq_len_q, extend_i)
            row_steps.append(torch.full((rows,), step, dtype=cu_seqlens_q.dtype))
            old_bs_to_new_bs_range[i + 1] = old_bs_to_new_bs_range[i] + rows

        sparse_page_table_bs = old_bs_to_new_bs_range[bs]
        sparse_page_table = torch.zeros(
            (sparse_page_table_bs, max_sparse_cache_len),
            dtype=sparse_page_table_dtype,
            device=sparse_page_table_device,
        )
        sparse_cu_seqlens_q_cpu = torch.zeros(
            sparse_page_table_bs + 1, dtype=cu_seqlens_q.dtype, device="cpu"
        )
        if row_steps:
            sparse_cu_seqlens_q_cpu[1:] = torch.cat(row_steps).cumsum(
                0, dtype=cu_seqlens_q.dtype
            )
        sparse_cu_seqlens_q = sparse_cu_seqlens_q_cpu.to(device=cu_seqlens_q.device)

        sparse_idx = []
        for sparse_bs in sparse_bs_list:
            sparse_idx.extend(
                range(
                    old_bs_to_new_bs_range[sparse_bs],
                    old_bs_to_new_bs_range[sparse_bs + 1],
                )
            )

        return {
            "sparse_page_table": sparse_page_table,
            "sparse_cu_seqlens_q_cpu": sparse_cu_seqlens_q_cpu,
            "sparse_cu_seqlens_q": sparse_cu_seqlens_q,
            "old_bs_to_new_bs_range": old_bs_to_new_bs_range,
            "sparse_max_seq_len_q": sparse_max_seq_len_q,
            "sparse_idx": sparse_idx,
        }

    def build_sparse_decode_metadata(
        self,
        forward_batch: ForwardBatch,
        base_metadata: FlashAttentionMetadata,
        head_group_num: int,
        dense_len: int,
        sparse_topk: int,
        block_size: int,
    ) -> dict:
        """Build sparse decode metadata.

        This method handles sparse attention metadata for decode mode.

        Args:
            forward_batch: The forward batch to analyze
            base_metadata: Base metadata with cache_seqlens_int32, page_table
            head_group_num: Number of head groups
            dense_len: Dense length threshold
            sparse_topk: Top-K value for sparse attention
            block_size: Block size

        Returns:
            Dictionary with decode metadata
        """
        bs = forward_batch.batch_size
        cache_seqlens = base_metadata.cache_seqlens_int32
        page_table = base_metadata.page_table
        max_sparse_cache_len = 0

        sparse_cache_seqlens_cpu = torch.zeros(
            (bs * head_group_num,), dtype=cache_seqlens.dtype, device="cpu"
        )

        for b in range(bs):
            if forward_batch.seq_lens_cpu[b] >= dense_len:
                if forward_batch.seq_lens_cpu[b] <= sparse_topk * block_size:
                    sparse_cache_len = forward_batch.seq_lens_cpu[b]
                elif cache_seqlens[b] % block_size == 0:
                    sparse_cache_len = sparse_topk * block_size
                else:
                    sparse_cache_len = block_size * (sparse_topk - 1) + (
                        cache_seqlens[b] % block_size
                    )

                if sparse_cache_len > max_sparse_cache_len:
                    max_sparse_cache_len = sparse_cache_len

                sparse_cache_seqlens_cpu[2 * b] = sparse_cache_len
                sparse_cache_seqlens_cpu[2 * b + 1] = sparse_cache_len
            else:
                if cache_seqlens[b] > max_sparse_cache_len:
                    max_sparse_cache_len = cache_seqlens[b]

                sparse_cache_seqlens_cpu[2 * b] = cache_seqlens[b]
                sparse_cache_seqlens_cpu[2 * b + 1] = cache_seqlens[b]

        sparse_cache_seqlens_int32 = sparse_cache_seqlens_cpu.to(
            device=cache_seqlens.device
        )
        sparse_cu_seqlens_k = F.pad(
            torch.cumsum(sparse_cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
        )
        sparse_cu_seqlens_q = torch.arange(
            0,
            bs * head_group_num + 1,
            dtype=torch.int32,
            device=base_metadata.cu_seqlens_q.device,
        )
        token_to_bs = torch.arange(0, bs, dtype=torch.int32, device="cuda")
        sparse_page_table = torch.zeros(
            (2 * bs, sparse_topk * block_size),
            dtype=page_table.dtype,
            device=page_table.device,
        )

        return {
            "sparse_cache_seqlens_int32": sparse_cache_seqlens_int32,
            "sparse_cu_seqlens_k": sparse_cu_seqlens_k,
            "sparse_cu_seqlens_q": sparse_cu_seqlens_q,
            "sparse_page_table": sparse_page_table,
            "token_to_bs": token_to_bs,
        }

    def build_prefill_topk_metadata(
        self,
        forward_batch: ForwardBatch,
        base_metadata: FlashAttentionMetadata,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        tp_q_head_num: int,
        head_dim: int,
        compress_k1_kernel_size: int,
        compress_k1_kernel_stride: int,
        compress_k2_kernel_size: int,
        compress_k2_kernel_stride: int,
        dense_len: int,
    ) -> dict:
        """Build prefill TopK metadata.

        This method prepares all metadata needed for TopK computation in prefill mode,
        including sparse batch identification, sequence lengths, and query preparation.

        Args:
            forward_batch: The forward batch
            base_metadata: Base metadata with cu_seqlens
            key_states: Key states from model layer
            query_states: Query states from model layer
            tp_q_head_num: Number of query heads
            head_dim: Head dimension
            compress_k1_kernel_size: K1 compression kernel size
            compress_k1_kernel_stride: K1 compression kernel stride
            compress_k2_kernel_size: K2 compression kernel size
            compress_k2_kernel_stride: K2 compression kernel stride
            dense_len: Dense length threshold

        Returns:
            Dictionary with prefill TopK metadata
        """
        bs, seqlens_q, seqlens_k = (
            forward_batch.batch_size,
            forward_batch.extend_seq_lens_cpu,
            effective_seq_lens_cpu(forward_batch, base_metadata),
        )
        seqlens_k_list = [int(seq_len) for seq_len in seqlens_k]

        token_num_sparse_k1_total = [
            (
                (seq_len - compress_k1_kernel_size) // compress_k1_kernel_stride + 1
                if seq_len >= compress_k1_kernel_size
                else 0
            )
            for seq_len in seqlens_k_list
        ]
        k1_lens = torch.tensor(token_num_sparse_k1_total, dtype=torch.int64)

        token_num_sparse_k2_total = [
            (
                (seq_len - compress_k2_kernel_size) // compress_k2_kernel_stride + 1
                if seq_len >= compress_k2_kernel_size
                else 0
            )
            for seq_len in seqlens_k_list
        ]
        k2_lens = torch.tensor(token_num_sparse_k2_total, dtype=torch.int64)

        sparse_bs = []
        seqlens_q_sparse_bs = []
        seqlens_k_sparse_bs = []

        for i in range(bs):
            if seqlens_k_list[i] >= dense_len:
                sparse_bs.append(i)
                seqlens_q_sparse_bs.append(seqlens_q[i])
                seqlens_k_sparse_bs.append(seqlens_k_list[i])

        cu_seqlens_q = torch.cumsum(
            torch.tensor(
                [0] + seqlens_q, dtype=torch.int32, device=query_states.device
            ),
            dim=0,
            dtype=torch.int32,
        )

        query_states_reshaped = query_states.reshape(-1, tp_q_head_num, head_dim)

        query_states = batched_gather(query_states_reshaped, cu_seqlens_q, sparse_bs)

        cu_seqlens_q_sparse = torch.cumsum(
            torch.tensor(
                [0] + seqlens_q_sparse_bs, dtype=torch.int32, device=query_states.device
            ),
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens_k_sparse = torch.cumsum(
            torch.tensor(
                [0] + seqlens_k_sparse_bs, dtype=torch.int32, device=query_states.device
            ),
            dim=0,
            dtype=torch.int32,
        )

        return {
            "sparse_bs": sparse_bs,
            "seqlens_q_sparse_bs": seqlens_q_sparse_bs,
            "seqlens_k_sparse_bs": seqlens_k_sparse_bs,
            "k1_lens": k1_lens,
            "k2_lens": k2_lens,
            "cu_seqlens_q": cu_seqlens_q_sparse,
            "cu_seqlens_k": cu_seqlens_k_sparse,
            "max_seqlen_q": max(seqlens_q_sparse_bs),
            "max_seqlen_k": max(seqlens_k_sparse_bs),
            "query_states": query_states,
        }

    def build_decode_topk_metadata(
        self,
        forward_batch: ForwardBatch,
        base_metadata: FlashAttentionMetadata,
        query_states: torch.Tensor,
    ) -> dict:
        """Build decode TopK metadata.

        This method prepares metadata needed for TopK computation in decode mode.

        Args:
            forward_batch: The forward batch
            base_metadata: Base metadata with cu_seqlens
            query_states: Query states from model layer

        Returns:
            Dictionary with decode TopK metadata
        """
        query_states = query_states.squeeze(0)
        cu_seqlens_k = base_metadata.cu_seqlens_k
        max_seqlen_in_batch_k = base_metadata.max_seq_len_k
        cu_seqlens_q = base_metadata.cu_seqlens_q
        max_seqlen_in_batch_q = 1

        return {
            "query_states": query_states,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_in_batch_q,
            "max_seqlen_k": max_seqlen_in_batch_k,
        }


__all__ = [
    "CompressionLevelMetadata",
    "SparseConfig",
    "SparseMetadataBuilder",
    "batched_gather",
    "get_compress_k_v2",
    "get_compress_k_v2_padded",
    "allocate_and_compress_keys",
    "compressed_attention",
]
