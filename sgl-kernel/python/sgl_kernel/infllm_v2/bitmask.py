from typing import Optional, Tuple

import torch


def topk_to_uint64(
    topk_idx: torch.Tensor,
    max_seqlen_k: int,
    block_size: int,
    memory_buffer: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int]:
    """Convert topk block indices directly to a uint64 bitset representation.

    Drop-in replacement for ``infllm_v2.topk_to_uint64``.

    Args:
        topk_idx: int32 tensor of shape ``[batch, num_heads, total_seqlen, k]``
            or ``[num_heads, total_seqlen, k]`` containing block indices.
        max_seqlen_k: Maximum key sequence length.
        block_size: Size of each block.

    Returns:
        Tuple of (uint64 tensor with last dim ``n_uint64_per_row``, ``k_blocks``).
    """
    assert topk_idx.dtype == torch.int32
    k_blocks = (max_seqlen_k + block_size - 1) // block_size

    original_shape = topk_idx.shape
    has_batch = len(original_shape) == 4
    if has_batch:
        batch_size, num_heads, total_seqlen, k = original_shape
        flat_dims = batch_size * num_heads * total_seqlen
        output_shape = (batch_size, num_heads, total_seqlen, -1)
    else:
        num_heads, total_seqlen, k = original_shape
        flat_dims = num_heads * total_seqlen
        output_shape = (num_heads, total_seqlen, -1)

    n_uint64_per_row = (k_blocks + 63) // 64

    result = torch.zeros(
        (flat_dims, n_uint64_per_row), dtype=torch.int64, device=topk_idx.device
    )
    flat_topk = topk_idx.reshape(flat_dims, k).contiguous()
    torch.ops.sgl_kernel.infllm_v2_topk_to_uint64.default(result, flat_topk)
    return result.reshape(output_shape), k_blocks


def uint64_to_bool(uint64_array: torch.Tensor, last_dim_size: int) -> torch.Tensor:
    """Convert a uint64 bitset back to a boolean mask.

    Drop-in replacement for ``infllm_v2.uint64_to_bool``.
    """
    original_shape = uint64_array.shape
    n_uint64_per_row = original_shape[-1]

    flat_dims = 1
    for d in original_shape[:-1]:
        flat_dims *= d
    flat_uint64 = uint64_array.reshape(flat_dims, n_uint64_per_row).contiguous()

    result = torch.zeros(
        (flat_dims, last_dim_size), dtype=torch.bool, device=uint64_array.device
    )
    torch.ops.sgl_kernel.infllm_v2_uint64_to_bool.default(result, flat_uint64)
    return result.reshape(original_shape[:-1] + (last_dim_size,))


def blockmask_to_uint64(blockmask: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Convert a boolean block mask to a uint64 bitset representation.

    Drop-in replacement for ``infllm_v2.blockmask_to_uint64``.

    Returns:
        Tuple of (uint64 tensor with last dim ``n_uint64_per_row``, ``last_dim_size``).
    """
    original_shape = blockmask.shape
    last_dim_size = original_shape[-1]
    n_uint64_per_row = (last_dim_size + 63) // 64

    flat_dims = 1
    for d in original_shape[:-1]:
        flat_dims *= d
    flat_blockmask = (
        blockmask.reshape(flat_dims, last_dim_size).to(torch.bool).contiguous()
    )

    result = torch.zeros(
        (flat_dims, n_uint64_per_row), dtype=torch.int64, device=blockmask.device
    )
    torch.ops.sgl_kernel.infllm_v2_blockmask_to_uint64.default(result, flat_blockmask)
    return result.reshape(original_shape[:-1] + (n_uint64_per_row,)), last_dim_size
