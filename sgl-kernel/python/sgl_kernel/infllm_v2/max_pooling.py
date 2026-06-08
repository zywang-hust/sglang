import torch


def max_pooling_1d(
    input: torch.Tensor,  # num_heads x q_len x k_len
    cache_len: int,
    local_blocks: int,
    init_blocks: int,
    block_size: int = 64,
    stride: int = 16,
) -> torch.Tensor:
    """Fixed-length 1D max pooling.

    Drop-in replacement for ``infllm_v2.max_pooling_1d``.
    """
    assert input.dtype in (torch.float16, torch.bfloat16)
    input = input.contiguous()

    stride = block_size // stride
    kernel_size = stride + 1
    padding = 1

    num_heads, q_len, _k_len = input.shape
    total_len = q_len + cache_len
    out_len = (total_len + block_size - 1) // block_size

    output = torch.zeros(
        num_heads, q_len, out_len, device=input.device, dtype=input.dtype
    )
    torch.ops.sgl_kernel.infllm_v2_max_pooling_1d.default(
        input,
        output,
        cache_len,
        kernel_size,
        stride,
        padding,
        block_size,
        local_blocks,
        init_blocks,
    )
    return output


def max_pooling_1d_varlen(
    input: torch.Tensor,  # num_heads x total_q x max_k
    cu_seqlens_q: torch.Tensor,  # batch_size + 1
    cu_seqlens_k: torch.Tensor,  # batch_size + 1
    cache_lens: torch.Tensor,  # batch_size
    max_seqlen_q: int,
    max_context_len: int,
    local_blocks: int,
    init_blocks: int,
    block_size: int = 64,
    stride: int = 16,
    total_q: int = -1,
) -> torch.Tensor:
    """Variable-length 1D max pooling over packed sequences.

    Drop-in replacement for ``infllm_v2.max_pooling_1d_varlen``.
    """
    assert input.dtype in (torch.float16, torch.bfloat16)
    assert cu_seqlens_q.dtype == torch.int32
    assert cu_seqlens_k.dtype == torch.int32
    assert cache_lens.dtype == torch.int32
    assert input.dim() == 3, f"Expected 3D input, got {input.dim()}D"

    input = input.contiguous()
    cu_seqlens_q = cu_seqlens_q.contiguous()
    cu_seqlens_k = cu_seqlens_k.contiguous()
    cache_lens = cache_lens.contiguous()

    max_seqlen_k = max_context_len // stride
    out_len = (max_context_len + block_size - 1) // block_size

    stride = block_size // stride
    kernel_size = stride + 1
    padding = 1

    num_heads = input.shape[0]
    total_q = input.shape[1]

    output = torch.zeros(
        num_heads, total_q, out_len, device=input.device, dtype=input.dtype
    )
    torch.ops.sgl_kernel.infllm_v2_max_pooling_1d_varlen.default(
        input,
        output,
        cu_seqlens_q,
        cu_seqlens_k,
        cache_lens,
        max_seqlen_q,
        max_seqlen_k,
        kernel_size,
        stride,
        padding,
        block_size,
        local_blocks,
        init_blocks,
        total_q,
    )
    return output
