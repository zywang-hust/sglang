"""Live-context tightening in the MiniCPM stage-1 selection path."""

from __future__ import annotations

import pytest
import torch

from sglang.srt.layers.attention.minicpm.sparse_utils import compressed_attention
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_HEADS = 32
_HEAD_KV = 2
_GROUPS = _HEADS // _HEAD_KV
_DIM = 128
_KERNEL_SIZE = 32
_KERNEL_STRIDE = 16
_BLOCK_SIZE = 64
_INIT_BLOCKS = 1
_LOCAL_BLOCKS = 2048 // _BLOCK_SIZE
_SPARSE_TOPK = 64 + _LOCAL_BLOCKS
_MAX_CONTEXT_LEN = 131072


def _compressed_len(tokens: int, kernel_size: int, stride: int) -> int:
    return max(0, (tokens - kernel_size) // stride + 1)


def _stage1_inputs(query_len: int, context_len: int) -> tuple:
    """The seeded q/k1/k2 construction shared by the tight-vs-loose tests."""
    torch.manual_seed(context_len + query_len)
    k1_len = _compressed_len(context_len, _KERNEL_SIZE, _KERNEL_STRIDE)
    k2_len = _compressed_len(context_len, _KERNEL_SIZE * 4, _KERNEL_STRIDE * 4)
    q = torch.randn(query_len, _HEADS, _DIM, dtype=torch.bfloat16, device="cuda")
    k1 = torch.randn(k1_len, _HEAD_KV, _DIM, dtype=torch.bfloat16, device="cuda")
    k2 = torch.randn(k2_len, _HEAD_KV, _DIM, dtype=torch.bfloat16, device="cuda")
    return q, k1, k2, k1_len, k2_len, query_len, context_len


def _run_selection(inputs: tuple, max_seqlen_k: int) -> torch.Tensor:
    """The cu_seqlens/cache_lens construction and selection call shared by the tests."""
    q, k1, k2, k1_len, k2_len, query_len, context_len = inputs
    cu_q = torch.tensor([0, query_len], dtype=torch.int32, device="cuda")
    cu_k1 = torch.tensor([0, k1_len], dtype=torch.int32, device="cuda")
    cu_k2 = torch.tensor([0, k2_len], dtype=torch.int32, device="cuda")
    cache_lens = torch.tensor(
        [context_len - query_len], dtype=torch.int32, device="cuda"
    )
    return compressed_attention(
        q=q,
        k=k1,
        k2=k2,
        kernel_stride=_KERNEL_STRIDE,
        block_size=_BLOCK_SIZE,
        topk=_SPARSE_TOPK,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k1,
        cu_seqlens_k2=cu_k2,
        max_seqlen_q=query_len,
        max_seqlen_k=max_seqlen_k,
        max_context_len=_MAX_CONTEXT_LEN,
        init_blocks=_INIT_BLOCKS,
        local_blocks=_LOCAL_BLOCKS,
        cache_lens=cache_lens,
        cu_seqlens_q_adjusted=cu_q * _GROUPS,
        max_seqlen_q_adjusted=query_len * _GROUPS,
    )


@pytest.mark.parametrize("context_len", [6144, 6500, 8192, 20000])
@pytest.mark.parametrize("query_len", [17, 100])
def test_live_context_selection_matches_max_context(context_len, query_len):
    """With >= topk live blocks,
    live-context and max-context shapes must select identical blocks."""
    inputs = _stage1_inputs(query_len, context_len)
    tight = _run_selection(inputs, max_seqlen_k=context_len)
    loose = _run_selection(inputs, max_seqlen_k=_MAX_CONTEXT_LEN)
    assert torch.equal(tight, loose)


@pytest.mark.parametrize("context_len", [48, 64, 80, 4096])
def test_short_context_selection_matches_max_context(context_len):
    """Below topk * block_size the clamp keeps at least topk columns,
    so max_seqlen_k must not change the selection."""
    query_len = min(17, context_len)
    inputs = _stage1_inputs(query_len, context_len)
    tight = _run_selection(inputs, max_seqlen_k=context_len)
    loose = _run_selection(inputs, max_seqlen_k=_MAX_CONTEXT_LEN)
    assert torch.equal(tight, loose)


def test_selection_emits_topk_blocks_in_index_order():
    """Each row emits its top-k blocks re-sorted by block id as int32, so a
    block with a maximal pooled score appears in every row and the indices
    come out strictly ascending regardless of topk's value order."""
    query_len = 17
    context_len = 8192
    max_score_block = 20
    generator = torch.Generator(device="cuda").manual_seed(0)
    k1_len = _compressed_len(context_len, _KERNEL_SIZE, _KERNEL_STRIDE)
    k2_len = _compressed_len(context_len, _KERNEL_SIZE * 4, _KERNEL_STRIDE * 4)
    direction = torch.ones(_DIM, dtype=torch.bfloat16, device="cuda")
    q = 5.0 * direction + 0.1 * torch.randn(
        (query_len, _HEADS, _DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k1 = 0.01 * torch.randn(
        (k1_len, _HEAD_KV, _DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k2 = 0.01 * torch.randn(
        (k2_len, _HEAD_KV, _DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    inputs = q, k1, k2, k1_len, k2_len, query_len, context_len
    # Compressed rows 4b..4b+3 pool keys from block b,
    # so aligning them with the query direction gives block b a maximal pooled score.
    k1[4 * max_score_block : 4 * max_score_block + 4] = 50.0 * direction
    topk_idx = _run_selection(inputs, max_seqlen_k=context_len)

    assert topk_idx.dtype == torch.int32
    assert (topk_idx == max_score_block).any(-1).all()
    assert (topk_idx[..., 1:] > topk_idx[..., :-1]).all()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
