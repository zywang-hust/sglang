"""Fused top-k selection kernels for the MiniCPM sparse attention path."""

from __future__ import annotations

import functools

import pytest
import torch

from sglang.srt.layers.attention.minicpm.fuse_kernel import (
    build_fused_topk_kernels,
    decode_k_splits,
)
from sglang.srt.layers.attention.minicpm.sparse_kernels import select_topk_blocks
from sglang.srt.layers.attention.minicpm.sparse_utils import (
    compressed_attention_tilelang,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

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
_MAX_CACHE_LEN = 40960

_POOLED_K_LEN = (_MAX_CACHE_LEN + _BLOCK_SIZE - 1) // _BLOCK_SIZE

_KERNEL_KWARGS = dict(
    groups=_GROUPS,
    heads=_HEADS,
    dim=_DIM,
    pooled_k_len=_POOLED_K_LEN,
    m_block_dim=_GROUPS,
    block_M=_GROUPS,
    block_stride=_BLOCK_SIZE // _KERNEL_STRIDE,
    pad_len=_KERNEL_SIZE // _KERNEL_STRIDE - 1,
    num_offs=_KERNEL_SIZE // _KERNEL_STRIDE + _BLOCK_SIZE // _KERNEL_STRIDE - 1,
    kernel_stride=_KERNEL_STRIDE,
    block_size=_BLOCK_SIZE,
    init_blocks=_INIT_BLOCKS,
    local_blocks=_LOCAL_BLOCKS,
    dtype_str="bfloat16",
)


@functools.cache
def _decode_kernels():
    return build_fused_topk_kernels(
        is_causal=False,
        k_splits=decode_k_splits(_MAX_CACHE_LEN // _KERNEL_STRIDE),
        batch_size=1,
        **_KERNEL_KWARGS,
    )


@functools.cache
def _prefill_kernels(max_seqlen_q_grid: int):
    return build_fused_topk_kernels(
        is_causal=True,
        k_splits=1,
        batch_size=1,
        max_seqlen_q_grid=max_seqlen_q_grid,
        **_KERNEL_KWARGS,
    )


def _run_topk(kernel, q, k, cu_seqlens_q, cu_seqlens_k, cache_lens):
    return compressed_attention_tilelang(
        q,
        k,
        _BLOCK_SIZE,
        _SPARSE_TOPK,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cache_lens=cache_lens,
        fused_kernel=kernel,
        max_cache_len=_MAX_CACHE_LEN,
    )


def _make_decode_inputs(seq_len: int, seed: int = 0) -> tuple[torch.Tensor, ...]:
    num_k = (seq_len - _KERNEL_SIZE) // _KERNEL_STRIDE + 1
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        (1, _HEADS, _DIM), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    k = torch.randn(
        (num_k, _HEAD_KV, _DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    cu_seqlens_k = torch.tensor([0, num_k], dtype=torch.int32, device="cuda")
    cache_lens = torch.tensor([seq_len - 1], dtype=torch.int32, device="cuda")
    return q, k, cu_seqlens_q, cu_seqlens_k, cache_lens


def test_decode_topk_deterministic_multi_round():
    """Identical inputs must produce identical pooled scores on every run."""
    kernels = _decode_kernels()
    q, k, cu_seqlens_q, cu_seqlens_k, cache_lens = _make_decode_inputs(8500)
    # Pins compressed_attention_tilelang's launch layout to reach the raw output.
    q_kernel = (
        q.view(1, _HEAD_KV, _GROUPS, _DIM)
        .transpose(1, 2)
        .reshape(_GROUPS, _HEAD_KV, _DIM)
        .contiguous()
    )

    def run_raw() -> torch.Tensor:
        stats = torch.empty(
            (_HEAD_KV, 1, _GROUPS, kernels.k_splits, 2),
            dtype=torch.float32,
            device="cuda",
        )
        pool_scores = torch.empty(
            (_HEAD_KV, 1, kernels.pooled_k_len), dtype=torch.float32, device="cuda"
        )
        kernels.stats(
            q_kernel,
            k,
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens,
            stats,
        )
        kernels.scores(
            q_kernel,
            k,
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens,
            stats,
            pool_scores,
        )
        torch.cuda.synchronize()
        return pool_scores

    ref_scores = run_raw()
    for _ in range(10):
        assert torch.equal(run_raw(), ref_scores)


_PREFILL_GRID = 256


def _run_prefill_topk(seq_len: int, seed: int = 0) -> torch.Tensor:
    num_k = (seq_len - _KERNEL_SIZE) // _KERNEL_STRIDE + 1
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        (seq_len, _HEADS, _DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn(
        (num_k, _HEAD_KV, _DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    cu_seqlens_k = torch.tensor([0, num_k], dtype=torch.int32, device="cuda")
    kernel = _prefill_kernels(_PREFILL_GRID)
    return _run_topk(kernel, q, k, cu_seqlens_q, cu_seqlens_k, None)


@pytest.mark.parametrize("seq_len", [78, 200])
def test_prefill_topk_selects_each_tokens_own_block(seq_len):
    """Every query token's own block must be selected even when that block
    has no compressed row yet (78: block 1 needs row 4 of 3; 200: block 3
    needs row 12 of 11).
    """
    topk_idx = _run_prefill_topk(seq_len)
    own = torch.arange(seq_len, device="cuda") // _BLOCK_SIZE
    assert (topk_idx == own[None, :, None]).any(-1).all()


# Chunked prefill past the sparse capacity (_SPARSE_TOPK * _BLOCK_SIZE tokens):
# every new token has more causally visible blocks than output slots.
_CHUNK_CACHE_LEN = _SPARSE_TOPK * _BLOCK_SIZE
_CHUNK_NEW_LEN = 384
_CHUNK_TOTAL_LEN = _CHUNK_CACHE_LEN + _CHUNK_NEW_LEN


def test_prefill_topk_subscribed_rows_select_only_causal_blocks():
    """Fully subscribed rows must never select a block past the token's own.

    A 32-token key window at stride 16 straddles block boundaries,
    so the last token of block b pools a real score into block b + 1;
    past sparse capacity that block can displace a causally valid one.
    """
    generator = torch.Generator(device="cuda").manual_seed(0)
    direction = torch.ones(_DIM, dtype=torch.bfloat16, device="cuda")
    q = (
        5.0 * direction
        + 0.1
        * torch.randn(
            (_CHUNK_NEW_LEN, _HEADS, _DIM),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
    ).contiguous()
    num_k = (_CHUNK_TOTAL_LEN - _KERNEL_SIZE) // _KERNEL_STRIDE + 1
    k = 0.01 * torch.randn(
        (num_k, _HEAD_KV, _DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    num_blocks = _CHUNK_TOTAL_LEN // _BLOCK_SIZE
    rows_per_block = _BLOCK_SIZE // _KERNEL_STRIDE
    # The last key before block b's first row is causally visible to the
    # last token of block b - 1 yet pools into block b.
    for future_block in range(_SPARSE_TOPK + 1, num_blocks):
        k[rows_per_block * future_block - 1] = 50.0 * direction
    cu_seqlens_q = torch.tensor([0, _CHUNK_NEW_LEN], dtype=torch.int32, device="cuda")
    cu_seqlens_k = torch.tensor([0, num_k], dtype=torch.int32, device="cuda")
    cache_lens = torch.tensor([_CHUNK_CACHE_LEN], dtype=torch.int32, device="cuda")

    kernel = _prefill_kernels(_CHUNK_NEW_LEN)
    topk_idx = _run_topk(kernel, q, k, cu_seqlens_q, cu_seqlens_k, cache_lens)

    pos = _CHUNK_CACHE_LEN + torch.arange(_CHUNK_NEW_LEN, device="cuda")
    own = (pos // _BLOCK_SIZE)[None, :, None]
    assert (topk_idx >= 0).sum(-1).eq(_SPARSE_TOPK).all()
    assert ((topk_idx <= own) | (topk_idx < 0)).all()


@pytest.mark.parametrize("seq_len", [6500, 8500, 12000])
def test_decode_topk_keeps_forced_blocks_beyond_output_topk(seq_len):
    """+inf-forced init/local blocks must survive the top-k selection when
    more than output_topk blocks are selectable.
    """
    q, k, cu_seqlens_q, cu_seqlens_k, cache_lens = _make_decode_inputs(seq_len)
    kernel = _decode_kernels()
    topk_idx = _run_topk(kernel, q, k, cu_seqlens_q, cu_seqlens_k, cache_lens)

    last_block = (seq_len - 1) // _BLOCK_SIZE
    sel = topk_idx[:, 0]
    forced = torch.cat(
        [
            torch.arange(_INIT_BLOCKS, device="cuda"),
            torch.arange(last_block - _LOCAL_BLOCKS, last_block + 1, device="cuda"),
        ]
    )
    assert (sel[:, None, :] == forced[None, :, None]).any(-1).all()


@pytest.mark.parametrize("num_blocks,topk", [(8, 8), (100, 96), (1536, 96)])
def test_select_topk_blocks_matches_torch_topk(num_blocks, topk):
    """The radix select must pick the same blocks as torch.topk, ascending,
    with -inf never selected, +inf always selected and ties going to the
    lower index.
    """
    generator = torch.Generator(device="cuda").manual_seed(0)
    scores = torch.randn((_HEAD_KV, 5, num_blocks), device="cuda", generator=generator)
    scores[:, :, int(num_blocks * 0.7) :] = float("-inf")
    scores[:, :, :2] = float("inf")
    scores[:, :, 3:6] = 0.5
    scores[:, 0] = float("-inf")
    scores[:, 1, num_blocks // 2 :] = float("-inf")

    values, indices = scores.topk(min(topk, num_blocks), dim=-1)
    expected = torch.where(values == float("-inf"), num_blocks, indices)
    expected = expected.sort(-1).values
    expected = torch.where(expected == num_blocks, -1, expected).to(torch.int32)
    if topk > num_blocks:
        pad = torch.full(
            (_HEAD_KV, 5, topk - num_blocks), -1, dtype=torch.int32, device="cuda"
        )
        expected = torch.cat([expected, pad], dim=-1)

    assert torch.equal(select_topk_blocks(scores, topk=topk), expected)


def test_prefill_grid_padding_preserves_unassigned_rows():
    """Padded query programs must not write rows outside their request."""
    kernels = build_fused_topk_kernels(
        is_causal=True,
        k_splits=1,
        batch_size=2,
        max_seqlen_q_grid=4,
        **_KERNEL_KWARGS,
    )
    q = torch.zeros((8 * _GROUPS, _HEAD_KV, _DIM), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((6, _HEAD_KV, _DIM), device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 6], device="cuda", dtype=torch.int32)
    cache = torch.full((2,), 64, device="cuda", dtype=torch.int32)
    stats = torch.ones((_HEAD_KV, 8, _GROUPS, 1, 2), device="cuda")
    scores = torch.full((_HEAD_KV, 8, kernels.pooled_k_len), 123.0, device="cuda")
    kernels.scores(q, k, cu_q, cu_k, cache, stats, scores)
    assert scores[:, 3:].eq(123.0).all()


def test_select_topk_ties_and_padding_on_graph_replay():
    """Selection resolves cutoff ties by index and clears reused output slots."""
    scores = torch.tensor([[1.0, 2.0, 2.0, 2.0, float("inf"), 0.0]], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        select_topk_blocks(scores, topk=3)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = select_topk_blocks(scores, topk=3)
    graph.replay()
    assert result.tolist() == [[1, 2, 4]]
    scores.fill_(float("-inf"))
    scores[0, 5] = 1.0
    graph.replay()
    assert result.tolist() == [[5, -1, -1]]


def _small_score_kernels(*, q_lens, causal, splits):
    return build_fused_topk_kernels(
        **dict(
            _KERNEL_KWARGS,
            pooled_k_len=96,
            init_blocks=1,
            local_blocks=2,
        ),
        batch_size=len(q_lens),
        is_causal=causal,
        k_splits=splits,
        max_seqlen_q_grid=max(q_lens),
    )


def _score_outputs(kernels, q, k, cu_q, cu_k, cache):
    rows = q.shape[0] // _GROUPS
    stats = torch.empty((_HEAD_KV, rows, _GROUPS, kernels.k_splits, 2), device="cuda")
    scores = torch.empty((_HEAD_KV, rows, kernels.pooled_k_len), device="cuda")
    kernels.stats(q, k, cu_q, cu_k, cache, stats)
    kernels.scores(q, k, cu_q, cu_k, cache, stats, scores)
    return stats, scores


def _reference_pooled_scores(q, k, *, cache, causal, num_blocks):
    # CPU fp32 avoids GPU matmul precision settings in the numerical oracle.
    q = q.cpu().float().reshape(-1, _GROUPS, _HEAD_KV, _DIM)
    k = k.cpu().float()
    logits = torch.einsum("qghd,khd->hqgk", q, k) / (_DIM**0.5)
    query_positions = cache + torch.arange(q.shape[0])
    key_starts = torch.arange(k.shape[0]) * _KERNEL_STRIDE
    if causal:
        # These fixtures end on stride boundaries; a compressed row becomes
        # visible when its first stride completes.
        visible = key_starts[None, :] + _KERNEL_STRIDE <= query_positions[:, None] + 1
        logits.masked_fill_(~visible[None, :, None, :], float("-inf"))
    head_sum = logits.softmax(-1).sum(2)
    block_starts = torch.arange(num_blocks) * _BLOCK_SIZE
    overlaps = (key_starts[:, None] < block_starts[None, :] + _BLOCK_SIZE) & (
        key_starts[:, None] + _KERNEL_SIZE > block_starts[None, :]
    )
    scores = head_sum[..., None].masked_fill(~overlaps, float("-inf")).amax(2)
    own = query_positions // _BLOCK_SIZE
    blocks = torch.arange(num_blocks)
    forced = (blocks[None, :] == 0) | (
        (blocks[None, :] >= own[:, None] - 2) & (blocks[None, :] <= own[:, None])
    )
    scores.masked_fill_(forced[None, :, :], float("inf"))
    scores.masked_fill_(blocks[None, None, :] > own[None, :, None], float("-inf"))
    return logits, scores


@pytest.mark.parametrize(
    "causal,q_len,k_len,cache,splits",
    [
        (False, 1, 257, 4127, 4),
        (True, 16, 252, 4032, 1),
    ],
)
def test_pooled_scores_match_headwise_softmax(causal, q_len, k_len, cache, splits):
    """Split statistics must reconstruct head-wise softmax before overlap pooling."""
    generator = torch.Generator().manual_seed(137)
    q = (
        (torch.randn(q_len * _GROUPS, _HEAD_KV, _DIM, generator=generator) * 0.25)
        .bfloat16()
        .cuda()
    )
    k = (
        (torch.randn(k_len, _HEAD_KV, _DIM, generator=generator) * 0.25)
        .bfloat16()
        .cuda()
    )
    kernels = _small_score_kernels(q_lens=[q_len], causal=causal, splits=splits)
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, k_len], dtype=torch.int32, device="cuda")
    cache_lens = torch.tensor([cache], dtype=torch.int32, device="cuda")
    stats, scores = _score_outputs(kernels, q, k, cu_q, cu_k, cache_lens)
    logits, expected = _reference_pooled_scores(
        q, k, cache=cache, causal=causal, num_blocks=96
    )
    stats = stats.cpu()
    maxima = stats[..., 0] / (_DIM**0.5)
    maximum = maxima.amax(-1)
    denominator = (stats[..., 1] * (maxima - maximum[..., None]).exp()).sum(-1)
    expected_maximum = logits.amax(-1)
    expected_denominator = (logits - expected_maximum[..., None]).exp().sum(-1)
    torch.testing.assert_close(maximum, expected_maximum, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(denominator, expected_denominator, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(scores.cpu(), expected, atol=3e-5, rtol=3e-5)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
