"""Fused top-k selection kernels for the MiniCPM sparse attention path."""

from __future__ import annotations

import functools

import pytest
import tilelang.math
import torch

from sglang.srt.layers.attention.minicpm.fuse_kernel import (
    fused_attn_pooling_online_topk_decode,
    fused_attn_pooling_online_topk_prefill,
)
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
_OUTPUT_TOPK = min(_SPARSE_TOPK, _POOLED_K_LEN)
_KERNEL_TOPK = tilelang.math.next_power_of_2(_OUTPUT_TOPK)

_KERNEL_KWARGS = dict(
    groups=_GROUPS,
    heads=_HEADS,
    dim=_DIM,
    topk=_KERNEL_TOPK,
    pooled_k_len=tilelang.math.next_power_of_2(_POOLED_K_LEN),
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
def _kernel(factory, **overrides):
    return factory(batch_size=1, **_KERNEL_KWARGS, **overrides)


def _run_topk(kernel, q, k, cu_seqlens_q, cu_seqlens_k, cache_lens):
    return compressed_attention_tilelang(
        q,
        k,
        _BLOCK_SIZE,
        _SPARSE_TOPK,
        _KERNEL_TOPK,
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
    """Same input must produce identical raw kernel output on every run.

    Pre-fix, this shape disagreed with itself on 50 of 50 runs; asserted on
    raw output because truncation can hide order differences.
    """
    kernel = _kernel(fused_attn_pooling_online_topk_decode)
    q, k, cu_seqlens_q, cu_seqlens_k, cache_lens = _make_decode_inputs(8500)
    # Pins compressed_attention_tilelang's launch layout to reach the raw output.
    q_kernel = (
        q.view(1, _HEAD_KV, _GROUPS, _DIM)
        .transpose(1, 2)
        .reshape(_GROUPS, _HEAD_KV, _DIM)
        .contiguous()
    )

    def run_raw() -> tuple[torch.Tensor, torch.Tensor]:
        topk_indices = torch.full(
            (_HEAD_KV, 1, _KERNEL_TOPK), -1, dtype=torch.int32, device="cuda"
        )
        topk_values = torch.full(
            (_HEAD_KV, 1, _KERNEL_TOPK),
            float("-inf"),
            dtype=torch.float32,
            device="cuda",
        )
        kernel(
            q_kernel,
            k,
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens,
            topk_indices,
            topk_values,
        )
        torch.cuda.synchronize()
        return topk_indices, topk_values

    ref_indices, ref_values = run_raw()
    for _ in range(10):
        repeat_indices, repeat_values = run_raw()
        assert torch.equal(repeat_indices, ref_indices)
        assert torch.equal(repeat_values, ref_values)


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
    kernel = _kernel(
        fused_attn_pooling_online_topk_prefill, max_seqlen_q_grid=_PREFILL_GRID
    )
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

    kernel = _kernel(
        fused_attn_pooling_online_topk_prefill, max_seqlen_q_grid=_CHUNK_NEW_LEN
    )
    topk_idx = _run_topk(kernel, q, k, cu_seqlens_q, cu_seqlens_k, cache_lens)

    pos = _CHUNK_CACHE_LEN + torch.arange(_CHUNK_NEW_LEN, device="cuda")
    own = (pos // _BLOCK_SIZE)[None, :, None]
    assert (topk_idx >= 0).sum(-1).eq(_SPARSE_TOPK).all()
    assert ((topk_idx <= own) | (topk_idx < 0)).all()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
