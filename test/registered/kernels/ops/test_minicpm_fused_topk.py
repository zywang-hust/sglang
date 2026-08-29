"""Fused top-k selection kernels for the MiniCPM sparse attention path."""

from __future__ import annotations

import functools

import pytest
import tilelang.math
import torch

from sglang.srt.layers.attention.minicpm.fuse_kernel import (
    fused_attn_pooling_online_topk_decode,
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
