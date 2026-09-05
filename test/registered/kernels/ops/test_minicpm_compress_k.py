import pytest
import torch

from sglang.srt.layers.attention.minicpm.fuse_kernel import (
    build_fused_topk_kernels,
)
from sglang.srt.layers.attention.minicpm.sparse_utils import (
    compress_k_core_new,
    compressed_attention_tilelang,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_compress_k_writes_each_head_once():
    """Each compressed output must be produced once even with multiple KV heads."""
    key_cache = torch.arange(
        9 * 2 * 6,
        dtype=torch.float32,
        device="cuda",
    ).reshape(9, 2, 6)
    original = key_cache.clone()
    token_table = torch.tensor([[0, 0, 0, 1, 2, 3]], dtype=torch.int32, device="cuda")
    compressed_table = torch.tensor([[6, 7, 8]], dtype=torch.int32, device="cuda")
    full_compressed = torch.empty((3, 2, 6), device="cuda")

    compress_k_core_new(
        full_compressed,
        1,
        key_cache,
        token_table,
        compressed_table,
        torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
        2,
        2,
        6,
    )

    expected = torch.stack(
        (
            original[6],
            original[0:2].mean(dim=0),
            original[2:4].mean(dim=0),
        )
    )
    torch.testing.assert_close(full_compressed, expected)
    torch.testing.assert_close(key_cache[7:9], expected[1:])


def test_fused_decode_topk_skips_dense_rows():
    kernels = build_fused_topk_kernels(
        is_causal=False,
        k_splits=1,
        batch_size=2,
        groups=16,
        heads=16,
        dim=128,
        pooled_k_len=8,
        dense_len=5,
        dtype_str="bfloat16",
    )
    topk_indices = compressed_attention_tilelang(
        torch.randn(2, 16, 128, dtype=torch.bfloat16, device="cuda"),
        torch.randn(4, 1, 128, dtype=torch.bfloat16, device="cuda"),
        64,
        8,
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
        cu_seqlens_k=torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda"),
        cache_lens=torch.tensor([3, 7], dtype=torch.int32, device="cuda"),
        fused_kernel=kernels,
        max_cache_len=8 * 64,
    )

    assert torch.all(topk_indices[:, 0] == -1)
    assert torch.any(topk_indices[:, 1] >= 0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
