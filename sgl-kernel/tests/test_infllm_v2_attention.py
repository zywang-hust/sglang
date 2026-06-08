"""Equivalence tests for the migrated InfLLM-V2 FlashAttention API.

These compare the ``sgl_kernel.infllm_v2`` implementations against the original
``infllm_v2`` package (3rdparty/infllmv2_cuda_impl). Both call the same CUDA
kernels, so outputs are expected to match closely. The whole module is skipped
if the reference ``infllm_v2`` package is not importable.
"""

import pytest
import torch

sgl = pytest.importorskip("sgl_kernel.infllm_v2")
ref = pytest.importorskip("infllm_v2")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for InfLLM-V2 kernels"
)


def _assert_close(a, b, name):
    a = a.float()
    b = b.float()
    assert a.shape == b.shape, f"{name}: shape mismatch {a.shape} vs {b.shape}"
    max_diff = (a - b).abs().max().item()
    assert torch.allclose(a, b, atol=1e-2, rtol=1e-2), f"{name}: max diff {max_diff}"


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(256, 16), (64, 17)])
def test_stage1_matches_reference(head_dim, causal, seqlen_q, seqlen_k):
    torch.manual_seed(0)
    n_heads, n_kv_heads = 32, 2
    dtype = torch.bfloat16

    q = torch.randn(n_heads, seqlen_q, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(n_kv_heads, seqlen_k, head_dim, dtype=dtype, device="cuda")

    cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device="cuda")
    cu_seqlens_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device="cuda")

    q = q.transpose(0, 1).contiguous()
    k = k.transpose(0, 1).contiguous()

    common = dict(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_v=cu_seqlens_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        causal=causal,
    )
    out_ref = ref.infllmv2_attn_stage1(q, k, k, **common)
    out_sgl = sgl.infllmv2_attn_stage1(q, k, k, **common)
    _assert_close(out_sgl, out_ref, "stage1")


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_varlen_func_dense_matches_reference(head_dim, causal):
    torch.manual_seed(1)
    n_heads, n_kv_heads = 16, 2
    seqlen = 128
    dtype = torch.bfloat16

    q = torch.randn(seqlen, n_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(seqlen, n_kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(seqlen, n_kv_heads, head_dim, dtype=dtype, device="cuda")
    cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device="cuda")

    kw = dict(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        causal=causal,
    )
    out_ref = ref.infllmv2_attn_varlen_func(q, k, v, **kw)
    out_sgl = sgl.infllmv2_attn_varlen_func(q, k, v, **kw)
    _assert_close(out_sgl, out_ref, "varlen_func")


@pytest.mark.parametrize("head_dim", [64, 128])
def test_with_kvcache_dense_matches_reference(head_dim):
    torch.manual_seed(2)
    batch_size, seq_len, cache_len = 1, 1, 256
    n_heads, n_kv_heads = 32, 2
    dtype = torch.bfloat16

    q = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=dtype, device="cuda")
    k_pad = torch.randn(
        batch_size,
        cache_len + seq_len,
        n_kv_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    v_pad = torch.randn(
        batch_size,
        cache_len + seq_len,
        n_kv_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    k = torch.randn(
        batch_size, seq_len, n_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size, seq_len, n_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    cache_seqlens = torch.full(
        (batch_size,), cache_len, dtype=torch.int32, device="cuda"
    )

    kw = dict(k=k, v=v, cache_seqlens=cache_seqlens)
    out_ref = ref.infllmv2_attn_with_kvcache(q, k_pad.clone(), v_pad.clone(), **kw)
    out_sgl = sgl.infllmv2_attn_with_kvcache(q, k_pad.clone(), v_pad.clone(), **kw)
    _assert_close(out_sgl, out_ref, "with_kvcache")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
