import pytest
import torch
from sgl_kernel import infllm_v2 as sgl


def _compressed_len(tokens, kernel_size, stride):
    return max(0, (tokens - kernel_size) // stride + 1)


def _trunc_div(a, b):
    """C++ integer division (truncates toward zero, unlike Python floor)."""
    return a // b if a >= 0 else -(-a // b)


def _ref_stage1(q, k1, k2, query_len):
    """Torch reference pinning the stage-1 score/normalization path.
    Mask limits intentionally mirror ``apply_mask_stage1``
    (the key frontier advances one key per 16 (k1) / 64 (k2) query tokens,
    anchored so the last query token sees every key);
    they are not an independent oracle."""
    n_kv_heads = k1.shape[1]
    groups = q.shape[1] // n_kv_heads
    scale = q.shape[-1] ** (-0.5)
    k1_len, k2_len = k1.shape[0], k2.shape[0]
    qf = q.float().view(query_len, n_kv_heads, groups, q.shape[-1])
    score_k = torch.einsum("qhgd,khd->hqgk", qf, k1.float()) * scale
    score_c = torch.einsum("qhgd,khd->hqgk", qf, k2.float()) * scale
    t = torch.arange(query_len, device=q.device)
    lim_k = ((t + 1) // 16 - 1 + k1_len - _trunc_div(query_len - 15, 16)).clamp(
        0, k1_len
    )
    lim_c = ((t + 1) // 64 - 1 + k2_len - _trunc_div(query_len - 63, 64)).clamp(
        0, k2_len
    )
    mask_c = torch.arange(k2_len, device=q.device) >= lim_c[:, None]
    score_c = score_c.masked_fill(mask_c[None, :, None, :], float("-inf"))
    row_max = score_c.amax(dim=-1, keepdim=True)
    row_sum = (score_c - row_max).exp().sum(dim=-1, keepdim=True)
    p = (score_k - row_max).exp() / row_sum
    mask_k = torch.arange(k1_len, device=q.device) >= lim_k[:, None]
    return p.masked_fill(mask_k[None, :, None, :], 0.0).sum(dim=2).to(q.dtype)


@pytest.mark.parametrize("context_len", [4096, 8500])
@pytest.mark.parametrize("query_len", [17, 19])
def test_stage1_matches_torch_reference(context_len, query_len):
    """In the retiled stage-1 kernel,
    query_len * 16 % 64 != 0 exercises partial 64-row query tiles."""
    torch.manual_seed(context_len + query_len)
    n_heads, n_kv_heads, head_dim = 32, 2, 128
    groups = n_heads // n_kv_heads
    k1_len = _compressed_len(context_len, 32, 16)
    k2_len = _compressed_len(context_len, 128, 64)
    q = torch.randn(query_len, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k1 = torch.randn(k1_len, n_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k2 = torch.randn(k2_len, n_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    cu_q = torch.tensor([0, query_len], dtype=torch.int32, device="cuda")
    cu_k1 = torch.tensor([0, k1_len], dtype=torch.int32, device="cuda")
    cu_k2 = torch.tensor([0, k2_len], dtype=torch.int32, device="cuda")

    score = sgl.infllmv2_attn_stage1(
        q,
        k1,
        k2,
        cu_seqlens_q=cu_q * groups,
        cu_seqlens_k=cu_k1,
        cu_seqlens_v=cu_k2,
        max_seqlen_q=query_len * 16,
        max_seqlen_k=k1_len,
        causal=True,
    )
    torch.testing.assert_close(
        score[..., :k1_len], _ref_stage1(q, k1, k2, query_len), rtol=1e-2, atol=1e-4
    )
    assert torch.all(score[..., k1_len:] == 0)


def test_varlen_segments_match_single_runs():
    """64-row query tiles straddling a segment end must not write into the next segment:
    a two-segment varlen batch must reproduce each segment's single-run scores bitwise
    (17 and 19 query tokens both leave a partial tile at their segment end)."""
    n_heads, n_kv_heads, head_dim = 32, 2, 128
    groups = n_heads // n_kv_heads
    context_len = 4096
    k1_len = _compressed_len(context_len, 32, 16)
    k2_len = _compressed_len(context_len, 128, 64)

    segments = []
    for query_len in (17, 19):
        torch.manual_seed(query_len)
        segments.append(
            (
                torch.randn(
                    query_len, n_heads, head_dim, dtype=torch.bfloat16, device="cuda"
                ),
                torch.randn(
                    k1_len, n_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"
                ),
                torch.randn(
                    k2_len, n_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"
                ),
            )
        )

    def run(q, k1, k2, cu_rows, cu_k1, cu_k2):
        return sgl.infllmv2_attn_stage1(
            q,
            k1,
            k2,
            cu_seqlens_q=cu_rows,
            cu_seqlens_k=cu_k1,
            cu_seqlens_v=cu_k2,
            max_seqlen_q=cu_rows[-1].item(),
            max_seqlen_k=k1_len,
            causal=True,
        )

    starts = [0]
    for query_len in (17, 19):
        starts.append(starts[-1] + query_len)

    expected = []
    for (q, k1, k2), (lo, hi) in zip(segments, zip(starts, starts[1:])):
        single = run(
            q,
            k1,
            k2,
            torch.tensor([0, hi - lo], dtype=torch.int32, device="cuda") * groups,
            torch.tensor([0, k1_len], dtype=torch.int32, device="cuda"),
            torch.tensor([0, k2_len], dtype=torch.int32, device="cuda"),
        )
        expected.append(single)

    q = torch.cat([seg[0] for seg in segments])
    k1 = torch.cat([seg[1] for seg in segments])
    k2 = torch.cat([seg[2] for seg in segments])
    cu_rows = [0]
    cu_k1 = [0]
    cu_k2 = [0]
    for lo, hi in zip(starts, starts[1:]):
        cu_rows.append(hi * groups)
        cu_k1.append(cu_k1[-1] + k1_len)
        cu_k2.append(cu_k2[-1] + k2_len)
    batch = run(
        q,
        k1,
        k2,
        torch.tensor(cu_rows, dtype=torch.int32, device="cuda"),
        torch.tensor(cu_k1, dtype=torch.int32, device="cuda"),
        torch.tensor(cu_k2, dtype=torch.int32, device="cuda"),
    )

    for (lo, hi), single in zip(zip(starts, starts[1:]), expected):
        assert torch.equal(batch[:, lo:hi], single)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
