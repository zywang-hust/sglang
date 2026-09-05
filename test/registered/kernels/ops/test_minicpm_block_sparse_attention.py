"""Q-tiled block-sparse prefill fast path against the paged dense reference."""

import unittest

import torch

from sglang.srt.layers.attention.minicpm.block_sparse_attention import (
    block_sparse_attention,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _dense_reference(q, k, v, page_table, topk, prefix_len, seq_len, block_size, scale):
    """Per-token softmax over the token's deduplicated valid blocks."""
    token_num, q_heads, _ = q.shape
    kv_heads = k.shape[1]
    gqa_group = q_heads // kv_heads
    expected = torch.empty_like(q)
    for token in range(token_num):
        limit = prefix_len + token + 1
        for kv_head in range(kv_heads):
            blocks = sorted(
                {block for block in topk[kv_head, token].tolist() if block >= 0}
            )
            positions = []
            for block in blocks:
                positions.extend(
                    range(
                        block * block_size,
                        min((block + 1) * block_size, limit, seq_len),
                    )
                )
            slots = page_table[positions].long()
            keys = k[slots, kv_head].float()
            values = v[slots, kv_head].float()
            first_head = kv_head * gqa_group
            last_head = first_head + gqa_group
            scores = q[token, first_head:last_head].float() @ keys.T * scale
            expected[token, first_head:last_head] = (
                torch.softmax(scores, dim=-1) @ values
            ).to(q.dtype)
    return expected


class TestMiniCPMBlockSparseAttention(CustomTestCase):
    def _run_case(self, token_num, seq_len, prefix_len, topk_rows):
        torch.manual_seed(0)
        device = "cuda"
        dtype = torch.bfloat16
        q_heads = 32
        kv_heads = 2
        head_dim = 128
        block_size = 64
        scale = head_dim**-0.5

        q = torch.randn(token_num, q_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(seq_len, kv_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn_like(k)
        page_table = torch.randperm(seq_len, device=device, dtype=torch.int32)
        topk = torch.tensor(topk_rows, device=device, dtype=torch.int32)
        if topk.dim() == 2:
            topk = topk.unsqueeze(0).expand(kv_heads, -1, -1).contiguous()

        actual = block_sparse_attention(
            q=q,
            k_cache=k,
            v_cache=v,
            page_table=page_table,
            topk_idx=topk,
            prefix_len=prefix_len,
            seq_len=seq_len,
            block_size=block_size,
            softmax_scale=scale,
        )
        expected = _dense_reference(
            q, k, v, page_table, topk, prefix_len, seq_len, block_size, scale
        )
        self.assertTrue(
            # bf16 kernel against an fp32 oracle (suite convention).
            torch.allclose(actual, expected, atol=2e-2, rtol=2e-2),
            msg=f"max diff: {(actual.float() - expected.float()).abs().max().item()}",
        )

    def test_matches_reference_with_query_specific_blocks(self):
        token_num = 11
        prefix_len = 12000
        block_size = 64
        sparse_topk = 96

        def token_blocks(head, token):
            frontier = (prefix_len + token) // block_size
            if token < 8:
                return [*range(sparse_topk - 1), frontier]
            start = (token - 8) * 45 + head * 15
            return [
                *((start + offset) % frontier for offset in range(sparse_topk - 1)),
                frontier,
            ]

        self._run_case(
            token_num,
            12288,
            prefix_len,
            [
                [token_blocks(head, token) for token in range(token_num)]
                for head in range(2)
            ],
        )

    def test_ignores_negative_padding_in_topk(self):
        # -1 padding at tail and interior positions must be dropped by the mask build.
        frontier = 1900 // 64
        self._run_case(
            8,
            2048,
            1900,
            [[0, 1, -1, 2, 3, 10 + token, frontier, -1] for token in range(8)],
        )

    def test_union_block_clamped_at_unaligned_seq_len(self):
        # seq_len 2050 is not block-aligned,
        # so the frontier block of the last token straddles the buffer end;
        # its columns past seq_len have no page-table entries and must clamp out.
        self._run_case(
            8,
            2050,
            2042,
            [[16, 31, 32] for _ in range(8)],
        )

    def test_pair_kernel_clamps_union_block_at_unaligned_seq_len(self):
        # Distinct per-token block lists push the tile union to 33,
        # past union_threshold (topk=5 -> 8), so the tile kernel declines
        # and tokens 6/7 reach their straddling frontier block through the pair kernel.
        self._run_case(
            8,
            2050,
            2042,
            [
                [token, 8 + token, 16 + token, 24 + token, (2042 + token) // 64]
                for token in range(8)
            ],
        )

    def test_tile_union_at_threshold_plus_one_splits_to_pairs(self):
        # The tile union is exactly union_threshold + 1 blocks (topk=5 -> 9):
        # the tile kernel declines it and the split kernel hands its pairs over.
        # A one-off on the split comparison strands this tile with no writer;
        # the tile-side one-off instead strands a union==threshold tile,
        # which test_first_union_block pins.
        shared = [0, 1, 2, 3]
        self._run_case(
            8,
            2050,
            2042,
            [[*shared, 8 + token] if token < 5 else [*shared, 0] for token in range(8)],
        )

    def test_pair_kernel_at_full_capacity(self):
        # Fully disjoint per-token lists give every pair the exact worst case --
        # 2 * topk distinct blocks -- so the pair metadata fills pair_capacity;
        # a shrunk capacity drops or misroutes a block.
        self._run_case(
            8,
            4096,
            4088,
            [
                [token, 8 + token, 16 + token, 24 + token, 32 + token]
                for token in range(8)
            ],
        )

    def test_first_union_block_invisible_to_some_rows(self):
        # Block 0 is selected only by token 0,
        # so the ascending union's first iteration has no valid column for tokens 1..7;
        # guards the finite m_i init.
        frontier = 1900 // 64
        self._run_case(
            8, 2048, 1900, [[0, frontier]] + [[20, frontier] for _ in range(7)]
        )


if __name__ == "__main__":
    unittest.main()
