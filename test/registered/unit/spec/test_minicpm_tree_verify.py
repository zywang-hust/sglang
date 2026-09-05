"""MiniCPM tree-verify kernels and recompression rollback against torch oracles."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.minicpm_fixtures import (
    pack_custom_mask,
    plan_verify,
    tree_mask_from_parents,
    visible_counts,
)
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.minicpm.sparse_kernels import (
    compact_sparse_tree_page_table,
    copy_eagle_draft_tree_mask,
    fill_dense_page_table_rows,
)
from sglang.srt.layers.attention.minicpm.sparse_utils import (
    CompressLevel,
    _build_k1_k2_compression_metadata,
    compress_k_core_new,
)

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

DEVICE = "cuda"
BLOCK_SIZE = 4
SPARSE_TOPK = 3
CAPACITY = SPARSE_TOPK * BLOCK_SIZE
HEAD_GROUP = 2
NUM_DRAFT_TOKENS = 3


def _verify_metadata(seq_lens, draft_visible, dense_len):
    _, _, metadata = plan_verify(
        seq_lens,
        NUM_DRAFT_TOKENS,
        head_group_num=HEAD_GROUP,
        heads_per_group=16,
        dense_len=dense_len,
        sparse_topk=SPARSE_TOPK,
        block_size=BLOCK_SIZE,
        num_draft_visible=draft_visible,
        device=DEVICE,
    )
    return metadata


def _copy_tree_mask(out, num_visible_out, custom_mask, seq_lens, bs, padded_bs):
    """The tree-mask copy call shared by the extraction and padding tests."""
    copy_eagle_draft_tree_mask(
        out=out,
        num_visible_out=num_visible_out,
        custom_mask=custom_mask,
        seq_lens=seq_lens,
        num_draft_tokens=NUM_DRAFT_TOKENS,
        bs=bs,
        padded_bs=padded_bs,
    )


class TestCopyEagleDraftTreeMask(CustomTestCase):
    def test_extraction_matches_packed_layout(self):
        """A base or stride slip would read a neighbouring request's visibility bits."""
        seq_lens = [7, 3, 12]
        squares = [
            tree_mask_from_parents([-1, 0, 0]),
            tree_mask_from_parents([-1, 0, 1]),
            tree_mask_from_parents([-1, -1, 1]),
        ]
        custom_mask = pack_custom_mask(seq_lens, squares, device=DEVICE)
        seq_lens_gpu = torch.tensor(seq_lens, dtype=torch.int64, device=DEVICE)
        bs = len(seq_lens)
        out = torch.zeros(
            bs * NUM_DRAFT_TOKENS * NUM_DRAFT_TOKENS, dtype=torch.bool, device=DEVICE
        )
        counts = torch.full(
            (bs * NUM_DRAFT_TOKENS,), -1, dtype=torch.int32, device=DEVICE
        )
        _copy_tree_mask(
            out=out,
            num_visible_out=counts,
            custom_mask=custom_mask,
            seq_lens=seq_lens_gpu,
            bs=bs,
            padded_bs=bs,
        )
        want = torch.stack(squares).view(-1).to(DEVICE)
        self.assertTrue(torch.equal(out, want))
        self.assertTrue(
            torch.equal(
                counts,
                visible_counts(mask=want, bs=bs, num_draft_tokens=NUM_DRAFT_TOKENS),
            )
        )

    def test_padding_rows_are_all_true(self):
        seq_lens = [5]
        squares = [tree_mask_from_parents([-1, 0, 1])]
        custom_mask = pack_custom_mask(seq_lens, squares, device=DEVICE)
        seq_lens_gpu = torch.tensor(seq_lens, dtype=torch.int64, device=DEVICE)
        padded_bs = 3
        out = torch.zeros(
            padded_bs * NUM_DRAFT_TOKENS * NUM_DRAFT_TOKENS,
            dtype=torch.bool,
            device=DEVICE,
        )
        counts = torch.full(
            (padded_bs * NUM_DRAFT_TOKENS,), -1, dtype=torch.int32, device=DEVICE
        )
        _copy_tree_mask(
            out=out,
            num_visible_out=counts,
            custom_mask=custom_mask,
            seq_lens=seq_lens_gpu,
            bs=1,
            padded_bs=padded_bs,
        )
        self.assertTrue(
            torch.equal(
                out[: NUM_DRAFT_TOKENS * NUM_DRAFT_TOKENS].cpu(), squares[0].view(-1)
            )
        )
        self.assertTrue(out[NUM_DRAFT_TOKENS * NUM_DRAFT_TOKENS :].all())
        # All-True padding rows popcount to the 1-based offsets,
        # so the graph row refresh subtracts exactly zero from them.
        self.assertEqual(
            counts[NUM_DRAFT_TOKENS:].cpu().tolist(),
            list(range(1, NUM_DRAFT_TOKENS + 1)) * (padded_bs - 1),
        )


def _fill_source_rows(page_table_cpu, topk_idx_cpu, token_to_bs, token_pos):
    """Torch oracle of the get_block_table fill: head-group encoded slots,
    zero past the causal position or for -1 blocks."""
    tokens = len(token_to_bs)
    rows = torch.zeros(
        (tokens * HEAD_GROUP, SPARSE_TOPK * BLOCK_SIZE), dtype=torch.int32
    )
    for token in range(tokens):
        batch = token_to_bs[token]
        pos = token_pos[token]
        for group in range(HEAD_GROUP):
            row = token * HEAD_GROUP + group
            for slot_block in range(SPARSE_TOPK):
                block = int(topk_idx_cpu[group, token, slot_block])
                if block < 0:
                    continue
                for offset in range(BLOCK_SIZE):
                    key_pos = block * BLOCK_SIZE + offset
                    if key_pos < pos:
                        rows[row, slot_block * BLOCK_SIZE + offset] = (
                            int(page_table_cpu[batch, key_pos]) * HEAD_GROUP + group
                        )
    return rows


def _tree_case(seq_lens, square=None):
    """The draft-tree case shared by the compaction and dense fill tests."""
    bs = len(seq_lens)
    if square is None:
        square = tree_mask_from_parents([-1, 0, -1])
    squares = [square] * bs
    draft_tree_mask = torch.stack(squares).view(-1).to(DEVICE)
    draft_visible = visible_counts(
        mask=draft_tree_mask, bs=bs, num_draft_tokens=NUM_DRAFT_TOKENS
    )

    token_to_bs = [b for b in range(bs) for _ in range(NUM_DRAFT_TOKENS)]
    token_pos = [
        seq_lens[b] + i + 1 for b in range(bs) for i in range(NUM_DRAFT_TOKENS)
    ]
    tokens = bs * NUM_DRAFT_TOKENS
    return bs, squares, draft_tree_mask, draft_visible, token_to_bs, token_pos, tokens


class TestCompactSparseTreePageTable(CustomTestCase):
    def test_compaction_matches_visibility_oracle_and_plan(self):
        """Kept count per row must equal the planned popcount row length,
        or attention reads planned slots the fill never wrote (loc 0)."""
        seq_lens = [5, 17]
        bs, squares, draft_tree_mask, draft_visible, token_to_bs, token_pos, tokens = (
            _tree_case(seq_lens)
        )

        generator = torch.Generator().manual_seed(7)
        page_table_cpu = torch.randint(
            1, 300, (bs, 32), dtype=torch.int32, generator=generator
        )
        # Mimic the fused selection: ascending blocks, forced tail block,
        # -1 padded.
        topk_idx_cpu = torch.full(
            (HEAD_GROUP, tokens, SPARSE_TOPK), -1, dtype=torch.int32
        )
        for token in range(tokens):
            num_blocks = -(-token_pos[token] // BLOCK_SIZE)
            for group in range(HEAD_GROUP):
                if num_blocks <= SPARSE_TOPK:
                    chosen = list(range(num_blocks))
                else:
                    lead = [0, 1] if group == 0 else [0, num_blocks - 2]
                    chosen = sorted(set(lead + [num_blocks - 1]))[:SPARSE_TOPK]
                for slot, block in enumerate(chosen):
                    topk_idx_cpu[group, token, slot] = block

        source_rows_cpu = _fill_source_rows(
            page_table_cpu, topk_idx_cpu, token_to_bs, token_pos
        )
        metadata = _verify_metadata(seq_lens, draft_visible, dense_len=0)
        source_lens = metadata.verify_source_row_lens
        row_lens = metadata.sparse_cache_seqlens_int32

        out = torch.full_like(source_rows_cpu, -7).to(DEVICE)
        compact_sparse_tree_page_table(
            topk_rows=source_rows_cpu.to(DEVICE),
            topk_idx=topk_idx_cpu.to(DEVICE),
            token_to_bs=torch.tensor(token_to_bs, dtype=torch.int32, device=DEVICE),
            token_pos_in_bs=torch.tensor(token_pos, dtype=torch.int32, device=DEVICE),
            prefix_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
            draft_tree_mask=draft_tree_mask,
            source_row_lens=source_lens,
            out_page_table=out,
            num_draft_tokens=NUM_DRAFT_TOKENS,
            block_size=BLOCK_SIZE,
            head_group_num=HEAD_GROUP,
        )

        for token in range(tokens):
            batch = token_to_bs[token]
            prefix = seq_lens[batch]
            draft_idx = token_pos[token] - prefix - 1
            for group in range(HEAD_GROUP):
                row = token * HEAD_GROUP + group
                kept = []
                for slot_block in range(SPARSE_TOPK):
                    block = int(topk_idx_cpu[group, token, slot_block])
                    if block < 0:
                        continue
                    for offset in range(BLOCK_SIZE):
                        key_pos = block * BLOCK_SIZE + offset
                        if key_pos >= token_pos[token]:
                            continue
                        if (
                            key_pos >= prefix
                            and not squares[batch][draft_idx, key_pos - prefix]
                        ):
                            continue
                        kept.append(
                            int(source_rows_cpu[row, slot_block * BLOCK_SIZE + offset])
                        )
                got = out[row].cpu()
                self.assertEqual(got[: len(kept)].tolist(), kept, f"row={row}")
                self.assertTrue((got[len(kept) :] == 0).all(), f"row={row}")
                self.assertEqual(len(kept), int(row_lens[row]), f"row={row}")


class TestFillDensePageTableRows(CustomTestCase):
    def test_dense_rows_place_visible_drafts_after_prefix(self):
        """The fill popcounts each row's causal prefix of the square, as the row
        planner does; an all-True square (capture seed, padded replay rows) is the
        case where an ungated popcount writes num_draft_tokens slots instead."""
        for square in (
            tree_mask_from_parents([-1, 0, -1]),
            torch.ones((NUM_DRAFT_TOKENS, NUM_DRAFT_TOKENS), dtype=torch.bool),
        ):
            with self.subTest(square=square.tolist()):
                self._assert_dense_fill(square)

    def _assert_dense_fill(self, square):
        dense_len = 32
        seq_lens = [5, 40]
        bs, squares, draft_tree_mask, draft_visible, token_to_bs, token_pos, tokens = (
            _tree_case(seq_lens, square)
        )
        width = max(dense_len, CAPACITY)

        generator = torch.Generator().manual_seed(3)
        page_table_cpu = torch.randint(
            1, 300, (bs, 64), dtype=torch.int32, generator=generator
        )
        original = torch.randint(
            1, 300, (tokens * HEAD_GROUP, width), dtype=torch.int32, generator=generator
        )
        out = original.clone().to(DEVICE)

        fill_dense_page_table_rows(
            page_table=page_table_cpu.to(DEVICE),
            token_to_bs=torch.tensor(token_to_bs, dtype=torch.int32, device=DEVICE),
            token_pos_in_bs=torch.tensor(token_pos, dtype=torch.int32, device=DEVICE),
            prefix_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
            draft_tree_mask=draft_tree_mask,
            out_page_table=out,
            dense_len=dense_len,
            num_draft_tokens=NUM_DRAFT_TOKENS,
            head_group_num=HEAD_GROUP,
        )

        metadata = _verify_metadata(seq_lens, draft_visible, dense_len=dense_len)
        row_lens = metadata.sparse_cache_seqlens_int32
        for token in range(tokens):
            batch = token_to_bs[token]
            prefix = seq_lens[batch]
            draft_idx = token_pos[token] - prefix - 1
            for group in range(HEAD_GROUP):
                row = token * HEAD_GROUP + group
                got = out[row].cpu()
                if token_pos[token] >= dense_len:
                    self.assertTrue(torch.equal(got, original[row]), f"row={row}")
                    continue
                expected = [
                    int(page_table_cpu[batch, col]) * HEAD_GROUP + group
                    for col in range(prefix)
                ]
                for key in range(NUM_DRAFT_TOKENS):
                    if squares[batch][draft_idx, key] and key <= draft_idx:
                        expected.append(
                            int(page_table_cpu[batch, prefix + key]) * HEAD_GROUP
                            + group
                        )
                self.assertEqual(got[: len(expected)].tolist(), expected, f"row={row}")
                self.assertTrue((got[len(expected) :] == 0).all(), f"row={row}")
                self.assertEqual(len(expected), int(row_lens[row]), f"row={row}")


class TestTreeVerifyRecompression(CustomTestCase):
    def test_history_rollback_recompresses_relocated_chunks(self):
        """Chunks past the previous prefix were pooled over tree-order draft slots
        the accept path relocated;
        history_lens = prefix - num_draft_tokens must recompute them,
        the chain default reads them back stale."""
        kernel_size, kernel_stride = 4, 2
        num_draft_tokens = 4
        prefix = 12
        seq_len = prefix + num_draft_tokens
        head_dim = 8

        pool = torch.zeros((64, 1, head_dim), dtype=torch.float32, device=DEVICE)
        pool[:seq_len] = (
            torch.arange(seq_len, dtype=torch.float32, device=DEVICE).view(-1, 1, 1)
            * 10.0
        )
        token_table = torch.arange(seq_len, dtype=torch.int32, device=DEVICE).view(
            1, -1
        )
        num_chunks = (seq_len - kernel_size) // kernel_stride + 1
        compressed_slots = torch.arange(
            32, 32 + num_chunks, dtype=torch.int32, device=DEVICE
        )
        req_to_sparse = compressed_slots.view(1, -1)
        # Chunk 4 spans tokens [8, 12) and crossed the previous round's prefix (10).
        stale_chunk = 4
        oracle = torch.stack(
            [
                pool[chunk * kernel_stride : chunk * kernel_stride + kernel_size]
                .mean(dim=0)
                .squeeze(0)
                for chunk in range(num_chunks)
            ]
        )
        pool[32 : 32 + num_chunks, 0] = oracle

        def compress(history_lens):
            forward_batch = SimpleNamespace(
                batch_size=1,
                seq_lens_cpu=torch.tensor([prefix], dtype=torch.int32),
                req_pool_indices=torch.tensor([0], device=DEVICE),
            )
            base_metadata = SimpleNamespace(
                cu_seqlens_q=torch.tensor(
                    [0, num_draft_tokens], dtype=torch.int32, device=DEVICE
                ),
                cu_seqlens_k=torch.tensor(
                    [0, seq_len], dtype=torch.int32, device=DEVICE
                ),
            )
            k1, _ = _build_k1_k2_compression_metadata(
                req_pool_indices=forward_batch.req_pool_indices,
                base_metadata=base_metadata,
                levels=(
                    CompressLevel(
                        name="k1",
                        kernel_size=kernel_size,
                        kernel_stride=kernel_stride,
                        token_table=req_to_sparse,
                    ),
                    CompressLevel(
                        name="k2",
                        kernel_size=kernel_size * 4,
                        kernel_stride=kernel_stride * 4,
                        token_table=req_to_sparse,
                    ),
                ),
                seq_lens_cpu=forward_batch.seq_lens_cpu + num_draft_tokens,
                history_lens=history_lens,
            )
            full_compressed = torch.zeros(
                (num_chunks, 1, head_dim), dtype=torch.float32, device=DEVICE
            )
            compress_k_core_new(
                full_compressed,
                1,
                pool,
                token_table,
                k1.table,
                k1.cu_new_token_nums,
                k1.history_compress_token_nums,
                k1.cu_total_compress_token_nums,
                kernel_size,
                kernel_stride,
                seq_len,
            )
            return full_compressed.squeeze(1)

        pool[32 + stale_chunk] = 999.0
        got = compress(torch.tensor([prefix - num_draft_tokens], device=DEVICE))
        torch.testing.assert_close(got, oracle, rtol=0, atol=0)
        # The persisted slot is refreshed too,
        # so later rounds read the recomputed value.
        torch.testing.assert_close(
            pool[32 + stale_chunk].squeeze(0), oracle[stale_chunk], rtol=0, atol=0
        )

        pool[32 + stale_chunk] = 999.0
        stale = compress(None)
        self.assertTrue((stale[stale_chunk] == 999.0).all())


if __name__ == "__main__":
    unittest.main()
