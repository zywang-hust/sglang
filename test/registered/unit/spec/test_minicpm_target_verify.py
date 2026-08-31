"""MiniCPM sparse target-verify: row planning, segment layouts, and verify
metadata construction."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.minicpm_fixtures import (
    make_spec_backend,
    make_verify_batch,
    plan_verify,
)
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.minicpm import backend as backend_module
from sglang.srt.layers.attention.minicpm.sparse_utils import (
    _build_sparse_decode_metadata,
    _plan_repeated_segments,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEAD_GROUP = 2
HEADS_PER_GROUP = 16
BLOCK_SIZE = 64
SPARSE_TOPK = 96
CAPACITY = SPARSE_TOPK * BLOCK_SIZE


def _plan(seq_lens, num_draft_tokens, dense_len):
    return plan_verify(
        seq_lens,
        num_draft_tokens,
        head_group_num=HEAD_GROUP,
        heads_per_group=HEADS_PER_GROUP,
        dense_len=dense_len,
        sparse_topk=SPARSE_TOPK,
        block_size=BLOCK_SIZE,
    )


class TestVerifyMetadataClosedForm(CustomTestCase):
    def test_rows_match_decode_metadata_at_each_draft_position(self):
        """Verify row (b, i) must plan exactly like a decode step:
        seq_len = prefix + i + 1 -- draft tokens see the same key set."""
        seq_lens = [100, 6142, 8190, 8192]
        num_draft_tokens = 4
        for dense_len in (8192, 0):
            _, _, metadata = _plan(seq_lens, num_draft_tokens, dense_len)
            row_lens = metadata.sparse_cache_seqlens_int32
            for batch_idx, seq_len in enumerate(seq_lens):
                for draft_idx in range(num_draft_tokens):
                    token_pos = seq_len + draft_idx + 1
                    decode = _build_sparse_decode_metadata(
                        seq_lens_cpu=torch.tensor([token_pos]),
                        base_metadata=SimpleNamespace(
                            cache_seqlens_int32=torch.tensor(
                                [token_pos], dtype=torch.int32
                            ),
                            page_table=torch.zeros((1, token_pos), dtype=torch.int32),
                            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
                        ),
                        head_group_num=HEAD_GROUP,
                        dense_len=dense_len,
                        sparse_topk=SPARSE_TOPK,
                        block_size=BLOCK_SIZE,
                    )
                    row_start = (batch_idx * num_draft_tokens + draft_idx) * HEAD_GROUP
                    self.assertTrue(
                        torch.equal(
                            row_lens[row_start : row_start + HEAD_GROUP],
                            decode.sparse_cache_seqlens_int32,
                        ),
                        f"seq_len={seq_len} draft_idx={draft_idx} "
                        f"dense_len={dense_len}",
                    )

    def test_row_lengths_match_selection_fill(self):
        """Planned sparse row length equals the page-table entries;
        the block selection fills them."""
        num_draft_tokens = 3
        seq_lens = [6141, 6144, 6205, 8189]
        _, _, metadata = _plan(seq_lens, num_draft_tokens, dense_len=0)
        row_lens = metadata.sparse_cache_seqlens_int32
        for batch_idx, seq_len in enumerate(seq_lens):
            for draft_idx in range(num_draft_tokens):
                token_pos = seq_len + draft_idx + 1
                num_blocks = -(-token_pos // BLOCK_SIZE)
                if num_blocks <= SPARSE_TOPK:
                    selected = range(num_blocks)
                else:
                    # Only the forced last block can be partial;
                    # which others score in is irrelevant.
                    selected = [*range(SPARSE_TOPK - 1), num_blocks - 1]
                fill = sum(
                    min(BLOCK_SIZE, token_pos - block * BLOCK_SIZE)
                    for block in selected
                )
                row = (batch_idx * num_draft_tokens + draft_idx) * HEAD_GROUP
                self.assertEqual(int(row_lens[row]), fill)

    def test_topk_selection_uses_decode_form(self):
        """Regression: selection must run the decode form (one row per draft token,
        cache len token_pos - 1);
        the prefill form stops at the last complete block and drops the draft slots."""
        seq_lens = [50, 9000]
        num_draft_tokens = 4
        _, _, metadata = _plan(seq_lens, num_draft_tokens, dense_len=8192)

        num_tokens = len(seq_lens) * num_draft_tokens
        self.assertTrue(
            torch.equal(
                metadata.topk_cu_seqlens_q,
                torch.arange(num_tokens + 1, dtype=torch.int32),
            )
        )
        self.assertEqual(metadata.topk_max_seqlen_q, 1)
        # The decode form stages each verify row's cache at its own causal prefix:
        # seq_len + draft_idx keys.
        self.assertTrue(
            torch.equal(
                metadata.cache_seqlens_int32_stage1,
                torch.tensor(
                    [s + d for s in seq_lens for d in range(num_draft_tokens)],
                    dtype=torch.int32,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.cu_seqlens_q_adjusted,
                torch.arange(num_tokens + 1, dtype=torch.int32) * HEADS_PER_GROUP,
            )
        )


class TestPlanRepeatedSegments(CustomTestCase):
    def test_segments_repeat_per_draft_token(self):
        flat = torch.arange(5, dtype=torch.float32).view(5, 1, 1)
        layout = _plan_repeated_segments(
            torch.tensor([0, 2, 5], dtype=torch.int32), total_tokens=5, repeats=3
        )
        self.assertEqual(layout.cu_seqlens.tolist(), [0, 2, 4, 6, 9, 12, 15])
        want = torch.cat([flat[0:2]] * 3 + [flat[2:5]] * 3)
        self.assertTrue(torch.equal(flat.index_select(0, layout.index), want))


def _verify_forward_batch(seq_lens, num_draft_tokens, spec_info=None):
    forward_batch, base_metadata = make_verify_batch(seq_lens, num_draft_tokens)
    forward_batch.spec_info = spec_info
    forward_batch.req_pool_indices = torch.arange(len(seq_lens), dtype=torch.int64)
    forward_batch.forward_mode = SimpleNamespace(
        is_draft_extend_v2=lambda: False,
        is_idle=lambda: False,
        is_target_verify=lambda: True,
        is_decode_or_idle=lambda: False,
    )
    return forward_batch, base_metadata


def _k1_k2_stub() -> tuple[SimpleNamespace, SimpleNamespace]:
    """The k1/k2 compression stubs shared by the verify tests."""
    k1 = SimpleNamespace(
        cu_seqlens=torch.tensor([0, 1, 3], dtype=torch.int32),
        cu_seqlens_cpu=[0, 1, 3],
    )
    k2 = SimpleNamespace(
        cu_seqlens=torch.tensor([0, 0, 1], dtype=torch.int32),
        cu_seqlens_cpu=[0, 0, 1],
    )
    return k1, k2


class TestBackendVerifyMetadata(CustomTestCase):

    def test_init_forward_metadata_builds_verify_metadata(self):
        num_draft_tokens = 2
        backend, flash_attn_backend = make_spec_backend(
            num_draft_tokens=num_draft_tokens, eagle_topk=1
        )
        forward_batch, base_metadata = _verify_forward_batch([3, 200], num_draft_tokens)
        flash_attn_backend.forward_metadata = base_metadata
        k1, k2 = _k1_k2_stub()

        with patch.object(
            backend_module,
            "_build_k1_k2_compression_metadata",
            return_value=(k1, k2),
        ) as build_k1_k2:
            backend.init_forward_metadata(forward_batch)

        metadata = backend.forward_metadata
        # K1/K2 tables cover prefix + draft tokens.
        kwargs = build_k1_k2.call_args.kwargs
        self.assertTrue(
            torch.equal(
                kwargs["seq_lens_cpu"], forward_batch.seq_lens_cpu + num_draft_tokens
            )
        )
        self.assertIs(kwargs["cu_seqlens_q"], base_metadata.cu_seqlens_q)

        self.assertEqual(metadata.k1_repeat.index.tolist(), [0, 0, 1, 2, 1, 2])
        self.assertEqual(metadata.k1_repeat.cu_seqlens.tolist(), [0, 1, 2, 4, 6])
        self.assertEqual(metadata.k2_repeat.index.tolist(), [0, 0])
        self.assertEqual(metadata.k2_repeat.cu_seqlens.tolist(), [0, 0, 0, 1, 2])

        # dense_len=128: request 0 rows are dense, request 1 rows sparse.
        self.assertEqual(
            metadata.sparse_cache_seqlens_int32.tolist(),
            # Sparse rows clamp to capacity 128 - block 64 + token_pos % 64.
            [4, 5, 73, 74],
        )
        self.assertEqual(
            metadata.verify_dense_mask[:, 0, :].sum(dim=1).tolist(), [4, 5, 0, 0]
        )
        self.assertEqual(metadata.max_seqlen_q_adjusted, backend.heads_per_group)

        prepare_kwargs = backend.attention_adapter.prepare_forward.call_args.kwargs
        self.assertFalse(prepare_kwargs["is_prefill"])

    def test_tree_drafts_raise(self):
        with self.assertRaisesRegex(NotImplementedError, "chain drafts only"):
            make_spec_backend(num_draft_tokens=4, eagle_topk=2)

    def test_draft_extend_raises(self):
        backend, _ = make_spec_backend(num_draft_tokens=4, eagle_topk=1)
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_draft_extend_v2=lambda: True)
        )
        with self.assertRaisesRegex(NotImplementedError, "draft extend"):
            backend.init_forward_metadata(forward_batch)


class TestFusedTopkRepeatGate(CustomTestCase):
    """Regression: the fused top-k scores the k1 level alone,
    so under minicpm_fuse_topk the verify plan leaves the k2 repeat layout unset
    while k1 keeps its per-draft repeats."""

    def test_eager_verify_plan_leaves_k2_repeat_unset(self):
        num_draft_tokens = 2
        with (
            backend_module.envs.SGLANG_MINICPM_FUSE_TOPK.override(True),
            patch.object(
                backend_module.MiniCPMSparseBackend,
                "_get_fused_topk_kernel",
                return_value=None,
            ),
        ):
            backend, flash_attn_backend = make_spec_backend(
                num_draft_tokens=num_draft_tokens, eagle_topk=1
            )
            self.assertTrue(backend.minicpm_fuse_topk)
            forward_batch, base_metadata = _verify_forward_batch(
                [3, 200], num_draft_tokens
            )
            flash_attn_backend.forward_metadata = base_metadata
            k1, k2 = _k1_k2_stub()
            with patch.object(
                backend_module,
                "_build_k1_k2_compression_metadata",
                return_value=(k1, k2),
            ):
                backend.init_forward_metadata(forward_batch)
            metadata = backend.forward_metadata
            self.assertEqual(metadata.k1_repeat.index.tolist(), [0, 0, 1, 2, 1, 2])
            self.assertEqual(metadata.k1_repeat.cu_seqlens.tolist(), [0, 1, 2, 4, 6])
            self.assertIsNone(metadata.k2_repeat)


if __name__ == "__main__":
    unittest.main()
