"""MiniCPM sparse target-verify: row planning, segment layouts,
verify metadata construction, and graph buffers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.minicpm_fixtures import (
    make_spec_backend,
    make_verify_batch,
    tree_mask_from_parents,
    visible_counts,
)
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.minicpm import backend as backend_module
from sglang.srt.layers.attention.minicpm.backend import _copy_dense_page_table
from sglang.srt.layers.attention.minicpm.sparse_utils import (
    MiniCPMSparseMetadata,
    _build_dense_verify_overwrite,
    _build_sparse_decode_metadata,
    _plan_repeated_segments,
    _plan_sparse_verify,
    _plan_uniform_slab_segments,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEAD_GROUP = 2
HEADS_PER_GROUP = 16
BLOCK_SIZE = 64
SPARSE_TOPK = 96
CAPACITY = SPARSE_TOPK * BLOCK_SIZE


def _get_sparse_cache_len(seq_len, sparse_capacity, block_size):
    """Per-element oracle for the sparse-capacity clamp:
    what the vectorized builders apply."""
    if seq_len <= sparse_capacity:
        return seq_len
    remainder = seq_len % block_size
    return (
        sparse_capacity if remainder == 0 else sparse_capacity - block_size + remainder
    )


def _plan(seq_lens, num_draft_tokens, dense_len, num_draft_visible=None):
    forward_batch, base_metadata = make_verify_batch(seq_lens, num_draft_tokens)
    metadata = MiniCPMSparseMetadata(base=base_metadata)
    _plan_sparse_verify(
        forward_batch=forward_batch,
        metadata=metadata,
        head_group_num=HEAD_GROUP,
        heads_per_group=HEADS_PER_GROUP,
        dense_len=dense_len,
        sparse_topk=SPARSE_TOPK,
        block_size=BLOCK_SIZE,
        num_draft_tokens=num_draft_tokens,
        num_draft_visible=num_draft_visible,
    )
    return forward_batch, base_metadata, metadata


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


class TestTreeVerifyMetadata(CustomTestCase):
    def test_chain_mask_degenerates_to_chain_rows(self):
        """A fully-visible tree mask must reproduce the chain builder's row lengths,
        bitwise."""
        num_draft_tokens = 4
        seq_lens = [1, 100, 126, 6141, 6144, 8189]
        bs = len(seq_lens)
        chain_mask = tree_mask_from_parents([-1, 0, 1, 2]).view(-1).repeat(bs)
        counts = visible_counts(
            mask=chain_mask, bs=bs, num_draft_tokens=num_draft_tokens
        )
        for dense_len in (8192, 128, 0):
            _, _, chain = _plan(seq_lens, num_draft_tokens, dense_len)
            _, _, tree = _plan(
                seq_lens, num_draft_tokens, dense_len, num_draft_visible=counts
            )
            self.assertTrue(
                torch.equal(
                    tree.sparse_cache_seqlens_int32,
                    chain.sparse_cache_seqlens_int32,
                ),
                f"dense_len={dense_len}",
            )
            self.assertTrue(
                torch.equal(tree.sparse_cu_seqlens_k, chain.sparse_cu_seqlens_k)
            )
            # The chain form also bounds the pre-compaction fill.
            self.assertTrue(
                torch.equal(
                    tree.verify_source_row_lens,
                    chain.sparse_cache_seqlens_int32,
                )
            )

    def test_tree_rows_subtract_invisible_draft_keys(self):
        """Tree row length = chain closed form -
        (token offset - |ancestors incl. self|)."""
        num_draft_tokens = 5
        parents = [-1, 0, 0, 2, 1]
        mask = tree_mask_from_parents(parents)
        seq_lens = [100, 6141, 6144, 8189]
        bs = len(seq_lens)
        counts = visible_counts(
            mask=mask.view(-1).repeat(bs), bs=bs, num_draft_tokens=num_draft_tokens
        )
        for dense_len in (8192, 0):
            _, _, verify = _plan(
                seq_lens, num_draft_tokens, dense_len, num_draft_visible=counts
            )
            row_lens = verify.sparse_cache_seqlens_int32
            for batch_idx, seq_len in enumerate(seq_lens):
                for draft_idx in range(num_draft_tokens):
                    token_pos = seq_len + draft_idx + 1
                    chain_len = (
                        token_pos
                        if token_pos < dense_len
                        else _get_sparse_cache_len(token_pos, CAPACITY, BLOCK_SIZE)
                    )
                    ancestors = 0
                    node = draft_idx
                    while node != -1:
                        ancestors += 1
                        node = parents[node]
                    expected = chain_len - (draft_idx + 1 - ancestors)
                    row = (batch_idx * num_draft_tokens + draft_idx) * HEAD_GROUP
                    self.assertEqual(
                        int(row_lens[row]),
                        expected,
                        f"seq_len={seq_len} draft_idx={draft_idx} "
                        f"dense_len={dense_len}",
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

    def test_uniform_slab_repeat_matches_packed_repeat(self):
        """Uniform-slab repeat must match the packed repeat on every live prefix,
        with padding past a request's chunks staying -inf."""
        segment_rows, num_draft_tokens = 8, 3
        chunk_lens = [3, 8, 0, 5]
        bs = len(chunk_lens)
        slab = torch.full((bs * segment_rows, 1, 2), float("-inf"))
        packed_segments = []
        for batch_idx, n in enumerate(chunk_lens):
            segment = (
                torch.arange(n * 2, dtype=torch.float32).view(n, 1, 2) + 100 * batch_idx
            )
            slab[batch_idx * segment_rows : batch_idx * segment_rows + n] = segment
            packed_segments.append(segment)
        flat = torch.cat(packed_segments)
        packed_cu = [0]
        for n in chunk_lens:
            packed_cu.append(packed_cu[-1] + n)

        eager_layout = _plan_repeated_segments(
            torch.tensor(packed_cu, dtype=torch.int32),
            total_tokens=packed_cu[-1],
            repeats=num_draft_tokens,
        )
        eager_rep = flat.index_select(0, eager_layout.index)
        layout = _plan_uniform_slab_segments(
            batch_size=bs,
            segment_rows=segment_rows,
            repeats=num_draft_tokens,
            device="cpu",
        )
        rep = slab.index_select(0, layout.index)

        self.assertTrue(
            torch.equal(
                layout.cu_seqlens,
                torch.arange(bs * num_draft_tokens + 1, dtype=torch.int32)
                * segment_rows,
            )
        )
        for row in range(bs * num_draft_tokens):
            n = chunk_lens[row // num_draft_tokens]
            live = rep[row * segment_rows : row * segment_rows + n]
            start = int(eager_layout.cu_seqlens[row])
            eager_live = eager_rep[start : start + n]
            self.assertTrue(torch.equal(live, eager_live), row)
            padding = rep[row * segment_rows + n : (row + 1) * segment_rows]
            self.assertTrue(torch.isinf(padding).all(), row)


def _verify_forward_batch(seq_lens, num_draft_tokens, spec_info=None, tree_base=False):
    forward_batch, base_metadata = make_verify_batch(seq_lens, num_draft_tokens)
    forward_batch.spec_info = spec_info
    forward_batch.req_pool_indices = torch.arange(len(seq_lens), dtype=torch.int64)
    forward_batch.forward_mode = SimpleNamespace(
        is_draft_extend_v2=lambda: False,
        is_idle=lambda: False,
        is_target_verify=lambda: True,
        is_decode_or_idle=lambda: False,
    )
    if tree_base:
        # Mimic FA's topk>1 two-phase layout:
        # prefix-only cache lengths and page-table rows.
        prefix = torch.tensor(seq_lens, dtype=torch.int32)
        base_metadata.cache_seqlens_int32 = prefix
        base_metadata.max_seq_len_k = int(prefix.max())
        base_metadata.cu_seqlens_k = F.pad(
            torch.cumsum(prefix, dim=0, dtype=torch.int32), (1, 0)
        )
        base_metadata.page_table = torch.zeros(
            (len(seq_lens), int(prefix.max())), dtype=torch.int32
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
        # K1/K2 tables cover prefix + draft tokens;
        # chain rounds keep the default reuse boundary.
        kwargs = build_k1_k2.call_args.kwargs
        self.assertTrue(
            torch.equal(
                kwargs["seq_lens_cpu"], forward_batch.seq_lens_cpu + num_draft_tokens
            )
        )
        self.assertIsNone(kwargs["history_lens"])

        self.assertEqual(metadata.k1_repeat.index.tolist(), [0, 0, 1, 2, 1, 2])
        self.assertEqual(metadata.k1_repeat.cu_seqlens.tolist(), [0, 1, 2, 4, 6])
        self.assertEqual(metadata.k2_repeat.index.tolist(), [0, 0])
        self.assertEqual(metadata.k2_repeat.cu_seqlens.tolist(), [0, 0, 0, 1, 2])

        # dense_len=128: request 0 rows are dense, request 1 rows sparse.
        self.assertEqual(
            metadata.sparse_cache_seqlens_int32.tolist(),
            [
                4,
                5,
                _get_sparse_cache_len(201, 128, 64),
                _get_sparse_cache_len(202, 128, 64),
            ],
        )
        self.assertEqual(
            metadata.verify_dense_mask[:, 0, :].sum(dim=1).tolist(), [4, 5, 0, 0]
        )
        self.assertEqual(metadata.max_seqlen_q_adjusted, backend.heads_per_group)

        prepare_kwargs = backend.attention_adapter.prepare_forward.call_args.kwargs
        self.assertFalse(prepare_kwargs["is_prefill"])

    def test_tree_drafts_build_popcount_rows(self):
        """eagle_topk > 1 must stage the tree mask and plan popcount row lengths."""
        num_draft_tokens = 3
        backend, flash_attn_backend = make_spec_backend(
            num_draft_tokens=num_draft_tokens, eagle_topk=2
        )
        seq_lens = [3, 200]
        bs = len(seq_lens)
        # Root plus two sibling branches: node 2 hides node 1.
        mask = tree_mask_from_parents([-1, 0, 0]).view(-1).repeat(bs)
        custom_mask = torch.ones(1, dtype=torch.bool)
        forward_batch, base_metadata = _verify_forward_batch(
            seq_lens,
            num_draft_tokens,
            spec_info=SimpleNamespace(custom_mask=custom_mask),
            tree_base=True,
        )
        flash_attn_backend.forward_metadata = base_metadata

        def _stage_mask(*, out, num_visible_out, **kwargs):
            out.copy_(mask)
            num_visible_out.copy_(
                visible_counts(mask=mask, bs=bs, num_draft_tokens=num_draft_tokens)
            )

        with (
            patch.object(
                backend_module,
                "_build_k1_k2_compression_metadata",
                return_value=_k1_k2_stub(),
            ) as build_k1_k2,
            patch.object(
                backend_module, "copy_eagle_draft_tree_mask", side_effect=_stage_mask
            ) as copy_mask,
        ):
            backend.init_forward_metadata(forward_batch)

        copy_kwargs = copy_mask.call_args.kwargs
        self.assertIs(copy_kwargs["custom_mask"], custom_mask)
        self.assertIs(copy_kwargs["seq_lens"], forward_batch.seq_lens)

        # Tree rounds roll the compression reuse boundary back by num_draft_tokens.
        history_lens = build_k1_k2.call_args.kwargs["history_lens"]
        self.assertEqual(history_lens.tolist(), [0, 200 - num_draft_tokens])

        metadata = backend.forward_metadata
        self.assertTrue(torch.equal(metadata.verify_draft_tree_mask, mask))
        sc = [_get_sparse_cache_len(200 + off, 128, 64) for off in (1, 2, 3)]
        self.assertEqual(
            metadata.sparse_cache_seqlens_int32.tolist(),
            [4, 5, 6 - 1, sc[0], sc[1], sc[2] - 1],
        )
        self.assertEqual(
            metadata.verify_source_row_lens.tolist(),
            [4, 5, 6, sc[0], sc[1], sc[2]],
        )
        self.assertTrue(
            torch.equal(
                metadata.verify_prefix_lens,
                torch.tensor(seq_lens, dtype=torch.int32),
            )
        )

        # Regression: FA's prefix-only tree metadata, consumed verbatim,
        # gates every draft slot out of the fill;
        # the backend must rebuild the uniform prefix + num_draft_tokens geometry.
        base = metadata.base
        self.assertEqual(
            base.cache_seqlens_int32.tolist(),
            [s + num_draft_tokens for s in seq_lens],
        )
        self.assertEqual(base.max_seq_len_k, max(seq_lens) + num_draft_tokens)
        self.assertTrue(
            torch.equal(
                base.cu_seqlens_k,
                F.pad(
                    torch.cumsum(base.cache_seqlens_int32, dim=0, dtype=torch.int32),
                    (1, 0),
                ),
            )
        )
        req_to_token = backend.req_to_token_pool.req_to_token
        self.assertEqual(
            base.page_table.shape,
            (len(seq_lens), max(seq_lens) + num_draft_tokens),
        )
        self.assertTrue(
            torch.equal(
                base.page_table,
                req_to_token[
                    forward_batch.req_pool_indices,
                    : max(seq_lens) + num_draft_tokens,
                ],
            )
        )
        self.assertIs(metadata.seqlen_k_sparse_bs_tensor, base.cache_seqlens_int32)

    def test_draft_extend_raises(self):
        backend, _ = make_spec_backend(num_draft_tokens=4, eagle_topk=1)
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_draft_extend_v2=lambda: True)
        )
        with self.assertRaisesRegex(NotImplementedError, "draft extend"):
            backend.init_forward_metadata(forward_batch)


class TestVerifyGraphBuffers(CustomTestCase):
    def test_target_only_graph_state_skips_tree_buffers(self):
        """Regression: a target-only server has eagle_topk None;
        graph-buffer sizing must treat it as chain drafts instead of comparing None."""
        backend, _ = make_spec_backend(num_draft_tokens=None, eagle_topk=None)
        backend.init_cuda_graph_state(max_bs=2, max_num_tokens=2)
        self.assertNotIn("verify_draft_tree_mask", backend.decode_cuda_graph_metadata)

    def test_dense_overwrite_matches_python_loop(self):
        """The masked overwrite must write, exactly,
        what the eager _copy_dense_page_table loop writes and nothing else."""
        num_draft_tokens = 3
        backend, _ = make_spec_backend(
            num_draft_tokens=num_draft_tokens, eagle_topk=1, num_kv_heads=2
        )
        hg = backend.head_group_num
        seq_lens = [60, 126, 500]
        bs = len(seq_lens)
        width = max(backend.dense_len, backend.num_sparse_topk_tokens)
        generator = torch.Generator().manual_seed(0)
        page_table = torch.randint(
            0, 500, (bs, 1024), dtype=torch.int32, generator=generator
        )
        original = torch.randint(
            0,
            500,
            (bs * num_draft_tokens * hg, width),
            dtype=torch.int32,
            generator=generator,
        )
        token_pos_in_bs = torch.tensor(
            [s + d + 1 for s in seq_lens for d in range(num_draft_tokens)],
            dtype=torch.int32,
        )
        token_to_bs = torch.repeat_interleave(
            torch.arange(bs, dtype=torch.int32), num_draft_tokens
        )
        dense_rows, dense_mask = _build_dense_verify_overwrite(
            token_pos_in_bs,
            token_to_bs,
            page_table,
            dense_len=backend.dense_len,
            head_group_num=hg,
        )
        metadata = SimpleNamespace(
            sparse_page_table=original.clone(),
            verify_dense_rows=dense_rows,
            verify_dense_mask=dense_mask,
        )
        reference = original.clone()
        for batch_idx, seq_len in enumerate(seq_lens):
            for draft_idx in range(num_draft_tokens):
                kv_len = seq_len + draft_idx + 1
                if kv_len < backend.dense_len:
                    _copy_dense_page_table(
                        reference,
                        (batch_idx * num_draft_tokens + draft_idx) * hg,
                        page_table,
                        batch_idx,
                        kv_len,
                        hg,
                    )

        backend._overwrite_dense_verify_rows(metadata)

        self.assertTrue(torch.equal(metadata.sparse_page_table, reference))
        # The batch spans both regimes, so some rows must actually change.
        self.assertFalse(torch.equal(reference, original))


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
