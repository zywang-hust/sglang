"""Pin the MiniCPM verify CUDA-graph capture seed and replay refresh bitwise;
the eager sparse-verify builders are the reference."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.minicpm_fixtures import (
    make_replay_backend,
    make_verify_batch,
    pack_custom_mask,
    plan_verify,
    tree_mask_from_parents,
    visible_counts,
)
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.layers.attention.minicpm.backend import MiniCPMSparseBackend
from sglang.srt.layers.attention.minicpm.sparse_utils import (
    MiniCPMSparseMetadata,
    _build_k1_k2_compression_metadata,
    _build_sparse_verify_rows,
    _build_tree_verify_base_geometry,
    _plan_repeated_segments,
)

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

DEVICE = "cuda"


def _eager_verify(backend, seq_lens, num_draft_tokens, num_draft_visible=None):
    """Eager-builder oracle for the real (unpadded) batch."""
    _, _, metadata = plan_verify(
        seq_lens,
        num_draft_tokens,
        head_group_num=backend.head_group_num,
        heads_per_group=backend.heads_per_group,
        dense_len=backend.dense_len,
        sparse_topk=backend.sparse_topk,
        block_size=backend.block_size,
        num_draft_visible=num_draft_visible,
        device=DEVICE,
    )
    return metadata


def _replay_verify_graph(
    backend,
    base,
    *,
    bs,
    real_bs,
    padded_seq_lens,
    req_pool_indices,
    spec_info,
):
    """Bind the replay metadata and run the verify replay for the padded batch;
    the caller controls what the base buffers hold in between."""
    forward_batch = SimpleNamespace(
        batch_size=bs,
        num_padding=bs - real_bs,
        seq_lens=torch.tensor(padded_seq_lens, dtype=torch.int32, device=DEVICE),
        seq_lens_cpu=torch.tensor(padded_seq_lens, dtype=torch.int32),
        req_pool_indices=req_pool_indices,
        spec_info=spec_info,
    )
    metadata = MiniCPMSparseMetadata(base=base)
    backend._bind_sparse_verify_graph_metadata(
        forward_batch, metadata, in_capture=False
    )
    backend._replay_sparse_verify_graph_metadata(forward_batch, metadata)
    return forward_batch, metadata


def _capture_seed(backend, base, *, bs):
    """Bind a capture-seed metadata for a spec_info=None batch of bs rows;
    returns (metadata, forward_batch)."""
    forward_batch = SimpleNamespace(batch_size=bs, spec_info=None)
    metadata = MiniCPMSparseMetadata(base=base)
    backend._bind_sparse_verify_graph_metadata(forward_batch, metadata, in_capture=True)
    return metadata, forward_batch


def _capture_seed_chain_rows(backend, num_draft_tokens):
    """The chain closed-form row lengths at the capture seed geometry."""
    _, chain_rows = _build_sparse_verify_rows(
        torch.full((2,), backend.config_dense_len, dtype=torch.int32, device=DEVICE),
        head_group_num=backend.head_group_num,
        dense_len=backend.dense_len,
        sparse_topk=backend.sparse_topk,
        block_size=backend.block_size,
        num_draft_tokens=num_draft_tokens,
    )
    return chain_rows


class TestVerifyGraphReplay(CustomTestCase):
    def _assert_replay_rebuilds_base_geometry(
        self, backend, base, forward_batch, padded_seq_lens, num_draft_tokens
    ):
        """The replay's in-place rebuild must reproduce the eager tree base builder,
        field for field; nothing else holds the two closed forms equal."""
        expected = _build_tree_verify_base_geometry(
            forward_batch,
            req_to_token=backend.req_to_token_pool.req_to_token,
            num_draft_tokens=num_draft_tokens,
            eager_max_k=max(padded_seq_lens),
        )
        self.assertTrue(
            torch.equal(base.cache_seqlens_int32, expected.cache_seqlens_int32)
        )
        self.assertTrue(torch.equal(base.cu_seqlens_k, expected.cu_seqlens_k))
        # The graph page table is max_context_len wide; only each row's live
        # prefix + draft span is written.
        for row, seq_len in enumerate(padded_seq_lens):
            width = seq_len + num_draft_tokens
            self.assertTrue(
                torch.equal(
                    base.page_table[row, :width], expected.page_table[row, :width]
                ),
                row,
            )

    def _assert_level_buffers_match_eager_builder(
        self,
        backend,
        metadata,
        forward_batch,
        real_seq_lens,
        num_draft_tokens,
        history_lens=None,
    ):
        """K1/K2 level buffers hold the eager compression builder's output for the
        real batch, with flat cumsum tails and zero history on the padded rows."""
        real_bs = len(real_seq_lens)
        _, eager_base = make_verify_batch(
            real_seq_lens, num_draft_tokens, device=DEVICE
        )
        expected = _build_k1_k2_compression_metadata(
            req_pool_indices=forward_batch.req_pool_indices[:real_bs],
            base_metadata=eager_base,
            levels=backend.compress_levels,
            seq_lens_cpu=torch.tensor(real_seq_lens, dtype=torch.int32)
            + num_draft_tokens,
            history_lens=history_lens,
        )
        for name, want in zip(("k1", "k2"), expected):
            level = getattr(metadata, name)
            for field in ("cu_seqlens", "cu_new_token_nums"):
                got, exp = getattr(level, field), getattr(want, field)
                self.assertTrue(torch.equal(got[: real_bs + 1], exp), f"{name}.{field}")
                self.assertTrue(
                    (got[real_bs + 1 :] == exp[-1]).all(), f"{name}.{field}"
                )
            self.assertTrue(
                torch.equal(
                    level.history_compress_token_nums[:real_bs],
                    want.history_compress_token_nums,
                ),
                name,
            )
            self.assertTrue(
                (level.history_compress_token_nums[real_bs:] == 0).all(), name
            )
            self.assertTrue(torch.equal(level.table[:real_bs], want.table), name)
        return expected

    def _assert_repeat_layouts_match_eager(
        self, backend, metadata, eager_levels, real_bs, num_draft_tokens
    ):
        """The retained repeat layouts gather, from the slab-laid compression
        buffers, the chunk rows the eager planner gathers from packed ones;
        the row bounds are equal and the padded tokens have empty rows."""
        num_real_tokens = real_bs * num_draft_tokens
        for name, eager_level in zip(("k1", "k2"), eager_levels):
            if name not in backend.verify_repeat_levels:
                continue
            level = getattr(metadata, name)
            eager_layout = _plan_repeated_segments(
                eager_level.cu_seqlens,
                total_tokens=eager_level.cu_seqlens_cpu[-1],
                repeats=num_draft_tokens,
            )
            retained = backend.decode_cuda_graph_metadata[f"{name}_repeat"]
            padded_tokens = level.cu_seqlens.numel() - 1
            padded_tokens *= num_draft_tokens
            cu_seqlens = retained.cu_seqlens[: padded_tokens + 1]
            self.assertTrue(
                torch.equal(cu_seqlens[: num_real_tokens + 1], eager_layout.cu_seqlens),
                name,
            )
            self.assertTrue(
                (
                    cu_seqlens[num_real_tokens + 1 :] == eager_layout.cu_seqlens[-1]
                ).all(),
                name,
            )
            # Lay the packed chunk rows out at the slab offsets the compression
            # kernel writes to, then gather both ways.
            packed_cu = eager_level.cu_seqlens
            device = retained.index.device
            packed = torch.arange(
                int(packed_cu[-1]), dtype=torch.float32, device=device
            )
            slab = torch.full(
                (int(level.cu_total_compress_token_nums[-1]),), -1.0, device=device
            )
            for batch_idx in range(real_bs):
                start = int(level.cu_total_compress_token_nums[batch_idx])
                rows = packed[packed_cu[batch_idx] : packed_cu[batch_idx + 1]]
                slab[start : start + rows.numel()] = rows
            gathered = slab[retained.index[: eager_layout.index.numel()]]
            self.assertTrue(torch.equal(gathered, packed[eager_layout.index]), name)

    def _assert_capture_frozen_aranges_intact(
        self, backend, metadata, bs, num_draft_tokens
    ):
        """Replay refreshes the row buffers;
        the capture-frozen selection arange tensors must stay intact."""
        self.assertTrue(
            torch.equal(
                metadata.topk_cu_seqlens_q,
                torch.arange(
                    bs * num_draft_tokens + 1, dtype=torch.int32, device=DEVICE
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.cu_seqlens_q_adjusted,
                torch.arange(
                    bs * num_draft_tokens + 1, dtype=torch.int32, device=DEVICE
                )
                * backend.heads_per_group,
            )
        )

    def _assert_uniform_slab_offsets_survive_replay(self, backend, metadata, bs):
        """Capture-seeded uniform slab offsets must survive replay;
        the static repeat index depends on them."""
        for name, kernel_stride in (
            ("k1", backend.k1_kernel_stride),
            ("k2", backend.k2_kernel_stride),
        ):
            self._assert_slab_offsets(
                level=getattr(metadata, name),
                backend=backend,
                bs=bs,
                name=name,
                kernel_stride=kernel_stride,
            )

    def _assert_slab_offsets(self, level, backend, bs, name, kernel_stride):
        """cu_total_compress_token_nums keeps the capture-seeded,
        uniform slab offsets."""
        self.assertTrue(
            torch.equal(
                level.cu_total_compress_token_nums,
                torch.arange(bs + 1, dtype=torch.int32, device=DEVICE)
                * (backend.max_context_len // kernel_stride),
            ),
            name,
        )

    def _assert_replay_rows(
        self, metadata, eager, real_bs, num_draft_tokens, head_group_num
    ):
        """Replay row lengths match the eager builder, padded rows zeroed;
        returns (real_rows, row_lens) for the follow-up assertions."""
        real_rows = real_bs * num_draft_tokens * head_group_num
        row_lens = metadata.sparse_cache_seqlens_int32
        self.assertTrue(
            torch.equal(row_lens[:real_rows], eager.sparse_cache_seqlens_int32)
        )
        self.assertTrue((row_lens[real_rows:] == 0).all())
        return real_rows, row_lens

    def _assert_row_lens_cumsum(self, metadata, row_lens):
        """The replayed sparse cu_seqlens_k is the cumsum of the row lengths."""
        self.assertTrue(
            torch.equal(
                metadata.sparse_cu_seqlens_k,
                F.pad(torch.cumsum(row_lens, dim=0, dtype=torch.int32), (1, 0)),
            )
        )

    def test_capture_seeds_consistent_plan(self):
        """Capture seed must be self-consistent (cu_seqlens_k == cumsum of row lengths,
        all rows live); otherwise the captured fill disagrees with the capture plan."""
        num_draft_tokens = 2
        # num_kv_heads=2 makes sparse_rows outnumber the verify tokens, so
        # the row-width and token-width binds below cannot swap silently.
        backend, base = make_replay_backend(
            [1, 1], num_draft_tokens, num_kv_heads=2, max_bs=2, device=DEVICE
        )
        metadata, forward_batch = _capture_seed(backend, base, bs=2)
        self.assertTrue((metadata.sparse_cache_seqlens_int32 > 0).all())
        self.assertTrue(
            torch.equal(
                metadata.sparse_cu_seqlens_k,
                F.pad(
                    torch.cumsum(
                        metadata.sparse_cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                ),
            )
        )
        assume_kv_len = backend.config_dense_len + num_draft_tokens
        self.assertTrue(
            torch.equal(
                base.cu_seqlens_k,
                torch.arange(3, dtype=torch.int32, device=DEVICE) * assume_kv_len,
            )
        )
        # Magnitude pin:
        # the seeded rows must equal the chain closed form at the capture seed geometry
        # -- a constant seed would pass every check above.
        chain_rows = _capture_seed_chain_rows(backend, num_draft_tokens)
        self.assertTrue(torch.equal(metadata.sparse_cache_seqlens_int32, chain_rows))
        # The decode-form selection geometry the captured plan freezes:
        # prefix-stable arange buffers bound at their verify-width slice,
        # with the head-group expansion applied to the adjusted one.
        arange = torch.arange(
            2 * num_draft_tokens + 1, dtype=torch.int32, device=DEVICE
        )
        self.assertTrue(torch.equal(metadata.topk_cu_seqlens_q, arange))
        self.assertTrue(
            torch.equal(
                metadata.cu_seqlens_q_adjusted, arange * backend.heads_per_group
            )
        )
        # Frozen across capture and replay: the captured selection recorded its
        # cache_lens branch from this bound, not from a per-batch maximum.
        self.assertEqual(metadata.topk_max_seqlen_k, backend.max_context_len)
        self.assertEqual(metadata.max_seqlen_q_adjusted, backend.heads_per_group)
        # Row width: the row-count prefix sum is the full sparse-row arange,
        # not the token-width slice.
        sparse_rows = (
            forward_batch.batch_size * num_draft_tokens * backend.head_group_num
        )
        self.assertTrue(
            torch.equal(
                metadata.sparse_cu_seqlens_q,
                torch.arange(sparse_rows + 1, dtype=torch.int32, device=DEVICE),
            )
        )

    def test_capture_binds_retained_repeat_layouts(self):
        """Regression: the captured gather records the repeat index's address,
        so binding must hand out prefix views of buffers init_cuda_graph_state retains,
        never per-capture allocations."""
        num_draft_tokens = 2
        bs = 2
        backend, base = make_replay_backend(
            [1] * bs, num_draft_tokens, max_bs=4, device=DEVICE
        )
        metadata, _ = _capture_seed(backend, base, bs=bs)
        for name in ("k1", "k2"):
            layout = getattr(metadata, f"{name}_repeat")
            retained = backend.decode_cuda_graph_metadata[f"{name}_repeat"]
            self.assertEqual(layout.index.data_ptr(), retained.index.data_ptr())
            self.assertEqual(
                layout.cu_seqlens.data_ptr(), retained.cu_seqlens.data_ptr()
            )

    def test_capture_skips_k2_repeat_binding_under_fused_topk(self):
        """Regression:
        under minicpm_fuse_topk the captured gather reads no k2 repeat layout,
        so capture must skip the k2 binding;
        k1 still binds prefix views of the retained layout."""
        num_draft_tokens = 2
        with (
            envs.SGLANG_MINICPM_FUSE_TOPK.override(True),
            patch.object(
                MiniCPMSparseBackend,
                "_get_fused_topk_kernel",
                return_value=None,
            ),
        ):
            backend, base = make_replay_backend(
                [1, 1], num_draft_tokens, max_bs=4, device=DEVICE
            )
            metadata = MiniCPMSparseMetadata(base=base)
            backend._bind_sparse_verify_graph_metadata(
                SimpleNamespace(batch_size=2, spec_info=None),
                metadata,
                in_capture=True,
            )
            retained = backend.decode_cuda_graph_metadata["k1_repeat"]
            self.assertEqual(
                metadata.k1_repeat.index.data_ptr(), retained.index.data_ptr()
            )
            self.assertIsNone(metadata.k2_repeat)
            self.assertNotIn("k2_repeat", backend.decode_cuda_graph_metadata)

    def test_capture_compression_view_covers_retained_slab(self):
        """Regression: the captured gather reads the repeat layout's index,
        which addresses the slab offsets the compression kernel writes at,
        so the compression view sized from cu_seqlens_cpu must cover the slab
        -- a dense-window seed lets the captured index_select read past the view."""
        num_draft_tokens = 2
        backend, base = make_replay_backend(
            [1] * 4, num_draft_tokens, max_bs=4, device=DEVICE
        )
        _capture_seed(backend, base, bs=4)
        # A smaller capture reuses the retained index after a larger one wrote
        # it; the gather reads the whole smaller view, tail included.
        _, small_base = make_verify_batch([1, 1], num_draft_tokens, device=DEVICE)
        small_base.max_seq_len_q = num_draft_tokens
        metadata, _ = _capture_seed(backend, small_base, bs=2)
        for name in ("k1", "k2"):
            level = getattr(metadata, name)
            self.assertLess(
                int(getattr(metadata, f"{name}_repeat").index.max()),
                level.cu_seqlens_cpu[-1],
            )

    def test_replay_refreshes_static_buffers_to_eager_geometry(self):
        """Replay must leave the static buffers,
        holding the eager builder's geometry for the real batch, padded rows zeroed,
        and cu_seqlens_k == cumsum(row lengths)."""
        num_draft_tokens = 2
        # 190 puts a token position on a block multiple past the sparse capacity,
        # the clamp's mod == 0 arm; the others cover identity, dense and remainder.
        real_seq_lens = [100, 130, 126, 190]
        padded_seq_lens = [*real_seq_lens, 1]
        bs, real_bs = len(padded_seq_lens), len(real_seq_lens)
        backend, base = make_replay_backend(
            padded_seq_lens, num_draft_tokens, max_bs=5, device=DEVICE
        )
        head_group_num = backend.head_group_num

        capture_batch = SimpleNamespace(batch_size=bs, spec_info=None)
        backend._bind_sparse_verify_graph_metadata(
            capture_batch, MiniCPMSparseMetadata(base=base), in_capture=True
        )
        # The FlashAttention replay refreshes the base buffers;
        # the capture seeding overwrote them.
        base.cache_seqlens_int32.copy_(
            torch.tensor(padded_seq_lens, dtype=torch.int32, device=DEVICE)
            + num_draft_tokens
        )
        base.cu_seqlens_k.copy_(
            F.pad(
                torch.cumsum(base.cache_seqlens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
        )

        forward_batch, metadata = _replay_verify_graph(
            backend,
            base,
            bs=bs,
            real_bs=real_bs,
            padded_seq_lens=padded_seq_lens,
            req_pool_indices=torch.tensor([2, 5, 1, 4, 0], device=DEVICE),
            spec_info=None,
        )

        eager = _eager_verify(
            backend, seq_lens=real_seq_lens, num_draft_tokens=num_draft_tokens
        )
        _, row_lens = self._assert_replay_rows(
            metadata=metadata,
            eager=eager,
            real_bs=real_bs,
            num_draft_tokens=num_draft_tokens,
            head_group_num=head_group_num,
        )
        self._assert_row_lens_cumsum(metadata, row_lens)
        self.assertTrue(
            torch.equal(
                metadata.token_pos_in_bs[: real_bs * num_draft_tokens],
                eager.token_pos_in_bs,
            )
        )
        num_real_tokens = real_bs * num_draft_tokens
        self.assertTrue(
            torch.equal(
                metadata.cache_seqlens_int32_stage1[:num_real_tokens],
                eager.cache_seqlens_int32_stage1,
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.verify_dense_mask[:num_real_tokens], eager.verify_dense_mask
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.verify_dense_rows[:num_real_tokens], eager.verify_dense_rows
            )
        )
        self._assert_capture_frozen_aranges_intact(
            backend=backend, metadata=metadata, bs=bs, num_draft_tokens=num_draft_tokens
        )
        self.assertEqual(metadata.topk_max_seqlen_k, backend.max_context_len)

        self._assert_uniform_slab_offsets_survive_replay(
            backend=backend, metadata=metadata, bs=bs
        )
        eager_levels = self._assert_level_buffers_match_eager_builder(
            backend, metadata, forward_batch, real_seq_lens, num_draft_tokens
        )
        self._assert_repeat_layouts_match_eager(
            backend, metadata, eager_levels, real_bs, num_draft_tokens
        )

    def test_all_padding_replay_zeroes_compression_buffers(self):
        """Bug regression:
        an all-padding verify replay must zero the k1/k2 per-round compression lengths
        and the repeat row bounds -- stale counts otherwise feed the captured
        compression and selection kernels -- while cu_total_compress_token_nums
        keeps the capture-time slab offsets the compression kernel writes at."""
        num_draft_tokens = 2
        bs = 2
        backend, base = make_replay_backend(
            [1] * bs, num_draft_tokens, max_bs=bs, device=DEVICE
        )
        backend._bind_sparse_verify_graph_metadata(
            SimpleNamespace(batch_size=bs, spec_info=None),
            MiniCPMSparseMetadata(base=base),
            in_capture=True,
        )
        forward_batch = SimpleNamespace(
            batch_size=bs,
            num_padding=bs,
            seq_lens=torch.ones(bs, dtype=torch.int32, device=DEVICE),
            seq_lens_cpu=torch.ones(bs, dtype=torch.int32),
            req_pool_indices=torch.arange(bs, dtype=torch.int64, device=DEVICE),
            spec_info=None,
        )
        metadata = MiniCPMSparseMetadata(base=base)
        backend._bind_sparse_verify_graph_metadata(
            forward_batch, metadata, in_capture=False
        )
        refreshed_fields = (
            "history_compress_token_nums",
            "cu_seqlens",
            "cu_new_token_nums",
        )
        # Dirty the refreshed fields as a previous non-padded replay would.
        for name in ("k1", "k2"):
            level = getattr(metadata, name)
            for field in refreshed_fields:
                getattr(level, field).fill_(7)
            backend.decode_cuda_graph_metadata[f"{name}_repeat"].cu_seqlens.fill_(7)

        backend._replay_sparse_verify_graph_metadata(forward_batch, metadata)

        self.assertTrue((metadata.sparse_cache_seqlens_int32 == 0).all())
        self.assertTrue((metadata.sparse_cu_seqlens_k == 0).all())
        self.assertTrue((metadata.cache_seqlens_int32_stage1 == 0).all())
        for name, kernel_stride in (
            ("k1", backend.k1_kernel_stride),
            ("k2", backend.k2_kernel_stride),
        ):
            level = getattr(metadata, name)
            for field in refreshed_fields:
                buf = getattr(level, field)
                self.assertTrue(
                    (buf == 0).all(), f"{name}.{field} stale: {buf.tolist()}"
                )
            self._assert_slab_offsets(
                level=level,
                backend=backend,
                bs=bs,
                name=name,
                kernel_stride=kernel_stride,
            )
            repeat = backend.decode_cuda_graph_metadata[f"{name}_repeat"]
            self.assertTrue(
                (repeat.cu_seqlens[: bs * num_draft_tokens + 1] == 0).all(), name
            )

    def test_graph_tree_capture_seeds_chain_degenerate_plan(self):
        """Tree capture must seed an all-True square
        so the seeded plan rows equal the chain closed form."""
        num_draft_tokens = 2
        backend, base = make_replay_backend(
            [1, 1], num_draft_tokens, max_bs=2, eagle_topk=2, device=DEVICE
        )
        # Dirty the static mask as a previous replay would.
        backend.decode_cuda_graph_metadata["verify_draft_tree_mask"].fill_(False)
        metadata, _ = _capture_seed(backend, base, bs=2)
        self.assertTrue(metadata.verify_draft_tree_mask.all())
        chain_rows = _capture_seed_chain_rows(backend, num_draft_tokens)
        self.assertTrue(torch.equal(metadata.sparse_cache_seqlens_int32, chain_rows))
        self.assertTrue(torch.equal(metadata.verify_source_row_lens, chain_rows))

    def test_graph_tree_replay_matches_eager_tree_builder(self):
        """Tree replay must match the eager tree builder: popcount row lengths,
        chain source lengths, base geometry rebuilt in place,
        reuse boundary rolled back by num_draft_tokens."""
        num_draft_tokens = 3
        # 200 clamps every draft row past the sparse capacity;
        # the other two stay in the identity regime.
        real_seq_lens = [100, 130, 200]
        padded_seq_lens = [*real_seq_lens, 1]
        bs, real_bs = len(padded_seq_lens), len(real_seq_lens)
        backend, base = make_replay_backend(
            padded_seq_lens, num_draft_tokens, max_bs=4, eagle_topk=2, device=DEVICE
        )
        head_group_num = backend.head_group_num
        # Node 2 hides node 1 in both real requests.
        square = tree_mask_from_parents([-1, 0, 0])
        squares = [square] * real_bs
        custom_mask = pack_custom_mask(real_seq_lens, squares, device=DEVICE)

        capture_batch = SimpleNamespace(batch_size=bs, spec_info=None)
        backend._bind_sparse_verify_graph_metadata(
            capture_batch, MiniCPMSparseMetadata(base=base), in_capture=True
        )
        # FA's topk>1 replay refresh leaves the two-phase tree layout in the
        # static base buffers: prefix-only cache lengths and page-table rows.
        prefix = torch.tensor(padded_seq_lens, dtype=torch.int32, device=DEVICE)
        base.cache_seqlens_int32.copy_(prefix)
        base.cu_seqlens_k.copy_(
            F.pad(torch.cumsum(prefix, dim=0, dtype=torch.int32), (1, 0))
        )
        base.page_table.zero_()

        forward_batch, metadata = _replay_verify_graph(
            backend,
            base,
            bs=bs,
            real_bs=real_bs,
            padded_seq_lens=padded_seq_lens,
            req_pool_indices=torch.tensor([2, 5, 1, 4], device=DEVICE),
            spec_info=SimpleNamespace(custom_mask=custom_mask),
        )

        self._assert_replay_rebuilds_base_geometry(
            backend=backend,
            base=base,
            forward_batch=forward_batch,
            padded_seq_lens=padded_seq_lens,
            num_draft_tokens=num_draft_tokens,
        )

        staged = metadata.verify_draft_tree_mask.view(
            bs, num_draft_tokens, num_draft_tokens
        )
        for b in range(real_bs):
            self.assertTrue(torch.equal(staged[b].cpu(), square), b)
        self.assertTrue(staged[real_bs:].all())

        counts = visible_counts(
            mask=metadata.verify_draft_tree_mask[
                : real_bs * num_draft_tokens * num_draft_tokens
            ],
            bs=real_bs,
            num_draft_tokens=num_draft_tokens,
        )
        eager = _eager_verify(
            backend,
            seq_lens=real_seq_lens,
            num_draft_tokens=num_draft_tokens,
            num_draft_visible=counts,
        )
        real_rows, row_lens = self._assert_replay_rows(
            metadata=metadata,
            eager=eager,
            real_bs=real_bs,
            num_draft_tokens=num_draft_tokens,
            head_group_num=head_group_num,
        )
        self.assertTrue(
            torch.equal(
                metadata.verify_source_row_lens[:real_rows],
                eager.verify_source_row_lens,
            )
        )
        self.assertTrue((metadata.verify_source_row_lens[real_rows:] == 0).all())
        self.assertTrue(
            torch.equal(
                metadata.verify_prefix_lens,
                torch.tensor(padded_seq_lens, dtype=torch.int32, device=DEVICE),
            )
        )
        self._assert_row_lens_cumsum(metadata, row_lens)
        # The tree reuse boundary rolls back by num_draft_tokens (keys moved).
        eager_levels = self._assert_level_buffers_match_eager_builder(
            backend,
            metadata,
            forward_batch,
            real_seq_lens,
            num_draft_tokens,
            history_lens=torch.tensor(real_seq_lens, device=DEVICE) - num_draft_tokens,
        )
        self._assert_repeat_layouts_match_eager(
            backend, metadata, eager_levels, real_bs, num_draft_tokens
        )


if __name__ == "__main__":
    unittest.main()
