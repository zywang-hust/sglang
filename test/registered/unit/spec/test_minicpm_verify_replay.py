"""Pin the MiniCPM verify CUDA-graph capture seed and replay refresh to
the eager sparse-verify builders."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.minicpm_fixtures import (
    make_replay_backend,
    make_verify_batch,
    tree_mask_from_parents,
    visible_counts,
)
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.minicpm import backend as backend_module
from sglang.srt.layers.attention.minicpm.sparse_utils import (
    MiniCPMSparseMetadata,
    _build_sparse_verify_replay_rows,
    _plan_sparse_verify,
    _plan_uniform_slab_segments,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
        seq_lens=torch.tensor(padded_seq_lens, dtype=torch.int32),
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


def _assert_level_buffers_match_closed_form(
    backend, metadata, forward_batch, real_seq_lens, num_draft_tokens
):
    """K1/K2 level buffers against the closed-form reference for the padded
    batch."""
    seq_tensor = torch.tensor(real_seq_lens, dtype=torch.int32)
    real_bs = len(real_seq_lens)
    for name, kernel_size, kernel_stride in (
        ("k1", backend.k1_kernel_size, backend.k1_kernel_stride),
        ("k2", backend.k2_kernel_size, backend.k2_kernel_stride),
    ):
        level = getattr(metadata, name)
        expected_lens = torch.clamp(
            (seq_tensor + num_draft_tokens - kernel_size) // kernel_stride + 1,
            min=0,
        )
        expected_cu = F.pad(
            torch.cumsum(expected_lens, dim=0, dtype=torch.int32), (1, 0)
        )
        assert torch.equal(level.cu_seqlens[: real_bs + 1], expected_cu), name
        assert (level.cu_seqlens[real_bs + 1 :] == expected_cu[-1]).all()
        assert (level.history_compress_token_nums[real_bs:] == 0).all()
        table = getattr(backend, f"req_to_sparse_{name}_token")
        assert torch.equal(level.table, table[forward_batch.req_pool_indices])


class TestVerifyGraphReplay(CustomTestCase):
    def test_capture_seeds_consistent_plan(self):
        """Capture-time binding must seed a self-consistent plan (cumulative
        lengths equal to the row lengths, all rows live) at the assumed
        prefix + draft geometry -- an inconsistent seed records a graph whose
        kernel-side fill disagrees with the capture plan."""
        num_draft_tokens = 2
        backend, base = make_replay_backend([1, 1], num_draft_tokens, max_bs=2)
        forward_batch = SimpleNamespace(batch_size=2, spec_info=None)
        metadata = MiniCPMSparseMetadata(base=base)
        backend._bind_sparse_verify_graph_metadata(
            forward_batch, metadata, in_capture=True
        )
        rows = 2 * num_draft_tokens * backend.head_group_num
        self.assertEqual(metadata.sparse_cache_seqlens_int32.shape[0], rows)
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
                torch.arange(3, dtype=torch.int32) * assume_kv_len,
            )
        )
        # Magnitude pin: the seeded rows must equal the chain closed form at
        # the capture seed geometry -- a constant seed would pass every
        # check above.
        _, chain_rows = _build_sparse_verify_replay_rows(
            torch.full((2,), backend.config_dense_len, dtype=torch.int32),
            head_group_num=backend.head_group_num,
            dense_len=backend.dense_len,
            sparse_topk=backend.sparse_topk,
            block_size=backend.block_size,
            num_draft_tokens=num_draft_tokens,
        )
        self.assertTrue(torch.equal(metadata.sparse_cache_seqlens_int32, chain_rows))

    def test_capture_binds_retained_repeat_layouts(self):
        """Regression: the captured selection gathers through the repeat
        layout's index, so the graph records that tensor's device address.
        Binding must hand out prefix views of the layouts
        init_cuda_graph_state retains -- a per-capture allocation is freed
        when the next capture rebinds forward_metadata, and every later
        replay of this tier then gathers through recycled memory."""
        num_draft_tokens = 2
        bs = 2
        backend, base = make_replay_backend([1] * bs, num_draft_tokens, max_bs=4)
        metadata = MiniCPMSparseMetadata(base=base)
        backend._bind_sparse_verify_graph_metadata(
            SimpleNamespace(batch_size=bs, spec_info=None), metadata, in_capture=True
        )
        for name in ("k1", "k2"):
            layout = getattr(metadata, f"{name}_repeat")
            retained = backend.decode_cuda_graph_metadata[f"{name}_repeat"]
            self.assertEqual(layout.index.data_ptr(), retained.index.data_ptr())
            self.assertEqual(
                layout.cu_seqlens.data_ptr(), retained.cu_seqlens.data_ptr()
            )
            fresh = _plan_uniform_slab_segments(
                batch_size=bs,
                segment_rows=backend.max_context_len
                // getattr(backend, f"{name}_kernel_stride"),
                repeats=num_draft_tokens,
                device="cpu",
            )
            self.assertTrue(torch.equal(layout.index, fresh.index))
            self.assertTrue(torch.equal(layout.cu_seqlens, fresh.cu_seqlens))

    def test_capture_skips_k2_repeat_binding_under_fused_topk(self):
        """Regression:
        under minicpm_fuse_topk the captured gather reads no k2 repeat layout,
        so capture must skip the k2 binding;
        k1 still binds prefix views of the retained slab."""
        num_draft_tokens = 2
        with (
            backend_module.envs.SGLANG_MINICPM_FUSE_TOPK.override(True),
            patch.object(
                backend_module.MiniCPMSparseBackend,
                "_get_fused_topk_kernel",
                return_value=None,
            ),
        ):
            backend, base = make_replay_backend([1, 1], num_draft_tokens, max_bs=4)
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
        which spans the uniform slab,
        so the compression view sized from cu_seqlens_cpu must cover that slab
        -- a dense-window seed lets the captured index_select read past the view."""
        num_draft_tokens = 2
        bs = 2
        backend, base = make_replay_backend([1] * bs, num_draft_tokens, max_bs=4)
        metadata = MiniCPMSparseMetadata(base=base)
        backend._bind_sparse_verify_graph_metadata(
            SimpleNamespace(batch_size=bs, spec_info=None), metadata, in_capture=True
        )
        for name in ("k1", "k2"):
            level = getattr(metadata, name)
            segment_rows = backend.max_context_len // getattr(
                backend, f"{name}_kernel_stride"
            )
            self.assertEqual(
                level.cu_seqlens_cpu,
                [row * segment_rows for row in range(bs + 1)],
            )
            self.assertLess(
                int(getattr(metadata, f"{name}_repeat").index.max()),
                level.cu_seqlens_cpu[-1],
            )

    def test_replay_refreshes_static_buffers_to_eager_geometry(self):
        """Replay must leave the static buffers holding the same verify
        geometry the eager builder computes for the real batch, zero the
        padded rows so they plan and fill nothing, and keep the cumulative
        row lengths equal to the row lengths the selection fill writes."""
        num_draft_tokens = 2
        real_seq_lens = [100, 130, 126]
        padded_seq_lens = [*real_seq_lens, 1]
        bs, real_bs = len(padded_seq_lens), len(real_seq_lens)
        backend, base = make_replay_backend(padded_seq_lens, num_draft_tokens, max_bs=4)
        hg = backend.head_group_num

        capture_batch = SimpleNamespace(batch_size=bs, spec_info=None)
        backend._bind_sparse_verify_graph_metadata(
            capture_batch, MiniCPMSparseMetadata(base=base), in_capture=True
        )
        # The FlashAttention replay refreshes the base buffers;
        # the capture seeding overwrote them.
        base.cache_seqlens_int32.copy_(
            torch.tensor(padded_seq_lens, dtype=torch.int32) + num_draft_tokens
        )
        base.cu_seqlens_k.copy_(
            F.pad(
                torch.cumsum(base.cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
            )
        )

        forward_batch, metadata = _replay_verify_graph(
            backend,
            base,
            bs=bs,
            real_bs=real_bs,
            padded_seq_lens=padded_seq_lens,
            req_pool_indices=torch.tensor([2, 5, 1, 0]),
            spec_info=None,
        )

        eager_batch, eager_base = make_verify_batch(real_seq_lens, num_draft_tokens)
        eager = MiniCPMSparseMetadata(base=eager_base)
        _plan_sparse_verify(
            forward_batch=eager_batch,
            metadata=eager,
            head_group_num=hg,
            heads_per_group=backend.heads_per_group,
            dense_len=backend.dense_len,
            sparse_topk=backend.sparse_topk,
            block_size=backend.block_size,
            num_draft_tokens=num_draft_tokens,
        )
        real_rows = real_bs * num_draft_tokens * hg
        row_lens = metadata.sparse_cache_seqlens_int32
        self.assertTrue(
            torch.equal(row_lens[:real_rows], eager.sparse_cache_seqlens_int32)
        )
        self.assertTrue((row_lens[real_rows:] == 0).all())
        self.assertTrue(
            torch.equal(
                metadata.sparse_cu_seqlens_k,
                F.pad(torch.cumsum(row_lens, dim=0, dtype=torch.int32), (1, 0)),
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.token_pos_in_bs[: real_bs * num_draft_tokens],
                eager.token_pos_in_bs,
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.token_to_bs,
                torch.repeat_interleave(
                    torch.arange(bs, dtype=torch.int32), num_draft_tokens
                ),
            )
        )

        # Decode-form selection geometry: one single-query row per draft
        # token, per-row cache length = causal position - 1 (matching the
        # eager builder), and the static arange buffers sliced per row.
        num_real_tokens = real_bs * num_draft_tokens
        self.assertTrue(
            torch.equal(
                metadata.cache_seqlens_int32_stage1[:num_real_tokens],
                eager.cache_seqlens_int32_stage1,
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.verify_dense_mask[:num_real_tokens],
                eager.verify_dense_mask,
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.topk_cu_seqlens_q,
                torch.arange(bs * num_draft_tokens + 1, dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.cu_seqlens_q_adjusted,
                torch.arange(bs * num_draft_tokens + 1, dtype=torch.int32)
                * backend.heads_per_group,
            )
        )
        self.assertEqual(metadata.topk_max_seqlen_q, 1)
        self.assertEqual(metadata.max_seqlen_q_adjusted, backend.heads_per_group)

        # The uniform slab write offsets seeded at capture must survive the
        # replay refresh -- the static repeat index depends on them.
        for name, kernel_stride in (
            ("k1", backend.k1_kernel_stride),
            ("k2", backend.k2_kernel_stride),
        ):
            self.assertTrue(
                torch.equal(
                    getattr(metadata, name).cu_total_compress_token_nums,
                    torch.arange(bs + 1, dtype=torch.int32)
                    * (backend.max_context_len // kernel_stride),
                ),
                name,
            )

        # K1/K2 static tables must cover prefix + draft tokens (closed-form
        # reference), pad their tails with the last cumulative value, and
        # gather the per-request token tables for the padded batch.
        _assert_level_buffers_match_closed_form(
            backend, metadata, forward_batch, real_seq_lens, num_draft_tokens
        )

    def test_all_padding_replay_zeroes_compression_buffers(self):
        """Bug regression:
        an all-padding verify replay must zero the k1/k2 per-round compression lengths
        -- stale counts otherwise feed the captured compression kernels --
        while cu_total_compress_token_nums keeps the capture-time slab offsets,
        which the static repeat index reads."""
        num_draft_tokens = 2
        bs = 2
        backend, base = make_replay_backend([1] * bs, num_draft_tokens, max_bs=bs)
        backend._bind_sparse_verify_graph_metadata(
            SimpleNamespace(batch_size=bs, spec_info=None),
            MiniCPMSparseMetadata(base=base),
            in_capture=True,
        )
        forward_batch = SimpleNamespace(
            batch_size=bs,
            num_padding=bs,
            seq_lens=torch.ones(bs, dtype=torch.int32),
            seq_lens_cpu=torch.ones(bs, dtype=torch.int32),
            req_pool_indices=torch.arange(bs, dtype=torch.int64),
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
            self.assertTrue(
                torch.equal(
                    level.cu_total_compress_token_nums,
                    torch.arange(bs + 1, dtype=torch.int32)
                    * (backend.max_context_len // kernel_stride),
                ),
                name,
            )

    def test_graph_tree_capture_seeds_chain_degenerate_plan(self):
        """Tree capture must seed an all-True visibility square so the
        captured compaction keeps every slot and the seeded plan rows stay
        the chain closed form -- an inconsistent seed records a graph whose
        in-graph fill disagrees with the capture plan."""
        num_draft_tokens = 2
        backend, base = make_replay_backend(
            [1, 1], num_draft_tokens, max_bs=2, eagle_topk=2
        )
        # Dirty the static mask as a previous replay would.
        backend.decode_cuda_graph_metadata["verify_draft_tree_mask"].fill_(False)
        forward_batch = SimpleNamespace(batch_size=2, spec_info=None)
        metadata = MiniCPMSparseMetadata(base=base)
        backend._bind_sparse_verify_graph_metadata(
            forward_batch, metadata, in_capture=True
        )
        self.assertTrue(metadata.verify_draft_tree_mask.all())
        _, chain_rows = _build_sparse_verify_replay_rows(
            torch.full((2,), backend.config_dense_len, dtype=torch.int32),
            head_group_num=backend.head_group_num,
            dense_len=backend.dense_len,
            sparse_topk=backend.sparse_topk,
            block_size=backend.block_size,
            num_draft_tokens=num_draft_tokens,
        )
        self.assertTrue(torch.equal(metadata.sparse_cache_seqlens_int32, chain_rows))
        self.assertTrue(torch.equal(metadata.verify_source_row_lens, chain_rows))

    def test_graph_tree_replay_matches_eager_tree_builder(self):
        """Tree replay must stage the live visibility square (padding rows
        all-True), refresh the planned rows to the popcount form the eager
        builder computes, keep the chain form in the source-length buffer,
        and roll the compression reuse boundary back by num_draft_tokens."""
        num_draft_tokens = 3
        real_seq_lens = [100, 130]
        padded_seq_lens = [*real_seq_lens, 1]
        bs, real_bs = len(padded_seq_lens), len(real_seq_lens)
        backend, base = make_replay_backend(
            padded_seq_lens, num_draft_tokens, max_bs=3, eagle_topk=2
        )
        hg = backend.head_group_num
        # Node 2 hides node 1 in both real requests.
        square = tree_mask_from_parents([-1, 0, 0])
        mask = torch.cat(
            [
                square.view(-1).repeat(real_bs),
                torch.ones(num_draft_tokens * num_draft_tokens, dtype=torch.bool),
            ]
        )

        capture_batch = SimpleNamespace(batch_size=bs, spec_info=None)
        backend._bind_sparse_verify_graph_metadata(
            capture_batch, MiniCPMSparseMetadata(base=base), in_capture=True
        )
        # FA's topk>1 replay refresh leaves the two-phase tree layout in the
        # static base buffers: prefix-only cache lengths and page-table rows.
        prefix = torch.tensor(padded_seq_lens, dtype=torch.int32)
        base.cache_seqlens_int32.copy_(prefix)
        base.cu_seqlens_k.copy_(
            F.pad(torch.cumsum(prefix, dim=0, dtype=torch.int32), (1, 0))
        )
        base.page_table.zero_()

        custom_mask = torch.ones(1, dtype=torch.bool)
        forward_batch = SimpleNamespace(
            batch_size=bs,
            num_padding=bs - real_bs,
            seq_lens=torch.tensor(padded_seq_lens, dtype=torch.int32),
            seq_lens_cpu=torch.tensor(padded_seq_lens, dtype=torch.int32),
            req_pool_indices=torch.tensor([2, 5, 1]),
            spec_info=SimpleNamespace(custom_mask=custom_mask),
        )
        metadata = MiniCPMSparseMetadata(base=base)
        backend._bind_sparse_verify_graph_metadata(
            forward_batch, metadata, in_capture=False
        )

        def _stage_mask(*, out, num_visible_out, **kwargs):
            out.copy_(mask)
            num_visible_out.copy_(
                visible_counts(mask=mask, bs=bs, num_draft_tokens=num_draft_tokens)
            )

        def _fill_page_table(
            *, req_to_token, req_pool_indices, cache_seqlens, page_table, page_size
        ):
            for row, (req, length) in enumerate(
                zip(req_pool_indices.tolist(), cache_seqlens.tolist())
            ):
                page_table[row, :length] = req_to_token[req, :length]

        with (
            patch.object(
                backend_module, "copy_eagle_draft_tree_mask", side_effect=_stage_mask
            ) as copy_mask,
            patch.object(
                backend_module,
                "build_trtllm_mha_page_table",
                side_effect=_fill_page_table,
            ),
        ):
            backend._replay_sparse_verify_graph_metadata(forward_batch, metadata)

        # The in-place normalization must restore the uniform prefix +
        # num_draft_tokens geometry in the static buffers the captured fill
        # kernels read.
        self.assertEqual(
            base.cache_seqlens_int32.tolist(), (prefix + num_draft_tokens).tolist()
        )
        self.assertTrue(
            torch.equal(
                base.cu_seqlens_k,
                F.pad(
                    torch.cumsum(prefix + num_draft_tokens, dim=0, dtype=torch.int32),
                    (1, 0),
                ),
            )
        )
        req_to_token = backend.req_to_token_pool.req_to_token
        for row, (req, seq_len) in enumerate(
            zip(forward_batch.req_pool_indices.tolist(), padded_seq_lens)
        ):
            self.assertTrue(
                torch.equal(
                    base.page_table[row, : seq_len + num_draft_tokens],
                    req_to_token[req, : seq_len + num_draft_tokens],
                ),
                row,
            )

        copy_kwargs = copy_mask.call_args.kwargs
        self.assertIs(copy_kwargs["custom_mask"], custom_mask)
        self.assertEqual(copy_kwargs["bs"], real_bs)
        self.assertEqual(copy_kwargs["padded_bs"], bs)

        eager_batch, eager_base = make_verify_batch(real_seq_lens, num_draft_tokens)
        counts = visible_counts(
            mask=square.view(-1).repeat(real_bs),
            bs=real_bs,
            num_draft_tokens=num_draft_tokens,
        )
        eager = MiniCPMSparseMetadata(base=eager_base)
        _plan_sparse_verify(
            forward_batch=eager_batch,
            metadata=eager,
            head_group_num=hg,
            heads_per_group=backend.heads_per_group,
            dense_len=backend.dense_len,
            sparse_topk=backend.sparse_topk,
            block_size=backend.block_size,
            num_draft_tokens=num_draft_tokens,
            num_draft_visible=counts,
        )
        real_rows = real_bs * num_draft_tokens * hg
        row_lens = metadata.sparse_cache_seqlens_int32
        self.assertTrue(
            torch.equal(row_lens[:real_rows], eager.sparse_cache_seqlens_int32)
        )
        self.assertTrue((row_lens[real_rows:] == 0).all())
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
                torch.tensor(padded_seq_lens, dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.sparse_cu_seqlens_k,
                F.pad(torch.cumsum(row_lens, dim=0, dtype=torch.int32), (1, 0)),
            )
        )
        # Compression reuse boundary rolled back by num_draft_tokens for the
        # real batch.
        kernel_size = backend.k1_kernel_size
        kernel_stride = backend.k1_kernel_stride
        expected_history = torch.clamp(
            (
                torch.tensor(real_seq_lens, dtype=torch.int32)
                - num_draft_tokens
                - kernel_size
            )
            // kernel_stride
            + 1,
            min=0,
        )
        self.assertTrue(
            torch.equal(
                metadata.k1.history_compress_token_nums[:real_bs], expected_history
            )
        )


if __name__ == "__main__":
    unittest.main()
