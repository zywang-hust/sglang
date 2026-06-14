"""Contract: vectorized sparse-prefill metadata equals the scalar-loop original.

``SparseMetadataBuilder.build_token_mappings`` and
``build_sparse_prefill_metadata`` build the flattened sparse query stream's
cu_seqlens / token maps on the host. They were per-element Python loops
(``cu_seqlens[pt+1] = cu_seqlens[pt] + step`` over chunk_tokens*head_group rows
~ 100ms/8K chunk); they are now ``arange``/``cumsum``/``repeat_interleave``.
This pins the vectorized output byte-for-byte against an independent
reimplementation of the original loops, across the sparse / dense / tree-verify
/ mixed-batch branches that decide each request's row layout and step.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.minicpm_sparse_utils import (
    SparseConfig,
    SparseMetadataBuilder,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

HEAD_GROUP = 2  # MiniCPM-SALA num_key_value_heads
DENSE_LEN = 8192
SPARSE_TOPK = 64
BLOCK_SIZE = 64


def _builder() -> SparseMetadataBuilder:
    config = SparseConfig.from_model_config(
        hf_config=SimpleNamespace(
            sparse_topk=SPARSE_TOPK,
            sparse_kernel_size=32,
            sparse_kernel_stride=16,
            sparse_block_size=BLOCK_SIZE,
            sparse_window_size=2048,
            sparse_dense_len=DENSE_LEN,
        ),
        model_config=SimpleNamespace(head_dim=128, num_key_value_heads=HEAD_GROUP),
    )
    return SparseMetadataBuilder(config, num_kv_heads=HEAD_GROUP)


def _ref_token_mappings(seqlens, prefixes, sparse_bs_list):
    """The original build_token_mappings: scalar fill per segment."""
    total = sum(seqlens)
    token_to_bs = torch.zeros(total, dtype=torch.int32)
    token_pos = torch.zeros(total, dtype=torch.int32)
    pt = 0
    for i, n in enumerate(seqlens):
        token_to_bs[pt : pt + n] = sparse_bs_list[i]
        token_pos[pt : pt + n] = torch.tensor(
            [idx + 1 + prefixes[i] for idx in range(n)], dtype=torch.int32
        )
        pt += n
    return token_to_bs, token_pos


def _ref_prefill_metadata(seq_lens, extend_lens, sparse_bs_list, tree_mode):
    """The original build_sparse_prefill_metadata loops (CPU scalars only)."""
    bs = len(seq_lens)
    max_cache = -1
    page_bs = 0
    old_to_new = [0] * (bs + 1)
    max_q = 1
    for i in range(bs):
        if seq_lens[i] >= DENSE_LEN:
            max_cache = max(max_cache, SPARSE_TOPK * BLOCK_SIZE)
            page_bs += extend_lens[i] * HEAD_GROUP
            old_to_new[i + 1] = old_to_new[i] + HEAD_GROUP * extend_lens[i]
        elif tree_mode:
            max_cache = max(max_cache, seq_lens[i])
            page_bs += extend_lens[i] * HEAD_GROUP
            old_to_new[i + 1] = old_to_new[i] + HEAD_GROUP * extend_lens[i]
        else:
            max_cache = max(max_cache, seq_lens[i])
            page_bs += HEAD_GROUP
            old_to_new[i + 1] = old_to_new[i] + HEAD_GROUP
            max_q = max(max_q, extend_lens[i])
    cu = torch.zeros(page_bs + 1, dtype=torch.int32)
    pt = 0
    for i in range(bs):
        if seq_lens[i] >= DENSE_LEN or tree_mode:
            for _ in range(extend_lens[i] * HEAD_GROUP):
                cu[pt + 1] = cu[pt] + 1
                pt += 1
        else:
            for _ in range(HEAD_GROUP):
                cu[pt + 1] = cu[pt] + extend_lens[i]
                pt += 1
    sparse_idx = []
    for b in sparse_bs_list:
        sparse_idx.extend(range(old_to_new[b], old_to_new[b + 1]))
    return {
        "cu": cu,
        "page_bs": page_bs,
        "max_cache": max_cache,
        "old_to_new": old_to_new,
        "max_q": max_q,
        "sparse_idx": sparse_idx,
    }


class TestBuildTokenMappings(CustomTestCase):
    def _check(self, seqlens, prefixes, sparse_bs_list):
        builder = _builder()
        got_bs, got_pos = builder.build_token_mappings(
            torch.tensor(prefixes, dtype=torch.long), seqlens, sparse_bs_list
        )
        ref_bs, ref_pos = _ref_token_mappings(seqlens, prefixes, sparse_bs_list)
        self.assertTrue(torch.equal(got_bs, ref_bs), f"token_to_bs {seqlens}")
        self.assertTrue(torch.equal(got_pos, ref_pos), f"token_pos {seqlens}")
        self.assertEqual(got_bs.dtype, torch.int32)
        self.assertEqual(got_pos.dtype, torch.int32)

    def test_matches_reference(self):
        cases = [
            ([5], [0], [0]),
            ([5], [10], [3]),  # nonzero prefix, original bs id != position
            ([3, 7], [0, 0], [0, 1]),
            ([3, 7, 2], [4, 0, 100], [1, 2, 5]),  # mixed prefixes and ids
            ([1, 1], [0, 1], [0, 1]),  # single-token segments
        ]
        for seqlens, prefixes, sparse_bs_list in cases:
            self._check(seqlens, prefixes, sparse_bs_list)

    def test_empty(self):
        got_bs, got_pos = _builder().build_token_mappings(
            torch.tensor([], dtype=torch.long), [], []
        )
        self.assertEqual(got_bs.numel(), 0)
        self.assertEqual(got_pos.numel(), 0)

    def test_defaults_sparse_bs_list(self):
        builder = _builder()
        got_bs, _ = builder.build_token_mappings(
            torch.tensor([0, 0], dtype=torch.long), [2, 3]
        )
        self.assertEqual(got_bs.tolist(), [0, 0, 1, 1, 1])


class TestBuildSparsePrefillMetadata(CustomTestCase):
    def _check(self, seq_lens, extend_lens, tree_mode):
        bs = len(seq_lens)
        sparse_bs_list = [i for i in range(bs) if seq_lens[i] >= DENSE_LEN]
        forward_batch = SimpleNamespace(batch_size=bs, extend_seq_lens_cpu=extend_lens)
        base_metadata = SimpleNamespace(
            seq_lens_cpu_for_sparse=torch.tensor(seq_lens, dtype=torch.int32)
        )
        out = _builder().build_sparse_prefill_metadata(
            forward_batch=forward_batch,
            base_metadata=base_metadata,
            sparse_bs_list=sparse_bs_list,
            head_group_num=HEAD_GROUP,
            dense_len=DENSE_LEN,
            sparse_topk=SPARSE_TOPK,
            block_size=BLOCK_SIZE,
            cu_seqlens_q=torch.zeros(bs + 1, dtype=torch.int32),
            sparse_page_table_dtype=torch.int32,
            sparse_page_table_device=torch.device("cpu"),
            tree_mode=tree_mode,
        )
        ref = _ref_prefill_metadata(seq_lens, extend_lens, sparse_bs_list, tree_mode)
        msg = f"seq_lens={seq_lens} extend={extend_lens} tree={tree_mode}"
        self.assertTrue(
            torch.equal(out["sparse_cu_seqlens_q_cpu"], ref["cu"]), f"cu_seqlens {msg}"
        )
        self.assertEqual(out["old_bs_to_new_bs_range"], ref["old_to_new"], msg)
        self.assertEqual(out["sparse_max_seq_len_q"], ref["max_q"], msg)
        self.assertEqual(list(out["sparse_idx"]), ref["sparse_idx"], msg)
        self.assertEqual(
            tuple(out["sparse_page_table"].shape),
            (ref["page_bs"], ref["max_cache"]),
            msg,
        )
        self.assertEqual(out["sparse_cu_seqlens_q_cpu"].dtype, torch.int32)

    def test_all_sparse(self):  # deployment: dense_as_sparse, every request sparse
        self._check([9000], [9000], tree_mode=False)
        self._check([9000, 10000], [9000, 10000], tree_mode=False)

    def test_all_dense(self):  # head-major rows, step = extend
        self._check([100], [100], tree_mode=False)
        self._check([100, 500], [100, 500], tree_mode=False)

    def test_mixed_sparse_dense(self):
        self._check([9000, 100, 12000], [9000, 100, 12000], tree_mode=False)

    def test_tree_mode_dense_is_token_major(self):
        # Under tree verify a sub-dense request becomes token-major (step 1) but
        # the page table holds its full prefix, unlike plain dense.
        self._check([100, 9000], [16, 16], tree_mode=True)
        self._check([200], [16], tree_mode=True)


if __name__ == "__main__":
    unittest.main()
