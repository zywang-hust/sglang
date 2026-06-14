"""Contract: sparse page_table -> FlashInfer CSR conversion (C6a cumsum fix).

``convert_sparse_page_table_to_flashinfer`` turns the dense per-row sparse page
table into FlashInfer's CSR triple (kv_indptr / kv_indices / kv_last_page_len).
The row offsets used to be built by a single-block serial ``cumsum_kernel`` that
walked all ``sparse_bs`` rows on one thread-block (``sparse_bs`` = query_tokens *
head_group = 16384 for an 8K chunk -> ~1ms/call); they are now ``torch.cumsum``.
This pins the CSR output byte-for-byte: ``kv_indptr`` is the exclusive prefix sum
of the per-row valid counts, ``kv_indices`` is each row's valid prefix
concatenated in row order, and ``kv_last_page_len`` is all ones -- across a
hand-checked case, empty/full rows, and the real 16384-row scale.
"""

import unittest

import torch

from sglang.srt.layers.attention.minicpm_sparse_kernels import (
    convert_sparse_page_table_to_flashinfer,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=3, stage="base-b", runner_config="1-gpu-small")

DEVICE = "cuda"


def _run_convert(sparse_page_table, cache_seqlens):
    sparse_bs, max_sparse_tokens = sparse_page_table.shape
    # Start the output buffers dirty: under CUDA graph capture these are
    # persistent, reused buffers, so convert must overwrite every used slot --
    # including kv_indptr[0], which it now zeroes on-device. Asserting against a
    # dirty start pins that (a scalar ``kv_indptr[0] = 0`` regressed graph capture).
    kv_indptr = torch.full((sparse_bs + 1,), -999, dtype=torch.int32, device=DEVICE)
    kv_indices = torch.full(
        (sparse_bs * max_sparse_tokens,), -999, dtype=torch.int32, device=DEVICE
    )
    kv_last_page_len = torch.full(
        (sparse_bs,), -999, dtype=torch.int32, device=DEVICE
    )
    return convert_sparse_page_table_to_flashinfer(
        sparse_page_table, cache_seqlens, kv_indptr, kv_indices, kv_last_page_len
    )


class TestPageTableConvert(CustomTestCase):
    def _check(self, sparse_page_table, cache_seqlens):
        kv_indptr, kv_indices, kv_last_page_len = _run_convert(
            sparse_page_table, cache_seqlens
        )
        counts = cache_seqlens.to(torch.int64)
        total = int(counts.sum())

        # kv_indptr is the exclusive prefix sum of the per-row valid counts.
        ref_indptr = torch.zeros(counts.numel() + 1, dtype=torch.int32, device=DEVICE)
        ref_indptr[1:] = torch.cumsum(counts, 0).to(torch.int32)
        self.assertTrue(torch.equal(kv_indptr, ref_indptr))

        # kv_indices is each row's valid prefix, concatenated in row order. A
        # boolean mask flattened row-major reproduces exactly that order.
        max_tok = sparse_page_table.shape[1]
        mask = torch.arange(max_tok, device=DEVICE)[None, :] < cache_seqlens[:, None]
        self.assertTrue(torch.equal(kv_indices[:total], sparse_page_table[mask]))

        self.assertTrue(
            torch.equal(kv_last_page_len, torch.ones_like(kv_last_page_len))
        )

    def test_small_handchecked(self):
        spt = torch.tensor(
            [[10, 11, 12, 0], [20, 0, 0, 0], [30, 31, 0, 0]],
            dtype=torch.int32,
            device=DEVICE,
        )
        cache = torch.tensor([3, 1, 2], dtype=torch.int32, device=DEVICE)
        kv_indptr, kv_indices, _ = _run_convert(spt, cache)
        self.assertEqual(kv_indptr.cpu().tolist(), [0, 3, 4, 6])
        self.assertEqual(kv_indices[:6].cpu().tolist(), [10, 11, 12, 20, 30, 31])
        self._check(spt, cache)

    def test_empty_and_full_rows(self):
        max_tok = 8
        gen = torch.Generator(device="cpu").manual_seed(1)
        spt = torch.randint(1, 1000, (5, max_tok), generator=gen, dtype=torch.int32).to(
            DEVICE
        )
        # leading zero-length row, a full row, and an all-empty row
        cache = torch.tensor([0, max_tok, 3, 0, 5], dtype=torch.int32, device=DEVICE)
        self._check(spt, cache)

    def test_realistic_16384_rows(self):
        # sparse_bs = 8192 tokens * 2 head_group, max_sparse_tokens = topk*block
        # = 96*64; the scale the old serial cumsum walked one row at a time.
        sparse_bs, max_tok = 16384, 96 * 64
        gen = torch.Generator(device="cpu").manual_seed(0)
        spt = torch.randint(
            0, 1_000_000, (sparse_bs, max_tok), generator=gen, dtype=torch.int32
        ).to(DEVICE)
        cache = torch.randint(
            0, max_tok + 1, (sparse_bs,), generator=gen, dtype=torch.int32
        ).to(DEVICE)
        self._check(spt, cache)


if __name__ == "__main__":
    unittest.main()
