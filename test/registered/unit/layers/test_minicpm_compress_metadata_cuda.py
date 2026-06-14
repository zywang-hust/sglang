"""Bit-exact checks for the fused MiniCPM verify-replay metadata kernels.

Two kernels collapse the host-bound verify-prepare window's per-bs integer math
into single launches; both must produce byte-identical results to the eager path,
since the in-graph compaction and the FlashInfer split-KV plan are derived from
these exact counts:

* ``_compress_level_metadata_kernel`` — a compression level's counts/cumsums
  (the ``(x - ksize)//kstride + 1`` chains, four cumsums, four count copies),
  mirror of ``_compute_single_compression_metadata``.
* ``_verify_plan_counts_kernel`` — the FlashInfer per-row KV counts
  (``sc(token_pos) - off + popcount``).
"""

import types
import unittest

import torch

from sglang.srt.layers.attention.minicpm_sparse_utils import (
    build_verify_plan_counts,
    fill_compress_level_metadata,
    sparse_block_quantized_count,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")

DEVICE = "cuda"

# realistic compressor (k1/k2) params plus edge cases (stride 1, ksize > seq, ...)
CONFIGS = [(32, 16), (64, 32), (32, 32), (16, 16), (128, 64), (1, 1), (5, 1), (3, 7)]
BATCH_SIZES = [1, 2, 3, 7, 8, 13, 16, 17, 32, 64, 128]
DRAFT_TOKEN_NUMS = [1, 2, 4, 7, 8, 16, 64]


def _eager_ref(prefix_i32, dtn, ksize, kstride, bs):
    """Exact arithmetic of copy_compress_metadata / the eager sparse builder."""
    prefix = prefix_i32[:bs]
    full = prefix + dtn
    seq_lens_k = ((full - ksize) // kstride + 1).clamp(min=0)
    history = ((prefix - ksize) // kstride + 1).clamp(min=0)
    new_token = full - history * kstride
    new_compress = ((new_token - ksize) // kstride + 1).clamp(min=0)
    total = history + new_compress

    def cu(x):
        out = torch.zeros(bs + 1, dtype=torch.int32, device=x.device)
        torch.cumsum(x, dim=0, dtype=torch.int32, out=out[1:])
        return out

    return {
        "history_compress_token_nums": history.to(torch.int32),
        "new_token_nums": new_token.to(torch.int32),
        "new_compress_token_nums": new_compress.to(torch.int32),
        "total_compress_token_nums": total.to(torch.int32),
        "cu_seqlens": cu(seq_lens_k),
        "cu_new_token_nums": cu(new_token),
        "cu_new_compress_token_nums": cu(new_compress),
        "cu_total_compress_token_nums": cu(total),
    }


def _make_level(bs):
    lv = types.SimpleNamespace()
    for f in (
        "history_compress_token_nums",
        "new_token_nums",
        "new_compress_token_nums",
        "total_compress_token_nums",
    ):
        setattr(lv, f, torch.zeros(bs, dtype=torch.int32, device=DEVICE))
    for f in (
        "cu_seqlens",
        "cu_new_token_nums",
        "cu_new_compress_token_nums",
        "cu_total_compress_token_nums",
    ):
        setattr(lv, f, torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE))
    return lv


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestCompressMetadataKernel(CustomTestCase):
    def test_bit_exact_vs_eager(self):
        gen = torch.Generator(device=DEVICE).manual_seed(0)
        for ksize, kstride in CONFIGS:
            for bs in BATCH_SIZES:
                for dtn in DRAFT_TOKEN_NUMS:
                    prefix = torch.randint(
                        0, 5000, (bs,), generator=gen, device=DEVICE, dtype=torch.int32
                    )
                    # pin boundary seq lens around the kernel size
                    n_edge = min(bs, 4)
                    prefix[:n_edge] = torch.tensor(
                        [0, ksize - 1, ksize, ksize + 1][:n_edge],
                        dtype=torch.int32,
                        device=DEVICE,
                    )
                    ref = _eager_ref(prefix, dtn, ksize, kstride, bs)
                    lv = _make_level(bs)
                    fill_compress_level_metadata(lv, prefix, dtn, ksize, kstride, bs)
                    for field, expected in ref.items():
                        got = getattr(lv, field)
                        self.assertTrue(
                            torch.equal(got, expected),
                            f"{field} mismatch at ks={ksize} kst={kstride} "
                            f"bs={bs} dtn={dtn}: ref={expected.tolist()} "
                            f"got={got.tolist()}",
                        )


# (block_size, sparse_topk, head_group_num)
PLAN_CONFIGS = [
    (64, 16, 1),
    (64, 16, 2),
    (32, 8, 4),
    (128, 32, 2),
    (16, 4, 8),
    (64, 1, 1),
]


def _plan_counts_ref(seq_lens, offsets, mask, dtn, block_size, sparse_topk, hg, tnum):
    num_visible = mask[: tnum * dtn].view(tnum, dtn).sum(dim=1).to(torch.int64)
    seq_per_token = torch.repeat_interleave(seq_lens.to(torch.int64), dtn)
    draft_offsets = offsets[:tnum].to(torch.int64)
    token_pos = seq_per_token + draft_offsets
    num_kv = (
        sparse_block_quantized_count(token_pos, block_size, sparse_topk)
        - draft_offsets
        + num_visible
    )
    return num_kv.repeat_interleave(hg).to(torch.int32)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestVerifyPlanCountsKernel(CustomTestCase):
    def test_bit_exact_vs_eager(self):
        gen = torch.Generator(device=DEVICE).manual_seed(0)
        for block_size, sparse_topk, hg in PLAN_CONFIGS:
            for bs in (1, 2, 7, 8, 16, 32):
                for dtn in (1, 2, 4, 7, 8, 16):
                    tnum = bs * dtn
                    seq_lens = torch.randint(
                        0, 9000, (bs,), generator=gen, device=DEVICE, dtype=torch.int32
                    )
                    offsets = torch.randint(
                        1,
                        dtn + 1,
                        (tnum,),
                        generator=gen,
                        device=DEVICE,
                        dtype=torch.int32,
                    )
                    mask = torch.rand(tnum * dtn, generator=gen, device=DEVICE) > 0.5
                    ref = _plan_counts_ref(
                        seq_lens, offsets, mask, dtn, block_size, sparse_topk, hg, tnum
                    )
                    got = build_verify_plan_counts(
                        seq_lens, offsets, mask, dtn, block_size, sparse_topk, hg, tnum
                    )
                    self.assertTrue(
                        torch.equal(got, ref),
                        f"mismatch bs={bs} dtn={dtn} bsz={block_size} "
                        f"stk={sparse_topk} hg={hg}: ref={ref.tolist()} "
                        f"got={got.tolist()}",
                    )


if __name__ == "__main__":
    unittest.main()
