import unittest

import torch
import triton

from sglang.srt.layers.attention.minicpm_sparse_kernels import (
    compress_k_complete_kernel_new,
    compress_k_complete_kernel_new_padded,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=4, stage="base-b", runner_config="1-gpu-small")

DEVICE = "cuda"
HEADS = 2
DIM = 8
CTX = 256


def _build_case(seed, hist_chunks, new_tokens, kernel_size, kernel_stride):
    """Per-seq history chunk counts and new token counts, scattered tables."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    bs = len(hist_chunks)
    max_chunks = max(0, (CTX - kernel_size) // kernel_stride + 1)
    pool = bs * (CTX + max_chunks) + 8
    key_cache = torch.randn(pool, HEADS, DIM, dtype=torch.bfloat16, device=DEVICE)
    token_table = (
        torch.randperm(bs * CTX, generator=gen)
        .to(torch.int32)
        .reshape(bs, CTX)
        .to(DEVICE)
    )
    comp_table = (
        (torch.randperm(bs * max_chunks, generator=gen).to(torch.int32) + bs * CTX)
        .reshape(bs, max_chunks)
        .to(DEVICE)
    )
    hist = torch.tensor(hist_chunks, dtype=torch.int32, device=DEVICE)
    new_k = torch.tensor(new_tokens, dtype=torch.int32, device=DEVICE)
    cu_new_k = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
    cu_new_k[1:] = torch.cumsum(new_k, 0)
    new_chunks = torch.where(
        new_k >= kernel_size, (new_k - kernel_size) // kernel_stride + 1, 0
    ).to(torch.int32)
    cu_new_chunks = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
    cu_new_chunks[1:] = torch.cumsum(new_chunks, 0)
    total_chunks = hist + new_chunks
    cu_total = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
    cu_total[1:] = torch.cumsum(total_chunks, 0)
    out_rows = max(bs * max_chunks, int(cu_total[-1].item()))
    out = torch.zeros(out_rows, HEADS, DIM, dtype=torch.bfloat16, device=DEVICE)
    return dict(
        bs=bs,
        key_cache=key_cache,
        token_table=token_table,
        comp_table=comp_table,
        hist=hist,
        new_k=new_k,
        cu_new_k=cu_new_k,
        new_chunks=new_chunks,
        cu_new_chunks=cu_new_chunks,
        total_chunks=total_chunks,
        cu_total=cu_total,
        out=out,
        kernel_size=kernel_size,
        kernel_stride=kernel_stride,
        max_chunks=max_chunks,
    )


def _launch(kernel, c):
    grid = (c["bs"], max(c["max_chunks"], 1), HEADS)
    kernel[grid](
        c["key_cache"],
        c["token_table"],
        c["cu_new_k"],
        c["hist"],
        c["kernel_stride"],
        c["comp_table"],
        c["cu_new_chunks"],
        c["cu_total"],
        c["total_chunks"],
        c["out"],
        c["bs"],
        c["max_chunks"],
        c["token_table"].shape[1],
        c["comp_table"].shape[1],
        HEADS,
        DIM,
        c["kernel_size"],
        c["kernel_stride"],
        BLOCK_SIZE=triton.next_power_of_2(DIM),
        max_grid_chunks=max(c["max_chunks"], 1),
    )


def _reference(c, padded):
    """Pure-torch reference; the mean accumulates sequentially in fp32 like
    the kernel, so all outputs must match bitwise."""
    key_cache = c["key_cache"].clone()
    out = c["out"].clone()
    for b in range(c["bs"]):
        hist = int(c["hist"][b])
        new_chunks = int(c["new_chunks"][b])
        base = b * c["max_chunks"] if padded else int(c["cu_total"][b])
        for i in range(hist):
            slot = int(c["comp_table"][b, i])
            out[base + i] = key_cache[slot]
        for j in range(new_chunks):
            slot = int(c["comp_table"][b, hist + j])
            y0 = j * c["kernel_stride"] + hist * c["kernel_stride"]
            acc = torch.zeros(HEADS, DIM, dtype=torch.float32, device=DEVICE)
            for t in range(c["kernel_size"]):
                token = int(c["token_table"][b, y0 + t])
                acc = acc + key_cache[token].to(torch.float32)
            mean = (acc / c["kernel_size"]).to(torch.bfloat16)
            key_cache[slot] = mean
            out[base + hist + j] = mean
    return out, key_cache


class TestMiniCPMCompressK(CustomTestCase):
    CASES = (
        # (hist_chunks, new_tokens, kernel_size, kernel_stride)
        ([5, 0, 9, 2], [4, 40, 0, 36], 32, 16),
        ([2, 1, 0, 1], [130, 64, 0, 7], 128, 64),
        ([0, 0], [0, 0], 32, 16),  # padding-only rows
        ([7], [70], 32, 16),  # multiple new chunks in one seq
    )

    def _run(self, kernel, padded):
        for idx, (hist, new, ks, stride) in enumerate(self.CASES):
            with self.subTest(case=idx, padded=padded):
                c = _build_case(idx + 1, hist, new, ks, stride)
                ref_out, ref_cache = _reference(c, padded)
                _launch(kernel, c)
                self.assertTrue(torch.equal(c["out"], ref_out))
                self.assertTrue(torch.equal(c["key_cache"], ref_cache))

    def test_plain_layout_matches_reference_bitwise(self):
        self._run(compress_k_complete_kernel_new, padded=False)

    def test_padded_layout_matches_reference_bitwise(self):
        self._run(compress_k_complete_kernel_new_padded, padded=True)


if __name__ == "__main__":
    unittest.main()
