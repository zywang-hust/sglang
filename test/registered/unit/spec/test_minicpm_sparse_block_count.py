"""Contract: sparse_block_quantized_count's piecewise formula.

The helper is the single source of truth for MiniCPM sparse "cache seqlens";
both the ScheduleBatch decode precompute and the target-verify CUDA-graph
split-KV plan must agree byte-for-byte with the counts the in-graph kernels
write, so its boundary behavior (identity below the budget, block-aligned
positions, just-over-budget positions) is pinned here against an independent
block-selection simulation and a hand-written truth table.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest

import torch

from sglang.srt.layers.attention.minicpm_sparse_utils import (
    sparse_block_quantized_count,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _reference_count(position: int, block_size: int, sparse_topk: int) -> int:
    """First-principles simulation of the sparse block selection.

    Keys [0, position) are grouped into blocks of ``block_size``. If they fit
    in ``sparse_topk`` blocks, every key is visible. Otherwise the local
    window always selects the frontier (most recent, possibly partial) block,
    and the remaining ``sparse_topk - 1`` selected blocks are full.
    """
    num_blocks = (position + block_size - 1) // block_size
    if num_blocks <= sparse_topk:
        return position
    frontier_keys = position - (num_blocks - 1) * block_size
    return (sparse_topk - 1) * block_size + frontier_keys


class TestSparseBlockQuantizedCount(CustomTestCase):
    def test_truth_table(self):
        block_size, sparse_topk = 4, 3  # budget = 12
        cases = {
            0: 0,
            1: 1,  # below budget: identity
            11: 11,  # budget - 1
            12: 12,  # exactly the budget
            13: 9,  # just over budget: (topk-1)*block + 1
            14: 10,  # over budget, p % block == 2
            15: 11,  # over budget, p % block == block - 1
            16: 12,  # block-aligned above budget: full budget
            20: 12,  # block-aligned, further out
            21: 9,  # p % block == 1 again
        }
        positions = torch.tensor(sorted(cases), dtype=torch.int64)
        expected = [cases[int(p)] for p in positions]
        actual = sparse_block_quantized_count(positions, block_size, sparse_topk)
        self.assertEqual(actual.tolist(), expected)

    def test_matches_block_selection_reference(self):
        for block_size, sparse_topk in ((2, 1), (4, 3), (64, 6)):
            budget = block_size * sparse_topk
            positions = torch.arange(0, 3 * budget + 2 * block_size, dtype=torch.int64)
            expected = [
                _reference_count(int(p), block_size, sparse_topk) for p in positions
            ]
            actual = sparse_block_quantized_count(positions, block_size, sparse_topk)
            self.assertEqual(
                actual.tolist(),
                expected,
                f"mismatch for block_size={block_size}, sparse_topk={sparse_topk}",
            )

    def test_preserves_dtype_and_device(self):
        for dtype in (torch.int32, torch.int64):
            positions = torch.tensor([5, 13, 16], dtype=dtype)
            result = sparse_block_quantized_count(positions, 4, 3)
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(result.device, positions.device)
            self.assertEqual(result.tolist(), [5, 9, 12])


if __name__ == "__main__":
    unittest.main()
