"""Contract: the sparse decode-wrapper plan warms once, then goes fast.

Sparse prefill plans the shared decode wrapper once per sparse layer per chunk.
begin_forward's Python-side max() over the chunk_tokens*head_group per-row kv
lengths dominates prefill host time, so after one backend-resolving standard
plan the wrapper switches to flashinfer's fast_decode_plan (same C++ schedule,
no Python max()). This pins the warmup dispatch and that both paths receive
identical plan arguments, without a GPU or a real flashinfer install.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.minicpm_attention_kernels import (
    FlashInferKernel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _bare_kernel() -> FlashInferKernel:
    # __new__ skips __init__, which would touch the distributed group and
    # allocate device workspaces; _plan_decode_wrapper only reads these fields.
    kernel = FlashInferKernel.__new__(FlashInferKernel)
    kernel._decode_plan_warmed = False
    kernel.head_dim = 128
    kernel.page_size = 1
    kernel.q_data_type = torch.bfloat16
    kernel.data_type = torch.bfloat16
    return kernel


class TestDecodePlanWarmup(CustomTestCase):
    def setUp(self):
        self.kernel = _bare_kernel()
        self.wrapper = mock.MagicMock()
        self.kv_indptr = torch.tensor([0, 1, 2], dtype=torch.int32)
        self.kv_indices = torch.tensor([7, 8], dtype=torch.int32)
        self.kv_last_page_len = torch.tensor([1, 1], dtype=torch.int32)
        self.plan_args = (16, 1)  # num_qo_heads, num_kv_heads

        self.fast_decode_plan = mock.MagicMock()
        fake_mod = types.ModuleType("flashinfer.decode")
        fake_mod.fast_decode_plan = self.fast_decode_plan
        patcher = mock.patch.dict(sys.modules, {"flashinfer.decode": fake_mod})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _plan(self):
        self.kernel._plan_decode_wrapper(
            self.wrapper,
            self.kv_indptr,
            self.kv_indices,
            self.kv_last_page_len,
            *self.plan_args,
        )

    def _expected_kwargs(self):
        return dict(
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            non_blocking=True,
        )

    def test_first_call_uses_standard_begin_forward(self):
        self.assertFalse(self.kernel._decode_plan_warmed)
        self._plan()
        self.assertTrue(self.kernel._decode_plan_warmed)
        self.wrapper.begin_forward.assert_called_once()
        self.fast_decode_plan.assert_not_called()

        pos = self.wrapper.begin_forward.call_args.args
        self.assertIs(pos[0], self.kv_indptr)
        self.assertIs(pos[1], self.kv_indices)
        self.assertIs(pos[2], self.kv_last_page_len)
        # num_qo_heads, num_kv_heads, head_dim, page_size
        self.assertEqual(pos[3:], (16, 1, 128, 1))
        self.assertEqual(
            self.wrapper.begin_forward.call_args.kwargs, self._expected_kwargs()
        )

    def test_subsequent_calls_use_fast_decode_plan(self):
        self._plan()  # warm
        self._plan()  # fast
        self._plan()  # fast

        self.wrapper.begin_forward.assert_called_once()  # never re-warmed
        self.assertEqual(self.fast_decode_plan.call_count, 2)

        # partial(fast_decode_plan, wrapper) -> wrapper is the first positional,
        # the rest matches begin_forward exactly.
        pos = self.fast_decode_plan.call_args.args
        self.assertIs(pos[0], self.wrapper)
        self.assertIs(pos[1], self.kv_indptr)
        self.assertEqual(pos[4:], (16, 1, 128, 1))
        self.assertEqual(
            self.fast_decode_plan.call_args.kwargs, self._expected_kwargs()
        )


if __name__ == "__main__":
    unittest.main()
