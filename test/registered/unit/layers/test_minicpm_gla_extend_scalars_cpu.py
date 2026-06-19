"""Contract: SimpleGLA hoists its per-layer extend scalars sync-free.

SimpleGLAAttnBackend.forward runs once per GLA layer (24 per chunk). Two of its
decisions — "does any request carry a cached prefix" (initial-state load) and
the extend length that picks the fused_recurrent/chunk kernel — are identical
for every layer, and evaluating them on the device tensors forces a GPU->CPU
sync each time (the dominant prefill host cost after C1). _hoist_extend_scalars
reads the CPU mirrors once per forward instead. This pins that the hoisted
values equal the device expressions they replace, including the gpu_only
fallback where the CPU mirrors are unset.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    SimpleGLAAttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _bare_backend() -> SimpleGLAAttnBackend:
    # __new__ skips __init__, which needs minicpm_hybrid_config / fla / the
    # distributed group; _hoist_extend_scalars only writes the two fields.
    backend = SimpleGLAAttnBackend.__new__(SimpleGLAAttnBackend)
    backend._extend_has_prefix = False
    backend._extend_max_seq_len = 0
    return backend


def _forward_batch(prefix_cpu, prefix_gpu, seq_cpu, seq_gpu):
    return SimpleNamespace(
        extend_prefix_lens_cpu=prefix_cpu,
        extend_prefix_lens=prefix_gpu,
        extend_seq_lens_cpu=seq_cpu,
        extend_seq_lens=seq_gpu,
    )


class TestHoistExtendScalars(CustomTestCase):
    def test_cpu_mirror_no_prefix(self):
        backend = _bare_backend()
        backend._hoist_extend_scalars(_forward_batch([0, 0], None, [8, 8], None))
        self.assertFalse(backend._extend_has_prefix)
        self.assertEqual(backend._extend_max_seq_len, 8)

    def test_cpu_mirror_has_prefix(self):
        backend = _bare_backend()
        backend._hoist_extend_scalars(_forward_batch([0, 5], None, [100, 1], None))
        self.assertTrue(backend._extend_has_prefix)
        self.assertEqual(backend._extend_max_seq_len, 100)

    def test_gpu_fallback_when_mirrors_absent(self):
        # gpu_only batch path: *_cpu unset, fall back to the device tensors.
        backend = _bare_backend()
        backend._hoist_extend_scalars(
            _forward_batch(
                None,
                torch.tensor([0, 3], dtype=torch.int32),
                None,
                torch.tensor([70, 8], dtype=torch.int32),
            )
        )
        self.assertTrue(backend._extend_has_prefix)
        self.assertEqual(backend._extend_max_seq_len, 70)

    def test_none_inputs_are_inert(self):
        # decode/verify hand in no extend tensors; the fields must stay safe.
        backend = _bare_backend()
        backend._hoist_extend_scalars(_forward_batch(None, None, None, None))
        self.assertFalse(backend._extend_has_prefix)
        self.assertEqual(backend._extend_max_seq_len, 0)

    def test_matches_original_device_expressions(self):
        for prefix, seq in (([0, 0], [8, 8]), ([0, 5, 0], [12, 4, 9]), ([1], [200])):
            prefix_gpu = torch.tensor(prefix, dtype=torch.int32)
            seq_gpu = torch.tensor(seq, dtype=torch.int32)
            backend = _bare_backend()
            backend._hoist_extend_scalars(
                _forward_batch(list(prefix), prefix_gpu, list(seq), seq_gpu)
            )
            self.assertEqual(
                backend._extend_has_prefix, bool((prefix_gpu > 0).any()), prefix
            )
            self.assertEqual(backend._extend_max_seq_len, int(torch.max(seq_gpu)), seq)


if __name__ == "__main__":
    unittest.main()
