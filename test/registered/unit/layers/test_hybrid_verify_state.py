from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    _build_retrieve_parent_token_ref,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestHybridVerifyState(CustomTestCase):
    def test_retrieve_parent_token_ref_builds_tree_parents(self):
        retrieve_next_token = torch.tensor(
            [
                [1, 3, -1, -1, 5, -1],
                [1, -1, 3, -1, -1, -1],
            ],
            dtype=torch.int32,
        )
        retrieve_next_sibling = torch.tensor(
            [
                [-1, 2, -1, 4, -1, -1],
                [-1, 2, -1, 4, 5, -1],
            ],
            dtype=torch.int32,
        )
        out = torch.full((3, 6), -1, dtype=torch.int32)

        parent = _build_retrieve_parent_token_ref(
            retrieve_next_token, retrieve_next_sibling, out=out
        )

        self.assertIs(parent, out)
        self.assertTrue(
            torch.equal(
                parent,
                torch.tensor(
                    [
                        [0, 0, 0, 1, 1, 4],
                        [0, 0, 0, 2, 2, 2],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    dtype=torch.int32,
                ),
            )
        )

    def test_retrieve_parent_token_ref_rejects_invalid_tree_index(self):
        retrieve_next_token = torch.tensor([[1, -1, -1]], dtype=torch.int32)
        retrieve_next_sibling = torch.tensor([[-1, 5, -1]], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "Invalid EAGLE sibling index"):
            _build_retrieve_parent_token_ref(retrieve_next_token, retrieve_next_sibling)

    def _backend_with_caches(self, *, has_conv: bool):
        backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
        caches = SimpleNamespace(
            conv=[torch.empty(1)] if has_conv else [],
            temporal=torch.empty(2, 3),
            intermediate_ssm=torch.empty(2, 2, 2, 3),
            intermediate_conv_window=[torch.empty(2, 2, 3)] if has_conv else [],
            gla_verify_k=None,
            gla_verify_v=None,
        )
        backend.linear_attn_backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(
                mamba_cache_indices=torch.tensor([0, 1], dtype=torch.int32)
            ),
            req_to_token_pool=SimpleNamespace(
                get_speculative_mamba2_params_all_layers=lambda: caches
            ),
        )
        return backend, caches

    def _run_verify_update(self, backend):
        calls = []

        def fake_scatter(dst, src, dst_indices, step_indices):
            calls.append((dst, src, dst_indices, step_indices))

        commit_steps = torch.tensor([0, 1], dtype=torch.int32)
        track_indices = torch.tensor([2, 3], dtype=torch.int32)
        track_steps = torch.tensor([1, -1], dtype=torch.int32)
        with patch(
            "sglang.srt.layers.attention.hybrid_linear_attn_backend.fused_mamba_state_scatter_with_mask",
            fake_scatter,
        ):
            backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=commit_steps,
                mamba_track_indices=track_indices,
                mamba_steps_to_track=track_steps,
                model=None,
            )
        return calls, commit_steps, track_indices, track_steps

    def _assert_scatter_call(self, call, dst, src, dst_indices, step_indices):
        self.assertIs(call[0], dst)
        self.assertIs(call[1], src)
        self.assertTrue(torch.equal(call[2], dst_indices))
        self.assertTrue(torch.equal(call[3], step_indices))

    def test_verify_update_skips_conv_scatter_when_conv_cache_is_empty(self):
        backend, caches = self._backend_with_caches(has_conv=False)

        calls, commit_steps, track_indices, track_steps = self._run_verify_update(
            backend
        )

        verify_indices = (
            backend.linear_attn_backend.forward_metadata.mamba_cache_indices
        )
        self.assertEqual(len(calls), 2)
        self._assert_scatter_call(
            calls[0],
            caches.temporal,
            caches.intermediate_ssm,
            verify_indices,
            commit_steps,
        )
        self._assert_scatter_call(
            calls[1],
            caches.temporal,
            caches.intermediate_ssm,
            track_indices,
            track_steps,
        )

    def test_verify_update_scatter_includes_conv_when_conv_cache_exists(self):
        backend, caches = self._backend_with_caches(has_conv=True)

        calls, commit_steps, track_indices, track_steps = self._run_verify_update(
            backend
        )

        verify_indices = (
            backend.linear_attn_backend.forward_metadata.mamba_cache_indices
        )
        self.assertEqual(len(calls), 4)
        self._assert_scatter_call(
            calls[0],
            caches.temporal,
            caches.intermediate_ssm,
            verify_indices,
            commit_steps,
        )
        self._assert_scatter_call(
            calls[1],
            caches.conv[0],
            caches.intermediate_conv_window[0],
            verify_indices,
            commit_steps,
        )
        self._assert_scatter_call(
            calls[2],
            caches.temporal,
            caches.intermediate_ssm,
            track_indices,
            track_steps,
        )
        self._assert_scatter_call(
            calls[3],
            caches.conv[0],
            caches.intermediate_conv_window[0],
            track_indices,
            track_steps,
        )


if __name__ == "__main__":
    unittest.main()
