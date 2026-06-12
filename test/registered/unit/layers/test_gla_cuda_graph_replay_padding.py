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

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeMambaReqToTokenPool:
    """Maps a req_pool_index to a distinct mamba state slot (req * 10)."""

    def get_mamba_indices(self, req_pool_indices):
        return req_pool_indices.clone() * 10


class _VerifyForwardMode:
    def is_decode_or_idle(self):
        return False

    def is_target_verify(self):
        return True


def _replay_view(real_bs: int):
    """A replay forward-batch view exposing the live (un-padded) request count
    through the ``_source_forward_batch`` side channel, as build_replay_fb_view
    sets it on the new out_graph protocol."""
    return SimpleNamespace(_source_forward_batch=SimpleNamespace(batch_size=real_bs))


def _make_backend(bs: int, draft_token_num: int) -> SimpleGLAAttnBackend:
    """A SimpleGLAAttnBackend stub carrying only the state _replay_metadata reads.

    The per-bs static buffers are indexed by ``[bs - 1]``, so the lists must be
    at least ``bs`` long with the active entry sized for this batch.
    """
    backend = SimpleGLAAttnBackend.__new__(SimpleGLAAttnBackend)
    backend.req_to_token_pool = _FakeMambaReqToTokenPool()
    backend.topk = 1  # topk==1 skips the retrieve_* tree-mask block
    backend.state_indices_list = [None] * bs
    backend.state_indices_list[bs - 1] = torch.full((bs,), -99, dtype=torch.int32)
    backend.query_start_loc_list = [None] * bs
    backend.query_start_loc_list[bs - 1] = torch.zeros(bs + 1, dtype=torch.int32)
    backend.cached_cuda_graph_verify_query_start_loc = torch.arange(
        bs * draft_token_num + 1, dtype=torch.int32
    )
    return backend


class TestGLACudaGraphReplayPadding(CustomTestCase):
    """Regression guard for the GLA CUDA-graph short-prompt padding bug.

    During CUDA-graph replay the batch is padded up to a captured size and the
    padding reqs occupy the tail [real_bs:bs]. Inferring the padding count from
    ``seq_lens == fill_value`` (fill_value == 1) misclassifies a real request
    whose verify prefix is a single token as padding, zeroing its mamba state
    index to -1 and corrupting the GLA recurrence. The fix derives the count
    from the live request count off ``_source_forward_batch`` instead, which is
    value-independent.

    ``_replay_metadata`` is defined once in MambaAttnBackendBase and not
    overridden by any subclass, so these topk==1 cases cover the shared padding
    logic for the whole mamba family. The only backend-specific branch
    (``builds_retrieve_parent_in_metadata``, set True only by SimpleGLA) is
    skipped at topk==1, so that branch is not exercised here.
    """

    def test_real_short_request_is_not_treated_as_padding(self):
        # bs == real batch size: a single real request whose seq_len collides
        # with the fill value must keep its mamba state index.
        backend = _make_backend(bs=1, draft_token_num=2)
        backend._replay_metadata(
            bs=1,
            req_pool_indices=torch.tensor([7], dtype=torch.int32),
            forward_mode=_VerifyForwardMode(),
            spec_info=SimpleNamespace(draft_token_num=2),
            seq_lens_cpu=torch.tensor([1], dtype=torch.int32),  # == fill_value
            forward_batch=_replay_view(real_bs=1),
        )
        # req 7 -> slot 70, preserved (not -1, not 0)
        self.assertEqual(int(backend.state_indices_list[0][0]), 70)

    def test_seq_len_heuristic_fallback_reproduces_the_bug(self):
        # Without the _source_forward_batch side channel the legacy seq_len
        # heuristic kicks in and misclassifies the same real request as padding
        # (-1). This documents exactly the corruption the fix removes.
        backend = _make_backend(bs=1, draft_token_num=2)
        backend._replay_metadata(
            bs=1,
            req_pool_indices=torch.tensor([7], dtype=torch.int32),
            forward_mode=_VerifyForwardMode(),
            spec_info=SimpleNamespace(draft_token_num=2),
            seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
            forward_batch=None,
        )
        self.assertEqual(int(backend.state_indices_list[0][0]), -1)

    def test_genuine_padding_is_still_marked(self):
        # bs=2 with one real request (batch_size=1) and one trailing padding
        # row: the real row keeps its slot, the padding row is marked -1.
        backend = _make_backend(bs=2, draft_token_num=2)
        backend._replay_metadata(
            bs=2,
            req_pool_indices=torch.tensor([7, 3], dtype=torch.int32),
            forward_mode=_VerifyForwardMode(),
            spec_info=SimpleNamespace(draft_token_num=2),
            seq_lens_cpu=torch.tensor([5, 1], dtype=torch.int32),
            forward_batch=_replay_view(real_bs=1),
        )
        self.assertEqual(int(backend.state_indices_list[1][0]), 70)  # real
        self.assertEqual(int(backend.state_indices_list[1][1]), -1)  # padding


if __name__ == "__main__":
    unittest.main()
