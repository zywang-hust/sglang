"""seg_la tree-verify links against torch and chain-kernel references."""

import unittest

import torch

from sglang.kernels.ops.attention.linear.seg_la import SegLaMeta, seg_la_fwd
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-large-amd")

HEAD_DIM = 128
NUM_HEADS = 2
# Same seg_la kernel family and value as the kit's LIGHTNING_ATOL.
ATOL = 3e-2
RTOL = 3e-2


def _reference_tree_verify(q, k, v, initial_state, decay, parents, softmax_scale):
    """Fp32 per-token recurrence with parent forks for one request;
    parents are local step indices, root == -1.

    Returns (outputs, per-token state snapshots).
    """
    step, num_heads, _ = q.shape
    outputs = torch.empty_like(q)
    token_states = []
    for token_idx in range(step):
        parent = parents[token_idx]
        base = initial_state if parent < 0 else token_states[parent]
        new_state = base * decay.view(num_heads, 1, 1) + torch.einsum(
            "hk,hv->hkv", k[token_idx], v[token_idx]
        )
        outputs[token_idx] = torch.einsum(
            "hk,hkv->hv", q[token_idx] * softmax_scale, new_state
        )
        token_states.append(new_state)
    return outputs, torch.stack(token_states)


def _links_from_parents(parents):
    """Build retrieve link rows from parents;
    children are linked in ascending token order,
    matching build_tree_kernel_efficient."""
    step = len(parents)
    next_token = [-1] * step
    next_sibling = [-1] * step
    children = {i: [] for i in range(-1, step)}
    for token_idx in range(step):
        children[parents[token_idx]].append(token_idx)
    for node, node_children in children.items():
        if node >= 0 and node_children:
            next_token[node] = node_children[0]
        for left, right in zip(node_children, node_children[1:]):
            next_sibling[left] = right
    return next_token, next_sibling


def _link_tensors(parents_rows, device):
    rows = [_links_from_parents(parents) for parents in parents_rows]
    return (
        torch.tensor([r[0] for r in rows], dtype=torch.int64, device=device),
        torch.tensor([r[1] for r in rows], dtype=torch.int64, device=device),
    )


def _make_mtp_inputs(bs, step, device, seed):
    torch.manual_seed(seed)
    length = bs * step
    q = torch.randn(length, NUM_HEADS, HEAD_DIM, dtype=torch.float32, device=device)
    k = torch.randn(length, NUM_HEADS, HEAD_DIM, dtype=torch.float32, device=device)
    v = torch.randn(length, NUM_HEADS, HEAD_DIM, dtype=torch.float32, device=device)
    s = (
        torch.randn(
            bs, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device
        )
        * 0.05
    )
    caches = torch.zeros(
        bs, step, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device
    )
    decay_scales = torch.tensor([0.5, 0.25], dtype=torch.float32, device=device)
    cache_indices = torch.arange(bs, dtype=torch.int32, device=device)
    meta = SegLaMeta(
        batch_size=bs,
        max_q_length=None,
        q_offsets=torch.arange(0, length + 1, step, dtype=torch.int32, device=device),
        s_offsets=torch.arange(bs, dtype=torch.int32, device=device),
        q_lengths=torch.full((bs,), step, dtype=torch.int32, device=device),
        s_scales=torch.ones(bs, dtype=torch.bool, device=device),
        mask=None,
    )
    return q, k, v, s, caches, decay_scales, cache_indices, meta


def _call_seg_la(inputs, caches, links):
    """Call seg_la_fwd on one _make_mtp_inputs bundle and the caches tensor;
    links is the (retrieve_next_token, retrieve_next_sibling) row pair."""
    q, k, v, s, _, decay_scales, cache_indices, meta = inputs
    retrieve_next_token, retrieve_next_sibling = links
    return seg_la_fwd(
        q=q,
        k=k,
        v=v,
        s=s,
        decay_scales=decay_scales,
        meta=meta,
        caches=caches,
        cache_indices=cache_indices,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
    )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestSegLaTreeKernels(CustomTestCase):
    # bs=3 (root_fan_out) also exercises the V_SPLIT_DIM=64 specialization.
    TREE_CASES = (
        ("deep_two_branch", ((-1, 0, 1, 2, 0, 4, 5), (-1, 0, 1, 2, 3, 0, 5))),
        (
            "root_fan_out",
            ((-1, 0, 0, 0, 0), (-1, 0, 1, 0, 3), (-1, 0, 0, 2, 2)),
        ),
        ("mixed_fork", ((-1, 0, 1, 1, 3, 0), (-1, 0, 0, 2, 2, 4))),
    )

    def test_tree_verify_matches_reference(self):
        """Each token restarts from its parent's snapshot;
        red if the in-loop parent derivation,
        the parent snapshot load, or the per-step snapshot store is wrong."""
        device = "cuda"
        for case_idx, (name, parents_rows) in enumerate(self.TREE_CASES):
            bs = len(parents_rows)
            step = len(parents_rows[0])
            with self.subTest(case=name, bs=bs, step=step):
                q, k, v, s, caches, decay_scales, cache_indices, meta = (
                    _make_mtp_inputs(bs=bs, step=step, device=device, seed=case_idx)
                )
                retrieve_next_token, retrieve_next_sibling = _link_tensors(
                    parents_rows, device
                )
                out = _call_seg_la(
                    inputs=(q, k, v, s, caches, decay_scales, cache_indices, meta),
                    caches=caches,
                    links=(retrieve_next_token, retrieve_next_sibling),
                )
                decay = torch.exp(-decay_scales)
                for req_idx in range(bs):
                    lo = req_idx * step
                    expected_out, expected_states = _reference_tree_verify(
                        q=q[lo : lo + step],
                        k=k[lo : lo + step],
                        v=v[lo : lo + step],
                        initial_state=s[req_idx],
                        decay=decay,
                        parents=parents_rows[req_idx],
                        softmax_scale=HEAD_DIM ** (-0.5),
                    )
                    torch.testing.assert_close(
                        out[lo : lo + step], expected_out, atol=ATOL, rtol=RTOL
                    )
                    torch.testing.assert_close(
                        caches[req_idx], expected_states, atol=ATOL, rtol=RTOL
                    )

    def test_chain_links_bitwise_match_chain_kernel(self):
        """A chain in tree-link form is bitwise identical to the links=None compilation.
        The fp32 snapshot round-trip is lossless,
        and the i != 0 guard keeps the tree branch off the chain path."""
        device = "cuda"
        bs, step = 2, 4
        q, k, v, s, caches, decay_scales, cache_indices, meta = _make_mtp_inputs(
            bs=bs, step=step, device=device, seed=42
        )
        caches_tree = caches.clone()
        out_chain = _call_seg_la(
            inputs=(q, k, v, s, caches, decay_scales, cache_indices, meta),
            caches=caches,
            links=(None, None),
        )
        chain_parents = [[-1] + list(range(step - 1))] * bs
        retrieve_next_token, retrieve_next_sibling = _link_tensors(
            chain_parents, device
        )
        out_tree = _call_seg_la(
            inputs=(q, k, v, s, caches, decay_scales, cache_indices, meta),
            caches=caches_tree,
            links=(retrieve_next_token, retrieve_next_sibling),
        )
        self.assertTrue(torch.equal(out_chain, out_tree))
        self.assertTrue(torch.equal(caches, caches_tree))

    def test_padded_row_ignores_stale_links(self):
        """Padded rows (s_offset == -1) must return before the tree preload,
        so stale links in the static buffers stay inert under CUDA-graph replay."""
        device = "cuda"
        bs, step = 2, 4
        q, k, v, s, caches, decay_scales, cache_indices, meta = _make_mtp_inputs(
            bs=bs, step=step, device=device, seed=3
        )
        meta.s_offsets[1] = -1
        caches[1].fill_(12345.0)
        sentinel = caches[1].clone()
        retrieve_next_token, retrieve_next_sibling = _link_tensors(
            [(-1, 0, 0, 1)], device
        )
        stale = torch.full((1, step), 7777, dtype=torch.int64, device=device)
        retrieve_next_token = torch.cat([retrieve_next_token, stale])
        retrieve_next_sibling = torch.cat([retrieve_next_sibling, stale])
        out = _call_seg_la(
            inputs=(q, k, v, s, caches, decay_scales, cache_indices, meta),
            caches=caches,
            links=(retrieve_next_token, retrieve_next_sibling),
        )
        self.assertTrue(torch.equal(caches[1], sentinel))
        decay = torch.exp(-decay_scales)
        expected_out, _ = _reference_tree_verify(
            q=q[:step],
            k=k[:step],
            v=v[:step],
            initial_state=s[0],
            decay=decay,
            parents=(-1, 0, 0, 1),
            softmax_scale=HEAD_DIM ** (-0.5),
        )
        torch.testing.assert_close(out[:step], expected_out, atol=ATOL, rtol=RTOL)

    def test_zero_link_rows_fork_from_first_token(self):
        """All-zero link rows are what CUDA-graph capture runs against.
        Until replay fills the static buffers, they must derive in-bounds parents.
        Every non-root token forks from token 0's snapshot,
        matching a root fan-out tree.
        Red if unreferenced tokens fall back to a negative parent,
        and read before the cache slot."""
        device = "cuda"
        bs, step = 2, 4
        q, k, v, s, caches, decay_scales, cache_indices, meta = _make_mtp_inputs(
            bs=bs, step=step, device=device, seed=7
        )
        zeros = torch.zeros((bs, step), dtype=torch.int64, device=device)
        out = _call_seg_la(
            inputs=(q, k, v, s, caches, decay_scales, cache_indices, meta),
            caches=caches,
            links=(zeros, zeros.clone()),
        )
        decay = torch.exp(-decay_scales)
        for req_idx in range(bs):
            lo = req_idx * step
            expected_out, _ = _reference_tree_verify(
                q=q[lo : lo + step],
                k=k[lo : lo + step],
                v=v[lo : lo + step],
                initial_state=s[req_idx],
                decay=decay,
                parents=(-1,) + (0,) * (step - 1),
                softmax_scale=HEAD_DIM ** (-0.5),
            )
            torch.testing.assert_close(
                out[lo : lo + step], expected_out, atol=ATOL, rtol=RTOL
            )


if __name__ == "__main__":
    unittest.main()
