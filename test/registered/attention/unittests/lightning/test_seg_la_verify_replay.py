import unittest

import torch
import triton

from sglang.kernels.ops.attention.linear.seg_la import (
    seg_la_mtp_kernel,
    seg_la_sum_kernel,
)
from sglang.kernels.ops.attention.linear.seg_la_verify_replay import (
    build_verify_parents,
    commit_verify_kv,
    verify_with_kv_cache,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


def _links(parents, bs, index_dtype=torch.int32):
    children, siblings = [-1] * len(parents), [-1] * len(parents)
    for node in range(len(parents)):
        row = [i for i, parent in enumerate(parents) if parent == node]
        if row:
            children[node] = row[0]
        for left, right in zip(row, row[1:]):
            siblings[left] = right
    return tuple(
        torch.tensor(x, dtype=index_dtype, device="cuda").repeat(bs, 1)
        for x in (children, siblings)
    )


def _snapshots(q, k, v, state, slots, slopes, steps, links):
    bs, heads, dim = slots.numel(), q.shape[1], q.shape[2]
    cache = torch.zeros(bs, steps, heads, dim, dim, device=q.device, dtype=state.dtype)
    partial = torch.zeros(4, bs * steps, heads, dim, device=q.device, dtype=q.dtype)
    out = torch.empty_like(q)
    bv = 32 if bs <= 2 else 64
    next_token, next_sibling = (None, None) if links is None else links
    seg_la_mtp_kernel[(bs, heads, 4 * (dim // bv))](
        q,
        k,
        v,
        state,
        cache,
        partial,
        dim**-0.5,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        state.stride(0),
        cache.stride(0),
        partial.stride(0),
        slots,
        torch.arange(bs, device=q.device),
        slopes,
        steps,
        next_token,
        next_sibling,
        steps if links is not None else 0,
        1,
        steps if links is not None else 0,
        1,
        HEAD_DIM=dim,
        K_SPLIT_DIM=32,
        V_SPLIT_DIM=bv,
        NP2_STEP=triton.next_power_of_2(steps),
        HAS_TREE=links is not None,
        num_warps=2,
        num_stages=3,
    )
    seg_la_sum_kernel[(bs * steps,)](partial, out, DIM=heads * dim, NUM_BLOCK=4)
    return out, cache


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestSegLaVerifyReplay(CustomTestCase):
    def test_pool_accounting_and_transfer_exclude_verify_scratch(self):
        """The solver must charge the allocated scratch without advertising it for transfer."""
        from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
        from sglang.srt.mem_cache.memory_pool import MambaPool

        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=0,
            n_groups=0,
            num_heads=2,
            head_dim=128,
            state_size=128,
            conv_kernel=1,
        )
        params = Mamba2CacheParams(shape=shape, layers=[0, 2], use_kv_verify_cache=True)
        pool = MambaPool(
            size=4,
            spec_state_size=3,
            cache_params=params,
            mamba_layer_ids=[0, 2],
            device="cuda",
            speculative_num_draft_tokens=7,
        )
        cache = pool.mamba_cache
        self.assertIsNone(cache.intermediate_ssm)
        kv = cache.intermediate_kv
        self.assertEqual(
            params.verify_cache_per_token * 4 * 7, kv.numel() * kv.element_size()
        )
        self.assertEqual(
            cache.at_layer_idx(1).intermediate_kv.data_ptr(), kv[1].data_ptr()
        )
        transferred = list(pool._iter_transfer_state_tensors())
        self.assertFalse(any(t.data_ptr() == kv.data_ptr() for _, t, _ in transferred))
        self.assertTrue(
            any(t.data_ptr() == cache.temporal.data_ptr() for _, t, _ in transferred)
        )

    def test_outputs_and_commit_match_snapshots(self):
        """Replay must preserve edge rounding, slot mapping and independent track paths."""
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for bs in (1, 3):
                for parents in (
                    None,
                    (-1, 0, 1, 2, 0, 4, 4),
                    (-1,),
                    tuple([-1] + [(i - 1) // 2 for i in range(1, 31)]),
                ):
                    with self.subTest(dtype=dtype, bs=bs, parents=parents):
                        self._check(dtype, bs, parents)

    def test_eager_int64_indices(self):
        """Eager tree links and commit indices must compile like int32 graph buffers."""
        self._check(torch.float32, 3, (-1, 0, 1, 2, 0, 4, 4), torch.int64)

    def _check(self, dtype, bs, parents, index_dtype=torch.int32):
        torch.manual_seed(92)
        steps = 7 if parents is None else len(parents)
        heads, dim, layers = 2, 128, 2
        state = torch.randn(
            layers, 2 * bs + 3, heads, dim, dim, device="cuda", dtype=dtype
        )
        expected = state.clone()
        slots = torch.arange(bs, device="cuda", dtype=index_dtype).flip(0) + 1
        if bs > 1:
            slots[-1] = -1
        track_slots = torch.arange(bs, device="cuda", dtype=index_dtype) + bs + 1
        if bs > 1:
            track_slots[1] = -1
        slopes = torch.rand(layers, heads, device="cuda") * 0.1
        # Capacity may exceed this graph's verify width and live batch size.
        kv = torch.full(
            (layers, bs + 2, steps + 2, 2, heads, dim), float("nan"), device="cuda"
        )
        links = None if parents is None else _links(parents, bs, index_dtype)
        parent_tensor = None if parents is None else torch.empty_like(links[0])
        if links is not None:
            build_verify_parents(
                next_token=links[0], next_sibling=links[1], parents=parent_tensor
            )
            self.assertEqual(parent_tensor[0, 1:].tolist(), list(parents[1:]))
        for end in (-1, 0, steps - 1):
            last = torch.full((bs,), end, device="cuda", dtype=index_dtype)
            track_steps = torch.full_like(last, steps // 2)
            for layer in range(layers):
                q, k, v = [
                    torch.randn(
                        bs * steps, heads, dim, device="cuda", dtype=torch.bfloat16
                    )
                    for _ in range(3)
                ]
                reference, snapshots = _snapshots(
                    q, k, v, state[layer], slots, slopes[layer], steps, links
                )
                actual = verify_with_kv_cache(
                    q=q,
                    k=k,
                    v=v,
                    state=state[layer],
                    kv_cache=kv[layer],
                    slots=slots,
                    slopes=slopes[layer],
                    parents=parent_tensor,
                    scale=dim**-0.5,
                )
                live_tokens = (slots >= 0).repeat_interleave(steps)
                self.assertTrue(
                    torch.equal(reference[live_tokens], actual[live_tokens])
                )
                for row, slot in enumerate(slots.tolist()):
                    if slot >= 0:
                        if end >= 0:
                            expected[layer, slot] = snapshots[row, end]
                        if track_slots[row] >= 0:
                            expected[layer, track_slots[row]] = snapshots[
                                row, track_steps[row]
                            ]
            commit_verify_kv(
                state=state,
                kv_cache=kv,
                slots=slots,
                last_steps=last,
                slopes=slopes,
                parents=parent_tensor,
                track_slots=track_slots,
                track_steps=track_steps,
            )
            self.assertTrue(torch.equal(state, expected))

    def test_capture_links_and_graph_replay(self):
        """Captured buffers must observe new links and a different live slot on replay."""
        bs, steps, heads, dim = 1, 4, 2, 128
        q, k, v = [
            torch.randn(steps, heads, dim, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        state = torch.randn(3, heads, dim, dim, device="cuda")
        slots = torch.tensor([1], device="cuda", dtype=torch.int32)
        slopes = torch.rand(heads, device="cuda")
        kv = torch.empty(bs, steps, 2, heads, dim, device="cuda")
        next_token, next_sibling, parents = [
            torch.zeros(bs, steps, device="cuda", dtype=torch.int32) for _ in range(3)
        ]

        def run():
            build_verify_parents(
                next_token=next_token, next_sibling=next_sibling, parents=parents
            )
            return verify_with_kv_cache(
                q=q,
                k=k,
                v=v,
                state=state,
                kv_cache=kv,
                slots=slots,
                slopes=slopes,
                parents=parents,
                scale=dim**-0.5,
            )

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        for slot, row in ((2, (-1, 0, 0, 2)), (1, (-1, 0, 1, 0))):
            slots.fill_(slot)
            nt, ns = _links(row, bs)
            next_token.copy_(nt)
            next_sibling.copy_(ns)
            graph.replay()
            reference, _ = _snapshots(q, k, v, state, slots, slopes, steps, (nt, ns))
            self.assertTrue(torch.equal(reference, actual))


if __name__ == "__main__":
    unittest.main()
