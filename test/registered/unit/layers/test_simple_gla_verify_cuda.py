import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2StateDType,
    SimpleGLACacheParams,
    SimpleGLAStateShape,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    SIMPLE_GLA_AVAILABLE,
    HybridLinearAttnBackend,
    _build_retrieve_parent_token,
    _build_retrieve_parent_token_ref,
    _fused_recurrent_gla_verify_tree_paths,
    _recompute_and_scatter_gla_state_from_tree_paths,
)
from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
    fused_mamba_state_scatter_with_mask,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    _simple_gla_verify_recompute_reservation_bytes,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")

# Shared EAGLE tree topology (2 requests, draft_token_num=6) for the
# parent-token tests; each test wraps these in a tensor on its own device.
_TREE_NEXT_TOKEN = [
    [1, 3, -1, -1, 5, -1],
    [1, -1, 3, -1, -1, -1],
]
_TREE_NEXT_SIBLING = [
    [-1, 2, -1, 4, -1, -1],
    [-1, 2, -1, 4, 5, -1],
]


def _make_gla_inputs(
    seed: int,
    batch_size: int,
    dtn: int,
    *,
    dtype: torch.dtype = torch.float32,
    num_heads: int = 2,
    head_dim: int = 8,
    value_dim: int = 8,
    state_rows: int = None,
) -> SimpleNamespace:
    """Build the q/k/v/g_gamma/initial_state/cu_seqlens/ht_buf fixture shared
    by the verify-kernel tests; ``state_rows`` sizes the initial-state pool
    when it differs from ``batch_size``."""
    torch.manual_seed(seed)
    device = "cuda"
    total_tokens = batch_size * dtn
    q = torch.randn(1, total_tokens, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, total_tokens, num_heads, value_dim, dtype=dtype, device=device)
    g_gamma = -torch.rand(num_heads, dtype=torch.float32, device=device) * 0.2
    initial_state = torch.randn(
        batch_size if state_rows is None else state_rows,
        num_heads,
        head_dim,
        value_dim,
        dtype=torch.float32,
        device=device,
    )
    cu_seqlens = torch.arange(
        0, total_tokens + 1, step=dtn, dtype=torch.int64, device=device
    )
    ht_buf = torch.empty(
        batch_size,
        dtn,
        num_heads,
        head_dim,
        value_dim,
        dtype=torch.float32,
        device=device,
    )
    return SimpleNamespace(
        q=q,
        k=k,
        v=v,
        g_gamma=g_gamma,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        ht_buf=ht_buf,
    )


def _reference_simple_gla(q, k, v, g_gamma, scale, initial_state, batch_size, dtn):
    out = torch.empty_like(v)
    states = torch.empty(
        batch_size,
        dtn,
        q.shape[2],
        q.shape[3],
        v.shape[3],
        dtype=torch.float32,
        device=q.device,
    )
    decay = torch.exp(g_gamma.float()).view(-1, 1, 1)

    for batch_idx in range(batch_size):
        h = initial_state[batch_idx].float().clone()
        for step in range(dtn):
            token_idx = batch_idx * dtn + step
            h = h * decay + k[0, token_idx].float().unsqueeze(-1) * v[
                0, token_idx
            ].float().unsqueeze(-2)
            out[0, token_idx] = (
                h * (q[0, token_idx].float() * scale).unsqueeze(-1)
            ).sum(dim=-2)
            states[batch_idx, step] = h
    return out, states


def _reference_simple_gla_tree(
    q, k, v, g_gamma, scale, initial_state, retrieve_parent_token
):
    batch_size, dtn = retrieve_parent_token.shape
    out = torch.empty_like(v)
    states = torch.empty(
        batch_size,
        dtn,
        q.shape[2],
        q.shape[3],
        v.shape[3],
        dtype=torch.float32,
        device=q.device,
    )
    decay = torch.exp(g_gamma.float()).view(-1, 1, 1)

    for batch_idx in range(batch_size):
        for step in range(dtn):
            token_idx = batch_idx * dtn + step
            if step == 0:
                h = initial_state[batch_idx].float().clone()
            else:
                h = states[batch_idx, int(retrieve_parent_token[batch_idx, step])]
            h = h * decay + k[0, token_idx].float().unsqueeze(-1) * v[
                0, token_idx
            ].float().unsqueeze(-2)
            out[0, token_idx] = (
                h * (q[0, token_idx].float() * scale).unsqueeze(-1)
            ).sum(dim=-2)
            states[batch_idx, step] = h
    return out, states


def _path_to_node(parent: torch.Tensor, node: int) -> list[int]:
    path = [node]
    while node != 0:
        node = int(parent[node].item())
        path.append(node)
    return list(reversed(path))


def _build_varlen_recompute_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    parent: torch.Tensor,
    selected_nodes: list[tuple[int, int]],
):
    batch_size, dtn = parent.shape
    q_seq = q.reshape(batch_size, dtn, q.shape[2], q.shape[3])
    k_seq = k.reshape(batch_size, dtn, k.shape[2], k.shape[3])
    v_seq = v.reshape(batch_size, dtn, v.shape[2], v.shape[3])

    q_parts = []
    k_parts = []
    v_parts = []
    initial_parts = []
    paths = []
    cu_lens = [0]
    max_depth = 0
    for req_idx, node in selected_nodes:
        path = _path_to_node(parent[req_idx].cpu(), node)
        paths.append(path)
        max_depth = max(max_depth, len(path))
        q_parts.append(q_seq[req_idx, path])
        k_parts.append(k_seq[req_idx, path])
        v_parts.append(v_seq[req_idx, path])
        initial_parts.append(initial_state[req_idx])
        cu_lens.append(cu_lens[-1] + len(path))

    ht = torch.empty(
        len(selected_nodes),
        max_depth,
        k.shape[2],
        k.shape[3],
        v.shape[3],
        dtype=torch.float32,
        device=k.device,
    )
    return (
        torch.cat(q_parts, dim=0).contiguous().unsqueeze(0),
        torch.cat(k_parts, dim=0).contiguous().unsqueeze(0),
        torch.cat(v_parts, dim=0).contiguous().unsqueeze(0),
        torch.stack(initial_parts, dim=0).contiguous(),
        torch.tensor(cu_lens, dtype=torch.int64, device=k.device),
        ht,
        paths,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
@unittest.skipUnless(SIMPLE_GLA_AVAILABLE, "fla simple_gla is required")
class TestSimpleGLAVerifyCuda(CustomTestCase):
    def test_retrieve_parent_token_cuda_matches_cpu_reference(self):
        retrieve_next_token = torch.tensor(_TREE_NEXT_TOKEN, dtype=torch.int32)
        retrieve_next_sibling = torch.tensor(_TREE_NEXT_SIBLING, dtype=torch.int32)
        expected = _build_retrieve_parent_token_ref(
            retrieve_next_token, retrieve_next_sibling
        )
        out = torch.full((3, 6), -7, dtype=torch.int32, device="cuda")

        _build_retrieve_parent_token(
            retrieve_next_token.cuda(),
            retrieve_next_sibling.cuda(),
            out=out,
        )

        torch.testing.assert_close(out[:2].cpu(), expected)
        torch.testing.assert_close(out[2].cpu(), torch.zeros(6, dtype=torch.int32))

    def test_verify_kernel_writes_intermediate_states(self):
        batch_size = 2
        dtn = 3
        head_dim = 8
        inputs = _make_gla_inputs(20260605, batch_size, dtn, head_dim=head_dim)

        out = _fused_recurrent_gla_verify_tree_paths(
            q=inputs.q,
            k=inputs.k,
            v=inputs.v,
            g_gamma=inputs.g_gamma,
            scale=head_dim**-0.5,
            initial_state=inputs.initial_state,
            cu_seqlens=inputs.cu_seqlens,
            ht_buf=inputs.ht_buf,
        )
        ref_out, ref_states = _reference_simple_gla(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.g_gamma,
            head_dim**-0.5,
            inputs.initial_state,
            batch_size,
            dtn,
        )

        torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(inputs.ht_buf, ref_states, rtol=1e-5, atol=1e-5)

    def test_tree_parent_token_drives_gla_verify_state(self):
        batch_size = 2
        dtn = 6
        head_dim = 8
        device = "cuda"

        retrieve_next_token = torch.tensor(
            _TREE_NEXT_TOKEN, dtype=torch.int32, device=device
        )
        retrieve_next_sibling = torch.tensor(
            _TREE_NEXT_SIBLING, dtype=torch.int32, device=device
        )
        retrieve_parent_token = _build_retrieve_parent_token(
            retrieve_next_token, retrieve_next_sibling
        )
        inputs = _make_gla_inputs(20260606, batch_size, dtn, head_dim=head_dim)

        out = _fused_recurrent_gla_verify_tree_paths(
            q=inputs.q,
            k=inputs.k,
            v=inputs.v,
            g_gamma=inputs.g_gamma,
            scale=head_dim**-0.5,
            initial_state=inputs.initial_state,
            cu_seqlens=inputs.cu_seqlens,
            ht_buf=inputs.ht_buf,
            retrieve_parent_token=retrieve_parent_token,
        )
        ref_out, ref_states = _reference_simple_gla_tree(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.g_gamma,
            head_dim**-0.5,
            inputs.initial_state,
            retrieve_parent_token,
        )

        torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(inputs.ht_buf, ref_states, rtol=1e-5, atol=1e-5)

    def test_verify_kernel_supports_indexed_initial_state(self):
        batch_size = 2
        dtn = 3
        head_dim = 8
        device = "cuda"
        inputs = _make_gla_inputs(
            20260607, batch_size, dtn, head_dim=head_dim, state_rows=5
        )

        state_indices = torch.tensor([4, 1], dtype=torch.int32, device=device)
        gathered_initial_state = inputs.initial_state[state_indices.long()].contiguous()
        gathered_ht_buf = torch.empty_like(inputs.ht_buf)

        indexed_out = _fused_recurrent_gla_verify_tree_paths(
            q=inputs.q,
            k=inputs.k,
            v=inputs.v,
            g_gamma=inputs.g_gamma,
            scale=head_dim**-0.5,
            initial_state=inputs.initial_state,
            initial_state_indices=state_indices,
            cu_seqlens=inputs.cu_seqlens,
            ht_buf=inputs.ht_buf,
        )
        gathered_out = _fused_recurrent_gla_verify_tree_paths(
            q=inputs.q,
            k=inputs.k,
            v=inputs.v,
            g_gamma=inputs.g_gamma,
            scale=head_dim**-0.5,
            initial_state=gathered_initial_state,
            cu_seqlens=inputs.cu_seqlens,
            ht_buf=gathered_ht_buf,
        )

        torch.testing.assert_close(indexed_out, gathered_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(inputs.ht_buf, gathered_ht_buf, rtol=1e-5, atol=1e-5)

    def test_verify_kernel_masks_negative_initial_state_index(self):
        # CUDA-graph replay marks padded verify rows with state index -1 while
        # they still carry draft_token_num tokens; the kernel must read a zero
        # initial state for them instead of garbage.
        batch_size = 2
        dtn = 3
        head_dim = 8
        device = "cuda"
        inputs = _make_gla_inputs(
            20260608, batch_size, dtn, head_dim=head_dim, state_rows=4
        )

        state_indices = torch.tensor([2, -1], dtype=torch.int32, device=device)
        valid = state_indices >= 0
        gathered_initial_state = torch.zeros_like(inputs.initial_state[:batch_size])
        gathered_initial_state[valid] = inputs.initial_state[
            state_indices[valid].long()
        ]

        indexed_out = _fused_recurrent_gla_verify_tree_paths(
            q=inputs.q,
            k=inputs.k,
            v=inputs.v,
            g_gamma=inputs.g_gamma,
            scale=head_dim**-0.5,
            initial_state=inputs.initial_state,
            initial_state_indices=state_indices,
            cu_seqlens=inputs.cu_seqlens,
            ht_buf=inputs.ht_buf,
        )
        ref_out, ref_states = _reference_simple_gla(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.g_gamma,
            head_dim**-0.5,
            gathered_initial_state,
            batch_size,
            dtn,
        )

        torch.testing.assert_close(indexed_out, ref_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(inputs.ht_buf, ref_states, rtol=1e-5, atol=1e-5)

    def test_scatter_uses_batch_rows_and_destination_slots(self):
        batch_size = 2
        dtn = 3
        num_heads = 1
        head_dim = 4
        value_dim = 4
        device = "cuda"

        src = torch.arange(
            batch_size * dtn * num_heads * head_dim * value_dim,
            dtype=torch.float32,
            device=device,
        ).reshape(1, batch_size, dtn, num_heads, head_dim, value_dim)
        dst = torch.full(
            (1, 10, num_heads, head_dim, value_dim),
            -1.0,
            dtype=torch.float32,
            device=device,
        )
        dst_indices = torch.tensor([7, 3], dtype=torch.int32, device=device)
        step_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)

        fused_mamba_state_scatter_with_mask(dst, src, dst_indices, step_indices)

        torch.testing.assert_close(dst[0, 7], src[0, 0, 2])
        torch.testing.assert_close(dst[0, 3], src[0, 1, 0])
        self.assertTrue(torch.all(dst[0, 0] == -1.0).item())

    def test_varlen_recompute_matches_tree_snapshots_for_commit_and_track_nodes(self):
        batch_size = 3
        dtn = 6
        num_heads = 2
        head_dim = 8
        value_dim = 8
        device = "cuda"

        parent = torch.tensor(
            [
                [0, 0, 0, 1, 1, 4],
                [0, 0, 1, 1, 3, 3],
                [0, 0, 0, 2, 2, 4],
            ],
            dtype=torch.int64,
            device=device,
        )
        inputs = _make_gla_inputs(
            20260609,
            batch_size,
            dtn,
            dtype=torch.bfloat16,
            num_heads=num_heads,
            head_dim=head_dim,
            value_dim=value_dim,
        )
        q, k, v = inputs.q, inputs.k, inputs.v
        g_gamma = inputs.g_gamma
        initial_state = inputs.initial_state
        snapshot = inputs.ht_buf

        _fused_recurrent_gla_verify_tree_paths(
            q=q,
            k=k,
            v=v,
            g_gamma=g_gamma,
            scale=head_dim**-0.5,
            initial_state=initial_state,
            cu_seqlens=inputs.cu_seqlens,
            ht_buf=snapshot,
            retrieve_parent_token=parent,
        )

        selected_nodes = [
            (0, 5),  # accepted terminal
            (1, 4),
            (2, 3),
            (0, 3),  # track interval point before accepted terminal
            (1, 2),
            (2, 1),
        ]
        (
            q_recompute,
            k_recompute,
            v_recompute,
            initial_recompute,
            cu_recompute,
            recompute_states,
            paths,
        ) = _build_varlen_recompute_inputs(
            q, k, v, initial_state, parent, selected_nodes
        )

        _fused_recurrent_gla_verify_tree_paths(
            q=q_recompute,
            k=k_recompute,
            v=v_recompute,
            g_gamma=g_gamma,
            scale=head_dim**-0.5,
            initial_state=initial_recompute,
            cu_seqlens=cu_recompute,
            ht_buf=recompute_states,
            tokens_per_seq=recompute_states.shape[1],
        )

        for row, (req_idx, node) in enumerate(selected_nodes):
            torch.testing.assert_close(
                recompute_states[row, len(paths[row]) - 1],
                snapshot[req_idx, node],
                rtol=0,
                atol=0,
            )

        # The production scatter kernel must recompute the same states from
        # the per-request k/v and write them to the commit/track slots.
        commit_nodes = torch.tensor([5, 4, 3], dtype=torch.int64, device=device)
        track_nodes = torch.tensor([3, 2, 1], dtype=torch.int64, device=device)
        initial_state_indices = torch.arange(
            batch_size, dtype=torch.int64, device=device
        )
        commit_dst_indices = initial_state_indices + batch_size
        track_dst_indices = initial_state_indices + 2 * batch_size
        states = torch.randn(
            1,
            3 * batch_size + 1,
            num_heads,
            head_dim,
            value_dim,
            dtype=torch.float32,
            device=device,
        )
        states[0, :batch_size] = initial_state
        states_before = states.clone()

        _recompute_and_scatter_gla_state_from_tree_paths(
            k=k.reshape(1, batch_size, dtn, num_heads, head_dim),
            v=v.reshape(1, batch_size, dtn, num_heads, value_dim),
            g_gamma=g_gamma,
            states=states,
            retrieve_parent_token=parent,
            selected_req_indices=torch.arange(
                batch_size, dtype=torch.int64, device=device
            ),
            initial_state_indices=initial_state_indices,
            commit_nodes=commit_nodes,
            commit_dst_indices=commit_dst_indices,
            track_nodes=track_nodes,
            track_dst_indices=track_dst_indices,
        )

        for req_idx in range(batch_size):
            torch.testing.assert_close(
                states[0, commit_dst_indices[req_idx]],
                snapshot[req_idx, commit_nodes[req_idx]],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                states[0, track_dst_indices[req_idx]],
                snapshot[req_idx, track_nodes[req_idx]],
                rtol=0,
                atol=0,
            )
        # Initial-state rows and the spare row stay untouched.
        torch.testing.assert_close(
            states[0, :batch_size], states_before[0, :batch_size], rtol=0, atol=0
        )
        torch.testing.assert_close(states[0, -1], states_before[0, -1], rtol=0, atol=0)

    def test_verify_recompute_pool_and_planner_shapes(self):
        cache_params = SimpleGLACacheParams(
            layers=[0, 1],
            shape=SimpleGLAStateShape.create(
                tp_world_size=1,
                num_heads=2,
                head_dim=4,
                state_size=6,
            ),
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
        )
        spec_rows = 3
        draft_token_num = 5

        pool = MambaPool(
            size=4,
            spec_state_size=spec_rows,
            cache_params=cache_params,
            mamba_layer_ids=cache_params.layers,
            device="cuda",
            speculative_num_draft_tokens=draft_token_num,
        )

        cache = pool.get_speculative_mamba2_params_all_layers()
        self.assertIsNone(cache.intermediate_ssm)
        self.assertEqual(
            tuple(cache.gla_verify_k.shape),
            (2, spec_rows + 1, draft_token_num, 2, 4),
        )
        self.assertEqual(
            tuple(cache.gla_verify_v.shape),
            (2, spec_rows + 1, draft_token_num, 2, 6),
        )

        layer_cache = pool.mamba2_layer_cache(1)
        self.assertEqual(
            tuple(layer_cache.gla_verify_k.shape),
            (spec_rows + 1, draft_token_num, 2, 4),
        )
        self.assertEqual(
            tuple(layer_cache.gla_verify_v.shape),
            (spec_rows + 1, draft_token_num, 2, 6),
        )

        expected_bytes = 2 * (spec_rows + 1) * draft_token_num * 2 * (4 + 6) * 4
        self.assertEqual(
            _simple_gla_verify_recompute_reservation_bytes(
                cache_params, spec_rows, draft_token_num
            ),
            expected_bytes,
        )

    def _run_verify_recompute_update(self, *, use_parent_table: bool, use_track: bool):
        """Drive _update_simple_gla_state_after_mtp_verify_with_recompute against
        snapshots from the verify kernel.

        ``use_parent_table=False`` exercises the topk==1 path where
        _forward_metadata builds no parent table and the update falls back to a
        linear draft chain; ``use_track=False`` exercises the HAS_TRACK=False
        kernel specialization, where only commit slots may change.
        """
        torch.manual_seed(20260612)
        device = "cuda"
        layers = 2
        batch_size = 3
        num_live_states = 8
        dtn = 6
        num_heads = 2
        head_dim = 8
        value_dim = 8

        parent = (
            torch.tensor(
                [
                    [0, 0, 0, 1, 1, 4],
                    [0, 0, 1, 1, 3, 3],
                    [0, 0, 0, 2, 2, 4],
                ],
                dtype=torch.int64,
                device=device,
            )
            if use_parent_table
            else None
        )
        g_gamma = -torch.rand(num_heads, dtype=torch.float32, device=device) * 0.2
        state_indices = torch.tensor([2, 4, 5], dtype=torch.int64, device=device)
        track_indices = torch.tensor([6, 7, 1], dtype=torch.int64, device=device)
        commit_steps = torch.tensor([5, 4, 3], dtype=torch.int64, device=device)
        track_steps = torch.tensor([3, -1, 1], dtype=torch.int64, device=device)
        initial_temporal = torch.randn(
            layers,
            num_live_states,
            num_heads,
            head_dim,
            value_dim,
            dtype=torch.float32,
            device=device,
        )
        saved_k = torch.empty(
            layers,
            batch_size,
            dtn,
            num_heads,
            head_dim,
            dtype=torch.float32,
            device=device,
        )
        saved_v = torch.empty(
            layers,
            batch_size,
            dtn,
            num_heads,
            value_dim,
            dtype=torch.float32,
            device=device,
        )
        snapshots = torch.empty(
            layers,
            batch_size,
            dtn,
            num_heads,
            head_dim,
            value_dim,
            dtype=torch.float32,
            device=device,
        )
        cu_seqlens = torch.arange(
            0, batch_size * dtn + 1, step=dtn, dtype=torch.int64, device=device
        )

        for layer_idx in range(layers):
            q = torch.randn(
                1,
                batch_size * dtn,
                num_heads,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            k = torch.randn_like(q)
            v = torch.randn(
                1,
                batch_size * dtn,
                num_heads,
                value_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            saved_k[layer_idx].copy_(k.reshape(batch_size, dtn, num_heads, head_dim))
            saved_v[layer_idx].copy_(v.reshape(batch_size, dtn, num_heads, value_dim))
            _fused_recurrent_gla_verify_tree_paths(
                q=q,
                k=k,
                v=v,
                g_gamma=g_gamma,
                scale=head_dim**-0.5,
                initial_state=initial_temporal[layer_idx],
                initial_state_indices=state_indices,
                cu_seqlens=cu_seqlens,
                ht_buf=snapshots[layer_idx],
                retrieve_parent_token=parent,
            )

        hybrid_backend = object.__new__(HybridLinearAttnBackend)
        hybrid_backend.linear_attn_backend = SimpleNamespace(
            g_gamma=g_gamma,
            forward_metadata=SimpleNamespace(
                retrieve_parent_token=parent,
                mamba_cache_indices=state_indices,
            ),
        )
        caches = SimpleNamespace(
            conv=[],
            temporal=initial_temporal.clone(),
            gla_verify_k=saved_k,
            gla_verify_v=saved_v,
        )
        hybrid_backend._update_simple_gla_state_after_mtp_verify_with_recompute(
            last_correct_step_indices=commit_steps,
            mamba_track_indices=track_indices if use_track else None,
            mamba_steps_to_track=track_steps if use_track else None,
            state_indices_tensor=state_indices,
            mamba_caches=caches,
            request_number=batch_size,
        )

        for layer_idx in range(layers):
            for req_idx, commit_step in enumerate(commit_steps.tolist()):
                torch.testing.assert_close(
                    caches.temporal[layer_idx, state_indices[req_idx]],
                    snapshots[layer_idx, req_idx, commit_step],
                    rtol=0,
                    atol=0,
                )
            for req_idx, track_step in enumerate(track_steps.tolist()):
                if not use_track or track_step < 0:
                    continue
                torch.testing.assert_close(
                    caches.temporal[layer_idx, track_indices[req_idx]],
                    snapshots[layer_idx, req_idx, track_step],
                    rtol=0,
                    atol=0,
                )

        untouched = torch.ones(num_live_states, dtype=torch.bool, device=device)
        untouched[state_indices] = False
        if use_track:
            untouched[track_indices[track_steps >= 0]] = False
        torch.testing.assert_close(
            caches.temporal[:, untouched],
            initial_temporal[:, untouched],
            rtol=0,
            atol=0,
        )

    def test_verify_recompute_update_commits_and_tracks_from_initial_state(self):
        self._run_verify_recompute_update(use_parent_table=True, use_track=True)

    def test_verify_recompute_update_falls_back_to_chain_without_parent_table(self):
        self._run_verify_recompute_update(use_parent_table=False, use_track=True)

    def test_verify_recompute_update_without_track_touches_only_commit_slots(self):
        self._run_verify_recompute_update(use_parent_table=True, use_track=False)


if __name__ == "__main__":
    unittest.main()
