from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.minicpm_backend import (
    MiniCPMSparseBackend,
    _build_mixed_tree_compaction_inputs,
    _compact_dense_tree_rows,
    _derive_sparse_cache_seqlens_from_topk,
)
from sglang.srt.layers.attention.minicpm_sparse_kernels import (
    compact_sparse_tree_page_table,
)
from sglang.srt.layers.attention.minicpm_sparse_utils import (
    SparseConfig,
    SparseMetadataBuilder,
    sparse_block_quantized_count,
)
from sglang.srt.mem_cache.common import (
    alloc_sparse_compressed_slots_for_range,
    alloc_token_slots,
    free_sparse_compressed_slots_for_range,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
# The CUDA-vs-CPU compaction tests below skip on the GPU-less stage-a runner;
# register the file on a CUDA runner too so the Triton kernel contract runs in CI.
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class _FakeAllocator:
    def __init__(self):
        self.next_slot = 100
        self.freed = []

    def alloc(self, num_tokens):
        ret = torch.arange(
            self.next_slot, self.next_slot + num_tokens, dtype=torch.int64
        )
        self.next_slot += num_tokens
        return ret

    def free(self, indices):
        self.freed.append(indices.clone())


class _FakeTreeCache:
    def __init__(self, req_to_token_pool):
        self.page_size = 1
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = _FakeAllocator()

    def is_chunk_cache(self):
        return True


class _FakeReqToTokenPool:
    def __init__(self):
        self.kernel_size = 4
        self.kernel_stride = 2
        self.req_to_token = torch.zeros((4, 64), dtype=torch.int32)
        self.req_to_sparse_k1_token = torch.zeros((4, 32), dtype=torch.int32)
        self.req_to_sparse_k2_token = torch.zeros((4, 32), dtype=torch.int32)

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def write_sparse_k1(self, indices, values):
        self.req_to_sparse_k1_token[indices] = values

    def write_sparse_k2(self, indices, values):
        self.req_to_sparse_k2_token[indices] = values


class _NotIdleForwardMode:
    def is_idle(self):
        return False


def _assign_req_to_token_pool_cpu(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    batch_size,
):
    pt = 0
    for i in range(batch_size):
        req_idx = int(req_pool_indices[i])
        start = int(start_offset[i])
        end = int(end_offset[i])
        length = end - start
        req_to_token[req_idx, start:end] = out_cache_loc[pt : pt + length].to(
            torch.int32
        )
        pt += length


def _prepare_sparse_verify_slots(
    batch,
    draft_token: torch.Tensor,
    draft_token_num: int,
    seq_lens_cpu: torch.Tensor,
):
    batch.input_ids = draft_token
    batch.out_cache_loc = alloc_token_slots(batch.tree_cache, len(batch.input_ids))
    end_offset = batch.seq_lens + draft_token_num
    _assign_req_to_token_pool_cpu(
        batch.req_pool_indices,
        batch.req_to_token_pool.req_to_token,
        batch.seq_lens,
        end_offset,
        batch.out_cache_loc,
        batch.batch_size(),
    )
    end_lens_cpu = seq_lens_cpu + draft_token_num
    (
        batch.token_num_sparse_k1_cpu,
        batch.token_num_sparse_k2_cpu,
    ) = alloc_sparse_compressed_slots_for_range(
        batch.tree_cache,
        batch.req_to_token_pool,
        [req.req_pool_idx for req in batch.reqs],
        seq_lens_cpu,
        end_lens_cpu,
    )


def _to_cuda(fixture):
    """Move every tensor in a compaction fixture to CUDA (ints pass through)."""
    return {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in fixture.items()
    }


def compact_sparse_tree_page_table_reference(
    page_table: torch.Tensor,
    topk_idx: torch.Tensor,
    token_to_bs: torch.Tensor,
    token_pos_in_bs: torch.Tensor,
    prefix_lens: torch.Tensor,
    mask_batch_offsets: Optional[torch.Tensor],
    custom_mask: torch.Tensor,
    source_cache_seqlens: torch.Tensor,
    draft_token_num: int,
    block_size: int,
    draft_tree_mask: Optional[torch.Tensor] = None,
    out_page_table: Optional[torch.Tensor] = None,
    out_cache_seqlens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch oracle for the Triton compaction kernel (CUDA-only in
    production); the CUDA tests below pin the kernel against this."""
    rows, max_sparse_tokens = page_table.shape
    head_group_num, token_num, sparse_topk = topk_idx.shape
    if rows != token_num * head_group_num:
        raise ValueError(
            "MiniCPM sparse tree compaction expects token-major rows: "
            f"rows={rows}, token_num={token_num}, head_group_num={head_group_num}."
        )

    device = page_table.device
    token_to_bs = token_to_bs.to(device=device, dtype=torch.int64)
    token_pos_in_bs = token_pos_in_bs.to(device=device, dtype=torch.int64)
    prefix_lens = prefix_lens.to(device=device, dtype=torch.int64)

    batch_for_token = token_to_bs
    prefix_for_token = prefix_lens[batch_for_token]
    kv_len_for_token = prefix_for_token + int(draft_token_num)
    query_idx = token_pos_in_bs - prefix_for_token - 1
    if torch.any((query_idx < 0) | (query_idx >= int(draft_token_num))):
        raise ValueError(
            "MiniCPM sparse tree compaction received token positions outside "
            f"the draft range: draft_token_num={draft_token_num}."
        )

    block_offsets = torch.arange(block_size, dtype=torch.int64, device=device)
    key_positions = (
        topk_idx.permute(1, 0, 2).to(torch.int64).unsqueeze(-1) * int(block_size)
        + block_offsets
    )
    valid = (
        (topk_idx.permute(1, 0, 2).unsqueeze(-1) >= 0)
        & (key_positions < kv_len_for_token.view(token_num, 1, 1, 1))
        & (key_positions < token_pos_in_bs.view(token_num, 1, 1, 1))
    )

    safe_key_positions = torch.where(
        valid, key_positions, torch.zeros_like(key_positions)
    )
    if draft_tree_mask is None:
        mask_batch_offsets = mask_batch_offsets.to(device=device, dtype=torch.int64)
        gather_offsets = (
            mask_batch_offsets[batch_for_token].view(token_num, 1, 1, 1)
            + query_idx.view(token_num, 1, 1, 1)
            * kv_len_for_token.view(token_num, 1, 1, 1)
            + safe_key_positions
        )
        projected = custom_mask[gather_offsets].view(
            token_num, head_group_num, sparse_topk * block_size
        )
    else:
        key_draft_idx = safe_key_positions - prefix_for_token.view(token_num, 1, 1, 1)
        draft_visible = torch.zeros_like(valid)
        draft_key_valid = (
            valid & (key_draft_idx >= 0) & (key_draft_idx < int(draft_token_num))
        )
        draft_offsets = (
            batch_for_token.view(token_num, 1, 1, 1) * int(draft_token_num) ** 2
            + query_idx.view(token_num, 1, 1, 1) * int(draft_token_num)
            + key_draft_idx.clamp(min=0)
        )
        draft_visible[draft_key_valid] = draft_tree_mask[draft_offsets][draft_key_valid]
        prefix_visible = valid & (
            safe_key_positions < prefix_for_token.view(token_num, 1, 1, 1)
        )
        projected = (prefix_visible | draft_visible).view(
            token_num, head_group_num, sparse_topk * block_size
        )
    row_mask = (
        projected & valid.view(token_num, head_group_num, sparse_topk * block_size)
    ).reshape(rows, max_sparse_tokens)

    col_offsets = torch.arange(max_sparse_tokens, device=device).view(1, -1)
    source_prefix = col_offsets < source_cache_seqlens.to(device=device).view(-1, 1)
    dense_mask = torch.zeros((rows, max_sparse_tokens), dtype=torch.bool, device=device)
    dense_mask[source_prefix] = row_mask[source_prefix]

    if out_cache_seqlens is None:
        out_cache_seqlens = torch.empty(
            (rows,), dtype=torch.int32, device=page_table.device
        )
    out_cache_seqlens.copy_(
        dense_mask.sum(dim=1, dtype=torch.int32).to(dtype=out_cache_seqlens.dtype)
    )

    if out_page_table is None:
        out_page_table = torch.zeros_like(page_table)
    else:
        out_page_table.zero_()
    ranks = torch.cumsum(dense_mask.to(torch.int32), dim=1) - 1
    row_ids = torch.arange(rows, device=device).view(-1, 1).expand_as(page_table)
    compact = dense_mask
    out_page_table[row_ids[compact], ranks[compact].to(torch.int64)] = page_table[
        compact
    ]

    return out_page_table, out_cache_seqlens


def _with_draft_tree_mask(fixture):
    """Swap the chain fixture's full custom mask for its draft-draft submask."""
    draft_token_num = fixture["draft_token_num"]
    prefix_len = int(fixture["prefix_lens"][0])
    draft_mask = (
        fixture["custom_mask"]
        .view(draft_token_num, prefix_len + draft_token_num)[:, prefix_len:]
        .contiguous()
        .flatten()
    )
    return {
        **fixture,
        "custom_mask": fixture["custom_mask"].new_empty((0,)),
        "draft_tree_mask": draft_mask,
    }


class TestMiniCPMSparseTargetVerify(CustomTestCase):
    def test_eagle_v1_verify_allocates_sparse_slots_and_releases_tail(self):
        pool = _FakeReqToTokenPool()
        tree_cache = _FakeTreeCache(pool)
        pool.req_to_sparse_k1_token[1, :6] = torch.arange(10, 16, dtype=torch.int32)
        batch = SimpleNamespace(
            forward_mode=_NotIdleForwardMode(),
            model_config=SimpleNamespace(has_sparse_attention=True, vocab_size=32000),
            tree_cache=tree_cache,
            req_to_token_pool=pool,
            req_pool_indices=torch.tensor([1], dtype=torch.int64),
            seq_lens=torch.tensor([15], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([15], dtype=torch.int64),
            reqs=[SimpleNamespace(req_pool_idx=1)],
            batch_size=lambda: 1,
        )
        _prepare_sparse_verify_slots(
            batch,
            draft_token=torch.arange(11, dtype=torch.int64),
            draft_token_num=11,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

        self.assertTrue(
            torch.equal(pool.req_to_token[1, 15:26], torch.arange(100, 111))
        )
        self.assertTrue(
            torch.equal(pool.req_to_sparse_k1_token[1, 6:12], torch.arange(111, 117))
        )
        self.assertTrue(
            torch.equal(pool.req_to_sparse_k2_token[1, :2], torch.arange(117, 119))
        )

        # Accepted length is 4, so sparse slots for logical [19, 26) are tail.
        free_sparse_compressed_slots_for_range(tree_cache, 1, 19, 26)

        self.assertEqual(len(tree_cache.token_to_kv_pool_allocator.freed), 2)
        self.assertTrue(
            torch.equal(
                tree_cache.token_to_kv_pool_allocator.freed[0],
                torch.arange(113, 117, dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                tree_cache.token_to_kv_pool_allocator.freed[1],
                torch.tensor([118], dtype=torch.int32),
            )
        )
        committed_k1 = pool.req_to_sparse_k1_token[1, :8]
        self.assertTrue(
            torch.equal(committed_k1[:6], torch.arange(10, 16, dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(committed_k1[6:], torch.arange(111, 113, dtype=torch.int32))
        )

    def test_sparse_compressed_slots_follow_overallocated_range(self):
        pool = _FakeReqToTokenPool()
        tree_cache = _FakeTreeCache(pool)

        k1_counts, k2_counts = alloc_sparse_compressed_slots_for_range(
            tree_cache,
            pool,
            req_pool_indices=[1, 3],
            start_lens=[15, 16],
            end_lens=[33, 40],
        )

        self.assertEqual(k1_counts.tolist(), [9, 12])
        self.assertEqual(k2_counts.tolist(), [3, 3])
        # Both levels draw from one allocation (k1 rows first, then k2 rows);
        # slot ids land in the request tables, not in the return value.
        self.assertTrue(
            torch.equal(pool.req_to_sparse_k1_token[1, 6:15], torch.arange(100, 109))
        )
        self.assertTrue(
            torch.equal(pool.req_to_sparse_k1_token[3, 7:19], torch.arange(109, 121))
        )
        self.assertTrue(
            torch.equal(pool.req_to_sparse_k2_token[1, 0:3], torch.arange(121, 124))
        )
        self.assertTrue(
            torch.equal(pool.req_to_sparse_k2_token[3, 1:4], torch.arange(124, 127))
        )

        free_sparse_compressed_slots_for_range(tree_cache, 3, 16, 40)

        self.assertEqual(len(tree_cache.token_to_kv_pool_allocator.freed), 2)
        self.assertTrue(
            torch.equal(
                tree_cache.token_to_kv_pool_allocator.freed[0],
                torch.arange(109, 121, dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                tree_cache.token_to_kv_pool_allocator.freed[1],
                torch.arange(124, 127, dtype=torch.int32),
            )
        )

    def _backend(self):
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        backend.page_size = 1
        backend.dense_as_sparse = True
        backend.dense_len = 0
        backend.head_group_num = 2
        backend.heads_per_group = 2
        backend.sparse_topk = 2
        backend.block_size = 4
        backend.k1_kernel_size = 4
        backend.k1_kernel_stride = 2
        backend.k2_kernel_size = 16
        backend.k2_kernel_stride = 8
        backend.req_to_sparse_k1_token = torch.arange(
            2 * 32, dtype=torch.int32
        ).reshape(2, 32)
        backend.req_to_sparse_k2_token = torch.arange(
            100, 100 + 2 * 32, dtype=torch.int32
        ).reshape(2, 32)
        sparse_config = SparseConfig(
            sparse_len=0,
            sparse_topk=1,
            kernel_size=4,
            kernel_stride=2,
            block_size=4,
            window_size=4,
            dense_len=0,
            head_dim=8,
            num_kv_heads=2,
            head_group_num=2,
            k1_kernel_size=4,
            k1_kernel_stride=2,
            k2_kernel_size=16,
            k2_kernel_stride=8,
        )
        backend.sparse_metadata_builder = SparseMetadataBuilder(
            sparse_config, num_kv_heads=2, max_context_len=64
        )
        return backend

    def _target_verify_graph_backend(self, max_bs=3, draft_token_num=4):
        backend = self._backend()
        backend.device = torch.device("cpu")
        backend.max_context_len = 64
        backend.num_sparse_topk_tokens = 8
        backend.attention_kernel_type = "unit-test-no-wrapper"
        backend.req_to_token = torch.arange(2 * 128, dtype=torch.int32).reshape(2, 128)

        sparse_rows = max_bs * draft_token_num * backend.head_group_num
        graph = {
            "verify_num_draft_tokens": draft_token_num,
            "strided_indices": torch.arange(backend.max_context_len, dtype=torch.int64),
            "page_table": torch.zeros(
                max_bs, backend.max_context_len, dtype=torch.int32
            ),
            "verify_cache_seqlens": torch.zeros(max_bs, dtype=torch.int32),
            "verify_cu_seqlens_k": torch.zeros(max_bs + 1, dtype=torch.int32),
            "verify_cu_seqlens_q": torch.arange(0, max_bs + 1, dtype=torch.int32)
            * draft_token_num,
            "verify_sparse_tree_prefix_lens": torch.zeros(max_bs, dtype=torch.int32),
            "verify_sparse_tree_draft_mask": torch.empty(
                max_bs * draft_token_num * draft_token_num, dtype=torch.bool
            ),
            "verify_token_to_bs": torch.repeat_interleave(
                torch.arange(max_bs, dtype=torch.int32), draft_token_num
            ),
            "verify_token_offsets": torch.arange(
                1, draft_token_num + 1, dtype=torch.int32
            ).repeat(max_bs),
            "verify_token_pos_in_bs": torch.zeros(
                max_bs * draft_token_num, dtype=torch.int32
            ),
            "verify_sparse_page_table": torch.zeros(
                sparse_rows, backend.max_context_len, dtype=torch.int32
            ),
            "verify_sparse_compact_page_table": torch.zeros(
                sparse_rows, backend.max_context_len, dtype=torch.int32
            ),
            "verify_sparse_cache_seqlens": torch.zeros(sparse_rows, dtype=torch.int32),
            "verify_sparse_compact_cache_seqlens": torch.zeros(
                sparse_rows, dtype=torch.int32
            ),
            "verify_sparse_cu_seqlens_q": torch.arange(
                0, sparse_rows + 1, dtype=torch.int32
            ),
            "verify_sparse_cu_seqlens_k": torch.zeros(
                sparse_rows + 1, dtype=torch.int32
            ),
            "cache_seqlens_int32_stage1": torch.zeros(max_bs, dtype=torch.int32),
            "compress_k1": torch.empty(
                max_bs * backend.max_context_len // backend.k1_kernel_stride,
                2,
                8,
                dtype=torch.bfloat16,
            ),
            "compress_k2": torch.empty(
                max_bs * backend.max_context_len // backend.k2_kernel_stride,
                2,
                8,
                dtype=torch.bfloat16,
            ),
        }
        for prefix in ("k1", "k2"):
            graph[f"{prefix}.cu_seqlens"] = torch.zeros(max_bs + 1, dtype=torch.int32)
            source_table = (
                backend.req_to_sparse_k1_token
                if prefix == "k1"
                else backend.req_to_sparse_k2_token
            )
            graph[f"{prefix}.table"] = torch.zeros(
                max_bs, source_table.shape[1], dtype=torch.int32
            )
            for name in (
                "history_compress_token_nums",
                "new_token_nums",
                "new_compress_token_nums",
                "total_compress_token_nums",
            ):
                graph[f"{prefix}.{name}"] = torch.zeros(max_bs, dtype=torch.int32)
            for name in (
                "cu_new_token_nums",
                "cu_new_compress_token_nums",
                "cu_total_compress_token_nums",
            ):
                graph[f"{prefix}.{name}"] = torch.zeros(max_bs + 1, dtype=torch.int32)
        backend.decode_cuda_graph_metadata = graph
        return backend

    def test_target_verify_metadata_uses_prefix_plus_draft_tokens(self):
        backend = self._backend()
        req_to_token = torch.arange(2 * 64, dtype=torch.int32).reshape(2, 64)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(topk=1, draft_token_num=3),
            seq_lens=torch.tensor([10, 20], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int64),
            batch_size=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            encoder_lens=None,
        )

        # New base: the sparse backend reads the pools off itself (cached in
        # __init__ from model_runner), not off forward_batch.
        backend.req_to_token_pool = forward_batch.req_to_token_pool
        backend.init_forward_metadata(forward_batch)

        metadata = backend.forward_metadata
        self.assertTrue(
            torch.equal(
                metadata.cache_seqlens_int32, torch.tensor([13, 23], dtype=torch.int32)
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.cu_seqlens_q, torch.tensor([0, 3, 6], dtype=torch.int32)
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.cu_seqlens_k, torch.tensor([0, 13, 36], dtype=torch.int32)
            )
        )
        self.assertTrue(
            torch.equal(metadata.seq_lens_cpu_for_sparse, torch.tensor([13, 23]))
        )
        self.assertEqual(forward_batch.extend_seq_lens_cpu, [3, 3])
        self.assertEqual(forward_batch.extend_prefix_lens_cpu, [10, 20])
        self.assertEqual(forward_batch.sparse_batch_size, 2)
        self.assertEqual(metadata.k1.max_seq_len, 10)

    def test_target_verify_metadata_accepts_tree_topk(self):
        backend = self._backend()
        draft_token_num = 3
        prefix_len = 10
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(
                topk=2,
                draft_token_num=draft_token_num,
                custom_mask=torch.ones(
                    draft_token_num * (prefix_len + draft_token_num),
                    dtype=torch.bool,
                ),
            ),
            seq_lens=torch.tensor([prefix_len], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([prefix_len], dtype=torch.int64),
            batch_size=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(64, dtype=torch.int32).reshape(1, 64)
            ),
            encoder_lens=None,
        )

        # New base: the sparse backend reads the pools off itself (cached in
        # __init__ from model_runner), not off forward_batch.
        backend.req_to_token_pool = forward_batch.req_to_token_pool
        backend.init_forward_metadata(forward_batch)

        metadata = backend.forward_metadata
        self.assertEqual(metadata.max_seq_len_q, draft_token_num)
        self.assertEqual(forward_batch.extend_seq_lens_cpu, [draft_token_num])

    def test_target_verify_sparse_topk_uses_prefix_cache_lens(self):
        backend = self._backend()
        backend.fuse_topk = False
        backend.kernel_size = backend.k1_kernel_size
        backend.kernel_stride = backend.k1_kernel_stride
        backend.max_context_len = 16384
        backend.init_blocks = 1
        backend.local_blocks = 2
        backend.split_stage1 = False

        prefix_len = 8192
        draft_token_num = 4
        req_to_token = torch.arange(
            prefix_len + draft_token_num, dtype=torch.int32
        ).reshape(1, -1)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(
                topk=2,
                draft_token_num=draft_token_num,
                custom_mask=torch.ones(
                    draft_token_num * (prefix_len + draft_token_num),
                    dtype=torch.bool,
                ),
            ),
            seq_lens=torch.tensor([prefix_len], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([prefix_len], dtype=torch.int64),
            batch_size=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            encoder_lens=None,
        )
        # New base: the sparse backend reads the pools off itself (cached in
        # __init__ from model_runner), not off forward_batch.
        backend.req_to_token_pool = forward_batch.req_to_token_pool
        backend.init_forward_metadata(forward_batch)
        metadata = backend.forward_metadata

        captured = {}

        def fake_compressed_attention(*args, **kwargs):
            captured["cache_lens"] = kwargs["cache_lens"].clone()
            captured["max_seqlen_q"] = args[10]
            return torch.zeros(
                (backend.head_group_num, draft_token_num, backend.sparse_topk),
                dtype=torch.int32,
            )

        with patch(
            "sglang.srt.layers.attention.minicpm_backend.compressed_attention",
            fake_compressed_attention,
        ):
            backend.sparse_get_topk_impl(
                query_layer=torch.empty(
                    (draft_token_num, backend.head_group_num, 8),
                    dtype=torch.bfloat16,
                ),
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k=metadata.cu_seqlens_k,
                max_seqlen_in_batch_q=metadata.max_seq_len_q,
                max_seqlen_in_batch_k=metadata.max_seq_len_k,
                compressed_k=torch.empty((1, 2, 8), dtype=torch.bfloat16),
                compressed_cu_seqlens=metadata.k1.cu_seqlens,
                compressed_k2=torch.empty((1, 2, 8), dtype=torch.bfloat16),
                compressed_cu_seqlens2=metadata.k2.cu_seqlens,
            )

        self.assertEqual(captured["max_seqlen_q"], draft_token_num)
        self.assertEqual(captured["cache_lens"].tolist(), [prefix_len])
        self.assertNotEqual(
            captured["cache_lens"].tolist(), [prefix_len + draft_token_num]
        )
        self.assertEqual(
            metadata.token_pos_in_bs.tolist(),
            [prefix_len + i + 1 for i in range(draft_token_num)],
        )

    def test_sparse_token_mapping_keeps_original_batch_indices(self):
        builder = self._backend().sparse_metadata_builder

        token_to_bs, token_pos_in_bs = builder.build_token_mappings(
            cu_seqlens_q_sparse_bs=torch.tensor([0, 2, 5], dtype=torch.int32),
            extend_prefix_lens_sparse=torch.tensor([10, 20], dtype=torch.int64),
            seqlen_q_sparse_bs=[2, 3],
            sparse_bs_list=[1, 3],
        )

        self.assertEqual(token_to_bs.tolist(), [1, 1, 3, 3, 3])
        self.assertEqual(token_pos_in_bs.tolist(), [11, 12, 21, 22, 23])

    def test_target_verify_cuda_graph_replay_refreshes_eagle_static_metadata(self):
        max_bs = 3
        draft_token_num = 4
        backend = self._target_verify_graph_backend(max_bs, draft_token_num)
        capture_spec = SimpleNamespace(
            draft_token_num=draft_token_num, custom_mask=None
        )
        metadata = backend._init_target_verify_graph_metadata(
            bs=max_bs,
            seq_lens=torch.tensor([1, 1, 1], dtype=torch.int64),
            req_pool_indices=torch.tensor([0, 1, 0], dtype=torch.int64),
            spec_info=capture_spec,
        )

        seq_lens = torch.tensor([10, 20, 1], dtype=torch.int64)
        draft_masks = torch.tensor(
            [
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, False, True, False],
                    [True, True, False, True],
                ],
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, False, True, False],
                    [True, False, True, True],
                ],
            ],
            dtype=torch.bool,
        )
        full_masks = []
        for prefix_len, draft_mask in zip(seq_lens[:2].tolist(), draft_masks):
            full_mask = torch.ones(
                draft_token_num,
                prefix_len + draft_token_num,
                dtype=torch.bool,
            )
            full_mask[:, prefix_len : prefix_len + draft_token_num] = draft_mask
            full_masks.append(full_mask.flatten())
        replay_spec = SimpleNamespace(
            draft_token_num=draft_token_num,
            custom_mask=torch.cat(full_masks),
        )

        backend._refresh_target_verify_graph_metadata(
            metadata,
            bs=max_bs,
            req_pool_indices=torch.tensor([0, 1, 0], dtype=torch.int64),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            spec_info=replay_spec,
            real_bs=2,
        )

        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [14, 24, 5])
        self.assertEqual(metadata.cu_seqlens_k.tolist(), [0, 14, 38, 43])
        self.assertEqual(metadata.cache_seqlens_int32_stage1.tolist(), [13, 23, 4])
        self.assertEqual(metadata.sparse_tree_prefix_lens.tolist(), [10, 20, 1])
        self.assertEqual(
            metadata.token_pos_in_bs.tolist(),
            [11, 12, 13, 14, 21, 22, 23, 24, 2, 3, 4, 5],
        )
        self.assertTrue(
            torch.equal(
                metadata.sparse_tree_draft_mask[
                    : 2 * draft_token_num * draft_token_num
                ].view(2, draft_token_num, draft_token_num),
                draft_masks,
            )
        )
        self.assertTrue(
            torch.all(
                metadata.sparse_tree_draft_mask[
                    2 * draft_token_num * draft_token_num :
                ].view(1, draft_token_num, draft_token_num)
            )
        )
        self.assertEqual(metadata.k1.cu_seqlens.tolist(), [0, 6, 17, 18])
        self.assertEqual(metadata.k1.history_compress_token_nums.tolist(), [4, 9, 0])
        self.assertEqual(metadata.k1.new_token_nums.tolist(), [6, 6, 5])
        self.assertEqual(metadata.k1.new_compress_token_nums.tolist(), [2, 2, 1])
        self.assertEqual(metadata.k1.total_compress_token_nums.tolist(), [6, 11, 1])
        self.assertEqual(metadata.k2.cu_seqlens.tolist(), [0, 0, 2, 2])
        self.assertEqual(metadata.k2.history_compress_token_nums.tolist(), [0, 1, 0])
        self.assertEqual(metadata.k2.new_token_nums.tolist(), [14, 16, 5])
        self.assertEqual(metadata.k2.new_compress_token_nums.tolist(), [0, 1, 0])
        self.assertEqual(metadata.k2.total_compress_token_nums.tolist(), [0, 2, 0])
        self.assertEqual(metadata.page_table[0, :5].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(metadata.page_table[1, :5].tolist(), [128, 129, 130, 131, 132])
        self.assertEqual(metadata.k1.table[1, :5].tolist(), [32, 33, 34, 35, 36])

    def test_sparse_tree_page_table_compaction_projects_tree_mask(self):
        fixture = self._sparse_tree_compaction_inputs()

        compact_table, compact_lens = compact_sparse_tree_page_table_reference(
            **fixture
        )

        self.assertEqual(compact_lens.tolist(), [3, 3, 4, 4, 4, 4])
        self.assertEqual(compact_table[0, :3].tolist(), [0, 1, 2])
        self.assertEqual(compact_table[2, :4].tolist(), [12, 13, 14, 15])
        self.assertEqual(compact_table[4, :4].tolist(), [24, 25, 26, 28])

    def test_sparse_tree_page_table_compaction_accepts_draft_mask(self):
        fixture = self._sparse_tree_compaction_inputs()
        full_table, full_lens = compact_sparse_tree_page_table_reference(**fixture)
        draft_table, draft_lens = compact_sparse_tree_page_table_reference(
            **_with_draft_tree_mask(fixture)
        )

        self.assertTrue(torch.equal(draft_lens, full_lens))
        self.assertTrue(torch.equal(draft_table, full_table))

    def test_sparse_tree_page_table_compaction_long_prefix_tree_visibility(self):
        fixture = self._long_prefix_sparse_tree_compaction_inputs()

        compact_table, compact_lens = compact_sparse_tree_page_table_reference(
            **fixture
        )

        self.assertEqual(
            fixture["source_cache_seqlens"].tolist(),
            [129, 129, 130, 130, 131, 131, 132, 132],
        )
        self.assertEqual(
            compact_lens.tolist(),
            [129, 129, 130, 130, 130, 130, 131, 131],
        )

        # token 2 can see draft 0 and itself, but not sibling draft 1.
        token2_head0_row = 4
        expected_token2 = list(range(128)) + [128, 130]
        self.assertEqual(
            compact_table[token2_head0_row, :130].tolist(),
            [token2_head0_row * 1000 + offset for offset in expected_token2],
        )

        # token 3 can see draft 0/1 and itself, but not sibling draft 2.
        token3_head1_row = 7
        expected_token3 = list(range(128)) + [128, 129, 131]
        self.assertEqual(
            compact_table[token3_head1_row, :131].tolist(),
            [token3_head1_row * 1000 + offset for offset in expected_token3],
        )

    def test_sparse_tree_page_table_compaction_multi_request_padded_topk(self):
        fixture = self._multi_request_sparse_tree_compaction_inputs()

        compact_table, compact_lens = compact_sparse_tree_page_table_reference(
            **fixture
        )

        self.assertEqual(compact_lens.tolist(), [5, 5, 6, 6, 5, 5, 3, 3])
        # request 0, token 1 keeps every key below its causal position.
        self.assertEqual(compact_table[2, :6].tolist(), [12, 13, 14, 15, 16, 17])
        # request 1, token 2 sees its prefix and itself; the later draft is
        # excluded by causality.
        self.assertEqual(compact_table[4, :5].tolist(), [24, 25, 26, 27, 28])
        # request 1, token 3: -1-padded slots dropped, non-ancestor draft masked.
        self.assertEqual(compact_table[6, :6].tolist(), [36, 37, 39, 0, 0, 0])

    def _dense_tree_rows_reference(
        self, page_table, prefix_lens, draft_tree_mask, head_group_num, draft_token_num
    ):
        """Brute-force oracle: a dense verify row is the full prefix followed by
        the causal, tree-visible drafts (full attention, no block selection)."""
        bs = page_table.shape[0]
        rows = bs * draft_token_num * head_group_num
        width = int((prefix_lens + draft_token_num).max())
        out = torch.zeros((rows, width), dtype=torch.int32)
        lens = torch.zeros((rows,), dtype=torch.int32)
        mask = draft_tree_mask.view(bs, draft_token_num, draft_token_num)
        for row in range(rows):
            token = row // head_group_num
            g = row % head_group_num
            b = token // draft_token_num
            qi = token % draft_token_num
            prefix = int(prefix_lens[b])
            kept = [int(page_table[b, c]) * head_group_num + g for c in range(prefix)]
            for j in range(draft_token_num):
                if j <= qi and bool(mask[b, qi, j]):
                    kept.append(int(page_table[b, prefix + j]) * head_group_num + g)
            out[row, : len(kept)] = torch.tensor(kept, dtype=torch.int32)
            lens[row] = len(kept)
        return out, lens

    def test_dense_tree_rows_match_full_prefix_and_tree_drafts(self):
        head_group_num = 2
        draft_token_num = 3
        prefix_lens = torch.tensor([4, 6], dtype=torch.int32)
        # request 0: chain (each draft sees all earlier ones); request 1: a tree
        # where draft 2's parent is draft 0, so it cannot see sibling draft 1.
        draft_tree_mask = torch.tensor(
            [
                [[True, False, False], [True, True, False], [True, True, True]],
                [[True, False, False], [True, True, False], [True, False, True]],
            ],
            dtype=torch.bool,
        ).flatten()
        page_table = torch.arange(2 * 9, dtype=torch.int32).reshape(2, 9) + 10

        rows = 2 * draft_token_num * head_group_num
        width = int((prefix_lens + draft_token_num).max())
        out_page_table = torch.full((rows, width + 5), -7, dtype=torch.int32)
        out_cache_seqlens = torch.full((rows,), -7, dtype=torch.int32)
        _compact_dense_tree_rows(
            out_page_table,
            out_cache_seqlens,
            page_table,
            torch.arange(rows),
            prefix_lens,
            draft_tree_mask,
            head_group_num,
            draft_token_num,
        )

        expected_table, expected_lens = self._dense_tree_rows_reference(
            page_table, prefix_lens, draft_tree_mask, head_group_num, draft_token_num
        )
        self.assertEqual(out_cache_seqlens.tolist(), expected_lens.tolist())
        self.assertTrue(
            torch.equal(out_page_table[:, : expected_table.shape[1]], expected_table)
        )
        # Tail past each row's length is zeroed (kernel/attention read up to len).
        for row in range(rows):
            self.assertTrue(
                torch.all(out_page_table[row, int(out_cache_seqlens[row]) :] == 0)
            )
        # request 1, draft 2 (qi=2) drops sibling draft 1: prefix 6 + drafts {0,2}.
        self.assertEqual(
            int(out_cache_seqlens[2 * 3 * head_group_num - head_group_num]), 8
        )

    def test_dense_tree_rows_match_identity_topk_compaction_reference(self):
        """Tie the dense fill to the sparse compaction reference: a dense row is a
        sparse row whose topk selects every block (identity), so feeding the
        reference identity topk over the full page table must reproduce it."""
        head_group_num = 2
        draft_token_num = 4
        block_size = 1
        prefix = 5
        kv_len = prefix + draft_token_num
        prefix_lens = torch.tensor([prefix], dtype=torch.int32)
        draft_tree_mask = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, False, True, False],
                [True, True, False, True],
            ],
            dtype=torch.bool,
        ).flatten()
        page_table_req = torch.arange(kv_len, dtype=torch.int32).reshape(1, kv_len) + 3

        rows = draft_token_num * head_group_num
        out_page_table = torch.zeros((rows, kv_len), dtype=torch.int32)
        out_cache_seqlens = torch.zeros((rows,), dtype=torch.int32)
        _compact_dense_tree_rows(
            out_page_table,
            out_cache_seqlens,
            page_table_req,
            torch.arange(rows),
            prefix_lens,
            draft_tree_mask,
            head_group_num,
            draft_token_num,
        )

        # Identity topk: block b (size 1) selects logical position b; the
        # candidate buffer holds the request's slots in identity order.
        identity = torch.arange(kv_len, dtype=torch.int32).repeat(draft_token_num, 1)
        topk_idx = torch.stack([identity, identity])
        token_to_bs = torch.zeros((draft_token_num,), dtype=torch.int32)
        token_pos_in_bs = torch.arange(
            prefix + 1, prefix + draft_token_num + 1, dtype=torch.int32
        )
        candidates = torch.stack(
            [page_table_req[0] * head_group_num + g for g in range(head_group_num)]
        )  # [head_group, kv_len]
        ref_page_table = torch.empty((rows, kv_len), dtype=torch.int32)
        for token in range(draft_token_num):
            for g in range(head_group_num):
                ref_page_table[token * head_group_num + g] = candidates[g]
        source = torch.full((rows,), kv_len, dtype=torch.int32)
        ref_table, ref_lens = compact_sparse_tree_page_table_reference(
            page_table=ref_page_table,
            topk_idx=topk_idx,
            token_to_bs=token_to_bs,
            token_pos_in_bs=token_pos_in_bs,
            prefix_lens=prefix_lens,
            mask_batch_offsets=None,
            custom_mask=torch.empty((0,), dtype=torch.bool),
            source_cache_seqlens=source,
            draft_token_num=draft_token_num,
            block_size=block_size,
            draft_tree_mask=draft_tree_mask,
        )
        self.assertEqual(out_cache_seqlens.tolist(), ref_lens.tolist())
        self.assertTrue(torch.equal(out_page_table, ref_table))

    def test_build_mixed_tree_compaction_inputs(self):
        head_group_num = 2
        draft_token_num = 2
        bs = 3
        sparse_bs_list = [0, 2]  # request 1 is dense
        prefix_lens = torch.tensor([100, 5, 200], dtype=torch.int32)
        sparse_topk = 4
        # topk_idx covers only the sparse subset, token-major: req0 tokens 0,1
        # then req2 tokens 2,3 (4 sparse tokens), 2 head groups.
        topk_idx = torch.arange(2 * 4 * sparse_topk, dtype=torch.int32).reshape(
            2, 4, sparse_topk
        )

        full_topk_idx, token_to_bs, token_pos_in_bs, dense_rows = (
            _build_mixed_tree_compaction_inputs(
                topk_idx,
                sparse_bs_list,
                prefix_lens,
                bs,
                head_group_num,
                draft_token_num,
                sparse_topk,
            )
        )

        # token-major across all requests: token t -> request t // draft.
        self.assertEqual(token_to_bs.tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(token_pos_in_bs.tolist(), [101, 102, 6, 7, 201, 202])
        # full topk: sparse tokens keep their blocks; dense request 1 is all -1.
        sparse_token = torch.tensor([True, True, False, False, True, True])
        self.assertTrue(torch.equal(full_topk_idx[:, sparse_token, :], topk_idx))
        self.assertTrue(torch.all(full_topk_idx[:, 2:4, :] == -1))
        # dense rows are request 1's token-major rows: tokens 2,3 x 2 head groups.
        self.assertEqual(dense_rows.tolist(), [4, 5, 6, 7])

    def test_mixed_tree_metadata_makes_dense_rows_token_major(self):
        backend = self._backend()
        backend.dense_len = 16  # request 0 (13 < 16) dense, request 1 (23) sparse
        draft_token_num = 3
        prefix_lens = [10, 20]
        req_to_token = torch.arange(2 * 64, dtype=torch.int32).reshape(2, 64)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(
                topk=2,
                draft_token_num=draft_token_num,
                custom_mask=torch.ones(
                    sum(draft_token_num * (p + draft_token_num) for p in prefix_lens),
                    dtype=torch.bool,
                ),
            ),
            seq_lens=torch.tensor(prefix_lens, dtype=torch.int64),
            seq_lens_cpu=torch.tensor(prefix_lens, dtype=torch.int64),
            batch_size=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            encoder_lens=None,
        )
        backend.req_to_token_pool = forward_batch.req_to_token_pool
        backend.init_forward_metadata(forward_batch)

        metadata = backend.forward_metadata
        # Only request 1 is sparse, but both are token-major under tree verify:
        # draft_token_num * head_group rows each.
        self.assertEqual(metadata.sparse_bs_list, [1])
        self.assertEqual(forward_batch.sparse_batch_size, 1)
        rows_per_req = draft_token_num * backend.head_group_num
        self.assertEqual(
            metadata.old_bs_to_new_bs_range, [0, rows_per_req, 2 * rows_per_req]
        )
        # Every token-major row carries a single query (step 1), dense included.
        self.assertEqual(
            metadata.sparse_cu_seqlens_q_cpu.tolist(),
            list(range(2 * rows_per_req + 1)),
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_sparse_tree_page_table_compaction_cuda_matches_cpu(self):
        fixture = self._sparse_tree_compaction_inputs()
        expected_table, expected_lens = compact_sparse_tree_page_table_reference(
            **fixture
        )
        actual_table, actual_lens = compact_sparse_tree_page_table(**_to_cuda(fixture))
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(actual_lens.cpu(), expected_lens))
        self.assertTrue(torch.equal(actual_table.cpu(), expected_table))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_sparse_tree_page_table_compaction_cuda_draft_mask_matches_full_mask(self):
        fixture = _to_cuda(self._sparse_tree_compaction_inputs())
        full_table, full_lens = compact_sparse_tree_page_table(**fixture)
        draft_table, draft_lens = compact_sparse_tree_page_table(
            **_with_draft_tree_mask(fixture)
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(draft_lens, full_lens))
        self.assertTrue(torch.equal(draft_table, full_table))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_sparse_tree_page_table_compaction_cuda_long_prefix_matches_cpu(self):
        fixture = self._long_prefix_sparse_tree_compaction_inputs()
        expected_table, expected_lens = compact_sparse_tree_page_table_reference(
            **fixture
        )
        actual_table, actual_lens = compact_sparse_tree_page_table(**_to_cuda(fixture))
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(actual_lens.cpu(), expected_lens))
        self.assertTrue(torch.equal(actual_table.cpu(), expected_table))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_sparse_tree_page_table_compaction_cuda_multi_request_matches_cpu(self):
        fixture = self._multi_request_sparse_tree_compaction_inputs()
        expected_table, expected_lens = compact_sparse_tree_page_table_reference(
            **fixture
        )
        actual_table, actual_lens = compact_sparse_tree_page_table(**_to_cuda(fixture))
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(actual_lens.cpu(), expected_lens))
        self.assertTrue(torch.equal(actual_table.cpu(), expected_table))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_sparse_tree_page_table_compaction_cuda_real_shape_repeated(self):
        """Production-width rows exercise the multi-warp store path.

        The toy fixtures above fit in one or two warps; this shape
        (rows=256, width 6144 -> BLOCK_N=8192, num_warps=8) is the regime
        where an ordering bug between the kernel's zero-fill and scatter
        stores would corrupt rows nondeterministically, so the CUDA result
        is checked against the CPU reference across repeated launches.
        """
        fixture = self._real_shape_sparse_tree_compaction_inputs()
        expected_table, expected_lens = compact_sparse_tree_page_table_reference(
            **fixture
        )
        cuda_fixture = _to_cuda(fixture)
        for repeat in range(4):
            actual_table, actual_lens = compact_sparse_tree_page_table(**cuda_fixture)
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(actual_lens.cpu(), expected_lens),
                f"cache seqlens diverged on repeat {repeat}",
            )
            self.assertTrue(
                torch.equal(actual_table.cpu(), expected_table),
                f"page table diverged on repeat {repeat}",
            )

    def test_cuda_graph_plan_counts_match_compaction_output(self):
        """Pin the plan==data invariant the CUDA-graph verify path relies on.

        `_refresh_target_verify_graph_metadata` plans FlashInfer's split-KV
        schedule with `sc(token_pos) - off + popcount` per row and the in-graph
        compaction must write exactly those counts (a mismatch flips greedy
        near-ties; see the plan-site comment). The identity holds only for
        production-shaped topk -- the local window always selects the frontier
        blocks -- so these fixtures build the selection by that rule and first
        assert the derived seqlens equal sc(token_pos).
        """
        scenarios = [
            # Below budget, prefix not block aligned, chain mask.
            dict(
                block_size=4,
                sparse_topk=3,
                prefix_lens=[5],
                draft_masks=[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]],
            ),
            # Above budget; token positions cross a block boundary and hit the
            # `token_pos % block_size == 0` branch; tree mask with siblings.
            dict(
                block_size=4,
                sparse_topk=2,
                prefix_lens=[13],
                draft_masks=[[[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 1]]],
            ),
            # Two requests: one below budget, one above (non-aligned prefix).
            dict(
                block_size=4,
                sparse_topk=2,
                prefix_lens=[3, 21],
                draft_masks=[
                    [[1, 0], [1, 1]],
                    [[1, 0], [0, 1]],
                ],
            ),
        ]
        for scenario in scenarios:
            with self.subTest(**{k: scenario[k] for k in ("prefix_lens",)}):
                self._assert_plan_counts_match_compaction(**scenario)

    def _assert_plan_counts_match_compaction(
        self, block_size, sparse_topk, prefix_lens, draft_masks
    ):
        head_group_num = 2
        draft_token_num = len(draft_masks[0])
        num_reqs = len(prefix_lens)
        max_sparse_tokens = block_size * sparse_topk

        token_to_bs, token_pos_list, topk_rows = [], [], []
        for req, prefix in enumerate(prefix_lens):
            for off in range(1, draft_token_num + 1):
                token_pos = prefix + off
                # Local-window selection: every causal block below budget, the
                # most recent sparse_topk blocks above it.
                num_causal_blocks = -(-token_pos // block_size)
                if num_causal_blocks <= sparse_topk:
                    row = list(range(num_causal_blocks))
                    row += [-1] * (sparse_topk - num_causal_blocks)
                else:
                    row = list(
                        range(num_causal_blocks - sparse_topk, num_causal_blocks)
                    )
                token_to_bs.append(req)
                token_pos_list.append(token_pos)
                topk_rows.append(row)

        token_num = num_reqs * draft_token_num
        topk_one_group = torch.tensor(topk_rows, dtype=torch.int32)
        topk_idx = torch.stack([topk_one_group] * head_group_num)
        token_to_bs = torch.tensor(token_to_bs, dtype=torch.int32)
        token_pos_in_bs = torch.tensor(token_pos_list, dtype=torch.int32)
        prefix_tensor = torch.tensor(prefix_lens, dtype=torch.int32)
        seqlen_k = prefix_tensor + draft_token_num

        derived = _derive_sparse_cache_seqlens_from_topk(
            topk_idx,
            token_to_bs,
            token_pos_in_bs,
            seqlen_k,
            block_size,
            torch.int32,
        )
        expected_unmasked = sparse_block_quantized_count(
            token_pos_in_bs.to(torch.int64), block_size, sparse_topk
        ).repeat_interleave(head_group_num)
        self.assertEqual(
            derived.tolist(),
            expected_unmasked.tolist(),
            "fixture violates the local-window invariant the plan encodes",
        )

        draft_mask = torch.tensor(
            [row for req_mask in draft_masks for row in req_mask], dtype=torch.bool
        )
        page_table = torch.arange(
            head_group_num * token_num * max_sparse_tokens, dtype=torch.int32
        ).reshape(head_group_num * token_num, max_sparse_tokens)
        _, compact_lens = compact_sparse_tree_page_table_reference(
            page_table=page_table,
            topk_idx=topk_idx,
            token_to_bs=token_to_bs,
            token_pos_in_bs=token_pos_in_bs,
            prefix_lens=prefix_tensor,
            mask_batch_offsets=None,
            custom_mask=torch.empty((0,), dtype=torch.bool),
            source_cache_seqlens=derived,
            draft_token_num=draft_token_num,
            block_size=block_size,
            draft_tree_mask=draft_mask.flatten(),
        )

        # The exact expression `_refresh_target_verify_graph_metadata` plans with.
        draft_offsets = torch.arange(1, draft_token_num + 1, dtype=torch.int64).repeat(
            num_reqs
        )
        num_visible_drafts = draft_mask.view(token_num, draft_token_num).sum(dim=1)
        planned = (
            sparse_block_quantized_count(
                token_pos_in_bs.to(torch.int64), block_size, sparse_topk
            )
            - draft_offsets
            + num_visible_drafts
        ).repeat_interleave(head_group_num)
        self.assertEqual(compact_lens.tolist(), planned.tolist())

    def _sparse_tree_compaction_inputs(self):
        """Single request, 3-token draft chain, full custom mask."""
        block_size = 2
        topk_for_one_group = torch.tensor(
            [[0, 2, 3], [0, 2, 3], [0, 2, 3]], dtype=torch.int32
        )
        topk_idx = torch.stack([topk_for_one_group, topk_for_one_group])
        token_to_bs = torch.tensor([0, 0, 0], dtype=torch.int32)
        token_pos_in_bs = torch.tensor([5, 6, 7], dtype=torch.int32)
        prefix_lens = torch.tensor([4], dtype=torch.int32)
        mask_batch_offsets = torch.tensor([0, 21], dtype=torch.int64)
        seqlen_k = torch.tensor([7], dtype=torch.int32)
        unmasked_cache_seqlens = _derive_sparse_cache_seqlens_from_topk(
            topk_idx,
            token_to_bs,
            token_pos_in_bs,
            seqlen_k,
            block_size,
            torch.int32,
        )
        self.assertEqual(unmasked_cache_seqlens.tolist(), [3, 3, 4, 4, 5, 5])
        custom_mask = torch.tensor(
            [
                [True, True, True, True, True, False, False],
                [True, True, True, True, True, True, False],
                [True, True, True, True, True, False, True],
            ],
            dtype=torch.bool,
        ).flatten()
        page_table = torch.arange(6 * 6, dtype=torch.int32).reshape(6, 6)
        return dict(
            page_table=page_table,
            topk_idx=topk_idx,
            token_to_bs=token_to_bs,
            token_pos_in_bs=token_pos_in_bs,
            prefix_lens=prefix_lens,
            mask_batch_offsets=mask_batch_offsets,
            custom_mask=custom_mask,
            source_cache_seqlens=unmasked_cache_seqlens,
            draft_token_num=3,
            block_size=block_size,
        )

    def _long_prefix_sparse_tree_compaction_inputs(self):
        """Single request, long prefix crossing block boundaries, draft tree mask."""
        block_size = 64
        draft_token_num = 4
        prefix_len = 8192
        topk_for_one_group = torch.tensor(
            [[0, 127, 128]] * draft_token_num,
            dtype=torch.int32,
        )
        topk_idx = torch.stack([topk_for_one_group, topk_for_one_group])
        token_to_bs = torch.zeros((draft_token_num,), dtype=torch.int32)
        token_pos_in_bs = torch.arange(
            prefix_len + 1,
            prefix_len + draft_token_num + 1,
            dtype=torch.int32,
        )
        prefix_lens = torch.tensor([prefix_len], dtype=torch.int32)
        seqlen_k = torch.tensor([prefix_len + draft_token_num], dtype=torch.int32)
        unmasked_cache_seqlens = _derive_sparse_cache_seqlens_from_topk(
            topk_idx,
            token_to_bs,
            token_pos_in_bs,
            seqlen_k,
            block_size,
            torch.int32,
        )
        page_table = torch.arange(3 * block_size, dtype=torch.int32).repeat(
            draft_token_num * 2, 1
        )
        page_table += (
            torch.arange(draft_token_num * 2, dtype=torch.int32).view(-1, 1) * 1000
        )
        draft_tree_mask = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, False, True, False],
                [True, True, False, True],
            ],
            dtype=torch.bool,
        ).flatten()
        return dict(
            page_table=page_table,
            topk_idx=topk_idx,
            token_to_bs=token_to_bs,
            token_pos_in_bs=token_pos_in_bs,
            prefix_lens=prefix_lens,
            mask_batch_offsets=torch.tensor([0], dtype=torch.int64),
            custom_mask=torch.empty((0,), dtype=torch.bool),
            source_cache_seqlens=unmasked_cache_seqlens,
            draft_token_num=draft_token_num,
            block_size=block_size,
            draft_tree_mask=draft_tree_mask,
        )

    def _real_shape_sparse_tree_compaction_inputs(self):
        """Production-width fixture: 16 requests x 8 drafts x 2 head groups
        (rows=256) over a 6144-wide page table, randomized but seeded."""
        block_size = 64
        sparse_topk = 96
        max_sparse_tokens = block_size * sparse_topk
        head_group_num = 2
        draft_token_num = 8
        num_reqs = 16
        block_pool = 256
        gen = torch.Generator().manual_seed(20260612)

        prefix_lens = torch.randint(
            100, 7000, (num_reqs,), generator=gen, dtype=torch.int32
        )
        token_num = num_reqs * draft_token_num
        token_to_bs = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int32), draft_token_num
        )
        token_pos_in_bs = prefix_lens.repeat_interleave(draft_token_num) + torch.arange(
            1, draft_token_num + 1, dtype=torch.int32
        ).repeat(num_reqs)
        seqlen_k = prefix_lens + draft_token_num

        def _random_topk_group():
            return torch.stack(
                [
                    torch.randperm(block_pool, generator=gen)[:sparse_topk].to(
                        torch.int32
                    )
                    for _ in range(token_num)
                ]
            )

        topk_idx = torch.stack([_random_topk_group(), _random_topk_group()])
        # Sprinkle -1 padding tails like short-prefix production rows.
        pad_lens = torch.randint(0, 8, (head_group_num, token_num), generator=gen)
        for group in range(head_group_num):
            for token in range(token_num):
                pad = int(pad_lens[group, token])
                if pad:
                    topk_idx[group, token, sparse_topk - pad :] = -1

        draft_mask = (
            torch.rand((token_num, draft_token_num), generator=gen) > 0.5
        )  # random tree visibility
        query_idx = (token_pos_in_bs - prefix_lens[token_to_bs.long()] - 1).long()
        draft_mask[torch.arange(token_num), query_idx] = True  # self-visibility

        source_cache_seqlens = _derive_sparse_cache_seqlens_from_topk(
            topk_idx,
            token_to_bs,
            token_pos_in_bs,
            seqlen_k,
            block_size,
            torch.int32,
        )
        page_table = torch.randint(
            0,
            1_000_000,
            (token_num * head_group_num, max_sparse_tokens),
            generator=gen,
            dtype=torch.int32,
        )
        return dict(
            page_table=page_table,
            topk_idx=topk_idx,
            token_to_bs=token_to_bs,
            token_pos_in_bs=token_pos_in_bs,
            prefix_lens=prefix_lens,
            mask_batch_offsets=None,
            custom_mask=torch.empty((0,), dtype=torch.bool),
            source_cache_seqlens=source_cache_seqlens,
            draft_token_num=draft_token_num,
            block_size=block_size,
            draft_tree_mask=draft_mask.flatten(),
        )

    def _multi_request_sparse_tree_compaction_inputs(self):
        """Two requests with different prefix lens and a -1-padded topk row."""
        block_size = 2
        draft_token_num = 2
        topk_for_one_group = torch.tensor(
            [[0, 1, 2], [0, 1, 2], [0, 2, 3], [0, 3, -1]], dtype=torch.int32
        )
        topk_idx = torch.stack([topk_for_one_group, topk_for_one_group])
        token_to_bs = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        token_pos_in_bs = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
        prefix_lens = torch.tensor([4, 6], dtype=torch.int32)
        # Per-request flat masks: [draft_token_num, prefix + draft_token_num].
        mask_batch_offsets = torch.tensor([0, 12, 28], dtype=torch.int64)
        seqlen_k = torch.tensor([6, 8], dtype=torch.int32)
        unmasked_cache_seqlens = _derive_sparse_cache_seqlens_from_topk(
            topk_idx,
            token_to_bs,
            token_pos_in_bs,
            seqlen_k,
            block_size,
            torch.int32,
        )
        # The -1-padded row (request 1, token 3) loses one whole block.
        self.assertEqual(unmasked_cache_seqlens.tolist(), [5, 5, 6, 6, 5, 5, 4, 4])
        custom_mask = torch.cat(
            [
                # request 0: chain (token 1 is a child of token 0)
                torch.tensor(
                    [
                        [True, True, True, True, True, False],
                        [True, True, True, True, True, True],
                    ],
                    dtype=torch.bool,
                ).flatten(),
                # request 1: siblings (token 3 does not see token 2)
                torch.tensor(
                    [
                        [True, True, True, True, True, True, True, False],
                        [True, True, True, True, True, True, False, True],
                    ],
                    dtype=torch.bool,
                ).flatten(),
            ]
        )
        page_table = torch.arange(8 * 6, dtype=torch.int32).reshape(8, 6)
        return dict(
            page_table=page_table,
            topk_idx=topk_idx,
            token_to_bs=token_to_bs,
            token_pos_in_bs=token_pos_in_bs,
            prefix_lens=prefix_lens,
            mask_batch_offsets=mask_batch_offsets,
            custom_mask=custom_mask,
            source_cache_seqlens=unmasked_cache_seqlens,
            draft_token_num=draft_token_num,
            block_size=block_size,
        )


if __name__ == "__main__":
    unittest.main()
