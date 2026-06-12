from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.triton_ops.common import (
    _get_last_loc_safe_kernel as _get_last_loc_safe_kernel,
)
from sglang.srt.mem_cache.triton_ops.common import (
    get_last_loc_kernel as get_last_loc_kernel,
)
from sglang.srt.mem_cache.triton_ops.common import (
    get_last_loc_triton,
    get_last_loc_triton_safe,
    write_req_to_token_pool_triton,
)
from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.utils import is_hip, support_triton
from sglang.srt.utils.common import ceil_align

_is_hip = is_hip()

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

# Needs 2 + 1 slots for mamba request with prefix cache. 2 for ping pong cache, 1 for running mamba state.
MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3
# Lazy mode: 1 + 1 slots (1 ping-pong + 1 running), second ping-pong allocated on demand at boundary.
MAMBA_STATE_PER_REQ_PREFIX_CACHE_LAZY = 2
MAMBA_STATE_PER_REQ_NO_CACHE = 1

logger = logging.getLogger(__name__)


def kv_to_page_indices(kv_indices: np.ndarray, page_size: int):
    # The page is guaranteed to be full except the last page.
    if page_size == 1:
        return kv_indices

    return kv_indices[::page_size] // page_size


def kv_to_page_num(num_kv_indices: int, page_size: int):
    return (num_kv_indices + page_size - 1) // page_size


def page_align_floor(length: int, page_size: int) -> int:
    return (length // page_size) * page_size


def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    if getattr(req, "skip_radix_cache_insert", False):
        return

    tree_cache.cache_unfinished_req(req, **kwargs)


def write_cache_indices(
    out_cache_loc: torch.Tensor,
    sparse_k1_loc: torch.Tensor,
    sparse_k2_loc: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    token_num_sparse_k1_cpu: torch.Tensor,
    token_num_sparse_k2_cpu: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    prefix_k1_tensors: list[torch.Tensor],
    prefix_k2_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
    kernel_size: Optional[int],
    kernel_stride: Optional[int],
):
    if support_triton(get_global_server_args().attention_backend):
        prefix_pointers = torch.tensor(
            [t.data_ptr() for t in prefix_tensors],
            device=req_to_token_pool.device,
            dtype=torch.uint64,
        )
        # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)
        write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
            req_to_token_pool.req_to_token,
            req_pool_indices_tensor,
            prefix_pointers,
            prefix_lens_tensor,
            seq_lens_tensor,
            extend_lens_tensor,
            out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
        )
    else:
        pt = 0
        for i in range(req_pool_indices_cpu.shape[0]):
            req_idx = req_pool_indices_cpu[i].item()
            prefix_len = prefix_lens_cpu[i].item()
            seq_len = seq_lens_cpu[i].item()
            extend_len = extend_lens_cpu[i].item()

            req_to_token_pool.write(
                (req_idx, slice(0, prefix_len)),
                prefix_tensors[i],
            )
            req_to_token_pool.write(
                (req_idx, slice(prefix_len, seq_len)),
                out_cache_loc[pt : pt + extend_len],
            )
            pt += extend_len

    bs = req_pool_indices_cpu.shape[0]
    pt = 0
    for i in range(bs):
        req_idx = req_pool_indices_cpu[i].item()
        prefix_len = prefix_lens_cpu[i].item()
        k1_len = _compressed_token_count(prefix_len, kernel_size, kernel_stride)
        if k1_len > 0:
            req_to_token_pool.write_sparse_k1(
                (req_idx, slice(0, k1_len)),
                prefix_k1_tensors[i],
            )
        if sparse_k1_loc is not None:
            req_to_token_pool.write_sparse_k1(
                (req_idx, slice(k1_len, token_num_sparse_k1_cpu[i] + k1_len)),
                sparse_k1_loc[pt : pt + token_num_sparse_k1_cpu[i]].to(torch.int32),
            )
            pt += token_num_sparse_k1_cpu[i]
    pt = 0
    k2_kernel_size = kernel_size * 4 if kernel_size is not None else None
    k2_kernel_stride = kernel_stride * 4 if kernel_stride is not None else None
    for i in range(bs):
        req_idx = req_pool_indices_cpu[i].item()
        prefix_len = prefix_lens_cpu[i].item()
        k2_len = _compressed_token_count(prefix_len, k2_kernel_size, k2_kernel_stride)
        if k2_len > 0:
            req_to_token_pool.write_sparse_k2(
                (req_idx, slice(0, k2_len)),
                prefix_k2_tensors[i],
            )
        if sparse_k2_loc is not None:
            req_to_token_pool.write_sparse_k2(
                (req_idx, slice(k2_len, token_num_sparse_k2_cpu[i] + k2_len)),
                sparse_k2_loc[pt : pt + token_num_sparse_k2_cpu[i]].to(torch.int32),
            )
            pt += token_num_sparse_k2_cpu[i]


def _compressed_token_count(
    seq_len: int, kernel_size: Optional[int], kernel_stride: Optional[int]
) -> int:
    if (
        kernel_size is None
        or kernel_stride is None
        or kernel_stride <= 0
        or seq_len < kernel_size
    ):
        return 0
    return (seq_len - kernel_size) // kernel_stride + 1


def _to_int_list(values) -> list[int]:
    if isinstance(values, torch.Tensor):
        return [int(v) for v in values.tolist()]
    return [int(v) for v in values]


def _pool_has_sparse_compressed_cache(req_to_token_pool) -> bool:
    """The pool carries addressable MiniCPM K1/K2 compressed-key tables."""
    return (
        getattr(req_to_token_pool, "req_to_sparse_k1_token", None) is not None
        and getattr(req_to_token_pool, "req_to_sparse_k2_token", None) is not None
        and getattr(req_to_token_pool, "kernel_size", None) is not None
        and getattr(req_to_token_pool, "kernel_stride", None) is not None
    )


def alloc_sparse_compressed_slots_for_range(
    tree_cache: BasePrefixCache,
    req_to_token_pool: ReqToTokenPool,
    req_pool_indices,
    start_lens,
    end_lens,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate MiniCPM sparse K1/K2 slots for a logical token range.

    Normal speculative decode over-allocates future KV slots by logical length. MiniCPM
    sparse attention needs the matching compressed-cache slots registered in the
    request table before target verify builds compression metadata.

    Release protocols differ by spec path. The v1 EAGLE worker allocates
    [seq_len, seq_len + draft_token_num) each verify round and trims the
    rejected tail right after sampling (free_sparse_compressed_slots_for_range
    in `EagleVerifyInput.sample`). The v2 worker allocates monotonically from
    `kv_allocated_len`, reuses rejected slots in place on later rounds, and
    releases the whole [committed, allocated) tail once when the request
    leaves (`release_kv_cache`); a per-round free here would free slots later
    rounds still address.

    Returns the per-request K1/K2 slot counts (CPU int64). The slot ids are
    written into the request tables here; callers do not need them.
    """
    start_lens = _to_int_list(start_lens)
    end_lens = _to_int_list(end_lens)
    req_pool_indices = _to_int_list(req_pool_indices)

    if not _pool_has_sparse_compressed_cache(req_to_token_pool):
        zeros = torch.zeros(len(start_lens), dtype=torch.int64)
        return zeros, zeros

    kernel_size = req_to_token_pool.kernel_size
    kernel_stride = req_to_token_pool.kernel_stride
    levels = (
        (kernel_size, kernel_stride, req_to_token_pool.write_sparse_k1),
        (kernel_size * 4, kernel_stride * 4, req_to_token_pool.write_sparse_k2),
    )

    level_starts = []
    level_counts = []
    for level_kernel_size, level_kernel_stride, _ in levels:
        starts = [
            _compressed_token_count(seq_len, level_kernel_size, level_kernel_stride)
            for seq_len in start_lens
        ]
        counts = [
            _compressed_token_count(seq_len, level_kernel_size, level_kernel_stride)
            - start
            for seq_len, start in zip(end_lens, starts)
        ]
        level_starts.append(starts)
        level_counts.append(counts)

    # One allocation covers both levels: a failure (alloc_token_slots raises
    # on OOM) cannot leave one level allocated or any table row half written.
    total = sum(level_counts[0]) + sum(level_counts[1])
    loc = alloc_token_slots(tree_cache, total) if total > 0 else None

    pt = 0
    for (_, _, write_sparse), starts, counts in zip(levels, level_starts, level_counts):
        for req_idx, start, count in zip(req_pool_indices, starts, counts):
            if count == 0:
                continue
            write_sparse(
                (req_idx, slice(start, start + count)),
                loc[pt : pt + count].to(torch.int32),
            )
            pt += count

    return (
        torch.tensor(level_counts[0], dtype=torch.int64),
        torch.tensor(level_counts[1], dtype=torch.int64),
    )


def free_sparse_compressed_slots_for_range(
    tree_cache: BasePrefixCache,
    req_pool_idx: int,
    start_len: int,
    end_len: int,
) -> None:
    req_to_token_pool = tree_cache.req_to_token_pool
    if start_len >= end_len or not _pool_has_sparse_compressed_cache(req_to_token_pool):
        return
    kernel_size = req_to_token_pool.kernel_size
    kernel_stride = req_to_token_pool.kernel_stride

    allocator = tree_cache.token_to_kv_pool_allocator
    for level_kernel_size, level_kernel_stride, req_to_sparse_token in (
        (kernel_size, kernel_stride, req_to_token_pool.req_to_sparse_k1_token),
        (kernel_size * 4, kernel_stride * 4, req_to_token_pool.req_to_sparse_k2_token),
    ):
        start = _compressed_token_count(
            start_len, level_kernel_size, level_kernel_stride
        )
        end = _compressed_token_count(end_len, level_kernel_size, level_kernel_stride)
        if start < end:
            allocator.free(req_to_sparse_token[req_pool_idx, start:end])


def get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    attn_backend = get_global_server_args().attention_backend
    uses_triton_dispatch = attn_backend not in ("ascend", "torch_native")

    if _is_hip and uses_triton_dispatch:
        # HIP-only: the legacy get_last_loc_triton kernel emits a
        # mixed-width int32->int64 store that Triton mis-compiles on HIP,
        # producing out-of-range last_loc values under EAGLE +
        # page_size>1 (e.g. with aiter unified attention or the triton
        # attention backend). The bug is in the Triton HIP codegen, not
        # in any particular attention backend, so route every HIP path
        # that would otherwise use get_last_loc_triton through the
        # int32-safe variant. Non-HIP hardware keeps the original
        # dispatcher below.
        return get_last_loc_triton_safe(
            req_to_token, req_pool_indices_tensor, prefix_lens_tensor
        )

    if uses_triton_dispatch:
        impl = get_last_loc_triton
    else:
        impl = get_last_loc_torch

    return impl(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)


def get_last_loc_torch(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        prefix_lens_tensor > 0,
        req_to_token[req_pool_indices_tensor, prefix_lens_tensor - 1],
        torch.full_like(prefix_lens_tensor, -1),
    )


def get_alloc_len_per_decode(server_args: Optional[ServerArgs] = None) -> int:
    if server_args is None:
        server_args = get_global_server_args()

    if server_args.speculative_algorithm is None:
        return 1

    # Spec v1:
    # 1) alloc topk * num_steps when draft decoding and then restore the allocation
    # 2) alloc num_draft_tokens when verifying the drafts
    # Sepc v2: allocate max(topk * num_steps, num_draft_tokens)

    spec_steps = server_args.speculative_num_steps or 1
    spec_topk = server_args.speculative_eagle_topk or 1
    spec_tokens = server_args.max_speculative_num_draft_tokens
    page_size = server_args.page_size

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    spec_algo = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
    if page_size == 1 or spec_topk == 1 or not spec_algo.has_draft_kv():
        return max(spec_steps * spec_topk, spec_tokens)
    else:
        # page_size > 1 + topk > 1 (spec v2 tree): worst-case page-aligned tree
        # footprint. Per topk branch needs ceil((last_page_len + num_steps) / page)
        # pages; the partial tail page can be up to page_size - 1, and each branch
        # gets its own (duplicated) copy -- so reserve for all topk branches.
        num_new_pages_per_topk = (
            (page_size - 1) + spec_steps + page_size - 1
        ) // page_size
        return max(num_new_pages_per_topk * page_size * spec_topk, spec_tokens)


def get_alloc_reserve_per_decode(server_args: Optional[ServerArgs] = None) -> int:
    """KV length reserved per request at each decode step.

    The 2x is a double-buffer that absorbs the kv_committed_len lag in overlap
    mode; see eagle_info_v2.prepare_for_decode.
    """
    return 2 * get_alloc_len_per_decode(server_args)


def get_req_to_token_extra_context_len(server_args: ServerArgs) -> int:
    """req_to_token row headroom beyond the model context length.

    Sized to hold the decode over-allocation (kv_committed_len +
    get_alloc_reserve_per_decode). The spec v2 page>1 topk>1 holey draft footprint
    can outgrow the default num_draft_tokens headroom (PR #26972).
    """
    # FIXME(lsyin): this is the temporary fix for the context length issue when
    # using speculative decoding
    extra = 4 + (server_args.max_speculative_num_draft_tokens or 0)
    if (
        server_args.speculative_algorithm is not None
        and server_args.page_size > 1
        and (server_args.speculative_eagle_topk or 1) > 1
    ):
        extra = max(extra, get_alloc_reserve_per_decode(server_args))
    return extra


def alloc_token_slots(
    tree_cache: BasePrefixCache,
    num_tokens: int,
    backup_state: bool = False,
):
    allocator = tree_cache.token_to_kv_pool_allocator
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc(num_tokens)

    if out_cache_loc is None:
        error_msg = (
            f"Out of memory. Try to lower your batch size.\n"
            f"Try to allocate {num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int):
    if tree_cache is None:
        return

    if tree_cache.is_chunk_cache():
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
    else:
        # Standard allocator
        if allocator.available_size() < num_tokens:
            tree_cache.evict(EvictParams(num_tokens=num_tokens))


def alloc_paged_token_slots_extend(
    tree_cache: BasePrefixCache,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    backup_state: bool = False,
):
    # Over estimate the number of tokens: assume each request needs a new page.
    allocator = tree_cache.token_to_kv_pool_allocator
    num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
    )

    if out_cache_loc is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def alloc_req_slots(
    req_to_token_pool: ReqToTokenPool,
    reqs: list[Req],
    tree_cache: BasePrefixCache | None,
) -> list[int]:
    """Allocate request slots from the pool."""
    num_reqs = len(reqs)
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        mamba_available_size = req_to_token_pool.mamba_allocator.available_size()
        if tree_cache.supports_mamba():
            factor = (
                MAMBA_STATE_PER_REQ_PREFIX_CACHE_LAZY
                if req_to_token_pool.enable_mamba_extra_buffer_lazy
                else MAMBA_STATE_PER_REQ_PREFIX_CACHE
            )
        else:
            factor = MAMBA_STATE_PER_REQ_NO_CACHE
        mamba_state_needed = num_reqs * factor
        if mamba_available_size < mamba_state_needed:
            if tree_cache is not None and tree_cache.supports_mamba():
                mamba_num = max(0, mamba_state_needed - mamba_available_size)
                tree_cache.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))
    req_pool_indices = req_to_token_pool.alloc(reqs)

    if req_pool_indices is None:
        raise RuntimeError(
            "alloc_req_slots runs out of memory. "
            "Please set a smaller number for `--max-running-requests`. "
            f"{req_to_token_pool.available_size()=}, "
            f"{num_reqs=}, "
        )
    return req_pool_indices


def alloc_for_extend(
    batch: ScheduleBatch,
) -> tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
        sparse_k1_loc: allocated sparse k1 cache locations (None if unused)
        sparse_k2_loc: allocated sparse k2 cache locations (None if unused)
        req_pool_indices_device: request pool indices as a device tensor
        req_pool_indices_cpu: request pool indices as a CPU tensor (host mirror)
    """
    # free out-of-window swa tokens
    batch.maybe_evict_swa()

    prefix_tensors = [r.prefix_indices for r in batch.reqs]
    prefix_k1_tensors = [r.prefix_k1_indices for r in batch.reqs]
    prefix_k2_tensors = [r.prefix_k2_indices for r in batch.reqs]

    # Create tensors for allocation
    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64)
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    # Allocate req slots
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    sparse_k1_loc, sparse_k2_loc = None, None
    if batch.tree_cache.page_size == 1:
        if batch.token_sum_sparse_k1 > 0:
            sparse_k1_loc = alloc_token_slots(
                batch.tree_cache, batch.token_sum_sparse_k1
            )
        if batch.token_sum_sparse_k2 > 0:
            sparse_k2_loc = alloc_token_slots(
                batch.tree_cache, batch.token_sum_sparse_k2
            )
        out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
    else:
        # Paged allocation - build last_loc
        last_loc = [
            (t[-1:] if len(t) > 0 else torch.tensor([-1], device=batch.device))
            for t in prefix_tensors
        ]
        out_cache_loc = alloc_paged_token_slots_extend(
            tree_cache=batch.tree_cache,
            prefix_lens=prefix_lens_device,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            last_loc=torch.cat(last_loc),
            extend_num_tokens=batch.extend_num_tokens,
        )

    # Write to req_to_token_pool
    write_cache_indices(
        out_cache_loc,
        sparse_k1_loc,
        sparse_k2_loc,
        req_pool_indices_device,
        req_pool_indices_cpu,
        prefix_lens_device,
        prefix_lens_cpu,
        batch.seq_lens,
        batch.seq_lens_cpu,
        batch.token_num_sparse_k1_cpu,
        batch.token_num_sparse_k2_cpu,
        extend_lens_device,
        extend_lens_cpu,
        prefix_tensors,
        prefix_k1_tensors,
        prefix_k2_tensors,
        batch.req_to_token_pool,
        (
            batch.req_to_token_pool.kernel_size
            if hasattr(batch.req_to_token_pool, "kernel_size")
            else None
        ),
        (
            batch.req_to_token_pool.kernel_stride
            if hasattr(batch.req_to_token_pool, "kernel_stride")
            else None
        ),
    )

    return (
        out_cache_loc,
        sparse_k1_loc,
        sparse_k2_loc,
        req_pool_indices_device,
        req_pool_indices_cpu,
    )


def alloc_paged_token_slots_decode(
    tree_cache: BasePrefixCache,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    token_per_req: int = 1,
) -> torch.Tensor:
    """Allocate paged KV cache for decode batch."""
    allocator = tree_cache.token_to_kv_pool_allocator
    # Over estimate the number of tokens: assume each request needs a new page.
    num_tokens = len(seq_lens) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    out_cache_loc = allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc)

    if out_cache_loc is None:
        error_msg = (
            f"Decode out of memory. Try to lower your batch size.\n"
            f"Try to allocate {len(seq_lens) * token_per_req} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return out_cache_loc


def alloc_for_decode(
    batch: ScheduleBatch, token_per_req: int
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Allocate KV cache for decode batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
        sparse_k1_loc: allocated sparse K1 cache locations (None if not needed)
        sparse_k2_loc: allocated sparse K2 cache locations (None if not needed)
    """

    batch.maybe_evict_swa()

    seq_lens_gpu = batch.seq_lens
    bs = seq_lens_gpu.shape[0]

    sparse_k1_loc, sparse_k2_loc = None, None
    if batch.tree_cache.page_size == 1:
        # Non-paged allocation
        out_cache_loc = alloc_token_slots(batch.tree_cache, bs * token_per_req)
        if batch.token_sum_sparse_k1 > 0:
            sparse_k1_loc = alloc_token_slots(
                batch.tree_cache, batch.token_sum_sparse_k1
            )
        if batch.token_sum_sparse_k2 > 0:
            sparse_k2_loc = alloc_token_slots(
                batch.tree_cache, batch.token_sum_sparse_k2
            )
    else:
        # Paged allocation
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, seq_lens_gpu - 1
        ]
        seq_lens_next = seq_lens_gpu + token_per_req
        out_cache_loc = alloc_paged_token_slots_decode(
            tree_cache=batch.tree_cache,
            seq_lens=seq_lens_next,
            seq_lens_cpu=batch.seq_lens_cpu + token_per_req,
            last_loc=last_loc,
            token_per_req=token_per_req,
        )

    # Write to req_to_token_pool
    if batch.model_config.is_encoder_decoder:
        locs = batch.encoder_lens + seq_lens_gpu
    else:
        locs = seq_lens_gpu.clone()

    batch.req_to_token_pool.write(
        (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
    )

    if sparse_k1_loc is not None:
        pt = 0
        k1_kernel_size = batch.req_to_token_pool.kernel_size
        k1_kernel_stride = batch.req_to_token_pool.kernel_stride
        for i in range(bs):
            if batch.token_num_sparse_k1_cpu[i] > 0:
                seq_len = batch.seq_lens_cpu[i].item()
                k1_len = (
                    (seq_len - k1_kernel_size) // k1_kernel_stride + 1
                    if seq_len >= k1_kernel_size
                    else 0
                )
                batch.req_to_token_pool.write_sparse_k1(
                    (
                        batch.req_pool_indices[i],
                        (k1_len, batch.token_num_sparse_k1_cpu[i] + k1_len),
                    ),
                    sparse_k1_loc[pt : pt + batch.token_num_sparse_k1_cpu[i]].to(
                        torch.int32
                    ),
                )
                pt += batch.token_num_sparse_k1_cpu[i]
    if sparse_k2_loc is not None:
        pt = 0
        k2_kernel_size = batch.req_to_token_pool.kernel_size * 4
        k2_kernel_stride = batch.req_to_token_pool.kernel_stride * 4
        for i in range(bs):
            if batch.token_num_sparse_k2_cpu[i] > 0:
                seq_len = batch.seq_lens_cpu[i].item()
                k2_len = (
                    (seq_len - k2_kernel_size) // k2_kernel_stride + 1
                    if seq_len >= k2_kernel_size
                    else 0
                )
                batch.req_to_token_pool.write_sparse_k2(
                    (
                        batch.req_pool_indices[i],
                        (k2_len, batch.token_num_sparse_k2_cpu[i] + k2_len),
                    ),
                    sparse_k2_loc[pt : pt + batch.token_num_sparse_k2_cpu[i]].to(
                        torch.int32
                    ),
                )
                pt += batch.token_num_sparse_k2_cpu[i]

    return out_cache_loc, sparse_k1_loc, sparse_k2_loc


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if req.req_pool_idx is None:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        # TODO (csy, hanming): clean up this early allocation logic
        if req.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_allocator.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        return

    tree_cache.cache_finished_req(
        req,
        is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
    )

    # StreamingSession.cache_finished_req handles speculative tail trim
    # and bookkeeping flag sync internally, then sets req_pool_idx = None.
    if req.req_pool_idx is None:
        return

    start_p, end_p = req.pop_overallocated_kv_cache()
    # Compressed K1/K2 slots are indexed by logical (unpaged) token count, so
    # their free range must use the pre-page-alignment start.
    sparse_start_p = start_p

    global_server_args = get_global_server_args()
    page_size = global_server_args.page_size
    spec_algo = global_server_args.speculative_algorithm

    # strip_thinking_cache intentionally reports output tokens as overallocated
    # so they fall into the free path below (#22373).
    if spec_algo is None and not global_server_args.strip_thinking_cache:
        assert (
            start_p == end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv_allocated_len=}"

    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            start_p:end_p
        ]
        tree_cache.token_to_kv_pool_allocator.free(indices_to_free)
    free_sparse_compressed_slots_for_range(
        tree_cache, req.req_pool_idx, sparse_start_p, end_p
    )
    # If the prefix cache doesn't manage mamba states, we must free them here.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert (
            req.mamba_pool_idx is not None
        ), "mamba state is freed while the tree cache does not manage mamba states"
        tree_cache.req_to_token_pool.free_mamba_cache(req)
    tree_cache.req_to_token_pool.free(req)


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    return tree_cache.available_and_evictable_str()
