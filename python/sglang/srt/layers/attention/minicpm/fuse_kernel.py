import msgspec
import tilelang
import tilelang.language as T

_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


class FusedTopkKernels(msgspec.Struct):
    """The two compiled halves of the fused block selection for one batch
    size: ``stats`` writes each query row's softmax max and sum per key split,
    ``scores`` merges them and writes the pooled block scores."""

    stats: object
    scores: object
    k_splits: int
    pooled_k_len: int


# Key rows per tile of both kernels, key tiles per decode statistics split
# and pool blocks per scores program; the last two were picked by
# microbenchmark on RTX 5090 at 4K to 32K decode and 96K prefill.
_BLOCK_N = 64
_DECODE_TILES_PER_SPLIT = 2
_DECODE_BLOCK_P = 32
_PREFILL_BLOCK_P = 512


def decode_k_splits(max_k_len: int) -> int:
    k_tiles = -(-max_k_len // _BLOCK_N)
    return max(1, -(-k_tiles // _DECODE_TILES_PER_SPLIT))


def _row_geometry(cu_seqlens_q, cu_seqlens_k, k_starts, cache_lens, batch_idx):
    return (
        cu_seqlens_q[batch_idx],
        cu_seqlens_q[batch_idx + 1],
        k_starts[batch_idx],
        cu_seqlens_k[batch_idx + 1] - cu_seqlens_k[batch_idx],
        cache_lens[batch_idx],
    )


@T.macro
def _load_masked_scores(
    K_unpad,
    K_shared,
    Q_shared,
    acc_s,
    k,
    kv_head_idx,
    k_start_idx,
    k_end_idx,
    k_current_seqlen,
    q_current_seqlen,
    original_q_idx,
    cache_len,
    is_causal,
    block_M,
    block_N,
    m_block_dim,
    kernel_stride,
):
    k_copy_end = T.min(k_start_idx + (k + 1) * block_N, k_end_idx)
    T.copy(K_unpad[k_start_idx + k * block_N : k_copy_end, kv_head_idx, :], K_shared)
    for i, d in T.Parallel(block_N, K_shared.shape[1]):
        if k * block_N + i >= k_current_seqlen:
            K_shared[i, d] = 0

    for i, j in T.Parallel(block_M, block_N):
        k_idx = k * block_N + j
        boundary_mask = (original_q_idx >= q_current_seqlen) or (
            k_idx >= k_current_seqlen
        )
        if is_causal:
            row_idx = original_q_idx * block_M + i + cache_len * block_M
            orig_row_idx = row_idx // m_block_dim
            orig_seqlen_q = ((q_current_seqlen + cache_len) * block_M) // m_block_dim
            compressed_seqlen_q = (orig_seqlen_q - kernel_stride + 1) // kernel_stride
            offset_row_idx = T.max(
                0,
                (orig_row_idx + 1) // kernel_stride
                - 1
                + k_current_seqlen
                - compressed_seqlen_q,
            )
            q_compress_clamped = T.min(k_current_seqlen, offset_row_idx)
            causal_mask = k_idx > q_compress_clamped
            acc_s[i, j] = T.if_then_else(boundary_mask or causal_mask, -1e9, 0)
        else:
            acc_s[i, j] = T.if_then_else(boundary_mask, -1e9, 0)

    T.gemm(
        Q_shared,
        K_shared,
        acc_s,
        transpose_B=True,
        policy=T.GemmWarpPolicy.FullRow,
    )


@tilelang.jit(pass_configs=_pass_configs)
def _fused_attn_stats(
    batch_size: int,
    groups: int,
    heads: int,
    dim: int,
    is_causal: bool,
    k_splits: int,
    max_seqlen_q_grid: int = 1,
    dense_len: int = 0,
    m_block_dim: int = 16,
    block_M: int = 16,
    block_N: int = _BLOCK_N,
    kernel_stride: int = 16,
    num_stages: int = 0,
    threads: int = 128,
    dtype_str: str = "bfloat16",
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups

    UQ = T.dynamic("UQ")
    UKV = T.dynamic("UKV")
    q_shape = [UQ * groups, head_kv, dim]
    kv_shape = [UKV, head_kv, dim]
    # Final axis stores (max, sum) for each query head and key split.
    stats_shape = [head_kv, UQ, block_M, k_splits, 2]
    dtype = dtype_str
    accum_dtype = "float"

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        K_unpad: T.Tensor(kv_shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], "int32"),
        cu_seqlens_k: T.Tensor([batch_size + 1], "int32"),
        k_starts: T.Tensor([batch_size], "int32"),
        cache_lens: T.Tensor([batch_size], "int32"),
        Stats: T.Tensor(stats_shape, "float32"),
    ):
        with T.Kernel(
            max_seqlen_q_grid, head_kv, batch_size * k_splits, threads=threads
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            batch_idx = bz // k_splits
            k_split = bz % k_splits
            kv_head_idx = by
            original_q_idx = bx

            q_start_idx, q_end_idx, k_start_idx, k_len, cache_len = _row_geometry(
                cu_seqlens_q, cu_seqlens_k, k_starts, cache_lens, batch_idx
            )
            k_end_idx = k_start_idx + k_len
            q_current_seqlen = T.alloc_var("int32", init=q_end_idx - q_start_idx)
            k_current_seqlen = T.alloc_var("int32", init=k_len)
            if not is_causal:
                active = cache_len + 1 >= dense_len
                q_current_seqlen = T.if_then_else(active, q_current_seqlen, 0)
                k_current_seqlen = T.if_then_else(active, k_current_seqlen, 0)

            loop_range_k = T.ceildiv(k_current_seqlen, block_N)
            tiles_per_split = T.ceildiv(loop_range_k, k_splits)
            k_tile_begin = k_split * tiles_per_split
            k_tile_end = T.min(loop_range_k, k_tile_begin + tiles_per_split)

            q_copy_end = T.min(
                q_start_idx * groups + (bx + 1) * block_M, q_end_idx * groups
            )
            T.copy(
                Q_unpad[
                    q_start_idx * groups + bx * block_M : q_copy_end, kv_head_idx, :
                ],
                Q_shared,
            )
            for i, d in T.Parallel(block_M, dim):
                if original_q_idx >= q_current_seqlen:
                    Q_shared[i, d] = 0

            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for k in T.Pipelined(k_tile_begin, k_tile_end, num_stages=num_stages):
                _load_masked_scores(
                    K_unpad,
                    K_shared,
                    Q_shared,
                    acc_s,
                    k,
                    kv_head_idx,
                    k_start_idx,
                    k_end_idx,
                    k_current_seqlen,
                    q_current_seqlen,
                    original_q_idx,
                    cache_len,
                    is_causal,
                    block_M,
                    block_N,
                    m_block_dim,
                    kernel_stride,
                )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )

                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            for i in T.Parallel(block_M):
                if original_q_idx < q_current_seqlen:
                    global_q_idx = q_start_idx + original_q_idx
                    Stats[kv_head_idx, global_q_idx, i, k_split, 0] = scores_max[i]
                    Stats[kv_head_idx, global_q_idx, i, k_split, 1] = logsum[i]

    return main


@tilelang.jit(pass_configs=_pass_configs)
def _fused_attn_pooled_scores(
    batch_size: int,
    groups: int,
    heads: int,
    dim: int,
    pooled_k_len: int,
    is_causal: bool,
    k_splits: int,
    max_seqlen_q_grid: int = 1,
    dense_len: int = 0,
    m_block_dim: int = 16,
    block_M: int = 16,
    block_N: int = _BLOCK_N,
    block_P: int = 128,
    block_stride: int = 4,
    pad_len: int = 1,
    num_offs: int = 5,
    kernel_stride: int = 16,
    block_size: int = 64,
    init_blocks: int = 0,
    local_blocks: int = 0,
    threads: int = 128,
    dtype_str: str = "bfloat16",
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    p_splits = -(-pooled_k_len // block_P)

    UQ = T.dynamic("UQ")
    UKV = T.dynamic("UKV")
    q_shape = [UQ * groups, head_kv, dim]
    kv_shape = [UKV, head_kv, dim]
    stats_shape = [head_kv, UQ, block_M, k_splits, 2]
    scores_shape = [head_kv, UQ, pooled_k_len]
    dtype = dtype_str
    accum_dtype = "float"

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        K_unpad: T.Tensor(kv_shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], "int32"),
        cu_seqlens_k: T.Tensor([batch_size + 1], "int32"),
        k_starts: T.Tensor([batch_size], "int32"),
        cache_lens: T.Tensor([batch_size], "int32"),
        Stats: T.Tensor(stats_shape, "float32"),
        PoolScores: T.Tensor(scores_shape, "float32"),
    ):
        with T.Kernel(
            max_seqlen_q_grid, head_kv, batch_size * p_splits, threads=threads
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            pool_max_shared = T.alloc_shared([block_P], "float32")
            scores_max = T.alloc_shared([block_M], accum_dtype)
            logsum = T.alloc_shared([block_M], accum_dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_output = T.alloc_fragment([block_N], accum_dtype)

            batch_idx = bz // p_splits
            p_block = bz % p_splits
            kv_head_idx = by
            original_q_idx = bx

            q_start_idx, q_end_idx, k_start_idx, k_len, cache_len = _row_geometry(
                cu_seqlens_q, cu_seqlens_k, k_starts, cache_lens, batch_idx
            )
            k_end_idx = k_start_idx + k_len
            q_current_seqlen = T.alloc_var("int32", init=q_end_idx - q_start_idx)
            k_current_seqlen = T.alloc_var("int32", init=k_len)
            if not is_causal:
                active = cache_len + 1 >= dense_len
                q_current_seqlen = T.if_then_else(active, q_current_seqlen, 0)
                k_current_seqlen = T.if_then_else(active, k_current_seqlen, 0)
            # The local-block boost selects the token's own block even before
            # its compressed row exists; the clamp bounds padded rows' pool rounds.
            query_token_pos = cache_len + (
                T.min(original_q_idx, q_current_seqlen - 1) if is_causal else 0
            )
            actual_pooled_k_len = query_token_pos // block_size + 1
            effective_pooled_k_len = T.min(actual_pooled_k_len, pooled_k_len)
            global_q_idx = q_start_idx + original_q_idx
            pool_block_start = p_block * block_P
            pool_block_end = T.min(pool_block_start + block_P, effective_pooled_k_len)

            q_copy_end = T.min(
                q_start_idx * groups + (bx + 1) * block_M, q_end_idx * groups
            )
            T.copy(
                Q_unpad[
                    q_start_idx * groups + bx * block_M : q_copy_end,
                    kv_head_idx,
                    :,
                ],
                Q_shared,
            )
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.fill(logsum, 0)
            if original_q_idx < q_current_seqlen:
                for i in T.Parallel(block_M):
                    row_max = T.alloc_var(accum_dtype, init=-T.infinity(accum_dtype))
                    for s in T.serial(k_splits):
                        row_max = T.max(
                            row_max, Stats[kv_head_idx, global_q_idx, i, s, 0]
                        )
                    row_sum = T.alloc_var(accum_dtype, init=0)
                    for s in T.serial(k_splits):
                        split_max = Stats[kv_head_idx, global_q_idx, i, s, 0]
                        # An empty split carries -inf; skip it instead of exp2(-inf + inf).
                        row_sum += T.if_then_else(
                            split_max > -T.infinity(accum_dtype),
                            Stats[kv_head_idx, global_q_idx, i, s, 1]
                            * T.exp2(split_max * scale - row_max * scale),
                            T.Cast(accum_dtype, 0.0),
                        )
                    scores_max[i] = row_max
                    logsum[i] = row_sum
            T.fill(pool_max_shared, float("-inf"))
            T.sync_threads()

            loop_range_k = T.ceildiv(k_current_seqlen, block_N)
            k_tile_begin = T.max(
                0, (pool_block_start * block_stride - pad_len) // block_N
            )
            # Rows past the request's length scan nothing; the syncs below
            # must not sit inside a data-dependent branch.
            # The last pool block is end - 1; its key window ends at start + num_offs.
            k_tile_end = T.if_then_else(
                original_q_idx < q_current_seqlen,
                T.min(
                    loop_range_k,
                    T.ceildiv(
                        (pool_block_end - 1) * block_stride - pad_len + num_offs,
                        block_N,
                    ),
                ),
                k_tile_begin,
            )
            for k in T.serial(k_tile_begin, k_tile_end):
                _load_masked_scores(
                    K_unpad,
                    K_shared,
                    Q_shared,
                    acc_s,
                    k,
                    kv_head_idx,
                    k_start_idx,
                    k_end_idx,
                    k_current_seqlen,
                    q_current_seqlen,
                    original_q_idx,
                    cache_len,
                    is_causal,
                    block_M,
                    block_N,
                    m_block_dim,
                    kernel_stride,
                )

                for i, j in T.Parallel(block_M, block_N):
                    normalized = (
                        T.exp2(acc_s[i, j] * scale - scores_max[i] * scale) / logsum[i]
                    )
                    # A row with no live keys has logsum 0; score it 0.
                    acc_s[i, j] = T.if_then_else(
                        (logsum[i] > 1e-10)
                        and (normalized >= 0)
                        and (normalized <= 1e10),
                        normalized,
                        T.Cast(accum_dtype, 0.0),
                    )

                T.fill(acc_output, 0)
                T.reduce_sum(acc_s, acc_output, dim=0)

                for j in T.Parallel(block_N):
                    k_idx = k * block_N + j
                    if k_idx < k_current_seqlen:
                        start_pool = T.max(
                            0,
                            (k_idx - num_offs + 1 + pad_len + block_stride - 1)
                            // block_stride,
                        )
                        end_pool = T.min(
                            effective_pooled_k_len - 1,
                            (k_idx + pad_len) // block_stride,
                        )
                        for p_off in T.serial(num_offs):
                            p_idx = start_pool + p_off
                            if (
                                p_idx >= pool_block_start
                                and p_idx < pool_block_end
                                and p_idx <= end_pool
                            ):
                                T.atomic_max(
                                    pool_max_shared[p_idx - pool_block_start],
                                    acc_output[j],
                                )
                T.sync_threads()

            # Dense rows write empty scores; grid padding owns no output row.
            for p_off in T.Parallel(block_P):
                p_idx = pool_block_start + p_off
                if (original_q_idx < q_end_idx - q_start_idx) and (
                    p_idx < pooled_k_len
                ):
                    off_bq = (original_q_idx + cache_len) // block_size
                    is_init_masked = (init_blocks > 0) and (p_idx < init_blocks)
                    is_local_masked = (
                        (local_blocks > 0)
                        and (off_bq >= p_idx)
                        and (off_bq <= p_idx + local_blocks)
                    )
                    PoolScores[kv_head_idx, global_q_idx, p_idx] = T.if_then_else(
                        (original_q_idx >= q_current_seqlen)
                        or (p_idx >= effective_pooled_k_len),
                        T.Cast("float32", float("-inf")),
                        T.if_then_else(
                            is_init_masked or is_local_masked,
                            T.Cast("float32", float("inf")),
                            pool_max_shared[p_off],
                        ),
                    )

    return main


def build_fused_topk_kernels(*, is_causal: bool, k_splits: int, **kwargs):
    stats_kwargs = {
        key: kwargs[key]
        for key in (
            "batch_size",
            "groups",
            "heads",
            "dim",
            "max_seqlen_q_grid",
            "dense_len",
            "m_block_dim",
            "block_M",
            "kernel_stride",
            "dtype_str",
        )
        if key in kwargs
    }
    return FusedTopkKernels(
        stats=_fused_attn_stats(is_causal=is_causal, k_splits=k_splits, **stats_kwargs),
        scores=_fused_attn_pooled_scores(
            is_causal=is_causal,
            k_splits=k_splits,
            block_P=_PREFILL_BLOCK_P if is_causal else _DECODE_BLOCK_P,
            **kwargs,
        ),
        k_splits=k_splits,
        pooled_k_len=kwargs["pooled_k_len"],
    )
