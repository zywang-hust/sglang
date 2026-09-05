"""Shared construction scaffold for MiniCPMSparseBackend unit tests."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.runtime_context import get_context, override_platform
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.minicpm import backend as backend_module
from sglang.srt.layers.attention.minicpm.backend import MiniCPMSparseBackend
from sglang.srt.layers.attention.minicpm.sparse_utils import (
    MiniCPMSparseMetadata,
    _plan_sparse_verify,
)

POOL_SIZE = 8


def _sparse_token_table(*, pages, device):
    """One compression level's request-to-chunk table, distinct value per slot."""
    return torch.arange(POOL_SIZE * pages, dtype=torch.int32, device=device).view(
        POOL_SIZE, pages
    )


def make_runner_scaffold(
    *,
    max_context_len=1024,
    max_running_requests=1,
    num_kv_heads=1,
    device="cpu",
    window_size=64,
):
    """Build the (model_runner, flash_attn_backend) SimpleNamespace pair
    that MiniCPMSparseBackend construction reads;
    callers add the fields only their tests consume
    before patching in the backend."""
    kernel_size, kernel_stride = 32, 16
    # attach_compressed_cache is stubbed out here, so the scaffold sizes the K1/K2
    # tables the way cache.py does; the graph buffers assert on that width.
    req_pool = SimpleNamespace(
        req_to_sparse_k1_token=_sparse_token_table(
            pages=(max_context_len - kernel_size) // kernel_stride + 1, device=device
        ),
        req_to_sparse_k2_token=_sparse_token_table(
            pages=(max_context_len - kernel_size * 4) // (kernel_stride * 4) + 1,
            device=device,
        ),
        req_to_token=torch.arange(
            POOL_SIZE * 1024, dtype=torch.int32, device=device
        ).view(POOL_SIZE, 1024),
    )
    flash_attn_backend = SimpleNamespace(
        max_context_len=max_context_len,
        device=device,
        decode_cuda_graph_metadata={},
        req_to_token_pool=req_pool,
        token_to_kv_pool=SimpleNamespace(),
        page_size=1,
    )
    server_args = SimpleNamespace(enable_memory_saver=False)
    model_runner = SimpleNamespace(
        dtype=torch.float16,
        max_running_requests=max_running_requests,
        token_to_kv_pool_allocator=SimpleNamespace(),
        server_args=server_args,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                has_minicpm_sparse_attention=True,
                sparse_config={
                    "kernel_size": kernel_size,
                    "kernel_stride": kernel_stride,
                    "init_blocks": 1,
                    "block_size": 64,
                    "window_size": window_size,
                    "dense_len": 128,
                    "topk": 1,
                },
            ),
            num_attention_heads=16 * num_kv_heads,
            head_dim=128,
            get_num_kv_heads=lambda _tp: num_kv_heads,
        ),
    )
    return model_runner, flash_attn_backend


class FakeFlashInferAdapter:
    def __init__(self, *args, **kwargs):
        pass


def make_sparse_backend(
    *,
    max_context_len=256,
    chunked_prefill_size=64,
    max_running_requests=1,
    use_flashinfer=False,
    blackwell=False,
    kv_cache_dtype_str="bfloat16",
    server_args_overrides=None,
    num_kv_heads=1,
    device="cpu",
    window_size=None,
):
    """A CPU MiniCPMSparseBackend over the mocked runner scaffold; returns
    (backend, model_runner, flash_attn_backend, flash_attention_ctor)."""
    if window_size is None:
        # Tree verify requires the local window to cover the draft tokens plus
        # one block; every other shape keeps the scaffold's one-block window.
        tree = ((server_args_overrides or {}).get("speculative_eagle_topk") or 1) > 1
        window_size = 128 if tree else 64
    model_runner, flash_attn_backend = make_runner_scaffold(
        max_context_len=max_context_len,
        max_running_requests=max_running_requests,
        num_kv_heads=num_kv_heads,
        device=device,
        window_size=window_size,
    )
    flash_attn_backend.kv_cache_dtype_str = kv_cache_dtype_str
    flash_attn_backend.init_forward_metadata = Mock()
    flash_attn_backend.init_cuda_graph_state = Mock()
    flash_attn_backend.forward_metadata = None
    # The constructor reads chunked_prefill_size and the draft shape off the
    # published context; publish them for the build.
    override = get_context().override_server_args(
        chunked_prefill_size=chunked_prefill_size, **(server_args_overrides or {})
    )
    override.install()
    try:
        with (
            patch.object(backend_module, "MiniCPMHybridConfig", SimpleNamespace),
            override_platform(is_blackwell=blackwell),
            patch.object(
                backend_module, "FlashAttentionBackend", return_value=flash_attn_backend
            ) as flash_attention,
            patch.object(
                backend_module, "MiniCPMFlashInferAdapter", FakeFlashInferAdapter
            ),
            patch.object(
                backend_module,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_size=1),
            ),
            patch.object(backend_module, "attach_compressed_cache"),
        ):
            backend = MiniCPMSparseBackend(model_runner, use_flashinfer=use_flashinfer)
    finally:
        override.restore()
    return backend, model_runner, flash_attn_backend, flash_attention


def make_spec_backend(
    *, num_draft_tokens, eagle_topk, num_kv_heads=1, device="cpu", window_size=None
):
    """Build a MiniCPMSparseBackend configured for one draft shape,
    with a mocked adapter. Returns (backend, flash_attn_backend)."""
    backend, _, flash_attn_backend, _ = make_sparse_backend(
        server_args_overrides={
            "speculative_num_draft_tokens": num_draft_tokens,
            "speculative_eagle_topk": eagle_topk,
        },
        num_kv_heads=num_kv_heads,
        device=device,
        window_size=window_size,
    )
    backend.attention_adapter = Mock()
    return backend, flash_attn_backend


def _verify_round_layout(*, seq_lens, num_draft_tokens, device):
    """The seq/cu_seqlens layout shared by the eager and replay builders."""
    bs = len(seq_lens)
    seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32)
    cache_seqlens = (seq_lens_cpu + num_draft_tokens).to(device)
    cu_seqlens_q = torch.arange(
        0, bs * num_draft_tokens + 1, num_draft_tokens, dtype=torch.int32, device=device
    )
    cu_seqlens_k = F.pad(torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32), (1, 0))
    return cache_seqlens, cu_seqlens_q, cu_seqlens_k


def make_verify_batch(seq_lens, num_draft_tokens, *, device="cpu"):
    """Build one verify round's eager inputs: (forward_batch, base)."""
    bs = len(seq_lens)
    seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32)
    cache_seqlens, cu_seqlens_q, cu_seqlens_k = _verify_round_layout(
        seq_lens=seq_lens, num_draft_tokens=num_draft_tokens, device=device
    )
    forward_batch = SimpleNamespace(
        batch_size=bs,
        seq_lens=seq_lens_cpu.clone().to(device),
        seq_lens_cpu=seq_lens_cpu,
        req_pool_indices=torch.arange(bs, dtype=torch.int64, device=device),
    )
    base = SimpleNamespace(
        cache_seqlens_int32=cache_seqlens,
        max_seq_len_k=int(cache_seqlens.max()),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        page_table=torch.zeros(
            (bs, int(cache_seqlens.max())), dtype=torch.int32, device=device
        ),
    )
    return forward_batch, base


def plan_verify(
    seq_lens,
    num_draft_tokens,
    *,
    head_group_num,
    heads_per_group,
    dense_len,
    sparse_topk,
    block_size,
    num_draft_visible=None,
    device="cpu",
):
    """Plan one eager verify round; returns (forward_batch, base, metadata)."""
    forward_batch, base = make_verify_batch(seq_lens, num_draft_tokens, device=device)
    metadata = MiniCPMSparseMetadata(base=base)
    _plan_sparse_verify(
        forward_batch=forward_batch,
        metadata=metadata,
        head_group_num=head_group_num,
        heads_per_group=heads_per_group,
        dense_len=dense_len,
        sparse_topk=sparse_topk,
        block_size=block_size,
        num_draft_tokens=num_draft_tokens,
        num_draft_visible=num_draft_visible,
    )
    return forward_batch, base, metadata


def make_replay_backend(
    seq_lens,
    num_draft_tokens,
    *,
    eagle_topk=1,
    num_kv_heads=1,
    max_bs,
    device="cpu",
):
    """Size a backend's CUDA-graph buffers for max_bs verify requests;
    return (backend, base) with the static base buffers the replay leaves."""
    bs = len(seq_lens)
    cache_seqlens, cu_seqlens_q, cu_seqlens_k = _verify_round_layout(
        seq_lens=seq_lens, num_draft_tokens=num_draft_tokens, device=device
    )
    backend, _ = make_spec_backend(
        num_draft_tokens=num_draft_tokens,
        eagle_topk=eagle_topk,
        num_kv_heads=num_kv_heads,
        device=device,
    )
    backend.init_cuda_graph_state(max_bs, max_bs * num_draft_tokens)
    base = SimpleNamespace(
        cache_seqlens_int32=cache_seqlens,
        max_seq_len_q=num_draft_tokens,
        max_seq_len_k=int(cache_seqlens.max()),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        # The graph page table spans the context; page_size is 1 in the scaffold.
        page_table=torch.zeros(
            (bs, backend.max_context_len), dtype=torch.int32, device=device
        ),
    )
    return backend, base


def tree_mask_from_parents(parents):
    """Ancestor-closure visibility square of one request's draft tree."""
    num_draft_tokens = len(parents)
    mask = torch.zeros((num_draft_tokens, num_draft_tokens), dtype=torch.bool)
    for token in range(num_draft_tokens):
        node = token
        while node != -1:
            mask[token, node] = True
            node = parents[node]
    return mask


def visible_counts(*, mask, bs, num_draft_tokens):
    """Causal-prefix popcount oracle of a staged visibility square."""
    causal = torch.tril(
        torch.ones(
            (num_draft_tokens, num_draft_tokens), dtype=torch.bool, device=mask.device
        )
    )
    return (
        (mask.view(bs, num_draft_tokens, num_draft_tokens) & causal)
        .sum(dim=-1, dtype=torch.int32)
        .view(-1)
    )


def pack_custom_mask(seq_lens, squares, device="cpu"):
    """Pack rows in EAGLE's custom_mask layout: prefix columns all-visible,
    trailing square from the tree."""
    rows = []
    for seq_len, square in zip(seq_lens, squares):
        num_draft_tokens = square.shape[0]
        request = torch.ones(
            (num_draft_tokens, seq_len + num_draft_tokens), dtype=torch.bool
        )
        request[:, seq_len:] = square
        rows.append(request.view(-1))
    return torch.cat(rows).to(device)
