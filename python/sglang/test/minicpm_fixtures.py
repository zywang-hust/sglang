"""Shared construction scaffold for MiniCPMSparseBackend unit tests."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.minicpm import backend as backend_module
from sglang.srt.layers.attention.minicpm.backend import MiniCPMSparseBackend


def make_runner_scaffold(
    *,
    max_context_len=1024,
    max_running_requests=1,
    server_args_overrides=None,
    num_kv_heads=1,
    device="cpu",
):
    """Build the (model_runner, flash_attn_backend) SimpleNamespace pair
    that MiniCPMSparseBackend construction reads;
    callers add the fields only their tests consume
    before patching in the backend."""
    req_pool = SimpleNamespace(
        req_to_sparse_k1_token=torch.empty(0),
        req_to_sparse_k2_token=torch.empty(0),
        req_to_token=torch.arange(8 * 1024, dtype=torch.int32, device=device).view(
            8, 1024
        ),
    )
    flash_attn_backend = SimpleNamespace(
        max_context_len=max_context_len,
        device=device,
        decode_cuda_graph_metadata={},
        req_to_token_pool=req_pool,
        token_to_kv_pool=SimpleNamespace(),
        page_size=1,
    )
    server_args_fields = {
        "enable_memory_saver": False,
        "chunked_prefill_size": 64,
        "speculative_num_draft_tokens": None,
        "speculative_eagle_topk": None,
    }
    server_args_fields.update(server_args_overrides or {})
    server_args = SimpleNamespace(**server_args_fields)
    model_runner = SimpleNamespace(
        dtype=torch.float16,
        max_running_requests=max_running_requests,
        token_to_kv_pool_allocator=SimpleNamespace(),
        server_args=server_args,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                has_minicpm_sparse_attention=True,
                sparse_config={
                    "kernel_size": 32,
                    "kernel_stride": 16,
                    "init_blocks": 1,
                    "block_size": 64,
                    "window_size": 64,
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


def make_sparse_backend(*, server_args_overrides=None, num_kv_heads=1, device="cpu"):
    """Build a MiniCPMSparseBackend with the runner scaffold mocked out.

    Returns (backend, flash_attn_backend).
    """
    model_runner, flash_attn_backend = make_runner_scaffold(
        server_args_overrides=server_args_overrides,
        num_kv_heads=num_kv_heads,
        device=device,
    )
    flash_attn_backend.init_forward_metadata = Mock()
    flash_attn_backend.init_cuda_graph_state = Mock()
    flash_attn_backend.forward_metadata = None
    with (
        patch.object(backend_module, "MiniCPMHybridConfig", SimpleNamespace),
        patch.object(backend_module, "is_blackwell_supported", return_value=False),
        patch.object(
            backend_module, "FlashAttentionBackend", return_value=flash_attn_backend
        ),
        patch.object(
            backend_module,
            "get_parallel",
            return_value=SimpleNamespace(attn_tp_size=1),
        ),
        patch.object(backend_module, "attach_compressed_cache"),
    ):
        backend = MiniCPMSparseBackend(model_runner, use_flashinfer=False)
    return backend, flash_attn_backend


def make_spec_backend(*, num_draft_tokens, eagle_topk, num_kv_heads=1, device="cpu"):
    """Build a MiniCPMSparseBackend configured for one draft shape,
    with a mocked adapter. Returns (backend, flash_attn_backend)."""
    backend, flash_attn_backend = make_sparse_backend(
        server_args_overrides={
            "speculative_num_draft_tokens": num_draft_tokens,
            "speculative_eagle_topk": eagle_topk,
        },
        num_kv_heads=num_kv_heads,
        device=device,
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
    return seq_lens_cpu, cache_seqlens, cu_seqlens_q, cu_seqlens_k


def make_verify_batch(seq_lens, num_draft_tokens, *, device="cpu"):
    """Build one verify round's eager inputs: (forward_batch, base)."""
    bs = len(seq_lens)
    seq_lens_cpu, cache_seqlens, cu_seqlens_q, cu_seqlens_k = _verify_round_layout(
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
    pool_size, max_pages = 8, 1024
    _, cache_seqlens, cu_seqlens_q, cu_seqlens_k = _verify_round_layout(
        seq_lens=seq_lens, num_draft_tokens=num_draft_tokens, device=device
    )
    backend, _ = make_spec_backend(
        num_draft_tokens=num_draft_tokens,
        eagle_topk=eagle_topk,
        num_kv_heads=num_kv_heads,
        device=device,
    )
    backend.init_cuda_graph_state(max_bs, max_bs * num_draft_tokens)
    for name in ("k1", "k2"):
        pages = backend.decode_cuda_graph_metadata[f"{name}.table"].shape[1]
        table = torch.arange(pool_size * pages, dtype=torch.int32, device=device).view(
            pool_size, pages
        )
        setattr(backend, f"req_to_sparse_{name}_token", table)
    backend.compress_levels = tuple(
        (name, size, stride, getattr(backend, f"req_to_sparse_{name}_token"))
        for name, size, stride, _ in backend.compress_levels
    )
    base = SimpleNamespace(
        cache_seqlens_int32=cache_seqlens,
        max_seq_len_q=num_draft_tokens,
        max_seq_len_k=int(cache_seqlens.max()),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        page_table=torch.zeros((bs, max_pages), dtype=torch.int32, device=device),
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
