"""Shared construction scaffold for MiniCPMSparseBackend unit tests."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch

from sglang.srt.layers.attention.minicpm import backend as backend_module
from sglang.srt.layers.attention.minicpm.backend import MiniCPMSparseBackend


def make_runner_scaffold(
    *,
    max_context_len=1024,
    max_running_requests=1,
    server_args_overrides=None,
    num_kv_heads=1,
):
    """Build the (model_runner, flash_attn_backend) SimpleNamespace pair
    that MiniCPMSparseBackend construction reads;
    callers add the fields only their tests consume
    before patching in the backend."""
    req_pool = SimpleNamespace(
        req_to_sparse_k1_token=torch.empty(0),
        req_to_sparse_k2_token=torch.empty(0),
    )
    flash_attn_backend = SimpleNamespace(
        max_context_len=max_context_len,
        device="cpu",
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


def make_sparse_backend(*, server_args_overrides=None, num_kv_heads=1):
    """Build a MiniCPMSparseBackend on CPU with the runner scaffold mocked out.

    Returns (backend, flash_attn_backend).
    """
    model_runner, flash_attn_backend = make_runner_scaffold(
        server_args_overrides=server_args_overrides, num_kv_heads=num_kv_heads
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
