"""Fused qk-norm + RoPE kernel against the eager rmsnorm -> rope chain."""

import pytest
import torch

from sglang.kernels.ops.attention.minicpm_qknorm_rope import minicpm_qknorm_rope
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding, get_rope
from sglang.srt.models.minicpm import MiniCPMLightningMixer
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
# The fused kernel and the eager module chain reduce the variance in fp32,
# in a different order, so rare single-ulp bf16 flips are expected.
REDUCTION_TOLERANCE = 2.0**-7


@pytest.fixture(autouse=True)
def _server_args():
    override = get_context().override_server_args()
    override.install()
    yield
    override.restore()


@pytest.mark.parametrize("num_tokens", [1, 16, 56, 333])
@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
def test_matches_production_module_chain(num_tokens, pos_dtype):
    num_heads, head_dim, eps = 32, 128, 1e-6
    width = num_heads * head_dim
    gen = torch.Generator(device="cpu").manual_seed(31 + num_tokens)
    qkv = (torch.randn((num_tokens, 3 * width), generator=gen) * 0.4).to(
        DEVICE, torch.bfloat16
    )
    positions = torch.randint(0, 4096, (num_tokens,), generator=gen).to(
        DEVICE, pos_dtype
    )

    q_norm = RMSNorm(head_dim, eps=eps).to(DEVICE, torch.bfloat16)
    k_norm = RMSNorm(head_dim, eps=eps).to(DEVICE, torch.bfloat16)
    with torch.no_grad():
        q_norm.weight.copy_(torch.randn((head_dim,), generator=gen) * 0.3)
        k_norm.weight.copy_(torch.randn((head_dim,), generator=gen) * 0.3)
    rope = get_rope(head_dim, rotary_dim=head_dim, max_position=8192, base=10000).to(
        DEVICE
    )

    q, k, v = qkv.clone().split([width, width, width], dim=-1)
    q_e = q_norm(q.reshape(-1, head_dim)).reshape(-1, width)
    k_e = k_norm(k.reshape(-1, head_dim)).reshape(-1, width)
    q_e, k_e = rope(positions, q_e, k_e)
    q_e = q_e.reshape(num_tokens, num_heads, head_dim)
    k_e = k_e.reshape(num_tokens, num_heads, head_dim)

    q_f, k_f, v_f = minicpm_qknorm_rope(
        qkv=qkv,
        q_weight=q_norm.weight,
        k_weight=k_norm.weight,
        cos_sin_cache=rope.cos_sin_cache,
        positions=positions,
        num_q_heads=num_heads,
        num_k_heads=num_heads,
        num_v_heads=num_heads,
        head_dim=head_dim,
        eps=eps,
    )
    for view in (q_f, k_f, v_f):
        assert view.untyped_storage().data_ptr() == qkv.untyped_storage().data_ptr()
    assert torch.equal(v_f, v.reshape(num_tokens, num_heads, head_dim))
    torch.testing.assert_close(
        q_f, q_e, atol=REDUCTION_TOLERANCE, rtol=REDUCTION_TOLERANCE
    )
    torch.testing.assert_close(
        k_f, k_e, atol=REDUCTION_TOLERANCE, rtol=REDUCTION_TOLERANCE
    )


def _make_mixer(head_dim=128, hidden_size=256, **kwargs):
    with get_parallel().override(tp_size=1, tp_rank=0):
        return MiniCPMLightningMixer(
            hidden_size=hidden_size,
            num_heads=2,
            num_kv_heads=2,
            head_dim=head_dim,
            **kwargs,
        )


def test_mrope_rope_keeps_unfused_path():
    # The asserted rope shape leaves the mrope exclusion as the only gate clause:
    # the fused kernel takes neither 2-D per-section positions nor a dtype-cast forward.
    mixer = _make_mixer(
        rope_scaling={"rope_type": "default", "mrope_section": [16, 24, 24]}
    )
    assert isinstance(mixer.rotary_emb, MRotaryEmbedding)
    assert mixer.rotary_emb.is_neox_style
    assert mixer.rotary_emb.rotary_dim == mixer.head_dim
    assert not mixer.rotary_emb.use_fallback_kernel
    assert mixer.use_fused_qknorm_rope is False


def test_fallback_rope_keeps_unfused_path():
    # CUDA only: use_fallback_kernel flips True
    # exactly when head_size is outside {64, 128, 256, 512};
    # head_dim=32 leaves every other condition holding.
    mixer = _make_mixer(head_dim=32, hidden_size=128)
    assert mixer.rotary_emb.use_fallback_kernel is True
    assert mixer.use_fused_qknorm_rope is False


def test_default_rope_takes_fused_path():
    # Positive control: the production SALA shape must stay fused.
    mixer = _make_mixer()
    assert mixer.use_fused_qknorm_rope is True


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
