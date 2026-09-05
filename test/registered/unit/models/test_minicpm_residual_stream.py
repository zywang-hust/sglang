"""Residual-stream reuse across the fused add-rmsnorm layers,
for fresh and carried residual entry forms."""

import math
import unittest

import torch

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.minicpm import MiniCPMDecoderLayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_NUM_LAYERS = 4
_HIDDEN = 8
_SCALE_DEPTH = 1.4


class _Branch:
    def __init__(self, shift: float):
        self.shift = shift

    def __call__(self, hidden_states=None, **_):
        return torch.tanh(hidden_states * 1.5 + self.shift)


def _cpu_rmsnorm(generator: torch.Generator) -> RMSNorm:
    norm = RMSNorm(_HIDDEN)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(_HIDDEN, generator=generator) + 0.5)
    # Platform dispatch targets CUDA whenever it is present;
    # pin the pure PyTorch path so the test runs on CPU tensors.
    norm.forward = norm.forward_native
    return norm


def _make_layer(seed: int) -> MiniCPMDecoderLayer:
    generator = torch.Generator().manual_seed(seed)
    layer = MiniCPMDecoderLayer.__new__(MiniCPMDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.input_layernorm = _cpu_rmsnorm(generator)
    layer.post_attention_layernorm = _cpu_rmsnorm(generator)
    layer.self_attn = _Branch(shift=0.25 * seed)
    layer.mlp = _Branch(shift=-0.1 * seed)
    layer.layer_scale = _SCALE_DEPTH / math.sqrt(_NUM_LAYERS)
    return layer


def _reference_layer(layer: MiniCPMDecoderLayer, x: torch.Tensor) -> torch.Tensor:
    """Pre-fusion semantics: explicit residual adds around single-arg norms."""
    residual = x
    normed = layer.input_layernorm(x)
    x = residual + layer.self_attn(hidden_states=normed) * layer.layer_scale
    residual = x
    normed = layer.post_attention_layernorm(x)
    return residual + layer.mlp(hidden_states=normed) * layer.layer_scale


class TestFusedResidualStream(CustomTestCase):
    """The split (hidden, residual) stream must equal the sequential stream
    x = x + branch(norm(x)) * layer_scale per branch;
    in fp32 both orderings are exact, so the check is bitwise."""

    def test_layer_chain_matches_sequential_reference(self):
        torch.manual_seed(0)
        layers = [_make_layer(seed=i + 1) for i in range(_NUM_LAYERS)]
        x0 = torch.randn(3, _HIDDEN, dtype=torch.float32)

        ref = x0
        hidden, residual = x0, None
        for i, layer in enumerate(layers):
            ref = _reference_layer(layer, ref)
            hidden, residual = layer.forward(
                positions=None,
                hidden_states=hidden,
                forward_batch=None,
                residual=residual,
            )
            self.assertTrue(
                torch.equal(hidden + residual, ref), f"stream diverged at layer {i}"
            )

        final_norm = _cpu_rmsnorm(torch.Generator().manual_seed(99))
        fused_out, _ = final_norm(hidden, residual)
        self.assertTrue(torch.equal(fused_out, final_norm(ref.clone())))


if __name__ == "__main__":
    unittest.main()
