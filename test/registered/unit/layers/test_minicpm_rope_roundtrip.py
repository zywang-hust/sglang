"""The bf16 rope kernel against the former fp32 round trip, bitwise."""

import unittest

import torch

from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")

DEVICE = "cuda"


class TestMiniCPMRopeRoundTrip(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def test_bf16_rope_matches_fp32_roundtrip(self):
        """bf16 into the jit rope kernel is bitwise-equal to the former round trip
        (float() -> rope -> to(bf16)): the kernel upcasts to fp32 and rounds once."""
        num_heads, head_dim, num_tokens = 32, 128, 173
        gen = torch.Generator(device="cpu").manual_seed(7)
        shape = (num_tokens, num_heads * head_dim)
        q = (torch.randn(shape, generator=gen) * 0.4).to(DEVICE, torch.bfloat16)
        k = (torch.randn(shape, generator=gen) * 0.4).to(DEVICE, torch.bfloat16)
        positions = torch.randint(0, 4096, (num_tokens,), generator=gen).to(DEVICE)
        rope = get_rope(
            head_dim, rotary_dim=head_dim, max_position=8192, base=10000
        ).to(DEVICE)

        q_rt, k_rt = rope(positions, q.float(), k.float())
        q_bf, k_bf = rope(positions, q.clone(), k.clone())
        self.assertTrue(torch.equal(q_bf, q_rt.to(torch.bfloat16)))
        self.assertTrue(torch.equal(k_bf, k_rt.to(torch.bfloat16)))


if __name__ == "__main__":
    unittest.main()
