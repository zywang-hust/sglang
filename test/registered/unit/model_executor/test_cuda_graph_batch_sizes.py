import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    get_batch_sizes_to_capture,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_MODULE = "sglang.srt.model_executor.runner.base_cuda_graph_runner"


def _model_runner(
    *,
    cuda_graph_bs,
    pool_size=16,
    enable_torch_compile=False,
    torch_compile_max_bs=4,
):
    server_args = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(bs=list(cuda_graph_bs))
        ),
        enable_two_batch_overlap=False,
        enable_torch_compile=enable_torch_compile,
        torch_compile_max_bs=torch_compile_max_bs,
    )
    return SimpleNamespace(
        server_args=server_args,
        req_to_token_pool=SimpleNamespace(size=pool_size),
    )


class TestCudaGraphBatchSizes(CustomTestCase):
    def setUp(self):
        self.patch_gather = patch(
            f"{_MODULE}.require_gathered_buffer",
            return_value=False,
        )
        self.patch_attn_cp = patch(
            f"{_MODULE}.get_attention_cp_size",
            return_value=1,
        )
        self.patch_gather.start()
        self.patch_attn_cp.start()
        self.addCleanup(self.patch_gather.stop)
        self.addCleanup(self.patch_attn_cp.stop)

    def test_compile_bs_truncates_to_torch_compile_max_bs(self):
        # The compile_bs branch is taken only when torch.compile is on, and it
        # is the one path get_batch_sizes_to_capture has that capture_bs does
        # not already exercise: it keeps only bs <= torch_compile_max_bs.
        # (The no-mutation/no-pollution invariant is covered by the test below.)
        runner = _model_runner(
            cuda_graph_bs=[1, 2, 4, 8],
            pool_size=16,
            enable_torch_compile=True,
            torch_compile_max_bs=4,
        )

        capture_bs, compile_bs = get_batch_sizes_to_capture(runner)

        self.assertEqual(capture_bs, [1, 2, 4, 8])
        self.assertEqual(compile_bs, [1, 2, 4])

    def test_clamped_runner_does_not_pollute_second_runner(self):
        shared_server_args = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(bs=[1, 2, 4, 8])),
            enable_two_batch_overlap=False,
            enable_torch_compile=False,
            torch_compile_max_bs=4,
        )
        clamped_runner = SimpleNamespace(
            server_args=shared_server_args,
            req_to_token_pool=SimpleNamespace(size=6),
        )
        full_runner = SimpleNamespace(
            server_args=shared_server_args,
            req_to_token_pool=SimpleNamespace(size=16),
        )

        clamped_bs, _ = get_batch_sizes_to_capture(clamped_runner)
        full_bs, _ = get_batch_sizes_to_capture(full_runner)

        self.assertEqual(clamped_bs, [1, 2, 4, 6])
        self.assertEqual(full_bs, [1, 2, 4, 8])
        self.assertEqual(shared_server_args.cuda_graph_config.decode.bs, [1, 2, 4, 8])


if __name__ == "__main__":
    unittest.main()
