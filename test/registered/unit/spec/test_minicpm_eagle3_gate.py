from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace

from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Backends with an explicit verify write-back (`_mamba_verify_update`).
# AGENTS.md hard rule: the gate must list these explicitly instead of using a
# `mambaish_config` catch-all, which would wrongly include KDA/kimi backends.
_GATED_CONFIGS = (
    "hybrid_gdn_config",
    "mamba2_config",
    "hybrid_lightning_config",
    "minicpm_hybrid_config",
)


def _extract_gate_condition(verify_fn):
    """Return the AST condition of the `if` guarding `_mamba_verify_update`."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(verify_fn)))
    conditions = [
        node.test
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and stmt.value.func.attr == "_mamba_verify_update"
            for stmt in node.body
        )
    ]
    if len(conditions) != 1:
        raise AssertionError(
            f"expected exactly one gated `_mamba_verify_update` call in "
            f"{verify_fn.__qualname__}, found {len(conditions)}"
        )
    return conditions[0]


def _gate_predicate(verify_fn):
    """Compile the gate condition into a predicate over a model-runner stub."""
    expr = ast.Expression(body=_extract_gate_condition(verify_fn))
    code = compile(ast.fix_missing_locations(expr), "<mamba-verify-gate>", "eval")

    def predicate(model_runner):
        worker_stub = SimpleNamespace(
            target_worker=SimpleNamespace(model_runner=model_runner)
        )
        # Evaluate with the worker module's globals so the gate may freely be
        # refactored into a (possibly shared) helper without breaking the test.
        return bool(eval(code, dict(verify_fn.__globals__), {"self": worker_stub}))

    return predicate


def _model_runner_stub(**configs):
    """Mimic ModelRunner's config properties for the verify gate.

    Mirrors the real coupling: `mambaish_config` is a catch-all that is set
    whenever any mamba-family config (including KDA/kimi) is set.
    """
    fields = dict.fromkeys(_GATED_CONFIGS + ("kimi_linear_config",))
    fields.update(configs)
    fields["mambaish_config"] = next(
        (value for value in fields.values() if value is not None), None
    )
    return SimpleNamespace(**fields)


class TestMiniCPMEagle3Gate(CustomTestCase):
    def _assert_gate_behavior(self, verify_fn):
        gate = _gate_predicate(verify_fn)

        # No hybrid/linear-attention state config: no verify write-back.
        self.assertFalse(gate(_model_runner_stub()))

        # Each explicitly supported backend (incl. MiniCPM) gets the write-back.
        for config_name in _GATED_CONFIGS:
            self.assertTrue(
                gate(_model_runner_stub(**{config_name: object()})),
                f"{config_name} must trigger _mamba_verify_update",
            )

        # KDA/kimi-style backends set mambaish_config but have no verify
        # write-back: a `mambaish_config is not None` catch-all must not pass.
        self.assertFalse(
            gate(_model_runner_stub(kimi_linear_config=object())),
            "kimi-style mambaish backend must not trigger _mamba_verify_update",
        )

    def test_v2_verify_gate_covers_minicpm_without_mambaish_catchall(self):
        self._assert_gate_behavior(EAGLEWorkerV2.verify)


if __name__ == "__main__":
    unittest.main()
