"""CPU-only contract checks for the A5 runner's CLI and matrix."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RUNNER = (
    Path(__file__).resolve().parents[1] / "benchmark" / "lim_a5_mtp_ab" / "bench.py"
)


class RunnerPreflight(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, str(RUNNER), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_help_without_accelerator(self):
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--original-library", result.stdout)
        self.assertIn("--inputs-file", result.stdout)
        self.assertIn("--save-inputs", result.stdout)

    def test_matrix_and_miss_targets(self):
        result = self.invoke("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        cases = json.loads(result.stdout)
        self.assertEqual(len(cases), 16)
        self.assertEqual(len({c["case_id"] for c in cases}), 16)
        expected = {(1, 91): 184, (1, 95): 102, (4, 91): 737, (4, 95): 410}
        for case in cases:
            self.assertEqual(
                case["union_misses_per_request"],
                expected[case["q"], case["hit_percent"]],
            )
            self.assertEqual(case["actual_length"], 65664)
            self.assertEqual(case["source_capacity"], 66944)

    def test_required_inputs(self):
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--case-id", result.stderr)


class PoolValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(RUNNER.parent))
        spec = importlib.util.spec_from_file_location("a5_bench_test", RUNNER)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.module.B, cls.module.K, cls.module.C = 1, 2, 2
        cls.module.PREFIX, cls.module.ACTUAL = 4, 6

    def test_tarball_source_does_not_inherit_enclosing_git_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", str(root)], capture_output=True, check=True)
            extracted = root / "extracted"
            extracted.mkdir()
            provenance = self.module.source_provenance(extracted)
            self.assertIsNone(provenance["source_commit"])
            self.assertIsNone(provenance["source_dirty_diff_sha256"])
            self.assertIn("no matching Git root", provenance["source_dirty_diff_scope"])

    def test_exact_pool_delta_and_rejection(self):
        module = self.module

        class Tensor:
            def __init__(self, value):
                self.value = value

            def cpu(self):
                return self

            def tolist(self):
                return self.value

        inv = module.INVALID
        initial = [[inv] * 6, [0, inv, 1, inv, inv, inv]]
        post = [[inv] * 6, [0, 1, inv, inv, inv, inv]]
        outputs = [Tensor(v) for v in ([[[0, 1]]], [[[0, 1]]], [1], [[1]], [[1]], [1])]
        checked = module.validate_result(
            None,
            [[[0, 1]]],
            {"missing": [[1]]},
            initial,
            Tensor(post),
            outputs,
            1,
            "test",
        )
        self.assertTrue(checked["passed"])
        post[1][2] = 1
        with self.assertRaisesRegex(AssertionError, "post-pool"):
            module.validate_result(
                None,
                [[[0, 1]]],
                {"missing": [[1]]},
                initial,
                Tensor(post),
                outputs,
                1,
                "test",
            )


if __name__ == "__main__":
    unittest.main()
