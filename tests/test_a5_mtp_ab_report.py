"""CPU-only tests for the A5 LIM MTP report gates and statistics."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "benchmark" / "lim_a5_mtp_ab" / "report.py"
SPEC = importlib.util.spec_from_file_location("a5_mtp_ab_report", REPORT_PATH)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def payload(case_id: str, *, diagnostic_only: bool = False) -> dict:
    q_text, mode, policy, hit_text = case_id.split("_")
    q = int(q_text[1:])
    hit = int(hit_text.removeprefix("hit"))
    graph_replay = {
        "passed": True,
        "request_miss_counts": [[0] * q],
        "route_miss_counts": [[0] * q],
    }
    return {
        "schema_version": 1,
        "case_id": case_id,
        "case": {
            "q": q,
            "mtp": q - 1,
            "mode": mode,
            "policy": policy,
            "hit_percent": hit,
            "hit_rate_basis": "selected_prefix_union",
        },
        "status": "ok",
        "diagnostic_only": diagnostic_only,
        "fixture": {
            "fixture_sha256": "fixture-a",
            "input_sha256": "input-a",
            "reference_sha256": "reference-a",
            "residency_sha256": "pool-a",
        },
        "hashes": {
            "fixture_sha256": "fixture-a",
            "input_sha256": "input-a",
            "reference_sha256": "reference-a",
            "residency_sha256": "pool-a",
        },
        "correctness": {
            name: {
                "passed": True,
                **(
                    {"graph_replay": graph_replay}
                    if mode == "graph" and name in ("original", "sglang")
                    else {}
                ),
            }
            for name in REPORT.BACKENDS
        },
        "rounds": [
            {
                "round": index,
                "correctness": {name: {"passed": True} for name in REPORT.BACKENDS},
                "samples_us": {
                    "li": [20.0, 20.0],
                    "original": [10.0, 10.0],
                    "sglang": [11.0, 11.0],
                },
            }
            for index in (1, 2, 3)
        ],
    }


class ReportTests(unittest.TestCase):
    def test_runner_graph_replay_details_are_measured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case_dir = Path(directory) / "q1_graph_low_hit91"
            (case_dir / "runner").mkdir(parents=True)
            (case_dir / "status").write_text("0\n", encoding="utf-8")
            data = payload("q1_graph_low_hit91")
            (case_dir / "runner" / "case.json").write_text(
                json.dumps(data), encoding="utf-8"
            )
            loaded, status = REPORT.load_case(case_dir)

        self.assertEqual(status, "measured")
        self.assertIsNotNone(loaded)

    def test_graph_replay_accepts_legacy_true_and_rejects_invalid_values(self) -> None:
        invalid_values = (
            False,
            None,
            {},
            {"other": True},
            {"passed": False},
            {"passed": 1},
            {"passed": "true"},
            "yes",
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            case_dir = Path(directory) / "q1_graph_low_hit91"
            (case_dir / "runner").mkdir(parents=True)
            (case_dir / "status").write_text("0\n", encoding="utf-8")
            data = payload("q1_graph_low_hit91")
            result_path = case_dir / "runner" / "case.json"
            for backend in ("original", "sglang"):
                data["correctness"][backend]["graph_replay"] = True
            result_path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(REPORT.load_case(case_dir)[1], "measured")

            for invalid in invalid_values:
                for backend in ("original", "sglang"):
                    data["correctness"][backend]["graph_replay"] = invalid
                result_path.write_text(json.dumps(data), encoding="utf-8")
                self.assertEqual(
                    REPORT.load_case(case_dir)[1],
                    "failed",
                    msg=f"accepted invalid graph_replay value: {invalid!r}",
                )

            for backend in ("original", "sglang"):
                data["correctness"][backend].pop("graph_replay")
            result_path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(REPORT.load_case(case_dir)[1], "failed")

    def test_missing_and_failed_cases_are_distinct_and_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            measured = root / "q1_eager_low_hit91"
            (measured / "runner").mkdir(parents=True)
            (measured / "runner" / "case.json").write_text(
                json.dumps(payload("q1_eager_low_hit91")), encoding="utf-8"
            )
            (measured / "status").write_text("0\n", encoding="utf-8")
            failed = root / "q1_eager_random_hit91"
            failed.mkdir()
            (failed / "status").write_text("17\n", encoding="utf-8")

            result = REPORT.build_report(root, 5.0)
            self.assertEqual(result["statuses"]["q1_eager_low_hit91"], "measured")
            self.assertEqual(result["statuses"]["q1_eager_random_hit91"], "failed")
            self.assertEqual(result["statuses"]["q1_eager_low_hit95"], "missing")
            report = Path(result["report"]).read_text(encoding="utf-8")
            self.assertIn("❌ 失败", report)
            self.assertIn("⏳ 缺失", report)
            self.assertNotIn("mean 0.000 µs", report)

    def test_recomputed_paired_ratios_and_threshold_warning(self) -> None:
        data = payload("q1_eager_low_hit91")
        ratios = REPORT.paired_ratio(data, "sglang", "original")
        self.assertEqual(ratios, [1.1, 1.1, 1.1])
        self.assertEqual(REPORT.paired_ratio(data, "original", "li"), [0.5, 0.5, 0.5])
        self.assertIn("超过5%告警阈值", REPORT._ratio_note(data, 5.0))

    def test_hash_mismatch_refuses_paired_comparison(self) -> None:
        data = payload("q1_eager_low_hit91")
        data["hashes"]["backends"] = {
            "li": {"input": "same"},
            "original": {"input": "different"},
        }
        with self.assertRaisesRegex(ValueError, "backend fixture/hash mismatch"):
            REPORT.paired_ratio(data, "original", "li")
        del data["hashes"]["backends"]
        del data["fixture"]["reference_sha256"]
        data["hashes"].pop("reference_sha256")
        with self.assertRaisesRegex(ValueError, "incomplete fixture hashes"):
            REPORT.paired_ratio(data, "original", "li")

    def test_formal_case_rejects_diagnostic_or_bad_round_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_dir = root / "q1_eager_low_hit91"
            (case_dir / "runner").mkdir(parents=True)
            (case_dir / "status").write_text("0\n", encoding="utf-8")
            data = payload("q1_eager_low_hit91", diagnostic_only=True)
            (case_dir / "runner" / "case.json").write_text(
                json.dumps(data), encoding="utf-8"
            )
            self.assertEqual(REPORT.load_case(case_dir)[1], "failed")
            data = payload("q1_eager_low_hit91")
            data["rounds"][0]["samples_us"]["sglang"] = [0.0, 11.0]
            (case_dir / "runner" / "case.json").write_text(
                json.dumps(data), encoding="utf-8"
            )
            self.assertEqual(REPORT.load_case(case_dir)[1], "failed")


if __name__ == "__main__":
    unittest.main()
