import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.brace.analyze_actor_failures import classify_failure, main, resolve_failures_path
from experiments.brace.collect_branches import bootstrap_lcb, select_branch_points, summarize_branch_rows
from experiments.brace.replay_audit import Candidate


class CollectBranchesTest(unittest.TestCase):
    def test_bootstrap_lcb_is_conservative(self):
        rng = __import__("random").Random(0)
        lcb = bootstrap_lcb([True, False, True], alpha=0.1, samples=200, rng=rng)
        self.assertLessEqual(lcb, 0.67)

    def test_summarize_branch_rows_detects_lift(self):
        rows = [
            {
                "env_seed": 1,
                "physics_step": 10,
                "point_type": "first_persistent_divergence",
                "branch_role": "candidate",
                "success": True,
            },
            {
                "env_seed": 1,
                "physics_step": 10,
                "point_type": "first_persistent_divergence",
                "branch_role": "control",
                "success": False,
            },
        ]
        summary = summarize_branch_rows(rows, alpha=0.1, delta=0.15)
        self.assertEqual(summary["matched_success_lift"], 1.0)
        self.assertTrue(summary["passed"])


class AnalyzeActorFailuresTest(unittest.TestCase):
    def test_classify_garbage_rotation(self):
        row = {
            "actor_metric_passed": {"garbage_1": {"translation": True, "rotation": False}},
            "errors": {"actor_errors": {"garbage_1": {"rotation_error": 0.1}}},
        }
        self.assertEqual(classify_failure(row), "garbage_rotation")

    def test_cli_writes_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            audit_dir = Path(directory) / "audit"
            audit_dir.mkdir()
            failures = audit_dir / "failures.jsonl"
            failures.write_text(
                json.dumps(
                    {
                        "check_type": "control_trace_replay",
                        "passed": False,
                        "worst_actor": "garbage_0",
                        "worst_metric": "rotation",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = Path(directory) / "summary.json"
            with mock.patch(
                "sys.argv",
                ["analyze", "--audit-dir", str(audit_dir), "--output", str(output)],
            ):
                self.assertEqual(main(), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["garbage_rotation_dominates"])

    def test_resolve_failures_json_to_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            audit_dir = Path(directory)
            (audit_dir / "failures.jsonl").write_text(
                json.dumps({"check_type": "control_trace_replay", "passed": False}) + "\n",
                encoding="utf-8",
            )
            path = resolve_failures_path(audit_dir / "failures.json", None)
            self.assertEqual(path.name, "failures.jsonl")


if __name__ == "__main__":
    unittest.main()
