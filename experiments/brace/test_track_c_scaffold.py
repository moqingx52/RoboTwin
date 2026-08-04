#!/usr/bin/env python3
"""Unit tests for artifact inventory, validation, and confirmatory gate stats."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.brace.anchor_smoke import anchor_gate_passed
from experiments.brace.anchor_unit_smoke import run_unit_anchor_smoke
from experiments.brace.evaluate_confirmatory_gate import merge_task_points
from experiments.brace.inventory_artifacts import build_inventory, inspect_artifact, ArtifactSpec
from experiments.brace.inventory_traced_rollouts import inventory_task
from experiments.brace.merge_replay_audit_summaries import merge_summaries
from experiments.brace.orchestrate import prepare_resume_state, summarize_screen
from experiments.brace.run_eval_group import seed_shards_from_partial
from experiments.brace.promote_run import copy_file
from experiments.brace.replay_audit import write_json_atomic
from experiments.brace.resolve_artifact import resolve_audit_summary, resolve_branch_dir
from experiments.brace.screen_gates import compute_forgetting_gate, evaluate_constraint_feasibility, summarize_feasibility_trajectory
from experiments.brace.stage_records import emit_stage_record, list_records, resolve_latest_record
from experiments.brace.validate_artifacts import run_validation


class TrackCScaffoldTest(unittest.TestCase):
    def test_hydra_config_path_is_relative_to_script(self) -> None:
        script_dir = Path(__file__).resolve().parent
        config_dir = (script_dir / "../../policy/DP/diffusion_policy/config").resolve()
        rel = os.path.relpath(config_dir, script_dir)
        self.assertEqual(rel, "../../policy/DP/diffusion_policy/config")
        self.assertTrue((config_dir / "robot_dp_14.yaml").is_file())

    def test_run_all_shell_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", "experiments/brace/run_all.sh"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_constraint_feasibility_fails_on_missing_groups(self) -> None:
        protocol = {
            "anchor_smoke": {"identity_epsilon": 1e-4},
            "constraint_feasibility": {
                "mean_tolerance_factor": 2.0,
                "p90_max_factor": 5.0,
                "violation_fraction_max": {"base_solved": 0.5, "boundary": 0.5},
            },
        }
        summary = evaluate_constraint_feasibility([], protocol)
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["missing_groups"], ["base_solved", "boundary"])

    def test_preservation_group_batch_sampler_small_dataset(self) -> None:
        from experiments.brace.preservation_sampler import PreservationGroupBatchSampler

        groups = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int64)
        sampler = PreservationGroupBatchSampler(
            groups,
            batch_size=4,
            preservation_group_ids={"base_solved": 1, "boundary": 2},
            samples_per_group=2,
            seed=0,
            num_batches=3,
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 3)
        for batch in batches:
            self.assertEqual(len(batch), 4)
            labels = groups[batch]
            self.assertEqual(int(np.sum(labels == 1)), 2)
            self.assertEqual(int(np.sum(labels == 2)), 2)

    def test_forgetting_gate_rejects_joint_catastrophic_forgetting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_solved_seeds = {100, 101, 102, 104, 105}

            def write_eval(path: Path, outcomes: dict[int, bool]) -> None:
                payload = {
                    "rows": [
                        {"env_seed": seed, "split": "train_seen", "success": outcome}
                        for seed, outcome in outcomes.items()
                    ]
                }
                path.write_text(json.dumps(payload), encoding="utf-8")

            base_path = root / "base.json"
            u1_path = root / "u1.json"
            b2_path = root / "b2.json"
            outcomes = {seed: True for seed in base_solved_seeds}
            write_eval(base_path, outcomes)
            write_eval(u1_path, {seed: False for seed in base_solved_seeds})
            write_eval(b2_path, {seed: False for seed in base_solved_seeds})
            protocol = {
                "preservation_metrics": {
                    "min_paired_seeds": 5,
                    "require_b2_forgetting_below_u1": True,
                }
            }
            summary = compute_forgetting_gate(
                base_eval=base_path,
                u1_eval=u1_path,
                b2_eval=b2_path,
                base_solved_seeds=base_solved_seeds,
                protocol=protocol,
            )
            self.assertFalse(summary["passed"])
            self.assertAlmostEqual(summary["forgetting_rate_u1"], 1.0)
            self.assertAlmostEqual(summary["forgetting_rate_b2"], 1.0)

    def test_forgetting_gate_prefers_b2_when_u1_forgets_more(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_solved_seeds = {100, 101, 102, 103, 104}

            def write_eval(path: Path, outcomes: dict[int, bool]) -> None:
                payload = {
                    "rows": [
                        {"env_seed": seed, "split": "train_seen", "success": outcome}
                        for seed, outcome in outcomes.items()
                    ]
                }
                path.write_text(json.dumps(payload), encoding="utf-8")

            base_path = root / "base.json"
            u1_path = root / "u1.json"
            b2_path = root / "b2.json"
            write_eval(base_path, {seed: True for seed in base_solved_seeds})
            write_eval(
                u1_path,
                {100: True, 101: True, 102: False, 103: False, 104: False},
            )
            write_eval(b2_path, {seed: True for seed in base_solved_seeds})
            protocol = {
                "preservation_metrics": {
                    "min_paired_seeds": 5,
                    "require_b2_forgetting_below_u1": True,
                }
            }
            summary = compute_forgetting_gate(
                base_eval=base_path,
                u1_eval=u1_path,
                b2_eval=b2_path,
                base_solved_seeds=base_solved_seeds,
                protocol=protocol,
            )
            self.assertTrue(summary["passed"])
            self.assertGreater(summary["forgetting_rate_u1"], summary["forgetting_rate_b2"])

    def test_summarize_feasibility_trajectory_reports_tail_window(self) -> None:
        rows = []
        for step in range(10):
            rows.append(
                {
                    "step": step,
                    "brace_constraint/base_solved": 0.02 if step < 8 else 0.001,
                    "brace_constraint/boundary": 0.02 if step < 8 else 0.001,
                    "brace_monitor/base_solved_ema_drift": 0.015 if step < 8 else 0.001,
                    "brace_dual/base_solved": 0.01 * step,
                }
            )
        summary = summarize_feasibility_trajectory(rows, epsilon=1e-4, tail_fraction=0.2)
        self.assertEqual(summary["rows"], 10)
        self.assertGreater(summary["base_solved"]["tail_mean"], 0.0)
        self.assertLess(summary["base_solved"]["tail_mean"], summary["base_solved"]["full_mean"])

    def test_summarize_feasibility_trajectory_honors_tail_steps(self) -> None:
        rows = [
            {
                "step": step,
                "brace_constraint/base_solved": 0.01 if step < 95 else 0.001,
                "brace_constraint/boundary": 0.01 if step < 95 else 0.001,
            }
            for step in range(100)
        ]
        summary = summarize_feasibility_trajectory(rows, epsilon=1e-4, tail_steps=50)
        self.assertEqual(summary["tail_mode"], "steps")
        self.assertEqual(summary["tail_rows"], 50)
        self.assertLess(summary["base_solved"]["tail_mean"], summary["base_solved"]["full_mean"])

    def test_multi_draw_probe_aggregation(self) -> None:
        import torch
        from experiments.brace.anchor_probe_eval import evaluate_probe_draws

        class _Student:
            def __init__(self, scale):
                self.scale = scale
                self._train = True

            def training(self):
                return self._train

            def train(self, mode=True):
                self._train = mode

            def eval(self):
                return self

            def denoise_action(self, obs, noisy_action, timesteps):
                return noisy_action * self.scale

        class _Teacher:
            def eval(self):
                return self

            def denoise_action(self, obs, noisy_action, timesteps):
                return noisy_action

        draws = [
            {
                "base_solved": {
                    "obs": {"x": torch.zeros(1, 1)},
                    "noisy_action": torch.tensor([[1.0]]),
                    "timesteps": torch.tensor([0]),
                    "teacher_pred": torch.tensor([[1.0]]),
                }
            },
            {
                "base_solved": {
                    "obs": {"x": torch.zeros(1, 1)},
                    "noisy_action": torch.tensor([[2.0]]),
                    "timesteps": torch.tensor([0]),
                    "teacher_pred": torch.tensor([[2.0]]),
                }
            },
        ]
        raw, monitor, stats = evaluate_probe_draws(_Student(1.1), _Teacher(), draws, cfg={})
        self.assertEqual(stats["probe_draw_count"], 2)
        self.assertIn("base_solved", raw)
        self.assertIn("probe_constraint_p90/base_solved", stats)

    def test_apply_brace_anchor_overrides_skips_lambda_init(self) -> None:
        try:
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("omegaconf not installed")
        from experiments.brace.anchor_diagnostic_loop import DiagnosticJobConfig, apply_brace_anchor_overrides

        cfg = OmegaConf.create(
            {
                "training": {
                    "brace_anchor": {
                        "enabled": True,
                        "dual_lr": 0.01,
                        "formulation": "dual_only",
                    }
                }
            }
        )
        job = DiagnosticJobConfig(anchor_enabled=True)
        apply_brace_anchor_overrides(
            cfg,
            job,
            {"lambda_init": 0.0, "dual_lr": 0.03, "formulation": "dual_only"},
        )
        self.assertEqual(float(cfg.training.brace_anchor.dual_lr), 0.03)
        self.assertNotIn("lambda_init", cfg.training.brace_anchor)

    def test_calibration_jobs_manifest_has_eight_jobs(self) -> None:
        manifest = json.loads(
            Path("experiments/brace/calibration_jobs.place_container_plate.v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest["jobs"]), 8)
        job_ids = {job["job_id"] for job in manifest["jobs"]}
        self.assertEqual(job_ids, {f"A{i}" for i in range(8)})

    def test_select_checkpoint_candidates_empty_when_missing(self) -> None:
        from experiments.brace.anchor_behavior_eval import select_checkpoint_candidates

        with tempfile.TemporaryDirectory() as tmp:
            rows = select_checkpoint_candidates(Path(tmp), task="place_container_plate")
            self.assertEqual(rows, [])

    def test_eval_output_path_uses_task_subdir(self) -> None:
        from experiments.brace.anchor_behavior_eval import eval_output_path

        with tempfile.TemporaryDirectory() as tmp:
            out = eval_output_path(Path(tmp), "place_container_plate", "calib_dual_low_drift")
            self.assertEqual(out, Path(tmp) / "place_container_plate" / "calib_dual_low_drift.json")

    def test_audit_base_solved_coverage_detects_missing_train_seen(self) -> None:
        from experiments.brace.anchor_behavior_eval import audit_base_solved_coverage

        with tempfile.TemporaryDirectory() as tmp:
            seeds_file = Path(tmp) / "seeds.json"
            seeds_file.write_text(
                json.dumps({"train_rollout": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}),
                encoding="utf-8",
            )
            audit = audit_base_solved_coverage(base_solved_seeds=[5, 99], seeds_file=seeds_file)
            self.assertTrue(audit["requires_base_solved_anchor_split"])
            self.assertEqual(audit["train_seen_missing"], [99])

    def test_pick_best_formulation_job_prefers_lower_probe_sum(self) -> None:
        from experiments.brace.anchor_behavior_eval import pick_best_formulation_job

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for job_id, score in (("A5", 0.9), ("A6", 0.4), ("A7", 0.6)):
                (root / job_id).mkdir()
                write_json_atomic(
                    root / job_id / "summary.json",
                    {
                        "complete": True,
                        "probe_summary": {
                            "base_solved": {"probe_tail_mean": score / 2},
                            "boundary": {"probe_tail_mean": score / 2},
                        }
                    },
                )
            self.assertEqual(pick_best_formulation_job(root), "A6")

    def test_finalize_behavior_eval_summary_merges_partials(self) -> None:
        from experiments.brace.anchor_behavior_eval import finalize_behavior_eval_summary

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            partials = output_dir / "partials"
            partials.mkdir()
            base_eval = output_dir / "place_container_plate" / "calib_base_original.json"
            sft_eval = output_dir / "place_container_plate" / "calib_sft_only.json"
            for path in (base_eval, sft_eval):
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomic(
                    path,
                    {
                        "rows": [
                            {"env_seed": 1, "split": "train_seen", "success": True},
                        ]
                    },
                )
            write_json_atomic(
                partials / "base_original.json",
                {"label": "base_original", "eval_output": str(base_eval)},
            )
            write_json_atomic(
                partials / "sft_only.json",
                {"label": "sft_only", "eval_output": str(sft_eval)},
            )
            summary = finalize_behavior_eval_summary(
                task="place_container_plate",
                calibration_run_dir=output_dir,
                output_dir=output_dir,
                protocol={},
                seeds_file=output_dir / "seeds.json",
                base_solved_seeds=[1],
                coverage_audit={"requires_base_solved_anchor_split": False},
                extra_splits_file=None,
            )
            self.assertEqual(summary["stage"], "anchor_behavior_eval")
            self.assertEqual(len(summary["candidates"]), 2)
            self.assertTrue((output_dir / "summary.json").is_file())

    def test_validate_recommended_behavior_candidates_fails_closed(self) -> None:
        from experiments.brace.anchor_behavior_eval import validate_recommended_candidates

        incomplete = [
            {
                "label": "base_original",
                "checkpoint_audit": {"deploys_ema": True},
            }
        ]
        with self.assertRaises(RuntimeError):
            validate_recommended_candidates(incomplete)

    def test_preservation_rate_uses_base_success_denominator(self) -> None:
        from experiments.brace.anchor_behavior_eval import aggregate_preservation_metrics

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.json"
            candidate = root / "candidate.json"
            write_json_atomic(
                base,
                {
                    "rows": [
                        {"env_seed": 1, "split": "train_seen", "success": True},
                        {"env_seed": 2, "split": "train_seen", "success": False},
                    ]
                },
            )
            write_json_atomic(
                candidate,
                {
                    "rows": [
                        {"env_seed": 1, "split": "train_seen", "success": False},
                        {"env_seed": 2, "split": "train_seen", "success": False},
                    ]
                },
            )
            result = aggregate_preservation_metrics(
                base_eval_path=base,
                candidate_eval_path=candidate,
                base_solved_seeds={1, 2},
                protocol={},
            )
            self.assertEqual(result["base_success_count"], 1)
            self.assertEqual(result["forgetting_rate"], 1.0)

    def test_episode_outcomes_requires_all_repeats(self) -> None:
        from experiments.brace.screen_gates import episode_outcomes

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            write_json_atomic(
                path,
                {
                    "rows": [
                        {"env_seed": 1, "split": "untouched_preservation", "repeat": 0, "success": True},
                        {"env_seed": 1, "split": "untouched_preservation", "repeat": 1, "success": False},
                        {"env_seed": 2, "split": "untouched_preservation", "repeat": 0, "success": True},
                        {"env_seed": 2, "split": "untouched_preservation", "repeat": 1, "success": True},
                    ]
                },
            )
            self.assertEqual(episode_outcomes(path, "untouched_preservation"), {1: False, 2: True})

    def test_checkpoint_probe_fields_uses_step_artifact(self) -> None:
        from experiments.brace.anchor_behavior_eval import checkpoint_probe_fields

        job_summary = {
            "checkpoint_artifacts": {
                "1235": {
                    "probe_constraints": {"base_solved": 0.0016, "boundary": 0.000646},
                    "probe_monitor": {"base_solved_ema_drift": 0.0012, "boundary_ema_drift": 0.00045},
                },
                "2470": {
                    "probe_constraints": {"base_solved": 0.00051, "boundary": 0.000386},
                    "probe_monitor": {"base_solved_ema_drift": 0.00055, "boundary_ema_drift": 0.00039},
                },
            },
            "probe_summary": {
                "boundary": {"probe_tail_mean": 0.999},
            },
        }
        mid = checkpoint_probe_fields(job_summary, 1235)
        low = checkpoint_probe_fields(job_summary, 2470)
        self.assertAlmostEqual(mid["probe_summary"]["boundary"]["probe_tail_mean"], 0.000646)
        self.assertAlmostEqual(low["probe_summary"]["boundary"]["probe_tail_mean"], 0.000386)
        self.assertEqual(mid["probe_summary"]["source"], "checkpoint_artifacts")
        with self.assertRaises(KeyError):
            checkpoint_probe_fields(job_summary, 999)

    def test_confirmatory_manifest_expands_all_training_seeds(self) -> None:
        manifest = json.loads(
            Path("experiments/brace/confirmatory_preservation_jobs.place_container_plate.v3.json").read_text(
                encoding="utf-8"
            )
        )
        jobs = manifest["jobs"]
        self.assertEqual(len(jobs), 10)
        self.assertEqual(
            {(row["method"], row["training_seed"]) for row in jobs},
            {(method, seed) for method in ("sft_only", "a1_dual") for seed in (1, 2, 3, 4, 5)},
        )

    def test_confirmatory_jobs_v3_manifest_protocol_provenance(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        manifest_path = repo_root / "experiments/brace/confirmatory_preservation_jobs.place_container_plate.v3.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        protocol_path = repo_root / manifest["protocol_path"]
        self.assertTrue(protocol_path.is_file())
        self.assertEqual(protocol_path.name, "screen_protocol.v1.4.2.confirmatory_preservation.json")
        protocol_sha_sidecar = protocol_path.parent / f"{protocol_path.name}.sha256"
        self.assertTrue(protocol_sha_sidecar.is_file())
        check_protocol = subprocess.run(
            ["sha256sum", "-c", protocol_sha_sidecar.name],
            cwd=protocol_path.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(check_protocol.returncode, 0, check_protocol.stdout + check_protocol.stderr)
        sha_sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
        check_manifest = subprocess.run(
            ["sha256sum", "-c", sha_sidecar.name],
            cwd=manifest_path.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(check_manifest.returncode, 0, check_manifest.stdout + check_manifest.stderr)

    def test_confirmatory_eval_matrix_has_base_plus_ten_candidates(self) -> None:
        from experiments.brace.confirmatory_preservation_eval import build_candidate_labels

        protocol = json.loads(
            Path("experiments/brace/screen_protocol.v1.4.2.confirmatory_preservation.json").read_text(
                encoding="utf-8"
            )
        )
        labels = build_candidate_labels(protocol)
        self.assertEqual(len(labels), 11)
        self.assertEqual(labels[0], "base_original")
        self.assertEqual(
            set(labels[1:]),
            {f"{method}_s{seed}" for method in ("c0", "c1") for seed in range(1, 6)},
        )

    def test_confirmatory_eval_gpu_is_mapped_exactly_once(self) -> None:
        from experiments.brace.orchestrate_confirmatory_eval import isolated_eval_launch

        command, env = isolated_eval_launch({"command": ["python", "eval.py"]}, 5)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "5")
        self.assertEqual(command, ["python", "eval.py"])
        self.assertNotIn("--gpu", command)

    def test_enrollment_integer_two_of_three(self) -> None:
        from experiments.brace.confirmatory_common import enrollment_eligible, eligible_seeds_from_eval

        self.assertTrue(enrollment_eligible(2, min_successes=2))
        self.assertFalse(enrollment_eligible(2, min_successes=3))
        self.assertFalse(0.6666666666666666 >= 0.6667)
        with tempfile.TemporaryDirectory() as tmp:
            eval_path = Path(tmp) / "eval.json"
            write_json_atomic(
                eval_path,
                {
                    "rows": [
                        {"env_seed": 1, "split": "id_heldout", "repeat": 0, "success": True},
                        {"env_seed": 1, "split": "id_heldout", "repeat": 1, "success": False},
                        {"env_seed": 1, "split": "id_heldout", "repeat": 2, "success": True},
                    ]
                },
            )
            self.assertEqual(eligible_seeds_from_eval(eval_path, split="id_heldout", min_successes=2, repeats_required=3), [1])

    def test_exact_sign_test_five_seeds(self) -> None:
        from experiments.brace.aggregate_confirmatory_preservation import exact_one_sided_sign_test

        passed = exact_one_sided_sign_test(5, 5)
        failed = exact_one_sided_sign_test(4, 5)
        self.assertTrue(passed["passed"])
        self.assertAlmostEqual(passed["p_value"], 1 / 32)
        self.assertFalse(failed["passed"])

    def test_forgetting_denominator_is_frozen_cohort(self) -> None:
        from experiments.brace.aggregate_confirmatory_preservation import forgetting_rate

        rows = [
            {"env_seed": seed, "split": "untouched_preservation", "repeat": repeat, "success": seed == 10}
            for seed in (10, 11)
            for repeat in range(3)
        ]
        result = forgetting_rate(
            candidate_rows=rows,
            cohort_seeds=[10, 11],
            split="untouched_preservation",
        )
        self.assertEqual(result["paired_seeds"], 2)
        self.assertEqual(result["forgetting_seeds"], [11])
        self.assertEqual(result["forgetting_rate"], 0.5)
        self.assertEqual(result["denominator_source"], "frozen_census_cohort")

    def test_repeat_completeness_rejects_wrong_policy_seed(self) -> None:
        from experiments.brace.confirmatory_common import repeat_completeness

        rows = [
            {
                "env_seed": 10,
                "split": "id_heldout",
                "repeat": repeat,
                "policy_seed": 3000 + repeat,
                "success": True,
            }
            for repeat in range(3)
        ]
        self.assertTrue(
            repeat_completeness(rows, "id_heldout", [10], 3, policy_seed_offset=3000)
        )
        rows[-1]["policy_seed"] = 4002
        self.assertFalse(
            repeat_completeness(rows, "id_heldout", [10], 3, policy_seed_offset=3000)
        )

    def test_census_eval_command_has_no_hard_and_offset_3000(self) -> None:
        from experiments.brace.anchor_behavior_eval import build_eval_command

        command = build_eval_command(
            task="place_container_plate",
            variant="census_base",
            checkpoint_path="base.ckpt",
            output_dir=Path("out"),
            seeds_file=Path("seeds.json"),
            hard_seeds_file=Path("hard.json"),
            extra_splits_file=None,
            workers_per_gpu=3,
            protocol={
                "census_eval": {
                    "id_seed_count": 100,
                    "train_seed_count": 100,
                    "hard_seed_count": 0,
                    "id_repeats": 3,
                    "train_repeats": 3,
                    "hard_repeats": 0,
                    "policy_seed_offset": 3000,
                }
            },
            eval_profile="census",
        )
        self.assertIn("--no-include-hard", command)
        self.assertEqual(command[command.index("--policy-seed-offset") + 1], "3000")

    def test_census_eval_command_v142_uses_census_candidate_split(self) -> None:
        from experiments.brace.anchor_behavior_eval import build_eval_command

        command = build_eval_command(
            task="place_container_plate",
            variant="census_base",
            checkpoint_path="base.ckpt",
            output_dir=Path("out"),
            seeds_file=Path("seeds.json"),
            hard_seeds_file=Path("hard.json"),
            extra_splits_file=None,
            workers_per_gpu=3,
            protocol={
                "census_eval": {
                    "census_candidate_id_count": 200,
                    "train_seed_count": 100,
                    "hard_seed_count": 0,
                    "id_repeats": 3,
                    "train_repeats": 3,
                    "hard_repeats": 0,
                    "policy_seed_offset": 3000,
                }
            },
            eval_profile="census",
        )
        self.assertIn("--census-candidate-split", command)
        self.assertIn("--census-candidate-id-count", command)
        self.assertEqual(command[command.index("--census-candidate-id-count") + 1], "200")
        self.assertNotIn("--id-seed-count", command)

    def test_eval_per_seed_census_work_items_use_census_candidate_id_split(self) -> None:
        from experiments.phase1.eval_per_seed import build_work_items

        seed_payload = {
            "eval_id": [1, 2],
            "census_candidate_id": [1, 2, 3, 4],
            "train_rollout": [10, 11],
        }
        items = build_work_items(
            seed_payload,
            hard_seeds=[99],
            id_repeats=2,
            train_repeats=2,
            hard_repeats=1,
            census_candidate_split=True,
            census_candidate_id_count=4,
        )
        splits = {split for split, _, _ in items}
        self.assertEqual(splits, {"census_candidate_id", "train_seen"})
        self.assertEqual(
            sum(1 for split, _, _ in items if split == "census_candidate_id"),
            8,
        )

    def test_preservation_cohort_v142_splits_census_and_id_heldout(self) -> None:
        from experiments.brace.select_preservation_cohort import select_preservation_cohort

        eval_id = list(range(100, 110))
        reserve = list(range(200, 210))
        census_ids = eval_id + reserve
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seeds_file = root / "seeds.json"
            base_eval = root / "base.json"
            write_json_atomic(
                seeds_file,
                {
                    "train_rollout": list(range(300, 310)),
                    "eval_id": eval_id,
                    "census_candidate_id": census_ids,
                },
            )
            write_json_atomic(
                base_eval,
                {
                    "rows": [
                        {"env_seed": seed, "split": "census_candidate_id", "repeat": repeat, "success": True}
                        for seed in census_ids
                        for repeat in range(3)
                    ]
                },
            )
            payload = select_preservation_cohort(
                task="place_container_plate",
                base_eval=base_eval,
                census_summary={
                    "schema_version": 2,
                    "eval_path": str(base_eval),
                    "policy_seed_offset": 3000,
                    "census_candidate_ids": census_ids,
                    "id_heldout": eval_id,
                    "train_seeds": list(range(300, 310)),
                },
                seeds_file=seeds_file,
                exclusions={},
                enrollment_rule={"min_successes": 2, "repeats_required": 3},
                min_untouched=8,
                boundary_count=3,
            )
            self.assertEqual(len(payload["cohorts"]["untouched_preservation"]), 8)
            self.assertEqual(payload["cohorts"]["id_heldout"], eval_id)
            self.assertEqual(len(payload["cohorts"]["id_heldout"]), 10)

    def test_validate_census_summary_rejects_v141_schema_on_v142_protocol(self) -> None:
        from experiments.brace.confirmatory_common import file_sha256, validate_census_summary

        protocol_path = Path("experiments/brace/screen_protocol.v1.4.2.confirmatory_preservation.json")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_path = root / "eval.json"
            ckpt_path = root / "base.ckpt"
            seeds_path = root / "seeds.json"
            eval_path.write_text(json.dumps({"progress": {"complete": True, "policy_seed_offset": 3000}, "rows": []}), encoding="utf-8")
            ckpt_path.write_text("ckpt", encoding="utf-8")
            seeds_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version < 2"):
                validate_census_summary(
                    {
                        "schema_version": 1,
                        "stage": "confirmatory_base_census",
                        "run_type": "confirmatory_base_census",
                        "task": "place_container_plate",
                        "policy_seed_offset": 3000,
                        "protocol_revision": protocol["protocol_revision"],
                        "protocol_sha256": file_sha256(protocol_path),
                        "enrollment_rule": {
                            "min_successes": 2,
                            "repeats_required": 3,
                            "display_fraction": protocol["base_solved_enrollment"]["display_fraction"],
                        },
                        "eval_path": str(eval_path),
                        "eval_sha256": file_sha256(eval_path),
                        "base_checkpoint": str(ckpt_path),
                        "base_checkpoint_sha256": file_sha256(ckpt_path),
                        "seeds_file": str(seeds_path),
                        "seeds_file_sha256": file_sha256(seeds_path),
                        "id_seeds": [1],
                        "train_seeds": [2],
                        "repeat_completeness": {"id_heldout": True, "train_seen": True},
                    },
                    protocol_path=protocol_path,
                    protocol=protocol,
                )

    def test_h1_conjunction_synthetic(self) -> None:
        from experiments.brace.aggregate_confirmatory_preservation import aggregate_confirmatory_preservation

        protocol = json.loads(
            Path("experiments/brace/screen_protocol.v1.4.1.confirmatory_preservation.json").read_text(encoding="utf-8")
        )
        protocol["training_seeds"] = [1, 2]
        protocol["preservation_cohorts"]["min_untouched_base_solved"] = 2
        protocol["training_seed_inference"]["effect_ci"]["replicates"] = 200
        cohort = {
            "cohorts": {
                "untouched_preservation": [10, 11],
                "id_heldout": [10, 11, 12],
            }
        }

        def write_eval(path: Path, untouched_success: dict[int, bool], id_rates: dict[int, float]) -> None:
            rows = []
            for seed in (10, 11):
                for repeat in range(3):
                    rows.append(
                        {
                            "env_seed": seed,
                            "split": "untouched_preservation",
                            "repeat": repeat,
                            "success": untouched_success.get(seed, True),
                        }
                    )
            for seed, rate in id_rates.items():
                for repeat in range(3):
                    rows.append(
                        {
                            "env_seed": seed,
                            "split": "id_heldout",
                            "repeat": repeat,
                            "success": repeat < round(rate * 3),
                        }
                    )
            write_json_atomic(path, {"rows": rows})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = {"base_original": root / "base.json"}
            write_eval(artifacts["base_original"], {10: True, 11: True}, {10: 0.8, 11: 0.8, 12: 0.8})
            for label, untouched, id_rate in (
                ("c0_s1", {10: True, 11: False}, 0.5),
                ("c1_s1", {10: True, 11: True}, 0.6),
                ("c0_s2", {10: True, 11: True}, 0.5),
                ("c1_s2", {10: True, 11: True}, 0.6),
            ):
                path = root / f"{label}.json"
                artifacts[label] = path
                write_eval(path, untouched, {10: id_rate, 11: id_rate, 12: id_rate})
            summary = aggregate_confirmatory_preservation(
                protocol=protocol,
                cohort=cohort,
                eval_artifacts=artifacts,
                provenance={"complete": True},
            )
            self.assertIn(summary["h1_status"], {"passed", "failed", "inconclusive"})
            self.assertIn("preservation_absolute_gate", summary["gates"])

    def test_preservation_cohort_uses_eval_id_and_caps_primary_n(self) -> None:
        from experiments.brace.select_preservation_cohort import select_preservation_cohort

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seeds_file = root / "seeds.json"
            base_eval = root / "base.json"
            write_json_atomic(
                seeds_file,
                {"train_rollout": list(range(200, 240)), "eval_id": list(range(100, 180))},
            )
            write_json_atomic(
                base_eval,
                {
                    "rows": [
                        {"env_seed": seed, "split": "id_heldout", "repeat": repeat, "success": True}
                        for seed in range(100, 180)
                        for repeat in range(3)
                    ]
                },
            )
            payload = select_preservation_cohort(
                task="place_container_plate",
                base_eval=base_eval,
                census_summary={"eval_path": str(base_eval), "policy_seed_offset": 3000},
                seeds_file=seeds_file,
                exclusions={"sft_chunk": [100]},
                enrollment_rule={"min_successes": 2, "repeats_required": 3},
                min_untouched=60,
                boundary_count=20,
            )
            self.assertTrue(payload["meets_min_untouched"])
            self.assertEqual(len(payload["cohorts"]["untouched_preservation"]), 60)
            self.assertEqual(payload["cohorts"]["id_heldout"], list(range(100, 180)))
            self.assertNotIn(100, payload["cohorts"]["untouched_preservation"])

    def test_behavior_eval_command_uses_frozen_eval_budget(self) -> None:
        from experiments.brace.anchor_behavior_eval import build_eval_command

        command = build_eval_command(
            task="place_container_plate",
            variant="candidate",
            checkpoint_path="candidate.ckpt",
            output_dir=Path("out"),
            seeds_file=Path("seeds.json"),
            hard_seeds_file=Path("hard.json"),
            extra_splits_file=None,
            workers_per_gpu=3,
            protocol={
                "eval": {
                    "id_seed_count": 100,
                    "train_seed_count": 100,
                    "hard_seed_count": 20,
                    "id_repeats": 3,
                    "train_repeats": 3,
                    "hard_repeats": 8,
                    "policy_seed_offset": 4000,
                }
            },
        )
        for flag, expected in (
            ("--id-seed-count", "100"),
            ("--train-seed-count", "100"),
            ("--id-repeats", "3"),
            ("--hard-repeats", "8"),
            ("--extra-split-repeats", "3"),
            ("--policy-seed-offset", "4000"),
        ):
            self.assertEqual(command[command.index(flag) + 1], expected)

    def test_repair_behavior_eval_probe_links_updates_a1_steps(self) -> None:
        from experiments.brace.anchor_behavior_eval import repair_behavior_eval_probe_links

        calib = Path("experiments/brace/runs/20260803T031036Z_anchor_calibration_place_container_plate")
        summary_path = Path(
            "experiments/brace/runs/20260803T073015Z_anchor_behavior_eval_place_container_plate_dump_bin_bigbin/summary.json"
        )
        if not (calib / "A1/summary.json").is_file() or not summary_path.is_file():
            self.skipTest("restored forensic artifacts missing")
        with tempfile.TemporaryDirectory() as tmp:
            copied_summary = Path(tmp) / "summary.json"
            copied_summary.write_bytes(summary_path.read_bytes())
            summary = repair_behavior_eval_probe_links(
                summary_path=copied_summary, calibration_run_dir=calib, seal=False
            )
            dual_mid = next(row for row in summary["candidates"] if row["label"] == "dual_mid")
            dual_low = next(row for row in summary["candidates"] if row["label"] == "dual_low_drift")
            self.assertAlmostEqual(
                dual_mid["probe_summary"]["boundary"]["probe_tail_mean"], 0.0006463114768848754
            )
            self.assertAlmostEqual(
                dual_low["probe_summary"]["boundary"]["probe_tail_mean"], 0.0003861767187724278
            )

    def test_select_preservation_cohort_excludes_anchor_splits(self) -> None:
        from experiments.brace.select_preservation_cohort import collect_exclusions

        calib = Path("experiments/brace/runs/20260803T031036Z_anchor_calibration_place_container_plate")
        behavior = Path(
            "experiments/brace/runs/20260803T073015Z_anchor_behavior_eval_place_container_plate_dump_bin_bigbin"
        )
        if not (calib / "A1/anchor_probe_split.json").is_file():
            self.skipTest("restored calibration artifacts missing")
        exclusions = collect_exclusions(
            task="place_container_plate",
            calibration_run_dir=calib,
            behavior_eval_dir=behavior,
            pilot_seeds_file=Path("experiments/brace/seeds/place_container_plate_pilot_seeds.json"),
            confirm_seeds_file=Path("experiments/brace/seeds/place_container_plate_confirm_seeds.json"),
            dataset_manifest=None,
        )
        self.assertIn(100005, exclusions["anchor_train"])
        self.assertIn(100024, exclusions["phase3c_behavior"])
        self.assertIn(100014, exclusions["anchor_probe"])

    def test_anchor_smoke_passes(self) -> None:
        protocol = json.loads(
            Path("experiments/brace/screen_protocol.v1.json").read_text(encoding="utf-8")
        )
        summary = run_unit_anchor_smoke(protocol)
        self.assertTrue(summary["passed"])
        self.assertFalse(summary["eligible_for_screen_gate"])
        self.assertTrue(summary["checks"]["dual_nonnegativity"])

    def test_inventory_mixed_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp)
            manifest = task_dir / "manifest_shard_00_of_01.jsonl"
            rows = [
                {"env_seed": 1, "rollout_id": 0, "success": True},
                {"env_seed": 1, "rollout_id": 1, "success": False},
                {"env_seed": 2, "rollout_id": 0, "success": True},
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            payload = inventory_task(task_dir)
            self.assertEqual(payload["mixed_outcome_count"], 1)
            self.assertEqual(payload["mixed_outcome_seeds"], [1])

    def test_confirmatory_gate_merge_seed_level(self) -> None:
        point = {
            "accepted": True,
            "candidate_success_rate": 1.0,
            "control_success_rate": 0.0,
            "env_seed": 100,
            "point_type": "first_persistent_divergence",
        }
        pilot = {
            "harness_valid": True,
            "tasks": {
                "place_container_plate": {
                    "points": [dict(point, env_seed=100 + i) for i in range(15)]
                }
            },
        }
        confirm = {
            "harness_valid": True,
            "tasks": {
                "place_container_plate": {
                    "points": [dict(point, env_seed=200 + i) for i in range(15)]
                }
            },
        }
        merged = merge_task_points(pilot, confirm, "place_container_plate")
        self.assertTrue(merged["checks"]["min_seeds"])
        self.assertTrue(merged["checks"]["min_points"])
        self.assertTrue(merged["checks"]["lift_positive"])
        self.assertAlmostEqual(merged["seed_level_mean_lift"], 1.0)
        self.assertIn("seed_level_bootstrap_ci", merged)
        self.assertIn("candidate_positive_rate_by_seed", merged)

    def test_inventory_detects_present_summary(self) -> None:
        spec = ArtifactSpec(
            "place_pilot_archive",
            "experiments/brace/archive/branches_place_pilot_valid_v2.3/summary.json",
        )
        entry = inspect_artifact(spec)
        self.assertTrue(entry["exists"])
        self.assertEqual(entry["status"], "present")

    def test_resolve_audit_summary_archive_fallback(self) -> None:
        path = resolve_audit_summary("dump_bin_bigbin")
        self.assertIsNotNone(path)
        self.assertTrue(path.is_file())

    def test_resolve_audit_summary_skips_failed_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            brace_dir = Path(tmp)
            runs = brace_dir / "runs"
            archive = brace_dir / "archive" / "replay_audit_v2_place_v2.3_gate"
            failed_run = brace_dir / "failed_run"
            runs.mkdir(parents=True)
            failed_run.mkdir(parents=True)
            archive.mkdir(parents=True)
            failed_summary = {
                "passed": False,
                "tasks": {"place_container_plate": {"replay_gate_passed": False}},
            }
            passed_summary = {
                "passed": True,
                "tasks": {"place_container_plate": {"replay_gate_passed": True}},
            }
            write_json_atomic(failed_run / "summary.json", failed_summary)
            write_json_atomic(archive / "summary.json", passed_summary)
            (runs / "LATEST_AUDIT_place_container_plate").write_text(
                str(failed_run), encoding="utf-8"
            )
            path = resolve_audit_summary("place_container_plate", brace_dir=brace_dir)
            self.assertIsNotNone(path)
            self.assertEqual(path, archive / "summary.json")

    def test_resolve_audit_summary_explicit_override_must_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            brace_dir = Path(tmp)
            failed_dir = brace_dir / "explicit_failed"
            failed_dir.mkdir(parents=True)
            write_json_atomic(
                failed_dir / "summary.json",
                {
                    "passed": False,
                    "tasks": {"place_container_plate": {"replay_gate_passed": False}},
                },
            )
            os.environ["BRACE_AUDIT_RUN_DIR"] = str(failed_dir)
            try:
                with self.assertRaises(ValueError):
                    resolve_audit_summary("place_container_plate", brace_dir=brace_dir)
            finally:
                os.environ.pop("BRACE_AUDIT_RUN_DIR", None)

    def test_resolve_audit_summary_passed_latest_preferred(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            brace_dir = Path(tmp)
            runs = brace_dir / "runs"
            archive = brace_dir / "archive" / "replay_audit_v2_place_v2.3_gate"
            passed_run = brace_dir / "passed_run"
            runs.mkdir(parents=True)
            passed_run.mkdir(parents=True)
            archive.mkdir(parents=True)
            passed_summary = {
                "passed": True,
                "tasks": {"place_container_plate": {"replay_gate_passed": True}},
            }
            write_json_atomic(passed_run / "summary.json", passed_summary)
            write_json_atomic(archive / "summary.json", passed_summary)
            (runs / "LATEST_AUDIT_place_container_plate").write_text(
                str(passed_run), encoding="utf-8"
            )
            path = resolve_audit_summary("place_container_plate", brace_dir=brace_dir)
            self.assertEqual(path, passed_run / "summary.json")

    def test_anchor_gate_requires_training_path(self) -> None:
        protocol_revision = "screen.v1.1"
        self.assertFalse(
            anchor_gate_passed(
                {"passed": True, "gate_level": "unit", "protocol_revision": protocol_revision},
                protocol_revision,
            )
        )
        self.assertTrue(
            anchor_gate_passed(
                {
                    "passed": True,
                    "gate_level": "training_path",
                    "protocol_revision": protocol_revision,
                    "checks": {
                        "unit_smoke_passed": True,
                        "training_path_smoke_passed": True,
                    },
                },
                protocol_revision,
            )
        )
        path = resolve_branch_dir("branches")
        self.assertIsNotNone(path)
        self.assertTrue((path / "summary.json").is_file())

    def test_stage_record_emit_and_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records_dir = Path(tmp) / "records"
            summary_path = Path(tmp) / "summary.json"
            write_json_atomic(
                summary_path,
                {"passed": True, "complete": True, "tasks": {"place_container_plate": {"passed": True}}},
            )
            import experiments.brace.stage_records as stage_records

            original = stage_records.RECORDS_DIR
            stage_records.RECORDS_DIR = records_dir
            try:
                record_path = emit_stage_record(
                    "branch",
                    summary_path=summary_path,
                    tasks=["place_container_plate"],
                    label="branches",
                )
                self.assertTrue(record_path.is_file())
                record = json.loads(record_path.read_text(encoding="utf-8"))
                bundle = records_dir / "bundles" / record["record_id"] / "summary.json"
                self.assertTrue(bundle.is_file())
                self.assertFalse(record["evidence_bundle"]["binary_artifacts_included"])
                latest = resolve_latest_record("branch", task="place_container_plate")
                self.assertIsNotNone(latest)
                rows = list_records(limit=5)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["stage"], "branch")
            finally:
                stage_records.RECORDS_DIR = original

    def test_frozen_promotion_refuses_different_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            destination = root / "archive" / "summary.json"
            source.write_text('{"value": 1}\n', encoding="utf-8")
            copy_file(source, destination)
            copy_file(source, destination)  # idempotent
            source.write_text('{"value": 2}\n', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                copy_file(source, destination)

    def test_build_inventory_has_bundle_summary(self) -> None:
        inventory = build_inventory()
        self.assertIn("bundle_summary", inventory)
        self.assertIn("artifacts", inventory)

    def test_merge_replay_audit_summaries(self) -> None:
        dump_summary = json.loads(
            Path("experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/summary.json").read_text(
                encoding="utf-8"
            )
        )
        merged = merge_summaries([dump_summary])
        self.assertTrue(merged["tasks"]["dump_bin_bigbin"]["replay_gate_passed"])

    def test_development_screen_summary_applies_v12_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = {}
            feasibility = {
                "passed": True,
                "missing_groups": [],
                "groups": {
                    "base_solved": {"passed": True},
                    "boundary": {"passed": True},
                },
            }
            base_solved_seeds = [100, 101, 102, 103, 104]

            def write_eval(job_id, method, epoch, values, seed_outcomes=None):
                path = root / f"{method}_{epoch}.json"
                payload = {
                    "splits": {
                        split: {"mean_sr": value}
                        for split, value in zip(("id_heldout", "train_seen", "hard_20"), values)
                    }
                }
                if seed_outcomes is not None:
                    payload["rows"] = [
                        {"env_seed": seed, "split": "train_seen", "success": outcome}
                        for seed, outcome in seed_outcomes.items()
                    ]
                path.write_text(json.dumps(payload), encoding="utf-8")
                jobs[job_id] = {"artifact": str(path)}

            def write_train(job_id, method, epoch):
                ckpt = root / f"{method}_{epoch}.ckpt"
                ckpt.write_bytes(b"ckpt")
                feas_path = ckpt.with_suffix(".feasibility.json")
                feas_path.write_text(json.dumps(feasibility), encoding="utf-8")
                jobs[job_id] = {"artifact": str(ckpt)}

            base_outcomes = {seed: True for seed in base_solved_seeds}
            write_eval("eval:base", "base", 0, (0.5, 0.5, 0.5), base_outcomes)
            u1_outcomes = {seed: True for seed in base_solved_seeds}
            u1_outcomes[104] = False
            scores = {
                "N1": (0.45, 0.40, 0.20),
                "B1": (0.46, 0.41, 0.30),
                "B2": (0.47, 0.43, 0.35),
                "B3": (0.49, 0.46, 0.40),
            }
            for method, values in scores.items():
                outcomes = u1_outcomes if method == "N1" else base_outcomes
                write_eval(f"eval:{method}:epoch1", method, 1, values, outcomes)
                if method in {"B2", "B3"}:
                    write_train(f"train:{method}:epoch1", method, 1)
            anchor_manifest_dir = root / "datasets" / "place_container_plate_anchor_replay.zarr"
            anchor_manifest_dir.mkdir(parents=True)
            anchor_manifest = {
                "episodes": [
                    {"env_seed": seed, "preservation_group": "base_solved"} for seed in base_solved_seeds
                ]
            }
            (anchor_manifest_dir / "brace_anchor_manifest.json").write_text(
                json.dumps(anchor_manifest), encoding="utf-8"
            )
            state = {
                "task": "place_container_plate",
                "run_label": "test",
                "run_dir": str(root),
                "screen_epochs": [1],
                "active_methods": ["N1", "B1", "B2", "B3"],
                "jobs": jobs,
            }
            protocol = {
                "screen_mode": "full",
                "screen_epochs": [1],
                "promotion": {"id_fraction_of_base": 0.8, "train_fraction_of_base": 0.75},
                "eval": {"hard_for_checkpoint_selection": False},
                "screens": {"credit": {"min_score_margin": 0.01}},
                "preservation_metrics": {
                    "base_solved_forgetting_max_delta": 0.05,
                    "min_paired_seeds": 5,
                    "require_b2_forgetting_below_u1": True,
                },
            }
            summary = summarize_screen(state, protocol)
            self.assertTrue(summary["passed"])
            self.assertTrue(summary["screens"]["preservation_U1_vs_B2"]["passed"])
            self.assertTrue(summary["screens"]["credit_B1_vs_N1"]["passed"])
            self.assertEqual(summary["screens"]["integration_B1_B2_B3"]["best_method"], "B3")
            self.assertTrue(summary["best_eligible"]["B2"]["constraint_feasibility"]["passed"])
            self.assertTrue(summary["best_eligible"]["B2"]["forgetting_gate"]["passed"])
            self.assertAlmostEqual(summary["methods"]["B1"]["1"]["selection_score"], 0.82, places=5)

    def test_validate_restored_branch_artifacts(self) -> None:
        result = run_validation(
            inventory_path=None,
            missing_only=False,
            strict_json=False,
            run_label="place_pilot_v2.3",
        )
        self.assertTrue(result["passed"], result)

    def test_orchestrate_launch_pins_cuda_visible_devices(self) -> None:
        from unittest.mock import MagicMock, patch

        from experiments.brace.orchestrate import Runner

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "job.log"
            state = {
                "jobs": {
                    "eval:test": {
                        "id": "eval:test",
                        "status": "pending",
                        "command": ["python", "-c", "print(0)"],
                        "log": str(log_path),
                        "attempts": 0,
                    }
                },
                "events": [],
            }
            args = MagicMock()
            args.max_retries = 1
            runner = Runner(args, Path(tmp) / "state.json", state, {})
            with patch("experiments.brace.orchestrate.subprocess.Popen") as popen:
                popen.return_value = MagicMock(pid=123, poll=MagicMock(return_value=None))
                runner.launch(3, state["jobs"]["eval:test"])
            _, kwargs = popen.call_args
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "3")

    def test_prepare_resume_state_resets_failed_jobs(self) -> None:
        state = {
            "status": "failed",
            "jobs": {
                "eval:base": {"status": "completed", "attempts": 1},
                "eval:N1:epoch1": {"status": "failed", "attempts": 2, "exit_code": 1, "gpu": 0},
            },
        }
        prepare_resume_state(state)
        self.assertEqual(state["status"], "running")
        self.assertEqual(state["jobs"]["eval:base"]["status"], "completed")
        self.assertEqual(state["jobs"]["eval:N1:epoch1"]["status"], "pending")
        self.assertEqual(state["jobs"]["eval:N1:epoch1"]["attempts"], 0)
        self.assertNotIn("exit_code", state["jobs"]["eval:N1:epoch1"])

    def test_partial_eval_rows_are_migrated_to_resume_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seeds = root / "seeds.json"
            hard = root / "hard.json"
            output = root / "eval"
            task_dir = output / "place_container_plate"
            task_dir.mkdir(parents=True)
            seeds.write_text(
                json.dumps({"eval_id": [10, 11], "train_rollout": [20, 21]}),
                encoding="utf-8",
            )
            hard.write_text(json.dumps({"hard_seeds": [30, 31]}), encoding="utf-8")
            rows = [
                {"split": "id_heldout", "env_seed": 10, "repeat": 0, "policy_seed": 2000, "success": True},
                {"split": "train_seen", "env_seed": 20, "repeat": 0, "policy_seed": 2000, "success": False},
            ]
            parent = {
                "task_name": "place_container_plate",
                "task_config": "demo_clean",
                "variant": "B1_epoch1",
                "ckpt_path": "checkpoint.ckpt",
                "hard_seeds": [30, 31],
                "hard_seed_source": str(hard),
                "rows": rows,
                "splits": {},
                "progress": {"complete": False, "completed_episodes": 2},
            }
            (task_dir / "B1_epoch1.json").write_text(json.dumps(parent), encoding="utf-8")
            command = [
                "python", "eval_per_seed.py",
                "--task", "place_container_plate",
                "--task-config", "demo_clean",
                "--variant", "B1_epoch1",
                "--ckpt-path", "checkpoint.ckpt",
                "--output-dir", str(output),
                "--seeds-file", str(seeds),
                "--hard-seeds-file", str(hard),
                "--id-seed-count", "2",
                "--train-seed-count", "2",
                "--hard-seed-count", "2",
                "--id-repeats", "1",
                "--train-repeats", "1",
                "--hard-repeats", "1",
                "--policy-seed-offset", "2000",
                "--resume",
            ]
            self.assertEqual(seed_shards_from_partial(command, 3), 2)
            migrated_rows = []
            for shard in range(3):
                path = task_dir / f"B1_epoch1_shard_{shard:02d}_of_03.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["progress"]["num_shards"], 3)
                self.assertEqual(payload["progress"]["shard_id"], shard)
                migrated_rows.extend(payload["rows"])
            self.assertCountEqual(migrated_rows, rows)

    def test_anchor_probe_split_disjoint_by_env_seed(self) -> None:
        from experiments.brace.anchor_probe_split import build_anchor_probe_split

        manifest = {
            "manifest_sha256": "abc123",
            "episodes": [
                {"preservation_group": "base_solved", "env_seed": 10},
                {"preservation_group": "base_solved", "env_seed": 11},
                {"preservation_group": "base_solved", "env_seed": 12},
                {"preservation_group": "boundary", "env_seed": 20},
                {"preservation_group": "boundary", "env_seed": 21},
                {"preservation_group": "boundary", "env_seed": 22},
            ],
        }
        split = build_anchor_probe_split(manifest, split_seed=7, holdout_fraction=0.5)
        self.assertTrue(split.train_env_seeds.isdisjoint(split.probe_env_seeds))
        self.assertGreater(len(split.train_env_seeds), 0)
        self.assertGreater(len(split.probe_env_seeds), 0)
        self.assertEqual(split.split_sha256, split.to_dict()["split_sha256"])

    def test_anchor_probe_split_filters_sampler_pools(self) -> None:
        from experiments.brace.anchor_probe_split import sequence_indices_for_env_seeds
        from experiments.brace.preservation_sampler import PreservationGroupBatchSampler

        groups = np.asarray([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        env_seeds = np.asarray([10, 11, 12, 13, 20, 21, 22, 23], dtype=np.int64)

        class _Dataset:
            sample_preservation_groups = groups
            sample_env_seeds = env_seeds

        train_indices = sequence_indices_for_env_seeds(_Dataset(), {10, 11, 20, 21})
        sampler = PreservationGroupBatchSampler(
            groups,
            batch_size=4,
            preservation_group_ids={"base_solved": 1, "boundary": 2},
            samples_per_group=2,
            seed=0,
            num_batches=5,
            allowed_indices=train_indices,
        )
        for batch in sampler:
            batch_seeds = set(env_seeds[batch].tolist())
            self.assertTrue(batch_seeds.issubset({10, 11, 20, 21}))

    def test_materialize_probe_draws_does_not_reset_global_rng(self) -> None:
        import torch
        from experiments.brace.anchor_probe_eval import materialize_probe_draws

        class _Teacher:
            noise_scheduler = type("NS", (), {"config": type("C", (), {"num_train_timesteps": 10})})()

            def eval(self):
                return self

            def predict_action(self, obs):
                batch = next(iter(obs.values())).shape[0]
                return {"action_pred": torch.zeros(batch, 2, device=obs["x"].device)}

            def make_noisy_action(self, clean_action, noise, timesteps):
                return clean_action + noise

            def denoise_action(self, obs, noisy_action, timesteps):
                return noisy_action

        class _Cfg:
            pass

        batch = {
            "obs": {"x": torch.zeros(4, 1)},
            "sample_preservation_group": torch.tensor([1, 1, 2, 2], dtype=torch.long),
        }
        torch.manual_seed(123)
        _ = torch.rand(10)
        state_before = torch.get_rng_state()
        materialize_probe_draws(_Teacher(), batch, _Cfg(), seed=99, device=torch.device("cpu"))
        state_after = torch.get_rng_state()
        self.assertTrue(torch.equal(state_before, state_after))


if __name__ == "__main__":
    unittest.main()
