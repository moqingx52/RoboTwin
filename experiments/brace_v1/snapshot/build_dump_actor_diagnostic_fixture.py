#!/usr/bin/env python3
"""Create synthetic dump actor-diagnostic failures for offline analysis tests."""

from __future__ import annotations

import json
from pathlib import Path

FAILURES = [
    ("dump_bin_bigbin", 100038, 1, 1, "garbage_0", 0.0509),
    ("dump_bin_bigbin", 100038, 1, 2, "garbage_1", 0.1131),
    ("dump_bin_bigbin", 100038, 3, 1, "garbage_2", 0.0616),
    ("dump_bin_bigbin", 100038, 3, 2, "garbage_3", 0.1080),
    ("dump_bin_bigbin", 100041, 5, 1, "garbage_0", 0.0581),
    ("dump_bin_bigbin", 100041, 5, 2, "garbage_4", 0.0849),
    ("dump_bin_bigbin", 100042, 2, 1, "garbage_1", 0.0518),
    ("dump_bin_bigbin", 100042, 2, 2, "garbage_2", 0.0732),
    ("dump_bin_bigbin", 100042, 6, 2, "garbage_0", 0.1186),
    ("dump_bin_bigbin", 100136, 3, 1, "garbage_3", 0.0738),
    ("dump_bin_bigbin", 100136, 3, 2, "garbage_4", 0.0793),
    ("dump_bin_bigbin", 100235, 5, 1, "garbage_1", 0.1120),
    ("dump_bin_bigbin", 100235, 5, 2, "garbage_2", 0.1103),
    ("place_container_plate", 100023, 2, 2, "container", 0.0503),
]


def main() -> None:
    output_dir = Path("experiments/brace/replay_audit_v2_dump_actor_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, env_seed, rollout_id, snapshot_id, actor, rotation_error in FAILURES:
        actor_errors = {
            "deskbin": {"translation_error": 0.001, "rotation_error": 0.01},
            "garbage_0": {"translation_error": 0.001, "rotation_error": 0.01},
            "garbage_1": {"translation_error": 0.001, "rotation_error": 0.01},
            "garbage_2": {"translation_error": 0.001, "rotation_error": 0.01},
            "garbage_3": {"translation_error": 0.001, "rotation_error": 0.01},
            "garbage_4": {"translation_error": 0.001, "rotation_error": 0.01},
        }
        if task == "place_container_plate":
            actor_errors = {
                "container": {"translation_error": 0.002, "rotation_error": rotation_error},
            }
        else:
            actor_errors[actor]["rotation_error"] = rotation_error
        actor_metric_passed = {
            name: {
                "translation": metrics["translation_error"] <= 0.005,
                "rotation": metrics["rotation_error"] <= 0.05,
            }
            for name, metrics in actor_errors.items()
        }
        rows.append(
            {
                "task": task,
                "env_seed": env_seed,
                "rollout_id": rollout_id,
                "snapshot_id": snapshot_id,
                "check_type": "control_trace_replay",
                "passed": False,
                "errors": {
                    "object_rotation_error": rotation_error,
                    "actor_errors": actor_errors,
                },
                "actor_metric_passed": actor_metric_passed,
                "worst_actor": actor,
                "worst_metric": "rotation",
            }
        )
    failures_path = output_dir / "failures.jsonl"
    with failures_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {failures_path}")


if __name__ == "__main__":
    main()
