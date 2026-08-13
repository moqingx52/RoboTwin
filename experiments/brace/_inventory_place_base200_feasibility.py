#!/usr/bin/env python3
"""Inventory Base200 Line A B2/B3 constraint-feasibility artifacts.

Rematches hydra training logs by checkpoint_name AND training.seed (the previous
finetune/reconcile path matched name only and could reuse the wrong seed log).

Writes a gate-facing inventory under the pilot run dir. Does not retune anchors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
BRACE = REPO / "experiments" / "brace"
DP = REPO / "policy" / "DP"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.brace.replay_audit import read_json, write_json_atomic

METHODS = ("B2", "B3")
SEEDS = (1, 2, 3, 4, 5)
CHECKPOINT_LABEL = "place_base200_v2"
EPOCHS = 10
PROTOCOL = BRACE / "screen_protocol.v1.2.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_hydra_log(*, checkpoint_name: str, train_seed: int) -> Path | None:
    """Prefer the newest logs.json.txt whose hydra config matches name and seed."""
    from omegaconf import OmegaConf

    outputs = DP / "data" / "outputs"
    if not outputs.is_dir():
        return None
    matches: list[tuple[float, Path]] = []
    for candidate in outputs.rglob("logs.json.txt"):
        hydra_cfg = candidate.parent / ".hydra" / "config.yaml"
        if not hydra_cfg.is_file():
            continue
        try:
            cfg = OmegaConf.load(hydra_cfg)
        except Exception:
            continue
        name = str(OmegaConf.select(cfg, "training.checkpoint_name", default="") or "")
        if name != checkpoint_name:
            continue
        seed = OmegaConf.select(cfg, "training.seed", default=None)
        if seed is None or int(seed) != int(train_seed):
            continue
        if candidate.stat().st_size <= 0:
            continue
        matches.append((candidate.stat().st_mtime, candidate))
    if not matches:
        return None
    matches.sort()
    return matches[-1][1]


def group_summary(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group, stats in (payload.get("groups") or {}).items():
        if not isinstance(stats, dict):
            out[group] = stats
            continue
        out[group] = {
            "passed": stats.get("passed"),
            "mean": stats.get("mean"),
            "mean_pass": stats.get("mean_pass"),
            "p90": stats.get("p90"),
            "p90_pass": stats.get("p90_pass"),
            "violation_fraction": stats.get("violation_fraction"),
            "violation_fraction_pass": stats.get("violation_fraction_pass"),
            "ema_mean": stats.get("ema_mean"),
            "ema_pass": stats.get("ema_pass"),
            "epsilon": stats.get("epsilon"),
        }
    return out


def recompute_one(*, method: str, seed: int, force: bool) -> dict[str, Any]:
    checkpoint_name = f"place_container_plate-brace-{CHECKPOINT_LABEL}-{method}"
    cdir = DP / "checkpoints" / f"{checkpoint_name}-{seed}"
    feas_path = cdir / f"{EPOCHS}.feasibility.json"
    log_path = find_hydra_log(checkpoint_name=checkpoint_name, train_seed=seed)
    row: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "checkpoint_dir": str(cdir),
        "feasibility_path": str(feas_path),
        "log_path": str(log_path) if log_path else None,
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": file_sha256(PROTOCOL),
    }
    if log_path is None:
        row["error"] = "missing_seed_matched_hydra_log"
        row["passed"] = False
        row["recomputed"] = False
        row["seed_matched"] = False
        # Keep any on-disk feasibility for archival, but mark it untrusted:
        # earlier finetune.sh matched by checkpoint_name only and could reuse
        # another seed's hydra log (disk pruning may have deleted the true log).
        if feas_path.is_file() and feas_path.stat().st_size > 0:
            payload = read_json(feas_path)
            row["on_disk_feasibility_untrusted"] = True
            row["on_disk_passed"] = bool(payload.get("passed"))
            row["groups"] = group_summary(payload)
            row["feasibility_sha256"] = file_sha256(feas_path)
        return row
    row["seed_matched"] = True
    if force or not feas_path.is_file() or feas_path.stat().st_size <= 0:
        cmd = [
            sys.executable,
            str(BRACE / "anchor_feasibility_test.py"),
            "--protocol",
            str(PROTOCOL),
            "--log",
            str(log_path),
            "--output",
            str(feas_path),
        ]
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        row["recomputed"] = True
        row["analyzer_exit"] = proc.returncode
    else:
        row["recomputed"] = False
    if not feas_path.is_file() or feas_path.stat().st_size <= 0:
        row["error"] = "feasibility_missing_after_analyzer"
        row["passed"] = False
        return row
    payload = read_json(feas_path)
    row["passed"] = bool(payload.get("passed"))
    row["missing_groups"] = payload.get("missing_groups") or []
    row["groups"] = group_summary(payload)
    row["feasibility_sha256"] = file_sha256(feas_path)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=BRACE / "runs/20260812T031200Z_place_base200_v2_line_a_pilot",
    )
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO / args.run_dir
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            rows.append(recompute_one(method=method, seed=seed, force=args.force_recompute))
    identical_payload_hashes: dict[str, list[str]] = {}
    for row in rows:
        if not row.get("seed_matched"):
            continue
        digest = row.get("feasibility_sha256")
        if not digest:
            continue
        identical_payload_hashes.setdefault(digest, []).append(f"{row['method']}:seed{row['seed']}")
    duplicate_groups = [ids for ids in identical_payload_hashes.values() if len(ids) > 1]
    missing_seed_matched_logs = [
        f"{row['method']}:seed{row['seed']}"
        for row in rows
        if row.get("error") == "missing_seed_matched_hydra_log"
    ]
    protocol = read_json(PROTOCOL)
    inventory = {
        "schema_version": 1,
        "kind": "place_base200_v2_constraint_feasibility_inventory",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_revision": protocol.get("protocol_revision"),
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": file_sha256(PROTOCOL),
        "constraint_feasibility": protocol.get("constraint_feasibility"),
        "preservation_requires_constraint_feasibility": bool(
            (protocol.get("screens") or {}).get("preservation", {}).get("requires_constraint_feasibility")
        ),
        "ineligible_if_infeasible": bool(
            (protocol.get("constraint_feasibility") or {}).get("ineligible_if_infeasible")
        ),
        "force_recompute": bool(args.force_recompute),
        "rows": rows,
        "summary": {
            "n_anchor_checkpoints": len(rows),
            "n_passed": sum(1 for row in rows if row.get("passed")),
            "n_failed": sum(1 for row in rows if not row.get("passed")),
            "n_seed_matched": sum(1 for row in rows if row.get("seed_matched")),
            "missing_seed_matched_logs": missing_seed_matched_logs,
            "all_failed": all(not row.get("passed") for row in rows),
            "duplicate_feasibility_payload_groups": duplicate_groups,
            "gate_note": (
                "Under screen.v1.2 with ineligible_if_infeasible=true and preservation "
                "requires_constraint_feasibility=true, these B2/B3 checkpoints are "
                "gate-ineligible until a versioned protocol revision changes that rule. "
                "Behavioral eval may still be completed and archived; do not freeze "
                "BRACE-v2 from behavioral wins alone."
            ),
        },
    }
    out = run_dir / "constraint_feasibility_inventory.json"
    write_json_atomic(out, inventory)
    print(json.dumps({"output": str(out), "summary": inventory["summary"]}, indent=2))
    return 0 if inventory["summary"]["n_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
