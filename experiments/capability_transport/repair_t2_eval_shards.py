#!/usr/bin/env python3
"""Globally repair T2 eval shards for slim unique-panel rollout.

Do NOT use launch_t2_eval.py --migrate-existing-shards. That path only renames
cross_cell_hard → hard in place and leaves rows on the wrong shard ownership,
which triggers eval_per_seed resume: unexpected work item.

Correct repair (this script):
  1. Gather all shards for a variant (global, not per-shard).
  2. Normalize labels to easy/medium/hard/memorization_hard.
     - Keep native easy/medium/hard/memorization_hard rows.
     - Remap cross_cell_hard → hard only when env_seed ∈ frozen hard panel.
     - Drop within_cell_hard / right_bowl_y1_y2_hard as standalone labels
       (they are hard subsets; do not invent hard coverage from them unless
       they were already native hard or remapped cross).
  3. Dedup globally by (split, env_seed, repeat); prefer native hard over
     remapped cross when both exist.
  4. Rebuild shard ownership from the canonical slim worklist:
       index % num_shards  (same as common.split_range / eval_per_seed).
  5. Write repaired shards + audit manifest. Resume only after gates pass.

Default is dry-run into an output directory; use --apply to overwrite the
live task dir. Never touch Zero/Self-Diverse while they are running unless
explicitly selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CT_DIR.parents[1]
if str(CT_DIR) not in sys.path:
    sys.path.insert(0, str(CT_DIR))

from common import split_range  # noqa: E402
from eval_per_seed import build_work_items, work_item_key  # noqa: E402

ROLLOUT_SPLITS = ("easy", "medium", "hard", "memorization_hard")
Key = Tuple[str, int, int]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any, *, sort_keys: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=sort_keys)
        f.write("\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return None


def canonical_worklist(slim_splits: Dict[str, Sequence[int]], repeats: int = 8) -> List[Key]:
    """Match eval_per_seed with id/train/hard repeats=0 and only extra_splits."""
    # Preserve insertion order of slim_splits (must be easy/medium/hard/mem).
    items = build_work_items(
        seed_payload={"eval_id": [], "train_rollout": []},
        hard_seeds=[],
        id_repeats=0,
        train_repeats=0,
        hard_repeats=0,
        extra_splits={k: list(v) for k, v in slim_splits.items()},
        extra_split_repeats=repeats,
    )
    return [work_item_key(*it) for it in items]


def normalize_row(
    row: Dict[str, Any],
    hard_seeds: set[int],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return (normalized_row_or_None, action)."""
    split = str(row.get("split"))
    seed = int(row["env_seed"])
    if split in ROLLOUT_SPLITS:
        out = dict(row)
        out["split"] = split
        out["env_seed"] = seed
        out["repeat"] = int(row["repeat"])
        return out, "kept"
    if split == "cross_cell_hard":
        if seed in hard_seeds:
            out = dict(row)
            out["split"] = "hard"
            out["env_seed"] = seed
            out["repeat"] = int(row["repeat"])
            out["remapped_from"] = "cross_cell_hard"
            return out, "remapped_cross_to_hard"
        return None, "dropped_cross_not_in_hard"
    # within/right are hard subsets; do not promote to hard here — only cross
    # was rolled out as a disjoint label ahead of hard in the fat worklist.
    if split in ("within_cell_hard", "right_bowl_y1_y2_hard"):
        return None, f"dropped_{split}"
    return None, f"dropped_unknown_{split}"


def row_preference(row: Dict[str, Any]) -> Tuple[int, int]:
    """Higher is better when resolving global duplicates."""
    # Prefer native hard over remapped cross; prefer evaluated rows.
    native = 1 if not row.get("remapped_from") else 0
    evaluated = 1 if row.get("evaluated", True) else 0
    return (native, evaluated)


def gather_variant_rows(shard_paths: Sequence[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    meta: Optional[Dict[str, Any]] = None
    for path in shard_paths:
        payload = load_json(path)
        if meta is None:
            meta = {
                "task_name": payload.get("task_name"),
                "task_config": payload.get("task_config"),
                "variant": payload.get("variant"),
                "ckpt_path": payload.get("ckpt_path"),
                "hard_seeds": payload.get("hard_seeds"),
                "hard_seed_source": payload.get("hard_seed_source"),
                "provenance": payload.get("provenance"),
                "progress_template": {
                    "id_repeats": payload.get("progress", {}).get("id_repeats", 0),
                    "train_repeats": payload.get("progress", {}).get("train_repeats", 0),
                    "hard_repeats": payload.get("progress", {}).get("hard_repeats", 0),
                    "extra_split_repeats": payload.get("progress", {}).get("extra_split_repeats", 8),
                    "policy_seed_offset": payload.get("progress", {}).get(
                        "policy_seed_offset", 8000
                    ),
                },
            }
        else:
            for key in ("task_name", "task_config", "variant", "ckpt_path"):
                if payload.get(key) != meta.get(key):
                    raise RuntimeError(
                        f"Incompatible shard {path}: {key}={payload.get(key)!r} "
                        f"expected {meta.get(key)!r}"
                    )
        for row in payload.get("rows", []):
            rows.append(dict(row))
    if meta is None:
        raise RuntimeError("no shard payloads")
    return rows, meta


def dedup_normalized(
    rows: Iterable[Dict[str, Any]],
    hard_seeds: set[int],
) -> Tuple[Dict[Key, Dict[str, Any]], Counter]:
    actions: Counter = Counter()
    chosen: Dict[Key, Dict[str, Any]] = {}
    for row in rows:
        norm, action = normalize_row(row, hard_seeds)
        actions[action] += 1
        if norm is None:
            continue
        key = work_item_key(norm["split"], norm["env_seed"], norm["repeat"])
        if key not in chosen or row_preference(norm) > row_preference(chosen[key]):
            if key in chosen:
                actions["dedup_replaced"] += 1
            chosen[key] = norm
        else:
            actions["dedup_kept_existing"] += 1
    return chosen, actions


def redistribute(
    chosen: Dict[Key, Dict[str, Any]],
    worklist: Sequence[Key],
    num_shards: int,
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
    ownership = {key: i % num_shards for i, key in enumerate(worklist)}
    workset = set(worklist)
    shard_rows: List[List[Dict[str, Any]]] = [[] for _ in range(num_shards)]
    extras = []
    for key, row in chosen.items():
        if key not in workset:
            extras.append(key)
            continue
        shard_rows[ownership[key]].append(row)
    for rows in shard_rows:
        rows.sort(key=lambda r: (r["split"], int(r["env_seed"]), int(r["repeat"])))
    present = set(chosen)
    missing = [k for k in worklist if k not in present]
    stats = {
        "worklist_size": len(worklist),
        "unique_kept": len(chosen),
        "missing": len(missing),
        "extra_outside_worklist": len(extras),
        "per_shard_rows": [len(r) for r in shard_rows],
        "per_shard_expected": [
            len(split_range(list(range(len(worklist))), sid, num_shards))
            for sid in range(num_shards)
        ],
        "missing_keys_sample": [list(k) for k in missing[:20]],
        "extra_keys_sample": [list(k) for k in extras[:20]],
    }
    return shard_rows, stats


def build_shard_payload(
    meta: Dict[str, Any],
    rows: List[Dict[str, Any]],
    *,
    shard_id: int,
    num_shards: int,
    expected_on_shard: int,
    slim_splits: Dict[str, Sequence[int]],
) -> Dict[str, Any]:
    prog = dict(meta["progress_template"])
    # Completeness vs expected keys is enforced in audit; mark incomplete unless full.
    is_complete = len(rows) == expected_on_shard
    return {
        "task_name": meta["task_name"],
        "task_config": meta["task_config"],
        "variant": meta["variant"],
        "ckpt_path": meta["ckpt_path"],
        "hard_seeds": meta.get("hard_seeds") or [],
        "hard_seed_source": meta.get("hard_seed_source"),
        "splits": {},
        "rows": rows,
        "provenance": meta.get("provenance") or {},
        "progress": {
            **prog,
            "complete": bool(is_complete),
            "completed_episodes": len(rows),
            "shard_id": int(shard_id),
            "num_shards": int(num_shards),
            "global_shard_repair": True,
            "global_shard_repair_at": utc_now(),
            "expected_episodes_on_shard": int(expected_on_shard),
            "expected_unique_episodes_per_job": sum(len(v) * 8 for v in slim_splits.values()),
            "rollout_splits": list(ROLLOUT_SPLITS),
        },
    }


def gate_repaired_variant(
    shard_payloads: Sequence[Dict[str, Any]],
    worklist: Sequence[Key],
    *,
    policy_seed_offset: int = 8000,
    expected_task: str,
    expected_variant: str,
    expected_ckpt: str,
) -> Dict[str, Any]:
    num_shards = len(shard_payloads)
    ownership = {key: i % num_shards for i, key in enumerate(worklist)}
    expected_keys = set(worklist)
    errors: List[str] = []
    warnings: List[str] = []
    seen: Dict[Key, int] = {}

    if len(worklist) != 1992:
        errors.append(f"worklist_size={len(worklist)} expected 1992")

    for sid, payload in enumerate(shard_payloads):
        if payload.get("task_name") != expected_task:
            errors.append(f"shard{sid} task_name mismatch")
        if payload.get("variant") != expected_variant:
            errors.append(f"shard{sid} variant mismatch")
        if payload.get("ckpt_path") != expected_ckpt:
            errors.append(f"shard{sid} ckpt_path mismatch")
        prog = payload.get("progress") or {}
        if int(prog.get("shard_id", -1)) != sid:
            errors.append(f"shard{sid} progress.shard_id={prog.get('shard_id')}")
        if int(prog.get("num_shards", -1)) != num_shards:
            errors.append(f"shard{sid} progress.num_shards={prog.get('num_shards')}")
        for row in payload.get("rows", []):
            key = work_item_key(row["split"], row["env_seed"], row["repeat"])
            if key not in expected_keys:
                errors.append(f"shard{sid} extra key {key}")
            if ownership.get(key) != sid:
                errors.append(f"shard{sid} owns {key} but ownership is {ownership.get(key)}")
            if key in seen:
                errors.append(f"duplicate key {key} on shards {seen[key]} and {sid}")
            seen[key] = sid
            expected_ps = policy_seed_offset + int(row["repeat"])
            if int(row.get("policy_seed", -1)) != expected_ps:
                errors.append(
                    f"shard{sid} {key} policy_seed={row.get('policy_seed')} expected {expected_ps}"
                )

    missing = [k for k in worklist if k not in seen]
    if missing:
        warnings.append(f"missing {len(missing)}/{len(worklist)} keys (ok pre-resume)")

    # No unexpected duplicates / wrong ownership already in errors.
    return {
        "ok_for_resume": len(errors) == 0,
        "ok_for_merge": len(errors) == 0 and len(missing) == 0,
        "errors": errors[:50],
        "error_count": len(errors),
        "warnings": warnings,
        "present": len(seen),
        "missing": len(missing),
        "expected": len(worklist),
    }


def discover_variants(source_dir: Path, prefix: str) -> List[str]:
    names = set()
    for path in source_dir.glob(f"{prefix}*_shard_*_of_*.json"):
        # t2_eval_Expert-Cover-12_seed00_shard_00_of_03.json
        stem = path.name
        marker = "_shard_"
        if marker not in stem:
            continue
        names.add(stem.split(marker, 1)[0])
    return sorted(names)


def shard_paths_for(source_dir: Path, variant: str, num_shards: int) -> List[Path]:
    paths = [
        source_dir / f"{variant}_shard_{sid:02d}_of_{num_shards:02d}.json"
        for sid in range(num_shards)
    ]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing shards for {variant}: {missing}")
    return paths


def repair_one(
    *,
    variant: str,
    source_dir: Path,
    out_dir: Path,
    slim_splits: Dict[str, Sequence[int]],
    worklist: Sequence[Key],
    num_shards: int,
    policy_seed_offset: int,
    expected_task: str,
    expected_ckpt: Optional[str],
) -> Dict[str, Any]:
    paths = shard_paths_for(source_dir, variant, num_shards)
    raw_rows, meta = gather_variant_rows(paths)
    hard_seeds = set(int(s) for s in slim_splits["hard"])
    chosen, actions = dedup_normalized(raw_rows, hard_seeds)
    shard_rows, redist_stats = redistribute(chosen, worklist, num_shards)

    if expected_ckpt is None:
        expected_ckpt = str(meta["ckpt_path"])
    else:
        meta = dict(meta)
        # Keep source ckpt_path; gate compares expected.
        pass

    expected_per_shard = [
        len(split_range(list(range(len(worklist))), sid, num_shards))
        for sid in range(num_shards)
    ]
    payloads = []
    written = []
    for sid, rows in enumerate(shard_rows):
        payload = build_shard_payload(
            meta,
            rows,
            shard_id=sid,
            num_shards=num_shards,
            expected_on_shard=expected_per_shard[sid],
            slim_splits=slim_splits,
        )
        # Ensure ckpt_path absolute string consistency with live eval.
        if expected_ckpt:
            # Prefer recorded path from source shards.
            payload["ckpt_path"] = str(meta["ckpt_path"])
        out_path = out_dir / f"{variant}_shard_{sid:02d}_of_{num_shards:02d}.json"
        write_json(out_path, payload, sort_keys=False)
        written.append(str(out_path))
        payloads.append(payload)

    gates = gate_repaired_variant(
        payloads,
        worklist,
        policy_seed_offset=policy_seed_offset,
        expected_task=expected_task,
        expected_variant=variant,
        expected_ckpt=str(meta["ckpt_path"]),
    )
    return {
        "variant": variant,
        "source_shards": [str(p) for p in paths],
        "written": written,
        "raw_rows": len(raw_rows),
        "actions": dict(actions),
        "redistribute": redist_stats,
        "gates": gates,
        "task_name": meta["task_name"],
        "ckpt_path": meta["ckpt_path"],
        "source_sha256": {str(p): sha256_file(p) for p in paths},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory with original (pre-migrate) shard JSONs.",
    )
    parser.add_argument(
        "--slim-splits",
        type=Path,
        required=True,
        help="Canonical slim splits JSON (easy/medium/hard/memorization_hard).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write repaired shards (and audit.json).",
    )
    parser.add_argument(
        "--variant-prefix",
        default="t2_eval_Expert-Cover-12_seed",
        help="Only repair variants whose name starts with this prefix.",
    )
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--policy-seed-offset", type=int, default=8000)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument(
        "--apply-to",
        type=Path,
        default=None,
        help="If set, copy repaired shards into this live task dir after gates pass.",
    )
    parser.add_argument(
        "--require-resume-gates",
        action="store_true",
        help="Exit non-zero if any variant fails ok_for_resume.",
    )
    args = parser.parse_args()

    slim = load_json(args.slim_splits)
    missing_keys = [k for k in ROLLOUT_SPLITS if k not in slim]
    if missing_keys:
        raise SystemExit(f"slim splits missing keys: {missing_keys}")
    # Drop any non-rollout keys if present.
    slim_ordered = {k: [int(x) for x in slim[k]] for k in ROLLOUT_SPLITS}
    n_seeds = sum(len(v) for v in slim_ordered.values())
    if n_seeds != 249:
        raise SystemExit(f"expected 249 unique seeds, got {n_seeds}")

    worklist = canonical_worklist(slim_ordered, repeats=8)
    if len(worklist) != 1992:
        raise SystemExit(f"expected 1992 work items, got {len(worklist)}")

    variants = [
        v
        for v in discover_variants(args.source_dir, args.variant_prefix)
        if v.startswith(args.variant_prefix)
    ]
    if not variants:
        raise SystemExit(f"no variants under {args.source_dir} with prefix {args.variant_prefix}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Preserve slim splits copy beside repaired shards for audit.
    write_json(args.output_dir / "t2_eval_splits.used.json", slim_ordered, sort_keys=False)

    reports = []
    for variant in variants:
        print(f"repairing {variant} ...", flush=True)
        report = repair_one(
            variant=variant,
            source_dir=args.source_dir,
            out_dir=args.output_dir,
            slim_splits=slim_ordered,
            worklist=worklist,
            num_shards=args.num_shards,
            policy_seed_offset=args.policy_seed_offset,
            expected_task=args.task,
            expected_ckpt=None,
        )
        reports.append(report)
        g = report["gates"]
        print(
            f"  kept={report['redistribute']['unique_kept']} "
            f"missing={g['missing']} errors={g['error_count']} "
            f"ok_resume={g['ok_for_resume']} actions={report['actions']}",
            flush=True,
        )

    audit = {
        "record": "capability_transport.t2_eval.global_shard_repair.v1",
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "source_dir": str(args.source_dir),
        "slim_splits": str(args.slim_splits),
        "output_dir": str(args.output_dir),
        "num_shards": args.num_shards,
        "policy_seed_offset": args.policy_seed_offset,
        "worklist_size": len(worklist),
        "unique_seeds": n_seeds,
        "episodes_per_job": n_seeds * 8,
        "rollout_splits": list(ROLLOUT_SPLITS),
        "variants": reports,
        "all_ok_for_resume": all(r["gates"]["ok_for_resume"] for r in reports),
        "note": (
            "Resume with original eval args after copying repaired shards. "
            "Do not use --migrate-existing-shards. After full completion, merge "
            "and run eval_per_seed --check-complete-result."
        ),
    }
    audit_path = args.output_dir / "repair_audit.json"
    write_json(audit_path, audit, sort_keys=False)
    print(f"wrote audit {audit_path}", flush=True)

    if args.apply_to is not None:
        if not audit["all_ok_for_resume"]:
            raise SystemExit("refusing --apply-to: resume gates failed")
        args.apply_to.mkdir(parents=True, exist_ok=True)
        for report in reports:
            for path in report["written"]:
                src = Path(path)
                dst = args.apply_to / src.name
                shutil.copy2(src, dst)
                print(f"applied {dst}", flush=True)
        write_json(args.apply_to / "repair_audit.applied.json", audit, sort_keys=False)

    if args.require_resume_gates and not audit["all_ok_for_resume"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
