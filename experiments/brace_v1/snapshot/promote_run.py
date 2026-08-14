#!/usr/bin/env python3
"""Promote immutable run outputs into frozen archive/ with source_run.json lineage."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BRACE_DIR = REPO_ROOT / "experiments" / "brace"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.brace.replay_audit import file_sha256, git_commit, read_json, repo_path, write_json_atomic
from experiments.brace.resolve_artifact import resolve_seeds_file

PROMOTE_TARGETS: dict[str, dict[str, Any]] = {
    "archive/branches_place_pilot_valid_v2.3": {
        "kind": "branch",
        "branch_label": "branches",
        "required": ["summary.json", "checks.jsonl"],
        "optional": ["budgets.json"],
        "extra_protocol": "protocol.v2.3.json",
        "validate_passed": True,
    },
    "archive/branches_dump_pilot_valid_v2.3": {
        "kind": "branch",
        "branch_label": "branches_dump",
        "required": ["summary.json", "checks.jsonl"],
        "optional": ["budgets.json"],
        "extra_protocol": "protocol.v2.3.json",
        "validate_passed": True,
    },
    "archive/branches_place_confirm_v2.3": {
        "kind": "branch",
        "branch_label": "branches_confirm",
        "required": ["summary.json", "checks.jsonl"],
        "optional": ["merged_gate.json", "confirm_seeds.json"],
        "extra_protocol": "protocol.v2.3.json",
        "validate_passed": False,
    },
    "archive/replay_audit_v2_place_v2.3_gate": {
        "kind": "audit",
        "task": "place_container_plate",
        "required": ["summary.json"],
        "validate_replay_gate": True,
    },
    "archive/replay_audit_v2_place_base200_v2_gate": {
        "kind": "audit",
        "task": "place_container_plate",
        "required": ["summary.json"],
        "validate_replay_gate": True,
        "require_passed": True,
        "base200_line_a": True,
        "traced_corpus": "experiments/brace/rollouts_traced_base200_v2",
        "ckpt_path": "policy/DP/checkpoints/place_container_plate-demo_clean-200-0/600.ckpt",
        "amendment": "experiments/brace/multitask_amendment.v2.1.json",
        "protocol": "experiments/brace/protocol.v2.3.json",
    },
    "archive/replay_audit_v2_dump_v2.3_gate": {
        "kind": "audit",
        "task": "dump_bin_bigbin",
        "required": ["summary.json"],
        "validate_replay_gate": True,
    },
    "archive/branches_place_base200_v2": {
        "kind": "branch",
        "branch_label": "branches_place_base200_v2",
        "required": ["summary.json", "checks.jsonl"],
        "optional": ["budgets.json"],
        "extra_protocol": "protocol.v2.3.json",
        "validate_passed": True,
    },
}

DATASET_PROMOTE_FILES = ["{run_label}_B1.jsonl", "{run_label}_N1.jsonl", "{run_label}_summary.json"]


def resolve_run_dir(run_dir: Path | None, target_key: str) -> Path:
    if run_dir is not None:
        return repo_path(run_dir)
    pointer = BRACE_DIR / "runs" / "LATEST"
    if pointer.is_file():
        return repo_path(Path(pointer.read_text(encoding="utf-8").strip()))
    raise ValueError(f"BRACE_PROMOTE_RUN not set and no runs/LATEST for target {target_key}")


def branch_source_dir(run_dir: Path, label: str) -> Path:
    nested = run_dir / label
    if (nested / "summary.json").is_file():
        return nested
    if (run_dir / "summary.json").is_file():
        return run_dir
    raise FileNotFoundError(f"no branch summary under {run_dir} (label={label})")


def audit_source_dir(run_dir: Path, task: str) -> Path:
    nested = run_dir / "replay_audit_v2" / task
    if (nested / "summary.json").is_file():
        return nested
    if (run_dir / "summary.json").is_file():
        return run_dir
    raise FileNotFoundError(f"no audit summary under {run_dir} for task {task}")


def validate_branch_summary(summary_path: Path, *, require_passed: bool) -> None:
    summary = read_json(summary_path)
    if require_passed and not summary.get("passed"):
        raise ValueError(f"branch summary passed=false, refuse promote: {summary_path}")
    if not summary.get("harness_valid"):
        raise ValueError(f"branch harness_valid=false, refuse promote: {summary_path}")


def validate_audit_summary(summary_path: Path, task: str, *, require_passed: bool = False) -> None:
    summary = read_json(summary_path)
    task_stats = summary.get("tasks", {}).get(task, {})
    if not summary.get("complete"):
        raise ValueError(f"audit complete=false, refuse promote: {summary_path}")
    if require_passed and not summary.get("passed"):
        raise ValueError(f"audit passed=false, refuse promote: {summary_path}")
    replay = summary.get("control_trace_replay", {})
    if int(replay.get("total_checks", 0)) == 0:
        raise ValueError(f"audit has zero replay checks, refuse promote: {summary_path}")
    if not task_stats.get("replay_gate_passed"):
        raise ValueError(f"audit replay_gate_passed=false for {task}: {summary_path}")


def copy_file(src: Path, dest: Path) -> dict[str, Any]:
    if not src.is_file():
        raise FileNotFoundError(f"missing promotion source: {src}")
    if dest.is_file() and file_sha256(dest) != file_sha256(src):
        raise FileExistsError(
            f"frozen artifact differs; refusing to overwrite {dest}. "
            "Choose a new versioned promote target."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        shutil.copy2(src, dest)
    return {
        "source": str(src),
        "dest": str(dest),
        "sha256": file_sha256(dest),
        "size_bytes": dest.stat().st_size,
    }


def write_manifest(archive_dir: Path, filenames: list[str]) -> None:
    lines: list[str] = []
    for name in sorted(filenames):
        path = archive_dir / name
        if path.is_file():
            lines.append(f"{file_sha256(path)}  {path}")
    manifest = archive_dir / "MANIFEST.sha256"
    manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def promote_branch(run_dir: Path, archive_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    label = spec["branch_label"]
    source = branch_source_dir(run_dir, label)
    validate_branch_summary(source / "summary.json", require_passed=bool(spec.get("validate_passed")))

    copied: list[dict[str, Any]] = []
    manifest_names: list[str] = []
    for name in spec["required"]:
        copied.append(copy_file(source / name, archive_dir / name))
        manifest_names.append(name)
    for name in spec.get("optional", []):
        src = source / name
        if name == "confirm_seeds.json":
            seeds = resolve_seeds_file("place_container_plate_confirm_seeds.json")
            if seeds is not None:
                copied.append(copy_file(seeds, archive_dir / name))
                manifest_names.append(name)
            continue
        if name == "merged_gate.json":
            merged = source / "merged_gate.json"
            if not merged.is_file():
                merged = run_dir / "merged_gate.json"
            if merged.is_file():
                copied.append(copy_file(merged, archive_dir / name))
                manifest_names.append(name)
            continue
        if src.is_file():
            copied.append(copy_file(src, archive_dir / name))
            manifest_names.append(name)
    protocol_name = spec.get("extra_protocol")
    if protocol_name:
        proto = BRACE_DIR / "protocol.v2.3.json"
        if proto.is_file():
            copied.append(copy_file(proto, archive_dir / protocol_name))
            manifest_names.append(protocol_name)
    budgets_src = source / "budgets.json"
    if not budgets_src.is_file():
        task = next(iter(read_json(source / "summary.json").get("tasks", {})), None)
        legacy = BRACE_DIR / "budgets" / f"{task}_pilot.json" if task else None
        if legacy and legacy.is_file():
            budgets_src = legacy
    if budgets_src.is_file() and "budgets.json" not in manifest_names:
        copied.append(copy_file(budgets_src, archive_dir / "budgets.json"))
        manifest_names.append("budgets.json")

    return {"source_dir": str(source), "copied": copied, "manifest_names": manifest_names}


def promote_audit(run_dir: Path, archive_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    task = spec["task"]
    source = audit_source_dir(run_dir, task)
    validate_audit_summary(
        source / "summary.json",
        task,
        require_passed=bool(spec.get("require_passed")),
    )
    if (archive_dir / "summary.json").is_file():
        raise FileExistsError(
            f"promote target already populated: {archive_dir / 'summary.json'}. "
            "Choose a new versioned promote target."
        )
    names = ["summary.json", "checks.jsonl", "failures.jsonl", "diagnostics.jsonl"]
    copied = [copy_file(source / name, archive_dir / name) for name in names]
    protocol_rel = spec.get("protocol") or "experiments/brace/protocol.v2.3.json"
    protocol = repo_path(Path(protocol_rel))
    copied.append(copy_file(protocol, archive_dir / "protocol.v2.3.json"))
    names.append("protocol.v2.3.json")

    if spec.get("base200_line_a"):
        traced_corpus = repo_path(Path(spec["traced_corpus"]))
        traced_prov = traced_corpus / task / "brace_base200_traced_provenance.json"
        seeds_file = traced_corpus / task / "base200_v2_rollout_train_seeds.json"
        ckpt = repo_path(Path(spec["ckpt_path"]))
        ckpt_prov = ckpt.with_name("brace_base200_provenance.json")
        amendment = repo_path(Path(spec["amendment"]))
        amendment_sidecar = Path(str(amendment) + ".sha256")
        for label, path in (
            ("traced_provenance", traced_prov),
            ("seeds_file", seeds_file),
            ("ckpt", ckpt),
            ("ckpt_provenance", ckpt_prov),
            ("amendment", amendment),
            ("amendment_sha256_sidecar", amendment_sidecar),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"missing Base200 promote evidence ({label}): {path}")
        for src, dest_name in (
            (traced_prov, "brace_base200_traced_provenance.json"),
            (seeds_file, "base200_v2_rollout_train_seeds.json"),
            (ckpt_prov, "brace_base200_ckpt_provenance.json"),
            (amendment, "multitask_amendment.v2.1.json"),
            (amendment_sidecar, "multitask_amendment.v2.1.json.sha256"),
        ):
            copied.append(copy_file(src, archive_dir / dest_name))
            names.append(dest_name)
        provenance_payload = {
            "schema_version": 1,
            "task": task,
            "passed": True,
            "replay_gate_passed": True,
            "traced_corpus": str(Path(spec["traced_corpus"])),
            "traced_provenance_sha256": file_sha256(traced_prov),
            "seeds_file_sha256": file_sha256(seeds_file),
            "ckpt_path": str(Path(spec["ckpt_path"])),
            "ckpt_sha256": file_sha256(ckpt),
            "ckpt_provenance_sha256": file_sha256(ckpt_prov),
            "protocol": str(Path(protocol_rel)),
            "protocol_sha256": file_sha256(protocol),
            "amendment": str(Path(spec["amendment"])),
            "amendment_sha256": file_sha256(amendment),
            "source_audit_summary_sha256": file_sha256(source / "summary.json"),
        }
        write_json_atomic(archive_dir / "base200_line_a_provenance.json", provenance_payload)
        names.append("base200_line_a_provenance.json")
        copied.append(
            {
                "source": "synthesized:base200_line_a_provenance",
                "dest": str(archive_dir / "base200_line_a_provenance.json"),
                "sha256": file_sha256(archive_dir / "base200_line_a_provenance.json"),
                "size_bytes": (archive_dir / "base200_line_a_provenance.json").stat().st_size,
                "synthesized": True,
            }
        )
    return {"source_dir": str(source), "copied": copied, "manifest_names": names}


def promote_datasets(run_dir: Path, dataset_dir: Path, run_label: str) -> dict[str, Any]:
    copied: list[dict[str, Any]] = []
    manifest_names: list[str] = []
    for pattern in DATASET_PROMOTE_FILES:
        name = pattern.format(run_label=run_label)
        src = run_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"missing export artifact: {src}")
        copied.append(copy_file(src, dataset_dir / name))
        manifest_names.append(name)
    export_summary = run_dir / "export_summary.json"
    if export_summary.is_file():
        export_name = f"{run_label}_export_summary.json"
        copied.append(copy_file(export_summary, dataset_dir / export_name))
        manifest_names.append(export_name)
    return {"source_dir": str(run_dir), "copied": copied, "manifest_names": manifest_names}


def promote(
    *,
    target: str,
    run_dir: Path | None,
    run_label: str | None,
    allow_failed_gate: bool,
) -> dict[str, Any]:
    if target.startswith("datasets/"):
        if not run_label:
            raise ValueError("--run-label required for datasets promote")
        source_run = resolve_run_dir(run_dir, target)
        dataset_dir = repo_path(BRACE_DIR / "datasets")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        result = promote_datasets(source_run, dataset_dir, run_label)
        archive_dir = dataset_dir
        spec = {"kind": "datasets"}
    else:
        if target not in PROMOTE_TARGETS:
            raise ValueError(f"unknown promote target: {target}")
        spec = PROMOTE_TARGETS[target]
        source_run = resolve_run_dir(run_dir, target)
        archive_dir = repo_path(BRACE_DIR / target)
        archive_dir.mkdir(parents=True, exist_ok=True)
        if allow_failed_gate and spec.get("validate_passed"):
            spec = {**spec, "validate_passed": False}
        if allow_failed_gate and spec.get("validate_replay_gate"):
            spec = {**spec, "validate_replay_gate": False}
        if spec["kind"] == "branch":
            if spec.get("validate_replay_gate"):
                pass
            result = promote_branch(source_run, archive_dir, spec)
        elif spec["kind"] == "audit":
            if not allow_failed_gate:
                validate_audit_summary(
                    audit_source_dir(source_run, spec["task"]) / "summary.json",
                    spec["task"],
                    require_passed=bool(spec.get("require_passed")),
                )
            result = promote_audit(source_run, archive_dir, spec)
        else:
            raise ValueError(spec["kind"])

    meta_path = source_run / "meta.json"
    if not meta_path.is_file():
        for parent in (source_run.parent, source_run.parent.parent):
            candidate = parent / "meta.json"
            if candidate.is_file():
                meta_path = candidate
                break
    source_run_payload = {
        "schema_version": 1,
        "promoted_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "git_commit": git_commit(),
        "target": target,
        "source_run_dir": str(source_run),
        "source_meta": read_json(meta_path) if meta_path.is_file() else None,
        "artifacts": result["copied"],
    }
    lineage_name = f"{run_label}_source_run.json" if spec["kind"] == "datasets" else "source_run.json"
    lineage_path = archive_dir / lineage_name
    if lineage_path.is_file():
        existing_lineage = read_json(lineage_path)
        if existing_lineage.get("source_run_dir") != str(source_run):
            raise FileExistsError(
                f"frozen lineage differs; refusing to overwrite {lineage_path}. "
                "Choose a new versioned promote target."
            )
    else:
        write_json_atomic(lineage_path, source_run_payload)
    if spec["kind"] != "datasets":
        write_manifest(archive_dir, result["manifest_names"] + ["source_run.json"])

    payload_out = {
        "target": target,
        "archive_dir": str(archive_dir),
        "source_run": str(source_run),
        **result,
    }
    try:
        from experiments.brace.stage_records import emit_stage_record

        emit_stage_record(
            "promote_run",
            summary=payload_out,
            run_dir=source_run,
            label=target.replace("/", "_"),
            extra={"promote_target": target},
        )
    except Exception:
        pass
    return payload_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="e.g. archive/branches_place_pilot_valid_v2.3 or datasets/")
    parser.add_argument("--run-dir", type=Path, default=None, help="defaults to runs/LATEST")
    parser.add_argument("--run-label", default=None, help="required for datasets/ promote")
    parser.add_argument("--allow-failed-gate", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source/target/gate only; do not write archive files.",
    )
    args = parser.parse_args()

    if args.dry_run:
        if args.target.startswith("datasets/"):
            raise SystemExit("dry-run not implemented for datasets/")
        if args.target not in PROMOTE_TARGETS:
            raise SystemExit(f"unknown promote target: {args.target}")
        spec = PROMOTE_TARGETS[args.target]
        source_run = resolve_run_dir(args.run_dir, args.target)
        archive_dir = repo_path(BRACE_DIR / args.target)
        if (archive_dir / "summary.json").is_file():
            raise SystemExit(f"dry-run fail: target already exists: {archive_dir}")
        if spec["kind"] == "audit":
            source = audit_source_dir(source_run, spec["task"])
            validate_audit_summary(
                source / "summary.json",
                spec["task"],
                require_passed=bool(spec.get("require_passed")),
            )
            summary = read_json(source / "summary.json")
            task_stats = summary["tasks"][spec["task"]]
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "ok": True,
                        "target": args.target,
                        "source_run": str(source_run),
                        "source_summary": str(source / "summary.json"),
                        "passed": summary.get("passed"),
                        "replay_gate_passed": task_stats.get("replay_gate_passed"),
                        "protocol_sha256": summary.get("protocol_sha256"),
                        "archive_dir_absent_or_empty": not (archive_dir / "summary.json").is_file(),
                        "base200_line_a": bool(spec.get("base200_line_a")),
                    },
                    indent=2,
                )
            )
            return 0
        raise SystemExit(f"dry-run not implemented for kind={spec['kind']}")

    payload = promote(
        target=args.target,
        run_dir=args.run_dir,
        run_label=args.run_label,
        allow_failed_gate=args.allow_failed_gate,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
