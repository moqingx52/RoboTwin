#!/usr/bin/env python3
"""Archive confirmatory v1.4.2 JSON/jsonl/log evidence (no ckpt/zarr payloads)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BRACE = REPO / "experiments" / "brace"
RUNS = BRACE / "runs"
ARCHIVE = BRACE / "archive"

ARCHIVES = [
    {
        "name": "confirmatory_base_census_v142_20260804",
        "source_run": RUNS / "20260804T070036Z_confirmatory_base_census_place_container_plate_dump_bin_bigbin",
        "readme": "P1a confirmatory base census under v1.4.2 (200 census_candidate_id + train, offset 3000). JSON/jsonl only.",
    },
    {
        "name": "confirmatory_preservation_cohort_v142_20260804",
        "source_run": RUNS / "20260804T213623Z_select_preservation_cohort_place_container_plate_dump_bin_bigbin",
        "readme": "P1b preservation cohort selection under v1.4.2 (meets_min_untouched=true). JSON only.",
    },
    {
        "name": "confirmatory_preservation_p1c_v142_20260805",
        "source_run": RUNS / "20260804T213700Z_confirmatory_preservation_place_container_plate_dump_bin_bigbin",
        "readme": "P1c confirmatory preservation training (C0/C1 x seeds 1-5, v1.4.2, 10/10 jobs complete). JSON/jsonl/log only.",
        "extra_logs": [
            RUNS / "p1_v142_full_chain_20260804T070036Z.log",
            RUNS / "p1_v142_from_c_20260804T213700Z.log",
            RUNS / "p1c_c1_resume_4a19bd8_20260805T062025Z.log",
        ],
    },
]

ALLOWED_SUFFIXES = {".json", ".jsonl", ".log"}
ALLOWED_MANIFEST_NAMES = {
    "brace_dataset_manifest.json",
    "brace_anchor_manifest.json",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_copy(path: Path) -> bool:
    if "/checkpoints/" in path.as_posix():
        return False
    if path.suffix in ALLOWED_SUFFIXES:
        return True
    if path.name in ALLOWED_MANIFEST_NAMES:
        return True
    return False


def collect_files(source_run: Path, extra_logs: list[Path] | None = None) -> list[Path]:
    files: list[Path] = []
    if source_run.is_dir():
        for path in sorted(source_run.rglob("*")):
            if path.is_file() and should_copy(path):
                files.append(path)
    for path in extra_logs or []:
        if path.is_file():
            files.append(path)
    return files


def write_manifest(archive_dir: Path, rel_names: list[str]) -> None:
    lines = []
    for name in sorted(rel_names):
        path = archive_dir / name
        if path.is_file():
            lines.append(f"{file_sha256(path)}  {name}")
    (archive_dir / "MANIFEST.sha256").write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def archive_one(spec: dict) -> dict:
    source_run: Path = spec["source_run"]
    archive_dir = ARCHIVE / spec["name"]
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    archive_dir.mkdir(parents=True)

    copied: list[str] = []
    for src in collect_files(source_run, spec.get("extra_logs")):
        if src.is_relative_to(source_run):
            rel = src.relative_to(source_run)
            dest = archive_dir / rel
        else:
            dest = archive_dir / "pipeline_logs" / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(str(dest.relative_to(archive_dir)))

    source_meta = {
        "schema_version": 1,
        "source_run": str(source_run.relative_to(REPO)),
        "archive_target": f"archive/{spec['name']}",
        "artifact_filter": "json_jsonl_log_and_zarr_manifests_excluding_checkpoints",
        "copied_files": sorted(copied),
    }
    (archive_dir / "source_run.json").write_text(
        json.dumps(source_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (archive_dir / "README.md").write_text(
        f"# {spec['name']}\n\n{spec['readme']}\n\nSource: `{source_meta['source_run']}`\n",
        encoding="utf-8",
    )
    all_rel = sorted(
        str(path.relative_to(archive_dir))
        for path in archive_dir.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    )
    write_manifest(archive_dir, all_rel)
    total_bytes = sum(path.stat().st_size for path in archive_dir.rglob("*") if path.is_file())
    return {"name": spec["name"], "files": len(copied), "bytes": total_bytes}


def main() -> int:
    summaries = [archive_one(spec) for spec in ARCHIVES]
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
