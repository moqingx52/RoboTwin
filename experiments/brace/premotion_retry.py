"""Bounded, evidence-logged pre-motion materialization for fixed saved seeds."""

from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAX_ATTEMPTS_ENV = "BRACE_PREMOTION_MAX_ATTEMPTS"
AMENDMENT_ENV = "BRACE_PREMOTION_RETRY_AMENDMENT"
REPO_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_retry_amendment(payload: dict[str, Any], *, max_attempts: int) -> list[str]:
    errors: list[str] = []
    retry = payload.get("operational_collection_retry", {})
    if payload.get("status") != "frozen":
        errors.append("retry amendment status must be 'frozen'")
    if int(retry.get("max_attempts_per_episode", -1)) != int(max_attempts):
        errors.append("retry amendment max_attempts_per_episode does not match the requested value")
    if retry.get("seed_substitution") is not False:
        errors.append("retry amendment must forbid seed substitution")
    if retry.get("episode_index_rule") != "frozen_cohort_index":
        errors.append("retry amendment must bind episode_index_rule=frozen_cohort_index")
    if retry.get("acceptance_rule") != "plan_success_and_check_success_and_trajectory_saved":
        errors.append("retry amendment has an unexpected acceptance_rule")
    if retry.get("log_all_attempts") is not True:
        errors.append("retry amendment must require logging every attempt")
    eligible_tasks = retry.get("eligible_tasks")
    if not isinstance(eligible_tasks, list) or not eligible_tasks or not all(isinstance(task, str) for task in eligible_tasks):
        errors.append("retry amendment must list eligible_tasks")
    return errors


def resolve_max_attempts(
    env: dict[str, str] | None = None,
    *,
    cwd: Path | None = None,
    task_name: str | None = None,
) -> tuple[int, str | None]:
    values = os.environ if env is None else env
    raw = values.get(MAX_ATTEMPTS_ENV, "1")
    try:
        max_attempts = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{MAX_ATTEMPTS_ENV} must be an integer, got {raw!r}") from exc
    if max_attempts < 1:
        raise RuntimeError(f"{MAX_ATTEMPTS_ENV} must be positive")
    if max_attempts == 1:
        return 1, None

    amendment_value = values.get(AMENDMENT_ENV)
    if not amendment_value:
        raise RuntimeError(f"{AMENDMENT_ENV} is required when {MAX_ATTEMPTS_ENV}>1")
    amendment_path = Path(amendment_value)
    if not amendment_path.is_absolute():
        amendment_path = (cwd or Path.cwd()) / amendment_path
    if not amendment_path.is_file():
        raise RuntimeError(f"retry amendment does not exist: {amendment_path}")
    payload = json.loads(amendment_path.read_text(encoding="utf-8"))
    errors = validate_retry_amendment(payload, max_attempts=max_attempts)
    errors.extend(validate_amendment_bindings(payload, amendment_path))
    if task_name is not None and task_name not in payload.get("operational_collection_retry", {}).get("eligible_tasks", []):
        errors.append(f"task {task_name!r} is not eligible for bounded retry under this amendment")
    if errors:
        raise RuntimeError("invalid retry amendment: " + "; ".join(errors))
    return max_attempts, str(amendment_path.resolve())


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_amendment_bindings(payload: dict[str, Any], amendment_path: Path) -> list[str]:
    errors: list[str] = []
    sidecar = amendment_path.with_suffix(amendment_path.suffix + ".sha256")
    if not sidecar.is_file() or not sidecar.read_text(encoding="utf-8").strip().split() or sidecar.read_text(encoding="utf-8").strip().split()[0] != sha256_file(amendment_path):
        errors.append("retry amendment SHA256 sidecar is missing or invalid")
    applies = payload.get("applies_to", {})
    protocol = REPO_ROOT / "experiments" / "brace" / "multitask_protocol.v1.json"
    if not protocol.is_file() or applies.get("protocol_sha256") != sha256_file(protocol):
        errors.append("retry amendment protocol_sha256 does not match multitask_protocol.v1.json")
    for path_key, sha_key in (
        ("isolated_parity_report", "isolated_parity_sha256"),
        ("packed_parity_report", "packed_parity_sha256"),
    ):
        value = payload.get("evidence", {}).get(path_key)
        evidence_path = REPO_ROOT / value if isinstance(value, str) else None
        if evidence_path is None or not evidence_path.is_file():
            errors.append(f"retry amendment evidence is missing: {path_key}")
        elif payload.get("evidence", {}).get(sha_key) != sha256_file(evidence_path):
            errors.append(f"retry amendment evidence SHA256 mismatch: {path_key}")
    return errors


def materialize_saved_seed_trajectory(
    task_env: Any,
    args: dict[str, Any],
    *,
    seed: int,
    episode_idx: int,
    max_attempts: int,
    amendment_path: str | None,
) -> dict[str, Any]:
    """Try one fixed seed repeatedly without substitution and log every attempt."""
    log_path = Path(args["save_path"]) / "premotion_attempts.jsonl"
    last_row: dict[str, Any] | None = None
    for attempt in range(1, max_attempts + 1):
        trajectory = Path(args["save_path"]) / "_traj_data" / f"episode{episode_idx}.pkl"
        row: dict[str, Any] = {
            "schema_version": 1,
            "stage": "saved_seed_premotion_materialization",
            "created_at": utc_now(),
            "task": args.get("task_name"),
            "task_config": args.get("task_config"),
            "seed": int(seed),
            "episode_idx": int(episode_idx),
            "attempt": attempt,
            "max_attempts": int(max_attempts),
            "seed_substitution": False,
            "amendment_path": amendment_path,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pid": os.getpid(),
            "plan_success": False,
            "check_success": False,
            "trajectory_saved": False,
            "passed": False,
            "error_type": None,
            "error_message": None,
        }
        try:
            task_env.setup_demo(now_ep_num=episode_idx, seed=seed, **args)
            task_env.play_once()
            row["plan_success"] = bool(task_env.plan_success)
            if not row["plan_success"]:
                row["error_type"] = "expert_plan_failed"
                row["error_message"] = "expert plan failed during saved-seed pre-motion materialization"
            else:
                row["check_success"] = bool(task_env.check_success())
                if not row["check_success"]:
                    row["error_type"] = "expert_check_failed"
                    row["error_message"] = "expert success check failed during saved-seed pre-motion materialization"
                else:
                    task_env.save_traj_data(episode_idx)
                    if not trajectory.is_file():
                        raise RuntimeError(f"trajectory was not created: {trajectory}")
                    row["trajectory_saved"] = True
                    row["passed"] = True
        except Exception as exc:
            row["error_type"] = row["error_type"] or type(exc).__name__
            row["error_message"] = row["error_message"] or (str(exc).strip() or type(exc).__name__)
        finally:
            try:
                task_env.close_env()
                if args.get("render_freq"):
                    viewer = getattr(task_env, "viewer", None)
                    if viewer is not None:
                        viewer.close()
            except Exception as exc:
                if row["passed"]:
                    row["passed"] = False
                    row["trajectory_saved"] = False
                row["error_type"] = row["error_type"] or "close_env_failed"
                row["error_message"] = row["error_message"] or (str(exc).strip() or type(exc).__name__)
            if not row["passed"] and trajectory.is_file():
                trajectory.unlink()
            append_jsonl(log_path, row)
        last_row = row
        if row["passed"]:
            return row

    assert last_row is not None
    raise RuntimeError(
        f"Saved seed {seed} exhausted {max_attempts} pre-motion attempts for episode {episode_idx}; "
        f"last_error={last_row.get('error_type')}: {last_row.get('error_message')}"
    )
