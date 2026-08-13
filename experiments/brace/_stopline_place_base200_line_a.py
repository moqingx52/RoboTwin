#!/usr/bin/env python3
"""Cloud stop-line for Base200 Line A place_container_plate.

Sequence:
  1. Wait for adaptation eval (25 jobs × easy/hard) to finish.
  2. Run independent preservation eval (Base + 25 arms = 26 jobs).
  3. Keep per-episode JSON, logs, and preservation_eval_state.json.
  4. Write developmental_summary.json.
  5. Stop. Do not freeze, promote, held-out, retrain, or start Track 4.

Does not overlap preservation with a still-running adaptation scheduler.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path("/workspace/RoboTwin")
BRACE = REPO / "experiments" / "brace"
PY = "/root/miniconda/envs/RoboTwin/bin/python"
DEFAULT_GPUS = "0 1 2 3 4 5 6 7"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_dir() -> Path:
    ptr = BRACE / "runs/LATEST_place_base200_v2_line_a_pilot"
    path = Path(ptr.read_text(encoding="utf-8").strip())
    return path if path.is_absolute() else REPO / path


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return False
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    if not proc.exists():
        return False
    try:
        raw = (proc / "stat").read_text()
        state = raw.split(")")[-1].split()[0]
        if state == "Z":
            return False
    except OSError:
        return False
    return True


def cmdline_of(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def find_scheduler(script_name: str) -> int | None:
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            exe = os.readlink(f"/proc/{pid}/exe")
        except OSError:
            continue
        if "python" not in exe:
            continue
        text = cmdline_of(pid)
        if script_name not in text:
            continue
        if "_stopline_place_base200_line_a.py" in text:
            continue
        if pid_alive(pid):
            return pid
    return None


def job_counts(state_path: Path) -> dict[str, int]:
    if not state_path.is_file():
        return {}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return dict(Counter(str(j.get("status")) for j in state.get("jobs") or []))


def log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] {message}\n")
        handle.flush()
    print(message, flush=True)


def launch_scheduler(script: str, log_path: Path, gpu_ids: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["BRACE_EVAL_GPU_IDS"] = gpu_ids
    env["PYTHONPATH"] = f"{REPO}:{REPO}/policy/DP"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\n=== stopline launch {utc_now()} {script} gpus={gpu_ids} ===\n")
    handle.flush()
    return subprocess.Popen(
        [PY, str(BRACE / script)],
        cwd=str(REPO),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_for_eval_state(
    *,
    state_path: Path,
    script_name: str,
    log_path: Path,
    n_jobs: int,
    launch_if_dead: bool,
    scheduler_log: Path,
    gpu_ids: str,
) -> dict[str, int]:
    while True:
        counts = job_counts(state_path)
        n_done = int(counts.get("completed", 0))
        n_failed = int(counts.get("failed", 0))
        n_running = int(counts.get("running", 0))
        n_pending = int(counts.get("pending", 0)) + int(counts.get("blocked", 0))
        alive = find_scheduler(script_name)
        log_line(
            log_path,
            f"wait {script_name} counts={counts} scheduler_pid={alive} n_jobs={n_jobs}",
        )
        if n_failed:
            raise SystemExit(f"{script_name} has failed jobs: {counts}")
        if n_done >= n_jobs and n_running == 0 and n_pending == 0:
            return counts
        if launch_if_dead and alive is None and (n_pending or n_running or n_done < n_jobs):
            proc = launch_scheduler(script_name, scheduler_log, gpu_ids)
            log_line(log_path, f"relaunched {script_name} pid={proc.pid}")
        time.sleep(60)


def main() -> int:
    os.chdir(REPO)
    gpu_ids = os.environ.get("BRACE_EVAL_GPU_IDS", DEFAULT_GPUS)
    rd = run_dir()
    log_path = rd / "logs" / "stopline.log"
    marker_path = rd / "STOP_LINE.json"
    log_line(log_path, f"stopline start run_dir={rd} gpus={gpu_ids}")
    log_line(
        log_path,
        "stop after developmental_summary; freeze/held-out/Track4/retrain will not run",
    )

    # 1. Adaptation
    adapt_state = rd / "eval_state.json"
    if not adapt_state.is_file():
        raise SystemExit(f"missing adaptation state: {adapt_state}")
    adapt_script = "_schedule_place_base200_v2_eval.py"
    if find_scheduler(adapt_script) is None:
        log_line(log_path, "adaptation scheduler not running; will resume if incomplete")
    wait_for_eval_state(
        state_path=adapt_state,
        script_name=adapt_script,
        log_path=log_path,
        n_jobs=25,
        launch_if_dead=True,
        scheduler_log=rd / "logs" / "eval_scheduler.log",
        gpu_ids=gpu_ids,
    )
    log_line(log_path, "adaptation eval complete")

    if find_scheduler(adapt_script) is not None:
        log_line(log_path, "waiting for adaptation scheduler process to exit")
        while find_scheduler(adapt_script) is not None:
            time.sleep(5)

    # 2. Preservation
    validate = subprocess.run(
        [PY, str(BRACE / "_validate_place_base200_line_a_cohort.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    (rd / "cohort_validation_stopline.json").write_text(validate.stdout or validate.stderr, encoding="utf-8")
    if validate.returncode != 0:
        raise SystemExit(f"cohort validation failed:\n{validate.stdout}\n{validate.stderr}")
    log_line(log_path, "cohort validation passed; starting preservation (26 jobs)")

    pres_script = "_schedule_place_base200_v2_preservation_eval.py"
    if find_scheduler(pres_script) is None:
        proc = launch_scheduler(pres_script, rd / "logs_preservation" / "preservation_scheduler.log", gpu_ids)
        log_line(log_path, f"started preservation scheduler pid={proc.pid}")
    wait_for_eval_state(
        state_path=rd / "preservation_eval_state.json",
        script_name=pres_script,
        log_path=log_path,
        n_jobs=26,
        launch_if_dead=True,
        scheduler_log=rd / "logs_preservation" / "preservation_scheduler.log",
        gpu_ids=gpu_ids,
    )
    log_line(log_path, "preservation eval complete")
    while find_scheduler(pres_script) is not None:
        time.sleep(5)

    # 4. Developmental summary
    agg = subprocess.run(
        [PY, str(BRACE / "_aggregate_place_base200_line_a_developmental.py"), "--run-dir", str(rd)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    (rd / "logs" / "developmental_aggregate.log").write_text(
        (agg.stdout or "") + (agg.stderr or ""), encoding="utf-8"
    )
    log_line(log_path, f"developmental aggregator exit={agg.returncode}")
    print(agg.stdout, flush=True)
    if agg.stderr:
        print(agg.stderr, flush=True)

    marker = {
        "schema_version": 1,
        "kind": "place_base200_line_a_stop_line",
        "created_at_utc": utc_now(),
        "run_dir": str(rd),
        "adaptation_eval_state": str(rd / "eval_state.json"),
        "preservation_eval_state": str(rd / "preservation_eval_state.json"),
        "preservation_output_dir": str(rd / "eval_preservation"),
        "preservation_logs_dir": str(rd / "logs_preservation"),
        "developmental_summary": str(rd / "developmental_summary.json"),
        "aggregator_exit": agg.returncode,
        "gpu_pipeline": "stopped",
        "not_run": [
            "method_freeze",
            "heldout_brace",
            "b1_n1_reexport",
            "25_retrain",
            "anchor_four_mechanism_comparison",
            "new_2x2",
            "track4_final_checkpoint_probe",
            "track4_positive_control",
            "track4_weight_interpolation",
        ],
    }
    (marker_path).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    log_line(log_path, f"STOP LINE written {marker_path}")
    log_line(log_path, "automatic GPU pipeline stopped")
    return 0 if agg.returncode in (0, 2, 3) else agg.returncode


if __name__ == "__main__":
    raise SystemExit(main())
