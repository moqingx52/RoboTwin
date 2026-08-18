#!/usr/bin/env python3
"""Run one T2 matched small-dose DP job from the frozen t2_launch_config.json.

Writes job_meta.json citing the launch_config sha256 before train.py starts.
Does not launch sibling jobs. Resume: skip if checkpoints/t2-{seed}/1.ckpt exists.

Usage (inside the cloud container; CUDA_VISIBLE_DEVICES set by the launcher):
    python experiments/capability_transport/train_t2_job.py \
        --point Expert-Cover-12 --seed 0 --run-dir experiments/capability_transport/runs/<run>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CT_DIR))

from common import repo_path  # noqa: E402
from t2_train_lib import (  # noqa: E402
    DEFAULT_MIXTURE_DIR,
    checkpoint_path,
    job_id,
    load_launch_config,
    mixture_zarr_path,
    python_bin,
    realized_batch_mix,
    sha256_file,
    w_traj_from_rho,
    write_json,
)

DP_DIR = repo_path("policy", "DP")
TRAIN_PY = DP_DIR / "train.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hydra_overrides(cfg: dict, point: str, seed: int, job_dir: Path, zarr_path: Path) -> list[str]:
    matched = cfg["matched_training"]
    mixture = cfg["mixtures"][point]
    rho = float(mixture["rho"])
    mix = realized_batch_mix(int(matched["batch_size"]), rho)
    ckpt = str(repo_path(matched["resume_from_ckpt"]))
    overrides = [
        f"task.name={cfg['task']}",
        f"task_name={cfg['task']}",
        f"task.dataset.zarr_path={zarr_path}",
        "task.dataset.val_ratio=0.0",
        f"training.resume_from_ckpt={ckpt}",
        "training.resume=false",
        f"training.seed={int(seed)}",
        "training.device=cuda:0",
        f"training.num_epochs=1",
        "training.stop_after_epoch=1",
        f"training.normalizer_source={matched['normalizer_source']}",
        f"training.lr_scheduler={matched['lr_scheduler']}",
        f"training.lr_warmup_steps={int(matched['lr_warmup_steps'])}",
        "training.debug=false",
        "training.checkpoint_name=t2",
        f"dataloader.num_batches={int(matched['s_star'])}",
        f"dataloader.batch_size={int(matched['batch_size'])}",
        f"+dataloader.seed={int(seed)}",
        "logging.mode=offline",
        f"exp_name=t2_{point}_k{int(seed)}",
        f"setting={cfg['task_config']}",
        "expert_data_num=200",
        "head_camera_type=D435",
        f"hydra.run.dir={job_dir}",
        "hydra.job.chdir=true",
    ]
    if mix["expert_ratio"] is not None:
        overrides.append(f"dataloader.expert_ratio={mix['expert_ratio']}")
    return overrides


def build_job_meta(
    cfg: dict,
    launch_sha: str,
    point: str,
    seed: int,
    job_dir: Path,
    zarr_path: Path,
) -> dict:
    mixture = cfg["mixtures"][point]
    matched = cfg["matched_training"]
    rho = float(mixture["rho"])
    q = int(mixture["Q"])
    mix = realized_batch_mix(int(matched["batch_size"]), rho)
    rho_hat = mix["rho_realized"]
    return {
        "record": "capability_transport.t2_job_meta.v1",
        "created_at": utc_now(),
        "point": point,
        "training_seed": int(seed),
        "job_id": job_id(point, seed),
        "job_dir": str(job_dir),
        "launch_config": str(CT_DIR / "t2_launch_config.json"),
        "launch_config_sha256": launch_sha,
        "protocol": cfg["protocol"],
        "protocol_sha256": cfg["depends_on"]["protocol.t2_training.v1.json"],
        "base_ckpt": cfg["base_policy"]["ckpt"],
        "base_ckpt_sha256": cfg["base_policy"]["ckpt_sha256"],
        "zarr_path": str(zarr_path),
        "resume_from_ckpt": matched["resume_from_ckpt"],
        "resume_mode": matched["resume_mode"],
        "s_base200": matched["s_base200"],
        "s_star": matched["s_star"],
        "batch_size": matched["batch_size"],
        "rho_target": rho,
        "rho_realized": rho_hat,
        "Q": q,
        "w_traj_frozen": mixture.get("w_traj"),
        "w_traj_from_rho_target": w_traj_from_rho(rho, q),
        "w_traj_from_rho_realized": w_traj_from_rho(rho_hat, q),
        "expert_ratio": mix["expert_ratio"],
        "expert_per_batch": mix["expert_per_batch"],
        "rollout_per_batch": mix["rollout_per_batch"],
        "normalizer_source": matched["normalizer_source"],
        "paired_training_seeds": cfg["training_grid"]["paired_training_seeds"],
        "checkpoint_path": str(checkpoint_path(job_dir, seed)),
        "expert_ratio_note": (
            "DP dataloader.expert_ratio is the Base200 (sample_sources==0) fraction. "
            "T2 rho is the new-source proportion, so expert_ratio = 1 - rho."
        ),
        "status": "pending",
    }


def run_train(overrides: list[str], job_dir: Path) -> int:
    cmd = [python_bin(), str(TRAIN_PY), "--config-name=robot_dp_14.yaml", *overrides]
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["WANDB_MODE"] = "offline"
    env["WANDB_DISABLED"] = "true"
    repo = str(repo_path())
    env["PYTHONPATH"] = repo + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log_path = job_dir / "train.log"
    job_dir.mkdir(parents=True, exist_ok=True)
    print(f"cwd={DP_DIR}")
    print(" ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n# start {utc_now()}\n")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(DP_DIR), env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n# end {utc_now()} exit={proc.returncode}\n")
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one T2 DP job from t2_launch_config.json")
    parser.add_argument("--point", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mixture-dir", type=Path, default=DEFAULT_MIXTURE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="retrain even if 1.ckpt exists")
    args = parser.parse_args()

    cfg, launch_sha = load_launch_config()
    if args.point not in cfg["training_grid"]["points"]:
        raise SystemExit(f"unknown point {args.point}")
    if int(args.seed) not in cfg["training_grid"]["paired_training_seeds"]:
        raise SystemExit(f"training seed {args.seed} is not in the frozen K=12 list")

    zarr_path = mixture_zarr_path(args.point, args.mixture_dir).resolve()
    if not zarr_path.exists():
        raise SystemExit(f"missing mixture zarr {zarr_path}; run build_t2_mixtures.py first")

    job_dir = (args.run_dir / "jobs" / job_id(args.point, args.seed)).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = checkpoint_path(job_dir, args.seed)
    meta = build_job_meta(cfg, launch_sha, args.point, args.seed, job_dir, zarr_path)
    meta["hydra_overrides"] = hydra_overrides(cfg, args.point, args.seed, job_dir, zarr_path)
    write_json(job_dir / "job_meta.json", meta)

    if ckpt_out.is_file() and not args.force:
        meta["status"] = "skipped_complete"
        meta["finished_at"] = utc_now()
        write_json(job_dir / "job_meta.json", meta)
        print(f"skip complete {job_id(args.point, args.seed)}: {ckpt_out}")
        return

    if args.dry_run:
        meta["status"] = "dry_run"
        write_json(job_dir / "job_meta.json", meta)
        print(f"dry-run {job_id(args.point, args.seed)}")
        for item in meta["hydra_overrides"]:
            print(f"  {item}")
        return

    meta["status"] = "running"
    meta["started_at"] = utc_now()
    write_json(job_dir / "job_meta.json", meta)
    rc = run_train(meta["hydra_overrides"], job_dir)
    meta["exit_code"] = rc
    meta["finished_at"] = utc_now()
    if rc == 0 and ckpt_out.is_file():
        meta["status"] = "completed"
        meta["checkpoint_sha256"] = sha256_file(ckpt_out)
    else:
        meta["status"] = "failed"
    write_json(job_dir / "job_meta.json", meta)
    if rc != 0:
        raise SystemExit(rc)
    if not ckpt_out.is_file():
        raise SystemExit(f"train exited 0 but missing checkpoint {ckpt_out}")
    print(f"completed {job_id(args.point, args.seed)} -> {ckpt_out}")


if __name__ == "__main__":
    main()
