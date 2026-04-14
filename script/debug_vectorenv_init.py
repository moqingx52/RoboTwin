#!/usr/bin/env python3
from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml


def _import_vector_env():
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from robotwin.envs.vector_env import VectorEnv  # type: ignore
    print(f"[dbg] imported VectorEnv from {repo_root_str}", flush=True)
    return VectorEnv


def _load_task_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    task_config = cfg.get("task_config")
    if not isinstance(task_config, dict):
        raise ValueError(f"task_config is missing or invalid in {config_path}")
    return task_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal smoke test for RoboTwin VectorEnv init + first get_obs"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to RLinf RoboTwin env yaml (contains task_config)",
    )
    parser.add_argument(
        "--assets-path",
        type=str,
        default=None,
        help="Override ASSETS_PATH (default: existing env var or config.assets_path)",
    )
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Enable verbose VectorEnv init debug logs",
    )
    parser.add_argument(
        "--dump-traceback-seconds",
        type=int,
        default=0,
        help="If >0, periodically dump Python traceback every N seconds",
    )
    parser.add_argument(
        "--skip-get-obs",
        action="store_true",
        help="Only test VectorEnv(...) constructor",
    )
    args = parser.parse_args()

    cfg_assets = None
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict):
        cfg_assets = cfg.get("assets_path") or cfg.get("ASSETS_PATH")

    assets = args.assets_path or os.environ.get("ASSETS_PATH") or cfg_assets
    if not assets:
        raise SystemExit("ASSETS_PATH not set: pass --assets-path or set it in env/config")
    os.environ["ASSETS_PATH"] = str(assets)
    if args.debug_mode:
        os.environ["VECTOR_ENV_INIT_DEBUG"] = "1"

    print(f"[dbg] ASSETS_PATH={os.environ['ASSETS_PATH']}", flush=True)
    print(f"[dbg] config={args.config}", flush=True)
    if args.debug_mode:
        print("[dbg] VECTOR_ENV_INIT_DEBUG=1", flush=True)
    if args.dump_traceback_seconds > 0:
        faulthandler.enable()
        faulthandler.dump_traceback_later(args.dump_traceback_seconds, repeat=True)
        print(
            f"[dbg] faulthandler dump every {args.dump_traceback_seconds}s",
            flush=True,
        )

    task_config = _load_task_config(args.config)
    print(
        "[dbg] task summary: "
        f"task_name={task_config.get('task_name')} "
        f"planner_backend={task_config.get('planner_backend')} "
        f"head_cam={((task_config.get('camera') or {}).get('collect_head_camera'))} "
        f"wrist_cam={((task_config.get('camera') or {}).get('collect_wrist_camera'))}",
        flush=True,
    )

    VectorEnv = _import_vector_env()

    env_seeds = [args.seed + i for i in range(args.n_envs)]
    t0 = time.time()
    print(f"[dbg] before VectorEnv n_envs={args.n_envs} env_seeds={env_seeds}", flush=True)
    venv = None
    try:
        venv = VectorEnv(task_config=task_config, n_envs=args.n_envs, env_seeds=env_seeds)
        print(f"[dbg] after VectorEnv dt={time.time() - t0:.2f}s", flush=True)

        if args.skip_get_obs:
            print("[dbg] skip get_obs by --skip-get-obs", flush=True)
            return

        t1 = time.time()
        print("[dbg] before get_obs", flush=True)
        obs = venv.get_obs()
        dt = time.time() - t1
        print(f"[dbg] after get_obs dt={dt:.2f}s", flush=True)
        if not obs:
            print("[dbg] get_obs returned empty list", flush=True)
            return
        keys = list(obs[0].keys())
        print(f"[dbg] obs_len={len(obs)} keys={keys}", flush=True)
        state = obs[0].get("state")
        if state is None:
            print("[dbg] state is None", flush=True)
        else:
            print(f"[dbg] state_shape={getattr(state, 'shape', None)}", flush=True)
    except Exception as e:
        print(f"[dbg] FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        if args.dump_traceback_seconds > 0:
            faulthandler.cancel_dump_traceback_later()
        if venv is not None:
            try:
                venv.close()
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    main()
