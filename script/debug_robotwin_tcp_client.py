#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import yaml


def _load_task_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    task_config = cfg.get("task_config")
    if not isinstance(task_config, dict):
        raise ValueError(f"task_config is missing or invalid in {config_path}")
    return task_config


def _append_rlinf_path(repo_path: Path) -> None:
    if not repo_path.exists():
        raise FileNotFoundError(f"RLinf path not found: {repo_path}")
    repo_path_s = str(repo_path.resolve())
    if repo_path_s not in sys.path:
        sys.path.insert(0, repo_path_s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal smoke test for RLinf ClientVectorEnv -> robotwin_env_server"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to RLinf RoboTwin env yaml (contains task_config)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--timeout", type=float, default=1800.0, help="request_timeout seconds")
    parser.add_argument("--heartbeat", type=float, default=20.0, help="heartbeat_interval seconds")
    parser.add_argument(
        "--rlinf-path",
        type=Path,
        default=Path("/workspace/RLinf"),
        help="RLinf repo root path for importing rlinf.envs.robotwin.robotwin_client",
    )
    args = parser.parse_args()

    _append_rlinf_path(args.rlinf_path)
    from rlinf.envs.robotwin.robotwin_client import ClientVectorEnv

    task_config = _load_task_config(args.config)
    env_seeds = [args.seed + i for i in range(args.n_envs)]
    print(
        f"[dbg] connecting tcp {args.host}:{args.port} n_envs={args.n_envs} "
        f"timeout={args.timeout}s heartbeat={args.heartbeat}s",
        flush=True,
    )
    print(
        "[dbg] task summary: "
        f"task_name={task_config.get('task_name')} "
        f"planner_backend={task_config.get('planner_backend')} "
        f"pointcloud={((task_config.get('data_type') or {}).get('pointcloud'))}",
        flush=True,
    )

    venv = None
    try:
        venv = ClientVectorEnv(
            host=args.host,
            port=args.port,
            n_envs=args.n_envs,
            env_seeds=env_seeds,
            request_timeout=args.timeout,
            heartbeat_interval=args.heartbeat,
            task_config=task_config,
        )
        print(f"[dbg] init ok remote_obs_schema={venv.remote_obs_schema}", flush=True)

        venv.reset(env_seeds=env_seeds)
        print(f"[dbg] reset ok env_seeds={env_seeds}", flush=True)

        obs = venv.get_obs()
        if not obs:
            print("[dbg] get_obs ok but obs is empty list", flush=True)
        else:
            state = obs[0].get("state")
            print(
                f"[dbg] get_obs ok obs_len={len(obs)} state_shape={getattr(state, 'shape', None)}",
                flush=True,
            )
    except Exception as e:
        print(f"[dbg] FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        if venv is not None:
            try:
                venv.close()
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    main()
