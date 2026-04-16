#!/usr/bin/env python3
# Copyright 2025 The RLinf / RoboTwin integration.
#
# Run from RoboTwin repository root (or any cwd if paths are absolute):
#   python script/robotwin_env_server.py --port 8765 --config /path/to/robotwin_place_empty_cup.yaml
#
# The YAML should contain ``assets_path`` and ``task_config`` (same shape as RLinf env yaml).
#
# Scaling: 1 RLinf EnvWorker process : 1 server port (set server_addr per rank via Hydra override_cfgs).

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import struct
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_VERSION = 1

try:
    from envs.utils.create_actor import UnStableError as _UnStableErr
except ImportError:
    _UnStableErr = None

_UNSTABLE_TYPES: tuple[type, ...] = (_UnStableErr,) if _UnStableErr is not None else tuple()


def _arr_stats(x: Any) -> dict[str, Any]:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return {"size": 0}
    return {
        "shape": list(arr.shape),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _debug_enabled(level: int, debug_level: int) -> bool:
    return debug_level >= level


def _recvn(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed connection")
        buf += chunk
    return buf


def _recv_frame(conn: socket.socket) -> tuple[dict[str, Any], bytes]:
    (lb,) = struct.unpack(">I", _recvn(conn, 4))
    body = _recvn(conn, lb)
    (lb2,) = struct.unpack(">I", _recvn(conn, 4))
    blob = _recvn(conn, lb2) if lb2 else b""
    return json.loads(body.decode("utf-8")), blob


def _send_frame(conn: socket.socket, meta: dict[str, Any], blob: bytes = b"") -> None:
    body = json.dumps(meta, separators=(",", ":"), default=str).encode("utf-8")
    conn.sendall(struct.pack(">I", len(body)) + body + struct.pack(">I", len(blob)) + blob)


def _serialize_obs_list(obs_list: list[dict]) -> tuple[dict, bytes]:
    serializable: list[dict[str, Any]] = []
    blob_parts: list[bytes] = []
    offset = 0
    keys = ("full_image", "left_wrist_image", "right_wrist_image", "state")
    for obs in obs_list:
        instr = obs.get("instruction", "")
        if isinstance(instr, np.ndarray):
            instr = str(instr)
        elif instr is None:
            instr = ""
        else:
            instr = str(instr)
        entry: dict[str, Any] = {"instruction": instr}
        for k in keys:
            v = obs.get(k)
            if v is None:
                entry[k] = None
            else:
                arr = np.ascontiguousarray(v)
                raw = arr.tobytes()
                entry[k] = {
                    "shape": list(arr.shape),
                    "dtype": arr.dtype.str,
                    "start": offset,
                    "len": len(raw),
                }
                offset += len(raw)
                blob_parts.append(raw)
        serializable.append(entry)
    return {"obs": serializable}, b"".join(blob_parts)


def _sanitize(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sanitize(v) for v in x]
    return x


def _classify_exception(e: BaseException) -> tuple[str, bool, bool]:
    """Returns (error_type, can_retry, should_recreate_env)."""
    et = type(e).__name__
    if _UNSTABLE_TYPES and isinstance(e, _UNSTABLE_TYPES):
        return "UnStableError", True, False
    if et == "UnStableError":
        return "UnStableError", True, False
    if isinstance(e, RuntimeError) and "SubEnv" in str(e):
        return "SubEnvError", True, True
    if isinstance(e, (ConnectionError, OSError, TimeoutError)):
        return et, True, False
    return et, False, True


def _build_obs_schema(
    task_config: dict[str, Any], sample_obs: Optional[dict[str, Any]]
) -> dict[str, Any]:
    cam = task_config.get("camera") or {}
    keys = ["full_image", "state", "instruction"]
    if cam.get("collect_wrist_camera", False):
        keys = ["full_image", "left_wrist_image", "right_wrist_image", "state", "instruction"]
    dr = task_config.get("domain_randomization")
    schema: dict[str, Any] = {
        "obs_keys": keys,
        "image_layout": "HWC",
        "task_name": task_config.get("task_name"),
        "embodiment": task_config.get("embodiment"),
        "step_lim": task_config.get("step_lim"),
        "planner_backend": task_config.get("planner_backend"),
        "language_num": task_config.get("language_num"),
        "domain_randomization": _sanitize(dr) if dr is not None else None,
        "collect_head_camera": cam.get("collect_head_camera", True),
        "collect_wrist_camera": cam.get("collect_wrist_camera", False),
    }
    if sample_obs is not None and sample_obs.get("state") is not None:
        st = np.asarray(sample_obs["state"])
        schema["state_dim"] = int(st.reshape(-1).shape[0])
    else:
        schema["state_dim"] = 14
    return schema


def load_task_config(config_path: Path) -> tuple[dict, Optional[str]]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assets = cfg.get("assets_path") or cfg.get("ASSETS_PATH")
    task_config = cfg.get("task_config")
    if task_config is None:
        task_config = {}
    return task_config, assets


def _error_reply(
    conn: socket.socket,
    req: dict[str, Any],
    exc: BaseException,
    *,
    include_tb: bool = True,
) -> None:
    et, can_retry, should_recreate = _classify_exception(exc)
    req_id = req.get("request_id", "")
    msg = f"{type(exc).__name__}: {exc}"
    meta: dict[str, Any] = {
        "ok": False,
        "api_version": API_VERSION,
        "request_id": req_id,
        "error_type": et,
        "error_message": msg,
        "error": msg,
        "can_retry": can_retry,
        "should_recreate_env": should_recreate,
    }
    if include_tb:
        meta["traceback"] = traceback.format_exc()
    _send_frame(conn, meta)


def _ok_reply(
    conn: socket.socket,
    req: dict[str, Any],
    extra: Optional[dict[str, Any]] = None,
    blob: bytes = b"",
) -> None:
    meta: dict[str, Any] = {
        "ok": True,
        "api_version": API_VERSION,
        "request_id": req.get("request_id", ""),
    }
    if extra:
        meta.update(extra)
    _send_frame(conn, meta, blob)


def handle_client_connection(
    conn: socket.socket,
    addr: Any,
    default_task_config: dict[str, Any],
    idle_timeout: float,
    *,
    log_ops: bool = True,
    debug_level: int = 0,
    debug_every: int = 1,
) -> None:
    from robotwin.envs.vector_env import VectorEnv

    venv: Optional[VectorEnv] = None
    stop_watch = threading.Event()
    processing = [False]
    last_touch = [time.monotonic()]

    def touch() -> None:
        last_touch[0] = time.monotonic()

    def watch_idle() -> None:
        while not stop_watch.wait(timeout=5.0):
            if processing[0]:
                continue
            if time.monotonic() - last_touch[0] > idle_timeout:
                print(f"[robotwin_env_server] idle timeout ({idle_timeout}s), closing {addr}", flush=True)
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                stop_watch.set()
                return

    watcher = threading.Thread(target=watch_idle, daemon=True)
    watcher.start()

    try:
        while True:
            try:
                req, _ = _recv_frame(conn)
            except (ConnectionError, json.JSONDecodeError, struct.error, EOFError) as e:
                print(f"[robotwin_env_server] disconnect {addr}: {e}", flush=True)
                break

            touch()
            api_ver = int(req.get("api_version", API_VERSION))
            if api_ver != API_VERSION:
                _send_frame(
                    conn,
                    {
                        "ok": False,
                        "api_version": API_VERSION,
                        "request_id": req.get("request_id", ""),
                        "error_type": "UnsupportedApiVersion",
                        "error_message": f"need api_version {API_VERSION}, got {api_ver}",
                        "error": "bad api_version",
                        "can_retry": False,
                        "should_recreate_env": False,
                    },
                )
                continue

            op = req.get("op")
            try:
                if op == "heartbeat":
                    _ok_reply(conn, req)
                    continue

                if op == "init":
                    processing[0] = True
                    try:
                        if venv is not None:
                            try:
                                venv.close()
                            except Exception:
                                pass
                            venv = None
                        n_envs = int(req["n_envs"])
                        env_seeds = req.get("env_seeds")
                        if env_seeds is not None:
                            env_seeds = [int(s) for s in env_seeds]
                        client_tc = req.get("task_config")
                        if client_tc is not None:
                            if not isinstance(client_tc, dict):
                                raise TypeError("task_config must be a JSON object / dict")
                            tc_use = copy.deepcopy(client_tc)
                        else:
                            if not default_task_config:
                                raise ValueError(
                                    "server yaml has empty task_config; client must send task_config in init"
                                )
                            tc_use = copy.deepcopy(default_task_config)
                        venv = VectorEnv(
                            task_config=tc_use,
                            n_envs=n_envs,
                            env_seeds=env_seeds,
                        )
                        sample = None
                        if venv.n_envs:
                            try:
                                sample = venv.get_obs()[0]
                            except Exception:
                                sample = None
                        obs_schema = _build_obs_schema(tc_use, sample)
                        _ok_reply(conn, req, {"obs_schema": obs_schema})
                        if log_ops or _debug_enabled(1, debug_level):
                            tn = tc_use.get("task_name", "?")
                            sd = env_seeds
                            if sd is not None and len(sd) > 8:
                                sd = f"{sd[:8]}...(+{len(env_seeds) - 8})"
                            rid = req.get("request_id", "")
                            rid_s = f" request_id={rid!r}" if rid else ""
                            print(
                                f"[robotwin_env_server] op=init ok{rid_s} n_envs={n_envs} "
                                f"task_name={tn!r} env_seeds={sd} state_dim={obs_schema.get('state_dim')}",
                                flush=True,
                            )
                    finally:
                        processing[0] = False
                        touch()

                elif venv is None:
                    _send_frame(
                        conn,
                        {
                            "ok": False,
                            "api_version": API_VERSION,
                            "request_id": req.get("request_id", ""),
                            "error_type": "NotInitialized",
                            "error_message": "send init before other ops",
                            "error": "send init before other ops",
                            "can_retry": False,
                            "should_recreate_env": False,
                        },
                    )

                elif op == "reset":
                    processing[0] = True
                    try:
                        env_idx = req.get("env_idx")
                        env_seeds = req.get("env_seeds")
                        if env_seeds is not None:
                            env_seeds = [int(s) for s in env_seeds]
                        venv.reset(env_idx=env_idx, env_seeds=env_seeds)
                        _ok_reply(conn, req)
                        if log_ops or _debug_enabled(1, debug_level):
                            es = env_seeds
                            if es is not None and len(es) > 8:
                                es = f"{es[:8]}...(+{len(env_seeds) - 8})"
                            rid = req.get("request_id", "")
                            rid_s = f" request_id={rid!r}" if rid else ""
                            print(
                                f"[robotwin_env_server] op=reset ok{rid_s} env_idx={env_idx!r} "
                                f"env_seeds={es}",
                                flush=True,
                            )
                    finally:
                        processing[0] = False
                        touch()

                elif op == "get_obs":
                    obs_list = venv.get_obs()
                    obs_meta, blob = _serialize_obs_list(obs_list)
                    _ok_reply(conn, req, {"obs_meta": obs_meta}, blob)
                    if log_ops or _debug_enabled(1, debug_level):
                        rid = req.get("request_id", "")
                        rid_s = f" request_id={rid!r}" if rid else ""
                        print(
                            f"[robotwin_env_server] op=get_obs ok{rid_s} n_envs={len(obs_list)} "
                            f"blob_bytes={len(blob)}",
                            flush=True,
                        )
                    touch()

                elif op == "step":
                    processing[0] = True
                    try:
                        actions = np.asarray(req["actions"], dtype=np.float32)
                        obs_list, rewards, terminated, truncated, infos = venv.step(actions)
                        obs_meta, blob = _serialize_obs_list(obs_list)
                        _ok_reply(
                            conn,
                            req,
                            {
                                "obs_meta": obs_meta,
                                "rewards": _sanitize(rewards),
                                "terminated": _sanitize(terminated),
                                "truncated": _sanitize(truncated),
                                "infos": _sanitize(infos),
                            },
                            blob,
                        )
                        if log_ops or _debug_enabled(1, debug_level):
                            rw = np.asarray(rewards)
                            term = np.asarray(terminated) if terminated is not None else None
                            trunc = np.asarray(truncated) if truncated is not None else None
                            rid = req.get("request_id", "")
                            rid_s = f" request_id={rid!r}" if rid else ""
                            term_s = (
                                f" any_term={bool(term.any())}" if term is not None and term.size else ""
                            )
                            trunc_s = (
                                f" any_trunc={bool(trunc.any())}" if trunc is not None and trunc.size else ""
                            )
                            print(
                                f"[robotwin_env_server] op=step ok{rid_s} actions_shape={tuple(actions.shape)} "
                                f"reward_mean={float(rw.mean()) if rw.size else 0.0}{term_s}{trunc_s}",
                                flush=True,
                            )
                        if _debug_enabled(1, debug_level) and (int(venv.envs[0].debug_step_count) % debug_every == 0):
                            diag: dict[str, Any] = {
                                "actions": _arr_stats(actions),
                                "rewards": _arr_stats(rewards),
                                "terminated_sum": int(np.asarray(terminated, dtype=np.int32).sum()),
                                "truncated_sum": int(np.asarray(truncated, dtype=np.int32).sum()),
                            }
                            if _debug_enabled(2, debug_level):
                                per_env = []
                                for i, inf in enumerate(infos):
                                    item = {"env_id": i}
                                    if isinstance(inf, dict):
                                        dbg_trace = inf.get("debug_trace")
                                        if dbg_trace is not None:
                                            item["debug_trace"] = dbg_trace
                                        else:
                                            keys = []
                                            for k in inf.keys():
                                                ks = str(k).lower()
                                                if (
                                                    "reward" in ks
                                                    or "success" in ks
                                                    or "done" in ks
                                                    or "fail" in ks
                                                ):
                                                    keys.append(str(k))
                                            item["info_keys_focus"] = keys
                                    per_env.append(item)
                                diag["per_env"] = per_env
                            if _debug_enabled(3, debug_level):
                                act = np.asarray(actions, dtype=np.float32)
                                rew = np.asarray(rewards, dtype=np.float32)
                                diag["actions_head"] = act.reshape(-1)[: min(24, act.size)].tolist()
                                diag["rewards_full"] = rew.reshape(-1).tolist()
                            print(
                                f"[robotwin_env_server][debug{debug_level}] step_diag={json.dumps(_sanitize(diag), separators=(',', ':'))}",
                                flush=True,
                            )
                    finally:
                        processing[0] = False
                        touch()

                elif op == "check_seeds":
                    seeds = [int(s) for s in req["seeds"]]
                    results = venv.check_seeds(seeds)
                    _ok_reply(conn, req, {"results": _sanitize(results)})
                    if log_ops or _debug_enabled(1, debug_level):
                        rid = req.get("request_id", "")
                        rid_s = f" request_id={rid!r}" if rid else ""
                        print(
                            f"[robotwin_env_server] op=check_seeds ok{rid_s} n_seeds={len(seeds)}",
                            flush=True,
                        )
                    touch()

                elif op == "close":
                    clear_cache = bool(req.get("clear_cache", True))
                    if venv is not None:
                        venv.close(clear_cache=clear_cache)
                        venv = None
                    _ok_reply(conn, req)
                    if log_ops or _debug_enabled(1, debug_level):
                        rid = req.get("request_id", "")
                        rid_s = f" request_id={rid!r}" if rid else ""
                        print(f"[robotwin_env_server] op=close ok{rid_s} clear_cache={clear_cache}", flush=True)
                    touch()

                else:
                    _send_frame(
                        conn,
                        {
                            "ok": False,
                            "api_version": API_VERSION,
                            "request_id": req.get("request_id", ""),
                            "error_type": "UnknownOp",
                            "error_message": f"unknown op: {op!r}",
                            "error": f"unknown op: {op!r}",
                            "can_retry": False,
                            "should_recreate_env": False,
                        },
                    )
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[robotwin_env_server] op={op} error: {e}\n{tb}", flush=True)
                _error_reply(conn, req, e, include_tb=True)
                touch()
    finally:
        stop_watch.set()
        try:
            conn.close()
        except OSError:
            pass
        if venv is not None:
            try:
                venv.close()
            except Exception:
                pass
            venv = None
        print(f"[robotwin_env_server] client {addr} disconnected", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="RoboTwin VectorEnv TCP server for RLinf client mode",
        epilog="Use one server port per RLinf EnvWorker; point env.train.server_addr (or override_cfgs per rank) accordingly.",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind address")
    p.add_argument("--port", type=int, required=True)
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML with assets_path; task_config optional fallback if client omits init.task_config",
    )
    p.add_argument(
        "--assets-path",
        type=str,
        default=None,
        help="Override ASSETS_PATH (otherwise from yaml or env)",
    )
    p.add_argument(
        "--idle-timeout",
        type=float,
        default=600.0,
        help="Seconds with no requests (heartbeats count) while not in step/reset/init before disconnect + env cleanup",
    )
    p.add_argument(
        "--no-log-ops",
        action="store_true",
        help="Disable per-op success logs (init/reset/get_obs/step/check_seeds/close)",
    )
    p.add_argument(
        "--debug-level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="0=off, 1=basic step diagnostics, 2=include per-env reward/success focus, 3=include action/reward samples",
    )
    p.add_argument(
        "--debug-every",
        type=int,
        default=1,
        help="Print debug diagnostics every N env steps",
    )
    args = p.parse_args()

    default_task_config, assets_from_yaml = load_task_config(args.config)

    assets = args.assets_path or assets_from_yaml or os.environ.get("ASSETS_PATH")
    if not assets:
        raise SystemExit("Set assets_path in yaml, use --assets-path, or export ASSETS_PATH")
    os.environ["ASSETS_PATH"] = assets
    os.environ["VECTOR_ENV_DEBUG_LEVEL"] = str(args.debug_level)
    os.environ["VECTOR_ENV_DEBUG_EVERY"] = str(max(1, int(args.debug_every)))

    if not default_task_config:
        print(
            "[robotwin_env_server] warning: yaml task_config empty; first init must include client task_config",
            flush=True,
        )

    listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen.bind((args.host, args.port))
    listen.listen(1)
    print(
        f"[robotwin_env_server] listening on {args.host}:{args.port} ASSETS_PATH={assets} "
        f"idle_timeout={args.idle_timeout}s api_version={API_VERSION} "
        f"debug_level={args.debug_level} debug_every={max(1, int(args.debug_every))}",
        flush=True,
    )

    while True:
        conn, addr = listen.accept()
        print(f"[robotwin_env_server] client connected from {addr}", flush=True)
        conn.settimeout(None)
        handle_client_connection(
            conn,
            addr,
            default_task_config,
            idle_timeout=args.idle_timeout,
            log_ops=not args.no_log_ops,
            debug_level=args.debug_level,
            debug_every=max(1, int(args.debug_every)),
        )


if __name__ == "__main__":
    main()
