#!/usr/bin/env python3
# Copyright 2025 The RLinf / RoboTwin integration.
#
# Standalone Diffusion Policy inference server (RoboTwin Python + diffusers/peft).
# RLinf actor/rollout connect here with expert_backend=remote to avoid importing DP stack.
#
#   python script/robotwin_dp_server.py --port 8767 --ckpt /path/to/600.ckpt
#
# Same length-prefixed JSON+blob framing as robotwin_env_server.py (api_version=1).

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_VERSION = 1

_POLICY_CACHE: dict[str, torch.nn.Module] = {}
_POLICY_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()


def _resolve_torch_device(map_location: str) -> torch.device:
    if map_location.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(map_location)


def _get_cached_policy(ckpt: str, dev: torch.device) -> torch.nn.Module:
    key = f"{ckpt}|{dev}"
    with _POLICY_LOCK:
        if key not in _POLICY_CACHE:
            from policy.DP.dp_model import load_diffusion_policy_from_checkpoint

            pol = load_diffusion_policy_from_checkpoint(ckpt, map_location=str(dev))
            pol.eval()
            _POLICY_CACHE[key] = pol
        return _POLICY_CACHE[key]


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


def _error_reply(conn: socket.socket, req: dict[str, Any], exc: BaseException) -> None:
    req_id = req.get("request_id", "")
    meta: dict[str, Any] = {
        "ok": False,
        "api_version": API_VERSION,
        "request_id": req_id,
        "error_type": type(exc).__name__,
        "error_message": f"{type(exc).__name__}: {exc}",
        "error": f"{type(exc).__name__}: {exc}",
        "can_retry": False,
        "should_recreate_env": False,
        "traceback": traceback.format_exc(),
    }
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


def _update_dp_history(
    hist_head: Optional[torch.Tensor],
    hist_state: Optional[torch.Tensor],
    head_bchw: torch.Tensor,
    state_bd: torch.Tensor,
    n_obs_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, dev, dt = head_bchw.shape[0], head_bchw.device, head_bchw.dtype
    T = n_obs_steps
    if hist_head is None or hist_head.shape[0] != B:
        hist_head = head_bchw.unsqueeze(1).expand(B, T, -1, -1, -1).clone()
        hist_state = state_bd.unsqueeze(1).expand(B, T, -1).clone()
    else:
        hist_head = torch.roll(hist_head, shifts=-1, dims=1)
        hist_state = torch.roll(hist_state, shifts=-1, dims=1)
        hist_head = hist_head.to(device=dev, dtype=dt)
        hist_state = hist_state.to(device=dev, dtype=state_bd.dtype)
        hist_head[:, -1] = head_bchw
        hist_state[:, -1] = state_bd
    return hist_head, hist_state


def handle_client(
    conn: socket.socket,
    addr: Any,
    default_ckpt: Optional[str],
    map_location: str,
) -> None:
    from policy.DP.rlinf_adapter import rlinf_main_state_to_dp_timestep

    policy: Optional[torch.nn.Module] = None
    hist_head: Optional[torch.Tensor] = None
    hist_state: Optional[torch.Tensor] = None
    dev = _resolve_torch_device(map_location)

    try:
        while True:
            try:
                req, blob = _recv_frame(conn)
            except (ConnectionError, json.JSONDecodeError, struct.error, EOFError) as e:
                print(f"[robotwin_dp_server] disconnect {addr}: {e}", flush=True)
                break

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
                    ckpt = req.get("ckpt_path") or default_ckpt
                    if not ckpt:
                        raise ValueError("ckpt_path required in init or set --ckpt on server")
                    policy = _get_cached_policy(ckpt, dev)
                    hist_head, hist_state = None, None
                    n_action_steps = int(getattr(policy, "n_action_steps", 0))
                    meta_out = {
                        "n_obs_steps": int(getattr(policy, "n_obs_steps", 0)),
                        "horizon": int(policy.horizon),
                        "action_dim": int(policy.action_dim),
                        "n_action_steps": n_action_steps,
                        "ckpt_path": ckpt,
                    }
                    _ok_reply(conn, req, {"dp_meta": meta_out})
                    print(
                        f"[robotwin_dp_server] loaded DP ckpt={ckpt} device={dev} meta={meta_out}",
                        flush=True,
                    )
                    continue

                if policy is None:
                    _send_frame(
                        conn,
                        {
                            "ok": False,
                            "api_version": API_VERSION,
                            "request_id": req.get("request_id", ""),
                            "error_type": "NotInitialized",
                            "error_message": "send init (with ckpt_path) before other ops",
                            "error": "not initialized",
                            "can_retry": False,
                            "should_recreate_env": False,
                        },
                    )
                    continue

                if op == "dp_reset_history":
                    hist_head, hist_state = None, None
                    _ok_reply(conn, req)
                    continue

                if op == "dp_predict":
                    main_shape = tuple(req["main_shape"])
                    state_shape = tuple(req["state_shape"])
                    main_dtype = np.dtype(req["main_dtype"])
                    state_dtype = np.dtype(req["state_dtype"])
                    has_noise = bool(req.get("has_init_noise", False))

                    off = 0
                    main_sz = int(np.prod(main_shape)) * int(main_dtype.itemsize)
                    st_sz = int(np.prod(state_shape)) * int(state_dtype.itemsize)
                    main_np = np.frombuffer(blob[off : off + main_sz], dtype=main_dtype).reshape(
                        main_shape
                    )
                    off += main_sz
                    state_np = np.frombuffer(blob[off : off + st_sz], dtype=state_dtype).reshape(
                        state_shape
                    )
                    off += st_sz

                    main_t = torch.from_numpy(np.ascontiguousarray(main_np)).to(dev)
                    state_t = torch.from_numpy(np.ascontiguousarray(state_np)).to(dev)
                    step = rlinf_main_state_to_dp_timestep(main_t, state_t)
                    head_bchw = step["head_cam"].to(dtype=torch.float32)
                    st_bd = step["agent_pos"].to(dtype=torch.float32)

                    exp_dtype = next(policy.parameters()).dtype
                    n_obs = int(getattr(policy, "n_obs_steps", 0))
                    hist_head, hist_state = _update_dp_history(
                        hist_head, hist_state, head_bchw, st_bd, n_obs
                    )

                    obs_dict = {
                        "head_cam": hist_head.to(dtype=exp_dtype),
                        "agent_pos": hist_state.to(dtype=exp_dtype),
                    }

                    init_noise = None
                    if has_noise:
                        nh = int(req["noise_horizon"])
                        nda = int(req["noise_action_dim"])
                        nb = int(req["noise_batch"])
                        noise_shape = (nb, nh, nda)
                        noise_dtype = np.dtype(req["noise_dtype"])
                        noise_sz = int(np.prod(noise_shape)) * int(noise_dtype.itemsize)
                        noise_np = np.frombuffer(
                            blob[off : off + noise_sz], dtype=noise_dtype
                        ).reshape(noise_shape)
                        off += noise_sz
                        init_noise = torch.from_numpy(np.ascontiguousarray(noise_np)).to(
                            device=dev, dtype=exp_dtype
                        )

                    with _INFER_LOCK, torch.inference_mode():
                        out = policy.predict_action(obs_dict, init_noise=init_noise)
                    act = out["action"].detach().float().cpu().numpy()
                    raw = np.ascontiguousarray(act).tobytes()
                    _ok_reply(
                        conn,
                        req,
                        {
                            "action_shape": list(act.shape),
                            "action_dtype": str(act.dtype),
                        },
                        raw,
                    )
                    continue

                if op == "close":
                    policy = None
                    hist_head, hist_state = None, None
                    _ok_reply(conn, req)
                    continue

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
                print(f"[robotwin_dp_server] op={op} error: {e}\n{traceback.format_exc()}", flush=True)
                _error_reply(conn, req, e)
    finally:
        try:
            conn.close()
        except OSError:
            pass
        print(f"[robotwin_dp_server] client {addr} disconnected", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="RoboTwin DP TCP inference server for RLinf remote expert")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, required=True)
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Default checkpoint if client init omits ckpt_path",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="map_location for DP weights (cuda:0 or cpu)",
    )
    args = p.parse_args()

    listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen.bind((args.host, args.port))
    listen.listen(8)
    print(
        f"[robotwin_dp_server] listening {args.host}:{args.port} default_ckpt={args.ckpt!r} "
        f"device={args.device} api_version={API_VERSION}",
        flush=True,
    )

    while True:
        conn, addr = listen.accept()
        print(f"[robotwin_dp_server] client connected from {addr}", flush=True)
        conn.settimeout(None)
        handle_client(conn, addr, args.ckpt, args.device)


if __name__ == "__main__":
    main()
