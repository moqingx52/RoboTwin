#!/usr/bin/env python3
# Copyright 2026 The RLinf / RoboTwin integration.
#
# Standalone RDT inference server for RLinf remote expert mode.
# Wire protocol is intentionally compatible with robotwin_dp_server.py.

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
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_VERSION = 1

_MODEL_CACHE: dict[str, dict[str, Any]] = {}
_MODEL_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()


def _resolve_torch_device(map_location: str) -> torch.device:
    if map_location.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(map_location)


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


def _create_rdt_model(
    *,
    ckpt_path: str,
    config_path: str,
    vision_encoder_path: str,
    text_encoder_path: str,
    lora_adapter_path: Optional[str],
    merge_lora: bool,
    control_frequency: int,
    device: torch.device,
) -> dict[str, Any]:
    from policy.RDT.multimodal_encoder.t5_encoder import T5Embedder
    from policy.RDT.scripts.agilex_model import create_model

    with open(config_path, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    cfg["arm_dim"] = {"left_arm_dim": 6, "right_arm_dim": 6}

    model = create_model(
        args=cfg,
        dtype=torch.bfloat16,
        pretrained=ckpt_path,
        pretrained_vision_encoder_name_or_path=vision_encoder_path,
        control_frequency=control_frequency,
        device=str(device),
    )
    model.policy.eval()
    model.vision_model.eval()
    model.policy = model.policy.to(device=device, dtype=torch.bfloat16)
    model.vision_model = model.vision_model.to(device=device, dtype=torch.bfloat16)
    lora_loaded = False
    if lora_adapter_path:
        try:
            from peft import PeftModel
        except ImportError as err:
            raise ImportError(
                "Loading an RDT LoRA adapter requires `peft` in the RoboTwin environment."
            ) from err
        model.policy = PeftModel.from_pretrained(
            model.policy,
            lora_adapter_path,
            is_trainable=False,
        )
        if merge_lora:
            model.policy = model.policy.merge_and_unload()
        model.policy = model.policy.to(device=device, dtype=torch.bfloat16)
        model.policy.eval()
        lora_loaded = True

    text_embedder = T5Embedder(
        from_pretrained=text_encoder_path,
        model_max_length=cfg["dataset"]["tokenizer_max_length"],
        device=device,
        use_offload_folder=None,
    )
    tokenizer = text_embedder.tokenizer
    text_encoder = text_embedder.model
    text_encoder.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "embed_cache": {},
        "pred_horizon": int(cfg["common"]["action_chunk_size"]),
        "action_dim": 14,
        "n_obs_steps": int(cfg["common"]["img_history_size"]),
        "lora_adapter_path": lora_adapter_path or "",
        "lora_loaded": lora_loaded,
        "lora_merged": bool(lora_loaded and merge_lora),
    }


def _get_cached_model(
    *,
    ckpt_path: str,
    config_path: str,
    vision_encoder_path: str,
    text_encoder_path: str,
    lora_adapter_path: Optional[str],
    merge_lora: bool,
    control_frequency: int,
    device: torch.device,
) -> dict[str, Any]:
    key = "|".join(
        [
            ckpt_path,
            config_path,
            vision_encoder_path,
            text_encoder_path,
            lora_adapter_path or "",
            str(bool(merge_lora)),
            str(control_frequency),
            str(device),
        ]
    )
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = _create_rdt_model(
                ckpt_path=ckpt_path,
                config_path=config_path,
                vision_encoder_path=vision_encoder_path,
                text_encoder_path=text_encoder_path,
                lora_adapter_path=lora_adapter_path,
                merge_lora=merge_lora,
                control_frequency=control_frequency,
                device=device,
            )
        return _MODEL_CACHE[key]


def _encode_instruction(bundle: dict[str, Any], instruction: str) -> torch.Tensor:
    instruction = instruction.strip()
    if not instruction:
        instruction = "Place the empty cup to the target area."
    cache: dict[str, torch.Tensor] = bundle["embed_cache"]
    if instruction in cache:
        return cache[instruction]
    tokenizer = bundle["tokenizer"]
    text_encoder = bundle["text_encoder"]
    dev = next(text_encoder.parameters()).device
    with torch.no_grad():
        tokens = tokenizer(
            instruction,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )["input_ids"].to(dev)
        embedding = text_encoder(tokens).last_hidden_state.detach()
    cache[instruction] = embedding
    return embedding


def _update_image_history(
    prev_imgs: Optional[list[dict[str, np.ndarray]]],
    main_np: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    bsz = int(main_np.shape[0])
    if prev_imgs is None or len(prev_imgs) != bsz:
        prev_imgs = [{"main": main_np[i].copy()} for i in range(bsz)]
    return prev_imgs


def _predict_actions(
    *,
    bundle: dict[str, Any],
    main_np: np.ndarray,
    state_np: np.ndarray,
    prev_imgs: list[dict[str, np.ndarray]],
    task_desc: str,
    init_noise: Optional[np.ndarray],
    n_action_steps: int,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    model = bundle["model"]
    pred_horizon = int(bundle["pred_horizon"])
    per_env_actions: list[np.ndarray] = []

    for i in range(main_np.shape[0]):
        prev = prev_imgs[i]["main"]
        curr = main_np[i]
        imgs = [
            Image.fromarray(prev),
            Image.fromarray(prev),
            Image.fromarray(prev),
            Image.fromarray(curr),
            Image.fromarray(curr),
            Image.fromarray(curr),
        ]
        proprio = torch.from_numpy(state_np[i : i + 1]).to(
            device=model.device, dtype=torch.float32
        )
        text_embeds = _encode_instruction(bundle, task_desc)
        noise_i = None
        if init_noise is not None:
            noise_i = torch.from_numpy(init_noise[i : i + 1]).to(
                device=model.device, dtype=torch.float32
            )
            if noise_i.shape[1] != pred_horizon:
                raise ValueError(
                    f"init_noise horizon mismatch: got {noise_i.shape[1]}, expect {pred_horizon}"
                )
        traj = model.step(
            proprio=proprio,
            images=imgs,
            text_embeds=text_embeds,
            init_noise=noise_i,
        )
        per_env_actions.append(traj[:, :n_action_steps, :].detach().cpu().numpy())
        prev_imgs[i]["main"] = curr.copy()

    actions = np.concatenate(per_env_actions, axis=0).astype(np.float32, copy=False)
    return actions, prev_imgs


def handle_client(
    conn: socket.socket,
    addr: Any,
    *,
    default_ckpt: Optional[str],
    config_path: str,
    vision_encoder_path: str,
    text_encoder_path: str,
    default_lora_adapter_path: Optional[str],
    merge_lora: bool,
    default_instruction: str,
    n_action_steps: int,
    control_frequency: int,
    map_location: str,
    log_ops: bool = True,
) -> None:
    bundle: Optional[dict[str, Any]] = None
    expected_batch_size: Optional[int] = None
    prev_imgs: Optional[list[dict[str, np.ndarray]]] = None
    dev = _resolve_torch_device(map_location)

    try:
        while True:
            try:
                req, blob = _recv_frame(conn)
            except (ConnectionError, json.JSONDecodeError, struct.error, EOFError):
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
                    lora_adapter_path = req.get("lora_adapter_path") or default_lora_adapter_path
                    bundle = _get_cached_model(
                        ckpt_path=ckpt,
                        config_path=config_path,
                        vision_encoder_path=vision_encoder_path,
                        text_encoder_path=text_encoder_path,
                        lora_adapter_path=lora_adapter_path,
                        merge_lora=merge_lora,
                        control_frequency=control_frequency,
                        device=dev,
                    )
                    expected_batch_size = None
                    prev_imgs = None
                    meta_out = {
                        "n_obs_steps": int(bundle["n_obs_steps"]),
                        "horizon": int(bundle["pred_horizon"]),
                        "action_dim": int(bundle["action_dim"]),
                        "n_action_steps": int(n_action_steps),
                        "ckpt_path": ckpt,
                        "lora_adapter_path": str(bundle.get("lora_adapter_path", "")),
                        "lora_loaded": bool(bundle.get("lora_loaded", False)),
                        "lora_merged": bool(bundle.get("lora_merged", False)),
                    }
                    _ok_reply(conn, req, {"dp_meta": meta_out})
                    if log_ops:
                        print(
                            f"[robotwin_rdt_server] op=init ok ckpt={ckpt} device={dev} meta={meta_out}",
                            flush=True,
                        )
                    continue

                if bundle is None:
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
                    env_idx = req.get("env_idx")
                    env_mask = req.get("env_mask")
                    if env_idx is None and env_mask is None:
                        prev_imgs = None
                        expected_batch_size = None
                        _ok_reply(conn, req, {"ack_batch_size": None, "selective": False})
                        continue
                    if prev_imgs is None or expected_batch_size is None:
                        raise ValueError(
                            "selective dp_reset_history requires at least one dp_predict first"
                        )
                    bsz = int(expected_batch_size)
                    mask = np.zeros((bsz,), dtype=np.bool_)
                    if env_mask is not None:
                        m = np.asarray(env_mask, dtype=np.bool_)
                        if m.size != bsz:
                            raise ValueError(
                                f"env_mask length {m.size} != ack_batch_size {bsz}"
                            )
                        mask |= m
                    if env_idx is not None:
                        for i in env_idx:
                            ii = int(i)
                            if ii < 0 or ii >= bsz:
                                raise ValueError(f"env_idx out of range: {ii} (B={bsz})")
                            mask[ii] = True
                    for ii, reset in enumerate(mask.tolist()):
                        if reset:
                            prev_imgs[ii]["main"] = None  # type: ignore[index]
                    _ok_reply(
                        conn,
                        req,
                        {
                            "ack_batch_size": bsz,
                            "selective": True,
                            "cleared_slots": int(mask.sum()),
                        },
                    )
                    continue

                if op == "dp_predict":
                    main_shape = tuple(req["main_shape"])
                    state_shape = tuple(req["state_shape"])
                    main_dtype = np.dtype(req["main_dtype"])
                    state_dtype = np.dtype(req["state_dtype"])
                    has_noise = bool(req.get("has_init_noise", False))
                    task_desc = str(req.get("task_description") or default_instruction)

                    off = 0
                    main_sz = int(np.prod(main_shape)) * int(main_dtype.itemsize)
                    state_sz = int(np.prod(state_shape)) * int(state_dtype.itemsize)
                    main_np = np.frombuffer(blob[off : off + main_sz], dtype=main_dtype).reshape(
                        main_shape
                    ).copy()
                    off += main_sz
                    state_np = np.frombuffer(
                        blob[off : off + state_sz], dtype=state_dtype
                    ).reshape(state_shape).copy()
                    off += state_sz
                    state_np = np.asarray(state_np, dtype=np.float32)

                    bsz = int(main_np.shape[0])
                    if expected_batch_size is None:
                        expected_batch_size = bsz
                    elif bsz != int(expected_batch_size):
                        raise ValueError(
                            f"dp_predict batch size {bsz} != locked {expected_batch_size}"
                        )
                    prev_imgs = _update_image_history(prev_imgs, main_np)

                    init_noise = None
                    if has_noise:
                        nh = int(req["noise_horizon"])
                        nda = int(req["noise_action_dim"])
                        nb = int(req["noise_batch"])
                        noise_shape = (nb, nh, nda)
                        noise_dtype = np.dtype(req["noise_dtype"])
                        noise_sz = int(np.prod(noise_shape)) * int(noise_dtype.itemsize)
                        init_noise = np.frombuffer(
                            blob[off : off + noise_sz], dtype=noise_dtype
                        ).reshape(noise_shape).copy()

                    # Reset slots that were marked by selective reset.
                    for ii in range(len(prev_imgs)):
                        if prev_imgs[ii]["main"] is None:
                            prev_imgs[ii]["main"] = main_np[ii].copy()

                    with _INFER_LOCK, torch.inference_mode():
                        actions, prev_imgs = _predict_actions(
                            bundle=bundle,
                            main_np=main_np,
                            state_np=state_np,
                            prev_imgs=prev_imgs,
                            task_desc=task_desc,
                            init_noise=init_noise,
                            n_action_steps=n_action_steps,
                        )
                    raw = np.ascontiguousarray(actions).tobytes()
                    _ok_reply(
                        conn,
                        req,
                        {
                            "action_shape": list(actions.shape),
                            "action_dtype": str(actions.dtype),
                        },
                        raw,
                    )
                    if log_ops:
                        print(
                            f"[robotwin_rdt_server] op=dp_predict ok B={bsz} "
                            f"main_shape={main_shape} state_shape={state_shape} "
                            f"has_init_noise={has_noise} action_shape={list(actions.shape)}",
                            flush=True,
                        )
                    continue

                if op == "close":
                    bundle = None
                    expected_batch_size = None
                    prev_imgs = None
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
                print(
                    f"[robotwin_rdt_server] op={op} error: {e}\n{traceback.format_exc()}",
                    flush=True,
                )
                _error_reply(conn, req, e)
    finally:
        try:
            conn.close()
        except OSError:
            pass
        print(f"[robotwin_rdt_server] client {addr} disconnected", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="RoboTwin RDT TCP inference server for RLinf remote expert"
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "policy" / "RDT" / "configs" / "base.yaml"),
    )
    p.add_argument(
        "--vision-encoder-path",
        type=str,
        default=str(ROOT / "weights" / "RDT" / "siglip-so400m-patch14-384"),
    )
    p.add_argument(
        "--text-encoder-path",
        type=str,
        default=str(ROOT / "weights" / "RDT" / "t5-v1_1-xxl"),
    )
    p.add_argument(
        "--lora-adapter-path",
        type=str,
        default=None,
        help="Optional PEFT LoRA adapter directory to mount on top of the RDT policy.",
    )
    p.add_argument(
        "--merge-lora",
        action="store_true",
        help="Merge the LoRA adapter into the policy after loading for inference.",
    )
    p.add_argument(
        "--instruction",
        type=str,
        default="Place the empty cup to the target area.",
    )
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=8,
        help="Execute only the first N actions from the predicted 64-step chunk.",
    )
    p.add_argument("--control-frequency", type=int, default=25)
    p.add_argument("--no-log-ops", action="store_true")
    args = p.parse_args()

    listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen.bind((args.host, args.port))
    listen.listen(8)
    print(
        f"[robotwin_rdt_server] listening {args.host}:{args.port} default_ckpt={args.ckpt!r} "
        f"device={args.device} api_version={API_VERSION} n_action_steps={args.n_action_steps}",
        flush=True,
    )

    while True:
        conn, addr = listen.accept()
        print(f"[robotwin_rdt_server] client connected from {addr}", flush=True)
        conn.settimeout(None)
        threading.Thread(
            target=handle_client,
            args=(conn, addr),
            kwargs={
                "default_ckpt": args.ckpt,
                "config_path": args.config,
                "vision_encoder_path": args.vision_encoder_path,
                "text_encoder_path": args.text_encoder_path,
                "default_lora_adapter_path": args.lora_adapter_path,
                "merge_lora": bool(args.merge_lora),
                "default_instruction": args.instruction,
                "n_action_steps": int(args.n_action_steps),
                "control_frequency": int(args.control_frequency),
                "map_location": args.device,
                "log_ops": not args.no_log_ops,
            },
            daemon=True,
        ).start()


if __name__ == "__main__":
    main()
