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

try:
    import cv2
except ImportError:  # pragma: no cover - RoboTwin envs normally include cv2.
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_VERSION = 1

_MODEL_CACHE: dict[str, dict[str, Any]] = {}
_MODEL_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()


def _resolve_ckpt_path(ckpt_path: str) -> str:
    p = Path(ckpt_path)
    if p.is_dir():
        fsdp_ckpt = p / "pytorch_model" / "mp_rank_00_model_states.pt"
        if fsdp_ckpt.is_file():
            return str(fsdp_ckpt)
    return str(p)


def _resolve_config_path(config_path: Optional[str], ckpt_path: str) -> str:
    if config_path:
        return config_path
    hint = ckpt_path.lower()
    if "170m" in hint or "rdt-170m" in hint:
        return str(ROOT / "policy" / "RDT" / "configs" / "base_170m.yaml")
    return str(ROOT / "policy" / "RDT" / "configs" / "base.yaml")


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
    rdt_root = ROOT / "policy" / "RDT"
    rdt_models = rdt_root / "models"
    rdt_scripts = rdt_root / "scripts"
    for module_path in (rdt_root, rdt_models, rdt_scripts):
        module_path_str = str(module_path)
        if module_path_str not in sys.path:
            sys.path.insert(0, module_path_str)

    from multimodal_encoder.t5_encoder import T5Embedder
    from agilex_model import create_model

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
    prev_imgs: Optional[list[dict[str, Optional[np.ndarray]]]],
    main_np: np.ndarray,
    left_wrist_np: Optional[np.ndarray] = None,
    right_wrist_np: Optional[np.ndarray] = None,
) -> list[dict[str, Optional[np.ndarray]]]:
    bsz = int(main_np.shape[0])
    if prev_imgs is None or len(prev_imgs) != bsz:
        prev_imgs = []
        for i in range(bsz):
            # Match policy/RDT/model.py warmup: the first observation window is
            # [dummy(None), current], not [current, current].
            prev_imgs.append(
                {
                    "prev_main": None,
                    "prev_left_wrist": None,
                    "prev_right_wrist": None,
                    "main": None,
                    "left_wrist": None,
                    "right_wrist": None,
                }
            )
    return prev_imgs


def _preprocess_rdt_image(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Match policy/RDT/model.py update_observation_window preprocessing."""
    if img is None:
        return None
    arr = np.asarray(img)
    if cv2 is None:
        pil = Image.fromarray(arr)
        return np.asarray(pil.resize((640, 480)))
    arr = cv2.resize(arr, (640, 480))
    ok, encoded = cv2.imencode(".jpg", arr)
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg', image) failed")
    return cv2.imdecode(np.frombuffer(encoded.tobytes(), np.uint8), cv2.IMREAD_COLOR)


def _assert_wrist_images_are_real(
    main_np: np.ndarray,
    left_wrist_np: Optional[np.ndarray],
    right_wrist_np: Optional[np.ndarray],
    *,
    context: str,
    min_mean_abs_diff: float = 1.0,
) -> None:
    if left_wrist_np is None or right_wrist_np is None:
        raise ValueError(
            f"{context} requires left/right wrist images. RDT-170M uses "
            "head + right wrist + left wrist views; refusing to duplicate "
            "the head camera into wrist slots."
        )
    if left_wrist_np.shape != main_np.shape or right_wrist_np.shape != main_np.shape:
        raise ValueError(
            f"{context} wrist image shapes must match main image shape: "
            f"main={main_np.shape}, left={left_wrist_np.shape}, right={right_wrist_np.shape}"
        )
    if main_np.shape[0] != left_wrist_np.shape[0] or main_np.shape[0] != right_wrist_np.shape[0]:
        raise ValueError(f"{context} image batch size mismatch")
    main_i = main_np.astype(np.int16, copy=False)
    left_i = left_wrist_np.astype(np.int16, copy=False)
    right_i = right_wrist_np.astype(np.int16, copy=False)
    left_diff = np.abs(main_i - left_i).reshape(main_np.shape[0], -1).mean(axis=1)
    right_diff = np.abs(main_i - right_i).reshape(main_np.shape[0], -1).mean(axis=1)
    bad_left = np.where(left_diff < min_mean_abs_diff)[0]
    bad_right = np.where(right_diff < min_mean_abs_diff)[0]
    if bad_left.size or bad_right.size:
        raise ValueError(
            f"{context} wrist images are nearly identical to the head camera "
            f"(left_bad={bad_left.tolist()}, right_bad={bad_right.tolist()}, "
            f"left_diff={left_diff.tolist()}, right_diff={right_diff.tolist()}). "
            "This usually means wrist rendering is disabled or head frames were copied."
        )


def _predict_actions(
    *,
    bundle: dict[str, Any],
    main_np: np.ndarray,
    state_np: np.ndarray,
    prev_imgs: list[dict[str, Optional[np.ndarray]]],
    task_desc: str | list[str],
    init_noise: Optional[np.ndarray],
    n_action_steps: int,
    left_wrist_np: Optional[np.ndarray] = None,
    right_wrist_np: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, list[dict[str, Optional[np.ndarray]]]]:
    model = bundle["model"]
    pred_horizon = int(bundle["pred_horizon"])
    per_env_actions: list[np.ndarray] = []
    _assert_wrist_images_are_real(
        main_np,
        left_wrist_np,
        right_wrist_np,
        context="dp_predict",
    )
    assert left_wrist_np is not None
    assert right_wrist_np is not None

    for i in range(main_np.shape[0]):
        curr_main = _preprocess_rdt_image(main_np[i])
        curr_left = _preprocess_rdt_image(left_wrist_np[i])
        curr_right = _preprocess_rdt_image(right_wrist_np[i])
        cold_start = prev_imgs[i]["main"] is None
        # The smoke/eval client can sync the true last two observations after
        # each executed chunk. Without that sync, fall back to the last observed
        # frame so older clients keep working.
        prev_main = None if cold_start else prev_imgs[i].get("prev_main")
        prev_left = None if cold_start else prev_imgs[i].get("prev_left_wrist")
        prev_right = None if cold_start else prev_imgs[i].get("prev_right_wrist")
        if not cold_start:
            if prev_main is None:
                prev_main = prev_imgs[i]["main"]
            if prev_left is None:
                prev_left = prev_imgs[i]["left_wrist"]
            if prev_right is None:
                prev_right = prev_imgs[i]["right_wrist"]
        imgs = [
            Image.fromarray(x) if x is not None else None
            for x in (
                prev_main,
                prev_right,
                prev_left,
                curr_main,
                curr_right,
                curr_left,
            )
        ]
        proprio = torch.from_numpy(state_np[i : i + 1]).to(
            device=model.device, dtype=torch.float32
        )
        if isinstance(task_desc, list):
            desc_i = task_desc[i] if i < len(task_desc) else ""
        else:
            desc_i = task_desc
        text_embeds = _encode_instruction(bundle, desc_i)
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
        prev_imgs[i]["prev_main"] = curr_main.copy()
        prev_imgs[i]["prev_left_wrist"] = curr_left.copy()
        prev_imgs[i]["prev_right_wrist"] = curr_right.copy()
        prev_imgs[i]["main"] = curr_main.copy()
        prev_imgs[i]["left_wrist"] = curr_left.copy()
        prev_imgs[i]["right_wrist"] = curr_right.copy()

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
    prev_imgs: Optional[list[dict[str, Optional[np.ndarray]]]] = None
    needs_history_sync = False
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
                    resolved_ckpt = _resolve_ckpt_path(str(ckpt))
                    resolved_config = _resolve_config_path(config_path, resolved_ckpt)
                    lora_adapter_path = req.get("lora_adapter_path") or default_lora_adapter_path
                    bundle = _get_cached_model(
                        ckpt_path=resolved_ckpt,
                        config_path=resolved_config,
                        vision_encoder_path=vision_encoder_path,
                        text_encoder_path=text_encoder_path,
                        lora_adapter_path=lora_adapter_path,
                        merge_lora=merge_lora,
                        control_frequency=control_frequency,
                        device=dev,
                    )
                    expected_batch_size = None
                    prev_imgs = None
                    needs_history_sync = False
                    meta_out = {
                        "n_obs_steps": int(bundle["n_obs_steps"]),
                        "horizon": int(bundle["pred_horizon"]),
                        "action_dim": int(bundle["action_dim"]),
                        "n_action_steps": int(n_action_steps),
                        "ckpt_path": ckpt,
                        "resolved_ckpt_path": resolved_ckpt,
                        "config_path": resolved_config,
                        "lora_adapter_path": str(bundle.get("lora_adapter_path", "")),
                        "lora_loaded": bool(bundle.get("lora_loaded", False)),
                        "lora_merged": bool(bundle.get("lora_merged", False)),
                    }
                    _ok_reply(conn, req, {"dp_meta": meta_out})
                    if log_ops:
                        print(
                            f"[robotwin_rdt_server] op=init ok ckpt={ckpt} "
                            f"resolved_ckpt={resolved_ckpt} config={resolved_config} "
                            f"device={dev} meta={meta_out}",
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
                        needs_history_sync = False
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
                            prev_imgs[ii]["prev_main"] = None  # type: ignore[index]
                            prev_imgs[ii]["prev_left_wrist"] = None  # type: ignore[index]
                            prev_imgs[ii]["prev_right_wrist"] = None  # type: ignore[index]
                            prev_imgs[ii]["main"] = None  # type: ignore[index]
                            prev_imgs[ii]["left_wrist"] = None  # type: ignore[index]
                            prev_imgs[ii]["right_wrist"] = None  # type: ignore[index]
                    _ok_reply(
                        conn,
                        req,
                        {
                            "ack_batch_size": bsz,
                            "selective": True,
                            "cleared_slots": int(mask.sum()),
                        },
                    )
                    needs_history_sync = False
                    continue

                if op == "dp_set_image_history":
                    main_shape = tuple(req["main_shape"])
                    left_wrist_shape = tuple(req["left_wrist_shape"])
                    right_wrist_shape = tuple(req["right_wrist_shape"])
                    main_dtype = np.dtype(req["main_dtype"])
                    left_wrist_dtype = np.dtype(req["left_wrist_dtype"])
                    right_wrist_dtype = np.dtype(req["right_wrist_dtype"])
                    if len(main_shape) != 5:
                        raise ValueError(
                            f"main history must be shaped (B,Hn,H,W,C), got {main_shape}"
                        )
                    if left_wrist_shape != main_shape or right_wrist_shape != main_shape:
                        raise ValueError(
                            "wrist history shapes must match main history shape"
                        )

                    off = 0
                    main_sz = int(np.prod(main_shape)) * int(main_dtype.itemsize)
                    left_sz = int(np.prod(left_wrist_shape)) * int(left_wrist_dtype.itemsize)
                    right_sz = int(np.prod(right_wrist_shape)) * int(
                        right_wrist_dtype.itemsize
                    )
                    main_np = np.frombuffer(
                        blob[off : off + main_sz], dtype=main_dtype
                    ).reshape(main_shape).copy()
                    off += main_sz
                    left_np = np.frombuffer(
                        blob[off : off + left_sz], dtype=left_wrist_dtype
                    ).reshape(left_wrist_shape).copy()
                    off += left_sz
                    right_np = np.frombuffer(
                        blob[off : off + right_sz], dtype=right_wrist_dtype
                    ).reshape(right_wrist_shape).copy()
                    _assert_wrist_images_are_real(
                        main_np.reshape((-1,) + main_np.shape[2:]),
                        left_np.reshape((-1,) + left_np.shape[2:]),
                        right_np.reshape((-1,) + right_np.shape[2:]),
                        context="dp_set_image_history",
                    )

                    bsz = int(main_np.shape[0])
                    hist_len = int(main_np.shape[1])
                    if hist_len < 1:
                        raise ValueError("image history length must be >= 1")
                    if expected_batch_size is None:
                        expected_batch_size = bsz
                    elif bsz != int(expected_batch_size):
                        raise ValueError(
                            f"dp_set_image_history batch size {bsz} != locked {expected_batch_size}"
                        )
                    prev_imgs = _update_image_history(prev_imgs, main_np[:, -1])
                    for i in range(bsz):
                        prev_i = max(0, hist_len - 2)
                        curr_i = hist_len - 1
                        prev_imgs[i]["prev_main"] = _preprocess_rdt_image(
                            main_np[i, prev_i]
                        )
                        prev_imgs[i]["prev_left_wrist"] = _preprocess_rdt_image(
                            left_np[i, prev_i]
                        )
                        prev_imgs[i]["prev_right_wrist"] = _preprocess_rdt_image(
                            right_np[i, prev_i]
                        )
                        prev_imgs[i]["main"] = _preprocess_rdt_image(main_np[i, curr_i])
                        prev_imgs[i]["left_wrist"] = _preprocess_rdt_image(
                            left_np[i, curr_i]
                        )
                        prev_imgs[i]["right_wrist"] = _preprocess_rdt_image(
                            right_np[i, curr_i]
                        )
                    _ok_reply(
                        conn,
                        req,
                        {
                            "ack_batch_size": bsz,
                            "history_len": hist_len,
                        },
                    )
                    needs_history_sync = False
                    if log_ops:
                        print(
                            f"[robotwin_rdt_server] op=dp_set_image_history ok "
                            f"B={bsz} history_len={hist_len}",
                            flush=True,
                        )
                    continue

                if op == "dp_predict":
                    if needs_history_sync:
                        raise RuntimeError(
                            "dp_predict called again before dp_set_image_history. "
                            "Remote RDT executes action chunks in the env, so the "
                            "server must receive the true post-chunk image history "
                            "before the next prediction."
                        )
                    main_shape = tuple(req["main_shape"])
                    state_shape = tuple(req["state_shape"])
                    main_dtype = np.dtype(req["main_dtype"])
                    state_dtype = np.dtype(req["state_dtype"])
                    has_noise = bool(req.get("has_init_noise", False))
                    task_descs = req.get("task_descriptions")
                    if isinstance(task_descs, list):
                        task_desc = [str(x) for x in task_descs]
                    else:
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

                    left_wrist_np = None
                    right_wrist_np = None
                    if bool(req.get("has_wrist_images", False)):
                        left_wrist_shape = tuple(req["left_wrist_shape"])
                        left_wrist_dtype = np.dtype(req["left_wrist_dtype"])
                        left_wrist_sz = int(np.prod(left_wrist_shape)) * int(
                            left_wrist_dtype.itemsize
                        )
                        left_wrist_np = np.frombuffer(
                            blob[off : off + left_wrist_sz], dtype=left_wrist_dtype
                        ).reshape(left_wrist_shape).copy()
                        off += left_wrist_sz

                        right_wrist_shape = tuple(req["right_wrist_shape"])
                        right_wrist_dtype = np.dtype(req["right_wrist_dtype"])
                        right_wrist_sz = int(np.prod(right_wrist_shape)) * int(
                            right_wrist_dtype.itemsize
                        )
                        right_wrist_np = np.frombuffer(
                            blob[off : off + right_wrist_sz], dtype=right_wrist_dtype
                        ).reshape(right_wrist_shape).copy()
                        off += right_wrist_sz
                    else:
                        raise ValueError(
                            "dp_predict requires left/right wrist images for RDT-170M. "
                            "Start RoboTwin with camera.collect_wrist_camera=true and "
                            "send left_wrist_images/right_wrist_images from the client."
                        )

                    bsz = int(main_np.shape[0])
                    if expected_batch_size is None:
                        expected_batch_size = bsz
                    elif bsz != int(expected_batch_size):
                        raise ValueError(
                            f"dp_predict batch size {bsz} != locked {expected_batch_size}"
                        )
                    prev_imgs = _update_image_history(
                        prev_imgs,
                        main_np,
                        left_wrist_np=left_wrist_np,
                        right_wrist_np=right_wrist_np,
                    )

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

                    # Reset slots that were marked by selective reset. Leave the
                    # previous frame as None for this prediction so reset matches
                    # the official RDT warmup window [dummy(None), current].
                    for ii in range(len(prev_imgs)):
                        if prev_imgs[ii]["main"] is None:
                            prev_imgs[ii]["left_wrist"] = None
                            prev_imgs[ii]["right_wrist"] = None

                    with _INFER_LOCK, torch.inference_mode():
                        actions, prev_imgs = _predict_actions(
                            bundle=bundle,
                            main_np=main_np,
                            state_np=state_np,
                            prev_imgs=prev_imgs,
                            task_desc=task_desc,
                            init_noise=init_noise,
                            n_action_steps=n_action_steps,
                            left_wrist_np=left_wrist_np,
                            right_wrist_np=right_wrist_np,
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
                    needs_history_sync = True
                    if log_ops:
                        print(
                            f"[robotwin_rdt_server] op=dp_predict ok B={bsz} "
                            f"main_shape={main_shape} state_shape={state_shape} "
                            f"has_wrist={left_wrist_np is not None and right_wrist_np is not None} "
                            f"has_init_noise={has_noise} action_shape={list(actions.shape)}",
                            flush=True,
                        )
                    continue

                if op == "close":
                    bundle = None
                    expected_batch_size = None
                    prev_imgs = None
                    needs_history_sync = False
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
        default=None,
        help="RDT base yaml. Defaults to base_170m.yaml when the checkpoint path hints 170M, otherwise base.yaml.",
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
