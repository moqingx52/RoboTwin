import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


REQUIRED_IMAGE_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def _decode_image_at(dataset, index: int = 0) -> tuple[np.ndarray | None, str | None]:
    if len(dataset) == 0:
        return None, "empty image dataset"
    raw = dataset[index]
    if isinstance(raw, str):
        raw = raw.encode("latin1")
    if isinstance(raw, np.bytes_):
        raw = bytes(raw)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "jpeg decode failed"
    if img.ndim != 3 or img.shape[-1] != 3:
        return None, f"decoded image shape is invalid: {img.shape}"
    return img, None


def _decode_first_image(dataset) -> tuple[bool, str | None]:
    img, err = _decode_image_at(dataset, 0)
    if img is None:
        return False, err
    return True, None


def _audit_episode(path: Path, min_steps: int, require_lang_embed: bool) -> dict[str, Any]:
    episode_dir = path.parent
    result: dict[str, Any] = {
        "episode_dir": str(episode_dir),
        "hdf5_path": str(path),
        "ok": True,
        "errors": [],
        "warnings": [],
    }

    instruction_json = episode_dir / "instruction.json"
    if not instruction_json.is_file():
        result["errors"].append("missing instruction.json")
    else:
        try:
            payload = json.loads(instruction_json.read_text(encoding="utf-8"))
            instruction = payload.get("instruction") or payload.get("seen")
            if not instruction:
                result["warnings"].append("instruction.json has no instruction/seen text")
        except Exception as exc:
            result["errors"].append(f"instruction.json is not valid json: {exc}")

    instructions_dir = episode_dir / "instructions"
    lang_embeds = list(instructions_dir.glob("*.pt")) if instructions_dir.is_dir() else []
    if require_lang_embed and not lang_embeds:
        result["errors"].append("missing precomputed instructions/*.pt language embedding")
    result["lang_embed_count"] = len(lang_embeds)

    try:
        with h5py.File(path, "r") as f:
            for key in ("action", "observations/qpos", "observations/left_arm_dim", "observations/right_arm_dim"):
                if key not in f:
                    result["errors"].append(f"missing hdf5 key: {key}")
            if result["errors"]:
                result["ok"] = False
                return result

            actions = np.asarray(f["action"][:], dtype=np.float32)
            qpos = np.asarray(f["observations"]["qpos"][:], dtype=np.float32)
            result["num_steps"] = int(actions.shape[0])
            result["action_shape"] = list(actions.shape)
            result["qpos_shape"] = list(qpos.shape)

            if actions.ndim != 2 or actions.shape[1] != 14:
                result["errors"].append(f"action must be (T,14), got {actions.shape}")
            if qpos.ndim != 2 or qpos.shape[1] != 14:
                result["errors"].append(f"qpos must be (T,14), got {qpos.shape}")
            if actions.shape[0] != qpos.shape[0]:
                result["errors"].append(
                    f"action/qpos length mismatch: {actions.shape[0]} vs {qpos.shape[0]}"
                )
            if actions.shape[0] < min_steps:
                result["errors"].append(f"episode too short: {actions.shape[0]} < {min_steps}")
            if not np.isfinite(actions).all():
                result["errors"].append("action contains NaN/Inf")
            if not np.isfinite(qpos).all():
                result["errors"].append("qpos contains NaN/Inf")

            motion = np.abs(qpos - qpos[:1]).max(axis=1) if qpos.size else np.array([])
            moving_idx = np.where(motion > 1e-2)[0]
            result["first_motion_step"] = int(moving_idx[0]) if moving_idx.size else None
            if not moving_idx.size:
                result["errors"].append("qpos never moves by more than 1e-2")

            result["action_min"] = actions.min(axis=0).tolist()
            result["action_max"] = actions.max(axis=0).tolist()
            gripper = actions[:, [6, 13]] if actions.ndim == 2 and actions.shape[1] >= 14 else np.empty((0, 2))
            if gripper.size:
                result["gripper_min"] = gripper.min(axis=0).tolist()
                result["gripper_max"] = gripper.max(axis=0).tolist()
                if gripper.min() < -1e-4 or gripper.max() > 1.0001:
                    result["warnings"].append("gripper values are outside [0,1]")

            images = f["observations/images"]
            for key in REQUIRED_IMAGE_KEYS:
                if key not in images:
                    result["errors"].append(f"missing observations/images/{key}")
                    continue
                if len(images[key]) != actions.shape[0]:
                    result["errors"].append(
                        f"image length mismatch for {key}: {len(images[key])} vs {actions.shape[0]}"
                    )
                ok, err = _decode_first_image(images[key])
                if not ok:
                    result["errors"].append(f"{key}: {err}")

            head_img, head_err = _decode_image_at(images.get("cam_high", []), 0)
            left_img, left_err = _decode_image_at(images.get("cam_left_wrist", []), 0)
            right_img, right_err = _decode_image_at(images.get("cam_right_wrist", []), 0)
            if head_img is not None and left_img is not None:
                left_diff = float(np.abs(head_img.astype(np.int16) - left_img.astype(np.int16)).mean())
                result["cam_high_vs_left_wrist_mean_abs_diff"] = left_diff
                if left_diff < 1.0:
                    result["errors"].append(
                        "cam_left_wrist is nearly identical to cam_high; wrist vision is likely missing"
                    )
            elif head_err or left_err:
                result["warnings"].append(
                    f"could not compare wrist vs head: head={head_err}, left={left_err}"
                )
            if head_img is not None and right_img is not None:
                right_diff = float(np.abs(head_img.astype(np.int16) - right_img.astype(np.int16)).mean())
                result["cam_high_vs_right_wrist_mean_abs_diff"] = right_diff
                if right_diff < 1.0:
                    result["errors"].append(
                        "cam_right_wrist is nearly identical to cam_high; wrist vision is likely missing"
                    )
    except Exception as exc:
        result["errors"].append(f"hdf5 read failed: {exc}")

    result["ok"] = not result["errors"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit processed RDT success/expert HDF5 data before LoRA training."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--min-episodes", type=int, default=50)
    parser.add_argument("--min-steps", type=int, default=32)
    parser.add_argument("--no-require-lang-embed", action="store_true")
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--max-print-errors", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    hdf5_files = sorted(dataset_dir.glob("**/*.hdf5"))
    episodes = [
        _audit_episode(
            path,
            min_steps=args.min_steps,
            require_lang_embed=not args.no_require_lang_embed,
        )
        for path in hdf5_files
    ]
    ok_episodes = [ep for ep in episodes if ep["ok"]]
    total_steps = sum(int(ep.get("num_steps", 0)) for ep in ok_episodes)
    summary = {
        "dataset_dir": str(dataset_dir),
        "episode_count": len(episodes),
        "ok_episode_count": len(ok_episodes),
        "bad_episode_count": len(episodes) - len(ok_episodes),
        "total_ok_steps": int(total_steps),
        "min_episodes_required": int(args.min_episodes),
        "min_steps_required": int(args.min_steps),
        "require_lang_embed": not args.no_require_lang_embed,
        "episodes": episodes,
    }

    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(
            json.dumps(summary, indent=2, default=_json_default, ensure_ascii=False),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "episodes"},
            indent=2,
            default=_json_default,
            ensure_ascii=False,
        ),
        flush=True,
    )

    printed = 0
    for ep in episodes:
        if ep["ok"]:
            continue
        print(
            json.dumps(
                {
                    "episode_dir": ep["episode_dir"],
                    "errors": ep["errors"],
                    "warnings": ep["warnings"],
                },
                default=_json_default,
                ensure_ascii=False,
            ),
            flush=True,
        )
        printed += 1
        if printed >= args.max_print_errors:
            break

    if len(ok_episodes) < args.min_episodes:
        raise SystemExit(
            f"Only {len(ok_episodes)} valid episodes; need at least {args.min_episodes}."
        )
    if len(ok_episodes) != len(episodes):
        raise SystemExit(f"{len(episodes) - len(ok_episodes)} episodes failed audit.")


if __name__ == "__main__":
    main()
