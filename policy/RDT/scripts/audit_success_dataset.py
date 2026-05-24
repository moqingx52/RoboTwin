import argparse
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - required for real audits.
    h5py = None

try:
    import cv2
except ImportError:  # pragma: no cover - RoboTwin env usually provides cv2.
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - cv2 path is preferred when available.
    Image = None


REQUIRED_IMAGE_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CAMERA_PAIR_KEYS = (
    ("cam_high", "cam_left_wrist"),
    ("cam_high", "cam_right_wrist"),
    ("cam_left_wrist", "cam_right_wrist"),
)


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
    raw = bytes(raw).rstrip(b"\0")
    if cv2 is not None:
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    elif Image is not None:
        try:
            img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception:
            img = None
    else:
        return None, "no JPEG decoder available; install opencv-python or Pillow"
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


def _sample_indices(length: int, sample_frames: int) -> list[int]:
    if length <= 0:
        return []
    n = max(1, min(int(sample_frames), int(length)))
    return sorted({int(x) for x in np.linspace(0, length - 1, n)})


def _image_gray_mean(img: np.ndarray) -> float:
    return float(img.astype(np.float32).mean())


def _decode_sampled_images(dataset, indices: list[int]) -> tuple[list[np.ndarray], list[str]]:
    images = []
    errors = []
    for idx in indices:
        img, err = _decode_image_at(dataset, idx)
        if img is None:
            errors.append(f"frame {idx}: {err}")
        else:
            images.append(img)
    return images, errors


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def _audit_image_health(
    images_group,
    action_len: int,
    *,
    sample_frames: int,
    black_mean_threshold: float,
    max_black_fraction: float,
    min_temporal_mean_abs_diff: float,
    min_camera_mean_abs_diff: float,
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "required_camera_keys": list(REQUIRED_IMAGE_KEYS),
        "sample_indices": _sample_indices(action_len, sample_frames),
        "cameras": {},
        "camera_pair_mean_abs_diff": {},
    }

    decoded_by_key: dict[str, list[np.ndarray]] = {}
    indices = report["sample_indices"]
    for key in REQUIRED_IMAGE_KEYS:
        if key not in images_group:
            errors.append(f"missing observations/images/{key}")
            continue
        dataset = images_group[key]
        if len(dataset) != action_len:
            errors.append(
                f"image length mismatch for {key}: {len(dataset)} vs {action_len}"
            )
        ok, err = _decode_first_image(dataset)
        if not ok:
            errors.append(f"{key}: {err}")
            continue

        decoded, decode_errors = _decode_sampled_images(dataset, indices)
        if decode_errors:
            errors.extend([f"{key}: {msg}" for msg in decode_errors])
        if not decoded:
            errors.append(f"{key}: no sampled frames decoded")
            continue
        decoded_by_key[key] = decoded

        means = np.asarray([_image_gray_mean(img) for img in decoded], dtype=np.float32)
        black_mask = means < float(black_mean_threshold)
        temporal_diffs = np.asarray(
            [
                _mean_abs_diff(decoded[i - 1], decoded[i])
                for i in range(1, len(decoded))
            ],
            dtype=np.float32,
        )
        camera_report = {
            "sample_count": int(len(decoded)),
            "mean_intensity_min": float(means.min()),
            "mean_intensity_mean": float(means.mean()),
            "mean_intensity_max": float(means.max()),
            "black_frame_count": int(black_mask.sum()),
            "black_frame_fraction": float(black_mask.mean()),
            "temporal_mean_abs_diff_mean": float(temporal_diffs.mean()) if temporal_diffs.size else None,
            "temporal_mean_abs_diff_min": float(temporal_diffs.min()) if temporal_diffs.size else None,
        }
        report["cameras"][key] = camera_report
        if camera_report["black_frame_fraction"] > max_black_fraction:
            errors.append(
                f"{key} has too many near-black sampled frames: "
                f"{camera_report['black_frame_count']}/{camera_report['sample_count']} "
                f"(threshold mean<{black_mean_threshold}, max_fraction={max_black_fraction})"
            )
        if temporal_diffs.size and float(temporal_diffs.mean()) < min_temporal_mean_abs_diff:
            errors.append(
                f"{key} changes too little over sampled frames: "
                f"temporal_mean_abs_diff={float(temporal_diffs.mean()):.4f} "
                f"< {min_temporal_mean_abs_diff}"
            )

    for key_a, key_b in CAMERA_PAIR_KEYS:
        imgs_a = decoded_by_key.get(key_a)
        imgs_b = decoded_by_key.get(key_b)
        if not imgs_a or not imgs_b:
            continue
        pair_count = min(len(imgs_a), len(imgs_b))
        diffs = np.asarray(
            [_mean_abs_diff(imgs_a[i], imgs_b[i]) for i in range(pair_count)],
            dtype=np.float32,
        )
        pair_name = f"{key_a}_vs_{key_b}"
        report["camera_pair_mean_abs_diff"][pair_name] = {
            "mean": float(diffs.mean()),
            "min": float(diffs.min()),
            "max": float(diffs.max()),
            "sample_count": int(pair_count),
        }
        if float(diffs.mean()) < min_camera_mean_abs_diff:
            errors.append(
                f"{key_b} is nearly identical to {key_a} over sampled frames: "
                f"mean_abs_diff={float(diffs.mean()):.4f} < {min_camera_mean_abs_diff}"
            )
        elif float(diffs.min()) < 1.0:
            warnings.append(
                f"{key_b} has at least one sampled frame nearly identical to {key_a}: "
                f"min_abs_diff={float(diffs.min()):.4f}"
            )

    return report, errors, warnings


def _audit_episode(
    path: Path,
    min_steps: int,
    require_lang_embed: bool,
    *,
    image_sample_frames: int,
    black_mean_threshold: float,
    max_black_fraction: float,
    min_temporal_mean_abs_diff: float,
    min_camera_mean_abs_diff: float,
) -> dict[str, Any]:
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

            if "observations/images" not in f:
                result["errors"].append("missing hdf5 key: observations/images")
            else:
                image_health, image_errors, image_warnings = _audit_image_health(
                    f["observations/images"],
                    int(actions.shape[0]),
                    sample_frames=image_sample_frames,
                    black_mean_threshold=black_mean_threshold,
                    max_black_fraction=max_black_fraction,
                    min_temporal_mean_abs_diff=min_temporal_mean_abs_diff,
                    min_camera_mean_abs_diff=min_camera_mean_abs_diff,
                )
                result["image_health"] = image_health
                result["errors"].extend(image_errors)
                result["warnings"].extend(image_warnings)
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
    parser.add_argument("--image-sample-frames", type=int, default=16)
    parser.add_argument("--black-mean-threshold", type=float, default=8.0)
    parser.add_argument("--max-black-fraction", type=float, default=0.20)
    parser.add_argument("--min-temporal-mean-abs-diff", type=float, default=0.05)
    parser.add_argument("--min-camera-mean-abs-diff", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if h5py is None:
        raise SystemExit("audit_success_dataset.py requires h5py to read RDT HDF5 data")
    dataset_dir = args.dataset_dir
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    hdf5_files = sorted(dataset_dir.glob("**/*.hdf5"))
    episodes = [
        _audit_episode(
            path,
            min_steps=args.min_steps,
            require_lang_embed=not args.no_require_lang_embed,
            image_sample_frames=args.image_sample_frames,
            black_mean_threshold=args.black_mean_threshold,
            max_black_fraction=args.max_black_fraction,
            min_temporal_mean_abs_diff=args.min_temporal_mean_abs_diff,
            min_camera_mean_abs_diff=args.min_camera_mean_abs_diff,
        )
        for path in hdf5_files
    ]
    episode_dir_counts = Counter(str(path.parent) for path in hdf5_files)
    duplicate_episode_dirs = {
        episode_dir: count
        for episode_dir, count in episode_dir_counts.items()
        if count > 1
    }
    if duplicate_episode_dirs:
        for ep in episodes:
            count = duplicate_episode_dirs.get(ep["episode_dir"])
            if count is not None:
                ep["errors"].append(
                    f"episode directory contains {count} HDF5 files; expected exactly one"
                )
                ep["ok"] = False
    ok_episodes = [ep for ep in episodes if ep["ok"]]
    total_steps = sum(int(ep.get("num_steps", 0)) for ep in ok_episodes)
    summary = {
        "dataset_dir": str(dataset_dir),
        "episode_count": len(episodes),
        "ok_episode_count": len(ok_episodes),
        "bad_episode_count": len(episodes) - len(ok_episodes),
        "duplicate_episode_dirs": duplicate_episode_dirs,
        "total_ok_steps": int(total_steps),
        "min_episodes_required": int(args.min_episodes),
        "min_steps_required": int(args.min_steps),
        "require_lang_embed": not args.no_require_lang_embed,
        "image_checks": {
            "required_camera_keys": list(REQUIRED_IMAGE_KEYS),
            "image_sample_frames": int(args.image_sample_frames),
            "black_mean_threshold": float(args.black_mean_threshold),
            "max_black_fraction": float(args.max_black_fraction),
            "min_temporal_mean_abs_diff": float(args.min_temporal_mean_abs_diff),
            "min_camera_mean_abs_diff": float(args.min_camera_mean_abs_diff),
        },
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
