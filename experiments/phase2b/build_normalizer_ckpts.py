#!/usr/bin/env python3
"""Build epoch-0 diagnose checkpoints with swapped normalizers."""

import argparse
import copy
import sys
from pathlib import Path

import dill
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DEFAULT_TASKS,
    DIAGNOSE_VARIANTS,
    atomic_write_json,
    base_checkpoint,
    dataset_path,
    diagnose_ckpt_path,
)


def load_payload(ckpt_path):
    return torch.load(open(ckpt_path, "rb"), pickle_module=dill)


def save_payload(payload, ckpt_path):
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt_path.open("wb"), pickle_module=dill)


def fit_normalizer(zarr_path):
    sys.path.append(str(Path(__file__).resolve().parents[2] / "policy" / "DP"))
    from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset

    dataset = RobotImageDataset(
        zarr_path=str(zarr_path),
        horizon=8,
        pad_before=2,
        pad_after=5,
        load_to_memory=False,
    )
    return dataset.get_normalizer()


def replace_policy_normalizer(payload, normalizer):
    payload = copy.deepcopy(payload)
    for key in ("model", "ema_model"):
        if key in payload["state_dicts"]:
            state = payload["state_dicts"][key]
            if "normalizer" in state:
                state["normalizer"] = normalizer.state_dict()
    return payload


def build_for_task(task):
    base_path = base_checkpoint(task)
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    payload = load_payload(base_path)
    outputs = {"A0_base": str(base_path)}

    expert_norm = fit_normalizer(dataset_path(task, "expert_only"))
    mixed_norm = fit_normalizer(dataset_path(task, "success"))

    for variant, normalizer in (
        ("A1_expert_norm", expert_norm),
        ("A2_mixed_norm", mixed_norm),
    ):
        out_path = diagnose_ckpt_path(task, variant)
        swapped = replace_policy_normalizer(payload, normalizer)
        save_payload(swapped, out_path)
        outputs[variant] = str(out_path)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Build diagnose normalizer swap checkpoints.")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "diagnose" / "normalizer_ckpts.json",
    )
    args = parser.parse_args()

    manifest = {"tasks": {}, "variants": list(DIAGNOSE_VARIANTS)}
    for task in args.tasks:
        manifest["tasks"][task] = build_for_task(task)
        print(f"Built diagnose checkpoints for {task}")
    atomic_write_json(args.output, manifest)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
