#!/usr/bin/env python3
"""Single-batch B3 dataloader preflight for place Base200 Line A."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DP = REPO / "policy" / "DP"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(DP) not in sys.path:
    sys.path.insert(0, str(DP))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--train-seed", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_ptr = REPO / "experiments" / "brace/runs/LATEST_place_base200_v2_line_a_pilot"
    run_dir = args.run_dir or Path(run_ptr.read_text(encoding="utf-8").strip())
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir

    import hydra
    from omegaconf import OmegaConf
    from diffusion_policy.dataset.base_dataset import BaseImageDataset
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.workspace.robotworkspace import create_dataloader

    from hydra import compose, initialize_config_dir

    config_dir = str(DP / "diffusion_policy/config")
    overrides = [
        "task.name=place_container_plate",
        f"task.dataset.zarr_path={run_dir / 'datasets/place_container_plate_B1.zarr'}",
        "task.dataset.load_to_memory=False",
        "dataloader.batch_size=128",
        "dataloader.num_workers=0",
        "dataloader.rollout_per_batch=16",
        f"training.seed={args.train_seed}",
        "training.brace_anchor.enabled=true",
        f"training.brace_anchor.dataset.zarr_path={run_dir / 'datasets/place_container_plate_anchor_replay.zarr'}",
    ]
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="robot_dp_14.yaml", overrides=overrides)

    dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    train_loader = create_dataloader(dataset, **cfg.dataloader)
    anchor_zarr = cfg.training.brace_anchor.dataset.zarr_path
    anchor_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.dataset, resolve=True))
    anchor_cfg.zarr_path = anchor_zarr
    anchor_loader_cfg = OmegaConf.select(cfg, "training.brace_anchor.dataloader", default=cfg.dataloader)
    anchor_cfg.batch_size = int(
        OmegaConf.select(anchor_loader_cfg, "batch_size", default=cfg.dataloader.batch_size)
    )
    anchor_dataset = hydra.utils.instantiate(anchor_cfg)
    anchor_loader = create_dataloader(
        anchor_dataset,
        preservation_stratified=True,
        preservation_group_ids={
            str(key): int(value)
            for key, value in dict(
                OmegaConf.select(cfg, "training.brace_anchor.groups", default={"base_solved": 1, "boundary": 2})
            ).items()
        },
        **anchor_loader_cfg,
    )
    main_batch = next(iter(train_loader))
    anchor_batch = next(iter(anchor_loader))
    summary = {
        "passed": True,
        "train_seed": args.train_seed,
        "main_batch_size": int(main_batch["action"].shape[0]),
        "anchor_batch_size": int(anchor_batch["action"].shape[0]),
        "anchor_dataset_batch_size": int(anchor_cfg.batch_size),
        "main_keys": sorted(main_batch.keys()),
        "anchor_keys": sorted(anchor_batch.keys()),
        "anchor_has_preservation_group": "sample_preservation_group" in anchor_batch,
    }
    out = args.output or (
        run_dir / "logs" / f"preflight_b3_dataloader_seed{args.train_seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
