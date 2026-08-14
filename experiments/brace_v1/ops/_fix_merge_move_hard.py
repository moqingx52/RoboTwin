#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

REPO = Path("/workspace/RoboTwin")
MARKER = "policy/DP/checkpoints/"


def norm_ckpt(v: str) -> str:
    if MARKER in v:
        return MARKER + v.split(MARKER, 1)[1]
    return v


def walk(obj, changed: list[bool]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "ckpt_path" and isinstance(v, str):
                nv = norm_ckpt(v)
                if nv != v:
                    obj[k] = nv
                    changed[0] = True
            else:
                walk(v, changed)
    elif isinstance(obj, list):
        for x in obj:
            walk(x, changed)


def main() -> None:
    run = REPO / "experiments/brace/runs/base200_frozen_eval_20260811T012027Z/results/move_can_pot"
    for p in sorted(run.glob("base200_confirm_hard_shard_*.json")):
        d = json.loads(p.read_text())
        changed = [False]
        walk(d, changed)
        if changed[0]:
            p.write_text(json.dumps(d, indent=2) + "\n")
            print("fixed", p.name)
        else:
            print("noop", p.name)
    subprocess.check_call(
        [
            "/root/miniconda/envs/RoboTwin/bin/python",
            "experiments/phase1/merge_eval_shards.py",
            "--task",
            "move_can_pot",
            "--variant",
            "base200_confirm_hard",
            "--output-dir",
            "experiments/brace/runs/base200_frozen_eval_20260811T012027Z/results",
            "--num-shards",
            "3",
        ],
        cwd=str(REPO),
    )
    print("merged_ok")


if __name__ == "__main__":
    main()
