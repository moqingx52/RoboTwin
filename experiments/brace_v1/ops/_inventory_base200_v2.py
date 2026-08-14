#!/usr/bin/env python3
import json
from pathlib import Path

protocol = json.loads(Path("experiments/brace/multitask_protocol.v2.json").read_text())
sidecar = Path("experiments/brace/multitask_protocol.v2.json.sha256")
protocol_sha = sidecar.read_text().split()[0] if sidecar.exists() else None
task_list = [protocol["development_task"]] + list(protocol["heldout_tasks"])
print("protocol_sha", protocol_sha)
header = "task hdf5 seed acq zarr zprov ckpt cprov"
print(header)
missing = []
for name in task_list:
    hdf5_dir = Path("data") / name / "demo_clean" / "data"
    hdf5 = len(list(hdf5_dir.glob("episode*.hdf5"))) if hdf5_dir.is_dir() else 0
    seed_path = Path("data") / name / "demo_clean" / "seed.txt"
    nseed = len(seed_path.read_text().split()) if seed_path.is_file() else 0
    acq = (Path("data") / name / "demo_clean" / "expert_acquisition_attempts.jsonl").is_file()
    zarr = Path(f"policy/DP/data/{name}-demo_clean-200.zarr")
    zarr_ok = zarr.is_dir()
    zprov = zarr / "brace_base200_provenance.json"
    zprov_ok = False
    if zprov.is_file():
        payload = json.loads(zprov.read_text())
        zprov_ok = payload.get("protocol_sha256") == protocol_sha and payload.get("task") == name
    ckpt_dir = Path(f"policy/DP/checkpoints/{name}-demo_clean-200-0")
    ckpt_ok = (ckpt_dir / "600.ckpt").is_file()
    cprov = ckpt_dir / "brace_base200_provenance.json"
    cprov_ok = False
    if cprov.is_file():
        raw = cprov.read_text().strip()
        if raw:
            payload = json.loads(raw)
            cprov_ok = payload.get("protocol_sha256") == protocol_sha and payload.get("task") == name
    print(
        f"{name:24} {hdf5:4} {nseed:4} {int(acq)} {int(zarr_ok)} {int(zprov_ok)} {int(ckpt_ok)} {int(cprov_ok)}"
    )
    if not (
        hdf5 == 200
        and nseed == 200
        and acq
        and zarr_ok
        and zprov_ok
        and ckpt_ok
        and cprov_ok
    ):
        missing.append(
            {
                "task": name,
                "hdf5": hdf5,
                "seed": nseed,
                "acq": acq,
                "zarr": zarr_ok,
                "zprov": zprov_ok,
                "ckpt": ckpt_ok,
                "cprov": cprov_ok,
            }
        )
print("not_fully_ready", json.dumps(missing, indent=2))
