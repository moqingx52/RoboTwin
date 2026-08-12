#!/usr/bin/env python3
import json, subprocess
from collections import Counter
from pathlib import Path
repo = Path("/workspace/RoboTwin")
st = json.loads((repo / "experiments/brace/runs/base200_place_traced_15shard_20260811.state.json").read_text())
c = Counter(m["status"] for m in st["shards"].values())
print("traced", dict(c), "gate", st.get("place_eval_gate_passed"), "verify", st.get("verify_done"))
for i, m in sorted(st["shards"].items(), key=lambda x: int(x[0])):
    print(f"  shard{i}: {m.get(\"status\")} gpu={m.get(\"gpu\")} pids={m.get(\"pids\")}")
run = repo / "experiments/brace/runs/base200_frozen_eval_20260811T012027Z/results"
for t in sorted(p.name for p in run.iterdir() if p.is_dir()):
    for v in ("base200_confirm_easy", "base200_confirm_hard"):
        p = run / t / f"{v}.json"
        if not p.exists():
            print(f"eval {t}/{v}: missing"); continue
        d = json.loads(p.read_text()); prog = d.get("progress") or {}
        held = ((d.get("splits") or {}).get("id_heldout") or {})
        print(f"eval {t}/{v}: complete={prog.get(\"complete\")} mean_sr={held.get(\"mean_sr\")} eps={held.get(\"episodes\")}")
print("collect_cvd:")
for line in subprocess.check_output(["ps", "-eo", "pid,args"], text=True).splitlines():
    if "collect_traced_rollouts.py" not in line or "place_container_plate" not in line: continue
    pid = int(line.strip().split(None, 1)[0])
    sid = line.split("--shard-id", 1)[1].split(None, 1)[0] if "--shard-id" in line else "?"
    try:
        env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        cvd = next((e.decode() for e in env if e.startswith(b"CUDA_VISIBLE_DEVICES=")), "?")
    except OSError:
        cvd = "?"
    print(f"  pid={pid} shard={sid} {cvd}")
tmp = repo / "experiments/brace/rollouts_traced_base200_v2/place_container_plate"
print("artifacts manifests=%d stats=%d" % (len(list(tmp.glob("manifest_shard_*"))), len(list(tmp.glob("seed_stats_shard_*")))))
print("tmp_eps", len(list((tmp/".tmp").glob("episode_*"))) if (tmp/".tmp").exists() else 0)
