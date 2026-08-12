#!/bin/bash
# Minimal Base200 frozen characterization eval.
#
# Evaluates each ready Base200 DP checkpoint on train-disjoint confirm_easy
# (demo_clean) and confirm_hard (demo_randomized) partitions from the
# multitask_v1 seed manifests. Seeds are verified disjoint from Base200 train
# seed.txt. Packing: one logical eval job per GPU, 3 shards via run_eval_group.
#
# Does not freeze BRACE-v2 method and does not launch held-out continued training.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

source /root/miniconda/etc/profile.d/conda.sh
conda activate RoboTwin
export PYTHONPATH="${repo_root}:${repo_root}/policy/DP${PYTHONPATH:+:${PYTHONPATH}}"

protocol=${BRACE_MULTITASK_V2_PROTOCOL:-experiments/brace/multitask_protocol.v2.json}
seed_manifest_dir=${BRACE_BASE200_EVAL_SEED_DIR:-experiments/brace/seeds/multitask_v1}
workers_per_gpu=${BRACE_EVAL_WORKERS_PER_GPU:-3}
read -r -a gpu_ids <<< "${BRACE_GPU_IDS:-2 4 5 6 7}"
read -r -a tasks <<< "${BRACE_BASE200_EVAL_TASKS:-place_container_plate beat_block_hammer click_alarmclock handover_mic lift_pot move_can_pot open_laptop put_object_cabinet}"

run_dir=${BRACE_BASE200_EVAL_RUN_DIR:-experiments/brace/runs/base200_frozen_eval_$(date -u +%Y%m%dT%H%M%SZ)}
log_root=${BRACE_BASE200_EVAL_LOG_DIR:-experiments/brace/logs/base200_frozen_eval}
mkdir -p "${run_dir}/seeds" "${run_dir}/results" "${log_root}"
run_id="$(basename "${run_dir}")"
manifest_log="${log_root}/${run_id}_manifest.log"
git_commit="$(git rev-parse HEAD)"

python - "${protocol}" "${seed_manifest_dir}" "${run_dir}" "${tasks[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path

protocol_path = Path(sys.argv[1])
seed_dir = Path(sys.argv[2])
run_dir = Path(sys.argv[3])
tasks = sys.argv[4:]
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
if protocol.get("protocol_revision") != "brace.multitask.v2":
    raise SystemExit("expected multitask protocol v2")
protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
jobs = []
for task in tasks:
    ckpt = Path(f"policy/DP/checkpoints/{task}-demo_clean-200-0/600.ckpt")
    prov = ckpt.with_name("brace_base200_provenance.json")
    if not ckpt.is_file() or not prov.is_file():
        raise SystemExit(f"missing Base200 ckpt+provenance for {task}")
    prov_payload = json.loads(prov.read_text(encoding="utf-8"))
    if prov_payload.get("protocol_sha256") != protocol_sha:
        raise SystemExit(f"provenance protocol hash mismatch for {task}")
    train_seeds = {int(x) for x in Path(f"data/{task}/demo_clean/seed.txt").read_text().split()}
    if len(train_seeds) != 200:
        raise SystemExit(f"{task}: expected 200 train seeds, got {len(train_seeds)}")
    manifest = json.loads((seed_dir / f"{task}.json").read_text(encoding="utf-8"))
    parts = manifest["partitions"]
    easy = [int(x) for x in parts["confirm_easy"]]
    hard = [int(x) for x in parts["confirm_hard"]]
    if len(easy) != 100 or len(hard) != 100:
        raise SystemExit(f"{task}: expected 100/100 confirm partitions")
    for name, seeds in (("confirm_easy", easy), ("confirm_hard", hard)):
        overlap = sorted(set(seeds) & train_seeds)
        if overlap:
            raise SystemExit(f"{task}: {name} overlaps Base200 train seeds: {overlap[:5]}")
    seed_easy = {
        "task": task,
        "source_manifest": str(seed_dir / f"{task}.json"),
        "source_manifest_status": manifest.get("status"),
        "eval_id": easy,
        "train_rollout": [],
        "split": "confirm_easy",
        "task_config": "demo_clean",
    }
    seed_hard = {
        "task": task,
        "source_manifest": str(seed_dir / f"{task}.json"),
        "source_manifest_status": manifest.get("status"),
        "eval_id": hard,
        "train_rollout": [],
        "split": "confirm_hard",
        "task_config": "demo_randomized",
    }
    easy_path = run_dir / "seeds" / f"{task}_confirm_easy.json"
    hard_path = run_dir / "seeds" / f"{task}_confirm_hard.json"
    easy_path.write_text(json.dumps(seed_easy, indent=2) + "\n", encoding="utf-8")
    hard_path.write_text(json.dumps(seed_hard, indent=2) + "\n", encoding="utf-8")
    # place first: emit place jobs before others
    for split, path, cfg, variant in (
        ("confirm_easy", easy_path, "demo_clean", "base200_confirm_easy"),
        ("confirm_hard", hard_path, "demo_randomized", "base200_confirm_hard"),
    ):
        jobs.append(
            {
                "task": task,
                "split": split,
                "seeds_file": str(path),
                "task_config": cfg,
                "variant": variant,
                "ckpt": str(ckpt),
                "priority": 0 if task == "place_container_plate" else 1,
            }
        )
jobs.sort(key=lambda j: (j["priority"], 0 if j["split"] == "confirm_easy" else 1, j["task"]))
(run_dir / "jobs.json").write_text(json.dumps({"protocol": str(protocol_path), "protocol_sha256": protocol_sha, "jobs": jobs}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"jobs": len(jobs), "tasks": tasks}, indent=2))
PY

echo "base200_frozen_eval start run_id=${run_id} gpus=${gpu_ids[*]} workers_per_gpu=${workers_per_gpu} commit=${git_commit}" | tee "${manifest_log}"

mapfile -t job_lines < <(python - "${run_dir}/jobs.json" <<'PY'
import json, sys
for job in json.load(open(sys.argv[1], encoding="utf-8"))["jobs"]:
    print("|".join([job["task"], job["split"], job["seeds_file"], job["task_config"], job["variant"], job["ckpt"]]))
PY
)

launch_eval_job() {
  local task=$1 split=$2 seeds_file=$3 task_config=$4 variant=$5 ckpt=$6 gpu=$7
  local log="${log_root}/${run_id}_${task}_${split}.log"
  {
    echo "LAUNCH ${task} ${split} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    python experiments/brace/run_eval_group.py --workers "${workers_per_gpu}" -- \
      python experiments/phase1/eval_per_seed.py \
        --task "${task}" \
        --task-config "${task_config}" \
        --variant "${variant}" \
        --ckpt-path "${ckpt}" \
        --seeds-file "${seeds_file}" \
        --output-dir "${run_dir}/results" \
        --id-repeats 1 \
        --train-seed-count 0 \
        --no-include-hard \
        --resume
  } >"${log}" 2>&1
}

job_complete() {
  local task=$1 variant=$2
  local path="${run_dir}/results/${task}/${variant}.json"
  [[ -f "${path}" ]] || return 1
  python - "${path}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if (payload.get("progress") or {}).get("complete") else 1)
PY
}

idx=0
failures=0
completed=0
skipped=0
while (( idx < ${#job_lines[@]} )); do
  pids=()
  names=()
  gpus_used=()
  for gpu in "${gpu_ids[@]}"; do
    while (( idx < ${#job_lines[@]} )); do
      IFS='|' read -r task split seeds_file task_config variant ckpt <<< "${job_lines[idx]}"
      idx=$((idx + 1))
      if job_complete "${task}" "${variant}"; then
        echo "SKIP ${task}:${split} (already complete)" | tee -a "${manifest_log}"
        skipped=$((skipped + 1))
        completed=$((completed + 1))
        continue
      fi
      echo "LAUNCH ${task} ${split} gpu=${gpu}" | tee -a "${manifest_log}"
      launch_eval_job "${task}" "${split}" "${seeds_file}" "${task_config}" "${variant}" "${ckpt}" "${gpu}" &
      pids+=("$!")
      names+=("${task}:${split}")
      gpus_used+=("${gpu}")
      break
    done
  done
  ((${#pids[@]})) || break
  for i in "${!pids[@]}"; do
    if wait "${pids[i]}"; then
      echo "OK ${names[i]}" | tee -a "${manifest_log}"
      completed=$((completed + 1))
    else
      echo "FAILED ${names[i]} gpu=${gpus_used[i]}" | tee -a "${manifest_log}"
      failures=$((failures + 1))
    fi
  done
done

python - "${run_dir}" "${git_commit}" "${failures}" "${completed}" <<'PY'
import json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
jobs = json.loads((run_dir / "jobs.json").read_text(encoding="utf-8"))["jobs"]
rows = []
for job in jobs:
    path = run_dir / "results" / job["task"] / f"{job['variant']}.json"
    row = {"task": job["task"], "split": job["split"], "variant": job["variant"], "complete": False, "success_rate": None, "path": str(path)}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        progress = payload.get("progress") or {}
        row["complete"] = bool(progress.get("complete"))
        splits = payload.get("splits") or {}
        heldout = splits.get("id_heldout") or {}
        row["success_rate"] = heldout.get("mean_sr")
        row["episodes"] = heldout.get("episodes")
        row["solved_coverage"] = heldout.get("solved_coverage")
        rows.append(row)
    else:
        rows.append(row)
summary = {
    "schema_version": 1,
    "stage": "base200_frozen_eval",
    "git_commit": sys.argv[2],
    "failures": int(sys.argv[3]),
    "completed_ok": int(sys.argv[4]),
    "tasks": rows,
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"summary": str(run_dir / "summary.json"), "failures": summary["failures"]}, indent=2))
PY

echo "base200_frozen_eval done failures=${failures} run_dir=${run_dir}" | tee -a "${manifest_log}"
exit "${failures}"
