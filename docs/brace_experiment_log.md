# BRACE 实验日志（滚动）

> 最后更新：**2026-08-04**
> 当前 branch 协议：**v2.3**（`experiments/brace/protocol.v2.3.json`）
> 设计文档：`docs/brace_cloud_experiment_plan.md`、`docs/brace_audit_v2_design.md`
> 三轨路线：`docs/brace_track_abc_roadmap.md`

本文件记录 BRACE 各阶段 **Gate 结论与文档索引**；详细分析见各 dated 结论文档。

---

## 里程碑总览

| 日期 | 阶段 | 任务 | Gate | 文档 / 归档 |
|------|------|------|------|-------------|
| 2026-08-04 | Confirmatory v1.4.2 | census feasibility revision | **待跑** P1a→P1b | `screen_protocol.v1.4.2.confirmatory_preservation.json` · `seeds/place_container_plate_confirmatory_v1.4.2_seeds.json` |
| 2026-08-03 | Confirmatory v1.4.1 | P1b preservation cohort | **protocol-feasibility failure**（untouched=42&lt;60） | `runs/20260803T235334Z_select_preservation_cohort_*` · [`brace_v1.4.1_census_feasibility_failure_20260803.md`](brace_v1.4.1_census_feasibility_failure_20260803.md) |
| 2026-08-03 | Confirmatory v1.4.1 | P1a base census | complete（600 ep, offset 3000） | `runs/20260803T144820Z_confirmatory_base_census_*` |
| 2026-08-03 | Phase 3C | anchor behavior eval | **进入确认实验**（非理论完成） | [`brace_phase3c_confirmatory_conclusion_20260803.md`](brace_phase3c_confirmatory_conclusion_20260803.md) · `runs/20260803T073015Z_anchor_behavior_eval_*` |
| 2026-08-03 | Phase 3A | anchor calibration (A0–A7) | 7/8 complete（A4 OOM） | `runs/20260803T031036Z_anchor_calibration_place_container_plate/` |
| 2026-07-27 | Stage 1 v1 replay | both | **NO-GO** (80%) | `archive/replay_audit_v1_no_go/` · `docs/brace_replay_audit_log_20260730.md` |
| 2026-07-30 | Stage 1 v2.0 diagnostic | both | NO-GO (mixed gate) | `archive/replay_audit_v2_mixed_gate_diagnostic/` |
| 2026-07-31 | Stage 1 v2.1 | both | per-task 分化 | `archive/replay_audit_v2_v2.1_gate/` |
| 2026-07-31 | Stage 1 v2.2 + actor diag | dump | garbage_rotation 主导 | `replay_audit_v2_dump_actor_diagnostic/` |
| 2026-08-01 | Stage 1 v2.3 replay | place pilot | **GO** (per-task) | `rollouts_traced_pilot/` |
| 2026-08-01 | Stage 2 branch v2.2 | place | **INVALID** (harness) | `archive/branches_place_pilot_invalid_v2.2/` |
| 2026-08-01 | Stage 2 branch v2.3 | place | **GO** (+62.2 pp) | `archive/branches_place_pilot_valid_v2.3/` · [`brace_stage2_place_pilot_conclusion.md`](brace_stage2_place_pilot_conclusion.md) |
| 2026-08-01 | Stage 1 v2.3 replay | dump-only (A1) | **GO** (57/60 replay) | `archive/replay_audit_v2_dump_v2.3_gate/` |
| 2026-08-01 | Stage 2 branch v2.3 | dump (A2) | **GO** (+36.1 pp) | `branches_dump/` · [`brace_stage2_dump_pilot_conclusion_20260801.md`](brace_stage2_dump_pilot_conclusion_20260801.md) |
| 2026-08-02 | Stage 2 confirm | place held-out | 单批 GO / **合并 NO-GO** | `branches_confirm/` · [`brace_track_bc_confirm_export_20260802.md`](brace_track_bc_confirm_export_20260802.md) |
| 2026-08-02 | Track C | B1/N1 export | **完成** (11+11) | `datasets/place_pilot_v2.3_*` |
| 2026-08-02 | Track C | anchor smoke | 待确认 | `anchor_smoke/summary.json` |
| 2026-08-02 | Screen trainer integration | place | **代码完成 / 未启动 GPU** | `screen_protocol.v1.1.json` · `orchestrate.py` |

---

## 2026-08-01 — A2 dump branch pilot（Track A）

**命令**：`bash experiments/brace/run_dump_pilot.sh`
**Git**：`37e2ddf86c93a5e830556ed487ac91f764782738`

| 指标 | 值 |
|------|-----|
| `harness_valid` | true |
| `passed` | true |
| `matched_success_lift` | +36.1 pp |
| `recovery_seed_fraction` | 100% (4/4 seeds) |
| `accepted_points` | 7 / 12 |
| 可分析 seeds | 4 / 10 pilot seeds |

**解读**：跨任务 feasibility 成立；样本偏紧（4 seed），归档后推进 Track B/C。
**全文**：[`brace_stage2_dump_pilot_conclusion_20260801.md`](brace_stage2_dump_pilot_conclusion_20260801.md)

**运维备注**：首次 A2 因 24 shard / 10 seed 调度导致 GPU0/1 双份负载失败；修复为 `BRACE_PILOT_NUM_SHARDS=8`、`WORKERS_PER_GPU=1` 后重跑成功。

---

## 2026-08-01 — place Stage 2 branch pilot（Track B 基础）

**Git**：`4753527e5b99709e2002f789b4b510f03030d342`

| 指标 | 值 |
|------|-----|
| `matched_success_lift` | +62.2 pp |
| `accepted_points` | 11 / 15 |
| 可分析 seeds | 5 / 10 |

**全文**：[`brace_stage2_place_pilot_conclusion.md`](brace_stage2_place_pilot_conclusion.md)

---

## 2026-08-01 — dump A1 replay audit（Track A 前置）

| 指标 | 值 |
|------|-----|
| restore | 60/60 |
| replay | 57/60 (95%) |
| `replay_gate_passed` | true |
| 失败类型 | deskbin/garbage translation（rotation 不参与 gate） |

**归档**：`experiments/brace/archive/replay_audit_v2_dump_v2.3_gate/`

---

## 2026-08-02 — place confirm + B1/N1 export + 合并 gate

**Git**：`24561055b61704fdd05cf290fc2ecae8593a30c2`

| 项 | 结果 |
|----|------|
| Confirm branch | +29.6 pp，1/5 seed，1/3 accepted |
| Merged gate | **NO-GO**（6 seeds，18 points） |
| B1/N1 export | 11 + 11 chunks |

**全文**：[`brace_track_bc_confirm_export_20260802.md`](brace_track_bc_confirm_export_20260802.md)

---

## 2026-08-03 — Phase 3C anchor behavior eval

**Run：** `experiments/brace/runs/20260803T073015Z_anchor_behavior_eval_place_container_plate_dump_bin_bigbin/`
**Git：** `fdb73b6`
**结论：** 进入确认实验信号；A1@2470 弱 Pareto；anchor 降低 forgetting 未区分
**封存：** `summary.json`（`sealed: true`，`probe_linkage_version: checkpoint_artifacts_v1`）
**全文：** [`brace_phase3c_confirmatory_conclusion_20260803.md`](brace_phase3c_confirmatory_conclusion_20260803.md)

---

## 当前阻塞与下一步

**v1.4.1 P1b 已于 2026-08-03 fail-closed（untouched=42&lt;60）。当前执行协议已升级为 v1.4.2（census 200 + adaptation ID 100）。**

1. **P1a（前置）**：`confirmatory-base-census`（offset **3000**，census_candidate_id + train，无 Hard）
2. **P1b**：`select-preservation-cohort`（整数 2/3 入组，校验 census provenance，`n>=60`）
3. **P1c**：`confirmatory-preservation`（**10** DP jobs：C0/C1 × seeds 1–5，协议 **v1.4.2**）
4. **P1d**：`confirmatory-preservation-eval` + `confirmatory-preservation-report`（offset **4000**，完整 H1 Pareto gate；seeds **v1.4.2**）
5. **P2–P5**：见 [`brace_phase3c_confirmatory_conclusion_20260803.md`](brace_phase3c_confirmatory_conclusion_20260803.md)

**可执行协议：** `screen_protocol.v1.4.2.confirmatory_preservation.json`
**Seeds manifest：** `seeds/place_container_plate_confirmatory_v1.4.2_seeds.json`
**Jobs manifest：** `confirmatory_preservation_jobs.place_container_plate.v3.json`

**Superseded（勿用于新跑）：** v1.4.1 协议、`confirmatory_preservation_jobs.place_container_plate.v2.json`

**暂缓：** 原 v1.4 + 直接 `select-preservation-cohort` / `confirmatory-preservation`（无 census）

---

## 文档索引

| 文档 | 用途 |
|------|------|
| `brace_cloud_experiment_plan.md` | 总体实验设计 |
| `brace_audit_v2_design.md` | Replay audit v2 设计 |
| `brace_replay_audit_log_20260730.md` | Stage 1 v1 详细记录 |
| `brace_stage2_place_pilot_conclusion.md` | Place Stage 2 GO |
| `brace_stage2_dump_pilot_conclusion_20260801.md` | Dump Stage 2 GO |
| `brace_track_bc_confirm_export_20260802.md` | Confirm + export + 合并 gate |
| `brace_track_abc_roadmap.md` | Track A/B/C 操作路线 |
| `brace_phase3c_confirmatory_conclusion_20260803.md` | Phase 3C 结论与确认实验路线图 |
