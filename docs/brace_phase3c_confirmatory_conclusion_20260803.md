# BRACE Phase 3C 结论与确认实验路线图（2026-08-03）

**Git commit（behavior eval）：** `fdb73b6`
**Behavior eval run：** `experiments/brace/runs/20260803T073015Z_anchor_behavior_eval_place_container_plate_dump_bin_bigbin/`
**Calibration source：** `experiments/brace/runs/20260803T031036Z_anchor_calibration_place_container_plate/`
**协议（exploratory）：** `experiments/brace/screen_protocol.v1.3.exploratory_calibration.json`

---

## 总论

**Phase 3C 是明确的「进入确认实验」信号，但不是 BRACE 理论验证完成。**

当前最强结果是 **A1@2470** 在单次开发性评估中呈弱 Pareto 优势；**「anchor 降低 forgetting」仍未被区分出来**，因为 SFT-only 本身也没有遗忘。

最合理的论文表述：

> A1@2470 获得了值得确认的 Pareto 信号；functional anchor 的机制路径可运行且稳定，但「降低遗忘」「verified credit 有效」「完整 BRACE 优于基线」三个理论命题仍需分别完成确认实验。

**决策（已冻结）：**

- 不补 A4 warm-start
- 不再根据当前 behavior outcome 调 optimizer
- 冻结 A1 超参数，进入 confirmatory preservation（P1）

---

## 当前结果的正确解读

### 完整 split 表（EMA 部署，单次 eval）

| 方法 | ID | Train | Hard-20 | Base-solved |
|------|---:|------:|--------:|------------:|
| Base | 0.65 | 0.60 | 0.00 | 1.00 |
| SFT-only (A0@1235) | 0.65 | 0.50 | 0.15 | 1.00 |
| A1@1235 | 0.65 | 0.55 | 0.25 | 1.00 |
| **A1@2470** | **0.65** | **0.60** | **0.25** | **1.00** |
| A3 high dual | 0.65 | 0.40 | 0.25 | 1.00 |
| A6 augmented | 0.65 | 0.50 | 0.10 | 1.00 |

### A1@2470 相对 SFT-only

| Split | 结果 |
|-------|------|
| ID | 完全相同，20/20 seed outcome 逐 seed 一致 |
| Train | +10 pp（0.60 vs 0.50），paired 3 胜、1 负 |
| Hard | +10 pp（0.25 vs 0.15），paired 2 胜、0 负 |
| Base-solved | 全部保持，但双方都是 8/8 |

这是很好的**方向性**结果：A1 没有通过牺牲 ID/base-solved 换取 Hard 改善，并且恢复了 SFT-only 在 train 上丢掉的 10 pp。

但 paired discordant 数量太少，**尚无统计显著性**。尤其 0/8 forgetting 的单侧 95% 上界仍约为 **31%**，不能据此声称遗忘率低于 5%。

### forgetting_gate 的 pass 含义

当前 gate 在 `0 == 0` 时允许通过。因此它证明的是：

> A1 **没有比** SFT-only **产生更多**遗忘。

它**不**证明：

> A1 **比** SFT-only **降低了**遗忘。

---

## 两个证据链 caveat

### 1. Base-solved 不是未见确认集

当前 8 个 behavior-eval seeds 正好来自 anchor corpus：

| 来源 | Seeds |
|------|-------|
| Anchor train | 100005, 100008, 100010, 100019 |
| Anchor probe / 候选选择 | 100014, 100016, 100021, 100024 |

因此 8/8 是合理的**开发性安全检查**，但不是独立**确认性** preservation 证据。下一阶段必须建立完全不与 anchor train、probe、SFT chunk 和候选选择重叠的新 preservation cohort。

### 2. A1@1235 的 probe summary 联接有误（已修复）

Behavior summary 曾给 A1@1235 复制了 A1 整条 2470-step run 的最终 tail。正确的 checkpoint-specific 指标应来自 `A1/summary.json` 的 `checkpoint_artifacts[step]`：

| Checkpoint | base_solved probe | boundary probe |
|------------|------------------:|---------------:|
| A1@1235 | 1.60e-3 | 6.46e-4 |
| A1@2470 | 5.10e-4 | 3.86e-4 |

行为评估结果未受影响；drift–behavior 图表必须使用 `checkpoint_artifacts[step]`。封存后的 `summary.json` 已用 `probe_linkage_version: checkpoint_artifacts_v1` 标记。

---

## BRACE 理论：三个可证伪命题

### H1：Functional anchor 改善 preservation–adaptation Pareto

对照：random chunks, anchor off **vs** random chunks, anchor on（U1/N1 vs B2）。当前 A0/A1 是开发性版本。

### H2：Branch verification 提升 credit quality

对照：matched random chunks (N1) **vs** verified chunks (B1)。必须匹配 seed、trajectory outcome、chunk index、accepted chunk 数和 optimizer budget。

### H3：两种机制组合的 2×2 因子实验

| | Anchor off | Anchor on |
|---|-----------|-----------|
| Random chunks | N1/U1 | B2 |
| Verified chunks | B1 | B3 |

理论量：

- \(\Delta_{\text{anchor, random}} = B2 - N1\)
- \(\Delta_{\text{anchor, verified}} = B3 - B1\)
- \(\Delta_{\text{credit, no-anchor}} = B1 - N1\)
- \(\Delta_{\text{credit, anchor}} = B3 - B2\)
- 交互项：\((B3-B1) - (B2-N1)\)

---

## 执行优先级

| 优先级 | 任务 | 状态 |
|--------|------|------|
| **P0** | 修正 probe 联接并封存 Phase 3C | 本地完成 |
| **P1** | 冻结 A1；建立未见 preservation cohort；3-seed A0 vs A1 确认实验 | 协议/脚本就绪，待云端 |
| **P2** | 已知-collapse 数据的 anchor on/off 压力测试 + rollout 剂量曲线 | 待预注册后云端 |
| **P3** | 扩充 branch confirm（≥10 seeds / ≥30 points） | 见 [`brace_track_bc_confirm_export_20260802.md`](brace_track_bc_confirm_export_20260802.md) |
| **P4** | N1/B1/B2/B3 正式 2×2 factorial | 待 H1/H2 各自过关 |
| **P5** | 原样迁移到 dump（`export BRACE_TASKS=place_container_plate`） | 待 place confirmatory 通过 |
| **P6** | \(\pi_0 \rightarrow \pi_1 \rightarrow \pi_2\) 多轮 self-improvement | 最后阶段 |

---

## P1：冻结 A1 的 confirmatory preservation

**冻结配置（来自 Phase 3A A1）：**

```json
{
  "formulation": "dual_only",
  "dual_lr": 0.01,
  "lambda_init": 0.0,
  "epochs": 10,
  "checkpoint_steps": [247, 741, 1235, 2470],
  "teacher_reference": "raw",
  "anchor_sampler": "frozen_from_calibration_run"
}
```

**Training seeds：** 1, 2, 3, 4, 5（exact sign test 需 5/5 同向）

**比较组：** C0 SFT-only vs C1 A1 dual（10 DP jobs）

**确认 eval 规模（v1.4.1）：**

- P1a census：offset **3000**，100 ID + 100 train × 3 repeats（**无 Hard**）
- P1d confirmatory：offset **4000**，preservation cohort 3 repeats **全部成功**
- ≥60 独立 untouched base-solved preservation seeds（整数入组：`success_count >= 2` of 3）
- 禁止复用 census rows 作为 confirmatory 主终点

**Preservation cohort 必须排除：**

- anchor train seeds
- anchor probe seeds
- SFT chunk source seeds
- branch pilot/confirm selection seeds
- 本次候选选择使用过的行为 seeds

**分层报告：** anchor-train · anchor-probe · untouched preservation · boundary · ID/train。主结论只能来自 **untouched cohort**。

**协议：** `experiments/brace/screen_protocol.v1.4.1.confirmatory_preservation.json`（冻结 + SHA256）
**Jobs：** `confirmatory_preservation_jobs.place_container_plate.v2.json`

**启动（P1a→P1d）：**

```bash
export BRACE_TASKS=place_container_plate
export BRACE_TRACED_ROLLOUT_DIR=experiments/brace/rollouts_traced
export BRACE_CONFIRMATORY_PROTOCOL_PATH=experiments/brace/screen_protocol.v1.4.1.confirmatory_preservation.json

bash experiments/brace/run_all.sh confirmatory-base-census
bash experiments/brace/run_all.sh select-preservation-cohort
bash experiments/brace/run_all.sh confirmatory-preservation
bash experiments/brace/run_all.sh confirmatory-preservation-eval
bash experiments/brace/run_all.sh confirmatory-preservation-report
```

---

## P2：已知-collapse 压力测试

复用 Phase 1 已知会产生 coverage relocation 的 successful-rollout 数据：

- 原 Phase 1 success SFT
- 相同数据/batch/steps/LR + A1 anchor
- 3 个新 training seeds
- 相同 paired eval

预注册 1×、2×、4× rollout dose，形成剂量曲线。若 anchor 在压力下把 place ID 从约 0.22 拉回、train 从约 0.08 拉回，同时保留 Hard 增益，将是比当前 8/8 ceiling 强得多的机制证据。

---

## P3：Branch credit 确认性证据

Place branch pilot/confirm 合并 gate 仍是 **NO-GO**（6 unique seeds < 10；18 points < 30）。当前 11 个 B1/N1 chunks 可用于开发性 falsification，不足以支持正式 H2。

建议：继续收集独立 mixed-outcome seeds → 保持原 gate → ≥10 seeds / ≥30 points → 重导出 matched B1/N1 → credit-only micro-screen → 仅 B1>N1 后进入 B3。

---

## P4：正式 2×2 factorial

H1 与 H2 各自过关后运行 N1/B1/B2/B3：

- ≥3 training seeds
- 数据量与 optimizer budget 严格匹配
- 固定 epoch 1/3/5/10
- 不用 Hard 调 epsilon 或选 checkpoint
- U4 作为外部系统基线，不参与 2×2 因果分解

**Primary endpoints：** untouched base-solved forgetting · ID/train non-inferiority · Hard/worst-group improvement · selection score · difference-in-differences interaction

**统计：** training seed 第一层 · env seed 第二层 · policy repeat 为 seed 内重复 · hierarchical paired bootstrap · McNemar 辅助 · 报绝对 pp、CI、worst-group

---

## P5：跨任务验证

当前 run 目录名含 `dump_bin_bigbin`，但 `summary.task` 是 `place_container_plate`——**不是跨任务结果**。后续必须：

```bash
export BRACE_TASKS=place_container_plate
```

Place confirmatory 通过后，将冻结的 A1 超参数**原样**迁移到 `dump_bin_bigbin`，不重新调参。

---

## Constraint budget 表述

不能再把 `identity_epsilon=1e-4` 当行为 gate。A1@2470 表明 train tail 已约 1.5e-4、held-out probe 约 3.9–5.6e-4，行为上尚未观察到损害。

新确认协议中分离：

| 层级 | 用途 |
|------|------|
| `identity_atol` | 实现 smoke |
| `training_target_epsilon=1e-4` | dual 优化目标 |
| `operational_probe_budget` | 确认前冻结的稳定性阈值 |
| behavior preservation | 最终资格 gate |

准确表述：

> Dual optimization 把功能漂移稳定在约 \(10^{-4}\)–\(10^{-3}\) 区间，并在开发性行为评估中未观察到能力损失。

---

## 相关产物

| 产物 | 路径 |
|------|------|
| Phase 3C forensic bundle | `behavior_eval_20260803_forensic.tar.gz` |
| 封存 summary | `.../summary.json`（`sealed: true`） |
| Confirmatory protocol | `experiments/brace/screen_protocol.v1.4.1.confirmatory_preservation.json` |
| Confirmatory jobs | `experiments/brace/confirmatory_preservation_jobs.place_container_plate.v2.json` |
| Preservation cohort selector | `experiments/brace/select_preservation_cohort.py` |
