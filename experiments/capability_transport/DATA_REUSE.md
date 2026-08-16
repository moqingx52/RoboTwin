# 旧数据复用审计（Data Reuse Audit）

日期：2026-08-16。基准：`exp/dp-self-improvement` @ 8286411（BRACE 代码删除前的最后完整状态）。

判定标准：新研究线（能力迁移层析）需要的是 **冻结难度定义（基线成功率）**、
**种子分区**、**数据来源 D_E/D_M/D_H/D_Q** 和 **层级统计协议**。据此逐项判定。

## 一、可直接复用

| 数据 | 位置 | 复用方式 |
| --- | --- | --- |
| 多任务任务清单（10 个留出任务 + 2 个开发任务） | `../brace/multitask_tasks.v1.json` (+.sha256) | T3 直接沿用；任务角色划分不变 |
| 多任务层级统计协议 | `../brace/multitask_protocol.v1.json` / `.v2.json` | 层级推断结构（task→train seed→env seed→repeat）沿用；门槛需按新主门 Γ 重写为新预注册 |
| 每任务预生成环境种子分区 | `../brace/seeds/multitask_v1/*.json` | 种子分区确定且互斥，可作为 240-seed 难度层析池的候选来源；注意其 `status=candidate_unvalidated`，冻结前需补可行性证据 |
| place 任务 v1.4.2 确认种子集 | `../brace/seeds/place_container_plate_confirmatory_v1.4.2_seeds.json` | T1 开发任务的评估种子池 |
| Base 200-seed 成功计数普查（π₀ 每种子成功数，place） | `../brace/archive/confirmatory_base_census_v142_20260804/`（200 个种子 × 重复成功计数） | **最有价值的复用件**：正是新方案第 3 步需要的 p₀(x) 估计原料，可直接喂 Beta–Binomial 软分组；但重复数少于新协议 R=8 时只能作为 T0/预筛，冻结难度仍需按新协议补采 |
| phase1 hard-seed 基线成功率（place，20 个种子，含多个 0.0） | 已拷贝到 `legacy_evidence/phase1_hard_eval_seeds_place_container_plate.json` | 零支持尾部（0/8 unsupported tail)候选种子来源；用于不可识别性定理的实证验证 |
| 专家演示数据 / Base checkpoints | 云端 `/workspace/RoboTwin`（本地 gitignore，不在仓库） | 50-demo 与 Base200 专家轨迹、π₀ checkpoint 全部继续使用：新方案 rehearsal 集与 π₀ 冻结策略即来自这里 |

## 二、可作动机 / T0 审计证据（不进入方法链）

| 数据 | 位置 | 用途 |
| --- | --- | --- |
| E0 v2 方差门 FAIL 记录（64 点/1536 回合） | `../brace/archive/e0_variance_place_20260815T160249Z/`（analysis/summary/checks） | 论文动机：状态内 Q 方差显著但不预测训练收益（因果点 Δ=0.0114 < 随机对照 0.0197）。保持只读，不重跑 |
| E0 v1 FAIL 记录 | `../brace/archive/e0_variance_place_20260814T154857Z/` | 同上 |
| B1/N1 验证片段 vs 随机对照数据集 | `../brace/datasets/place_*_{B1,N1}.jsonl` | T0 审计：说明"好片段分数"路线的样本量与重复访问问题 |
| 确认性保持评估 p1c/p1d、锚定行为评估 | `../brace/archive/confirmatory_preservation_*`、`anchor_behavior_eval_phase3c_*` | 负迁移/保持失败的动机证据 |
| replay/branch 审计门（v2.1–v2.3）、种子可行性扫描 | `../brace/archive/replay_audit_v2_*`、`seed_feasibility_*` | 环境可行性与仿真器行为的背景证据；种子可行性结论（哪些任务专家脚本可产数）在设计 D_Q 补采时仍然有参考价值 |
| 旧数学推导（w≤3 约束来源） | `/brace_v2_derivation.md`（仓库根） | w=ρ/q≤3 硬约束直接继承 |

## 三、不可复用（原因）

| 数据 | 原因 |
| --- | --- |
| E0 的 8×3 动作-续跑 rollouts 本体（chunks/*.npz，云端 runs/ 目录，归档时已省略） | 是 Q 方差估计的中间量，与 T_{gh} 估计无关；且其难度取样不符合新的冻结难度协议 |
| B1/N1 已导出的 6–8 帧 chunk 训练集 | 新协议明确禁止 chunk 级导出（须完整成功续跑）；且这批数据违反 w≤3（历史 w≈99） |
| 旧 screen/calibration/credit-micro 各版协议门槛值 | 均绑定 BRACE 干预臂定义；新线须另立预注册。协议文件保留只作审计 |
| 微调后 checkpoint（B1/N1/锚定各臂） | 处理后分组偏差风险：任何"用微调后模型重划难度"的用法被新协议禁止。仅可作 T0 展示 |

## 四、需要新采集的

1. 每开发任务 240 个环境种子 × R=8 次 π₀ 基线 rollout（冻结难度 + 软分组）；
   census v142 的 200×3 数据只能预筛，不满足 R=8。
2. 每任务 100 个独立 rollout seeds × 8 次的自生成数据（完整成功续跑，构成 D_E/D_M/D_H）。
3. 困难零支持环境的专家补充轨迹 D_Q（Q ∈ {5,10,20} 递增可行性搜索）。
4. 9 点 D-optimal 小剂量训练（4 种子/点，10–20% 预算）× 2 个开发任务。

## 五、本次清理动作记录

- 新分支 `exp/capability-transport`（自 `exp/dp-self-improvement` @ 8286411）。
- `git rm`：`experiments/{phase1,phase2,phase2b,phase3,brace_v1,brace_v2_legacy}/`
  全部，以及 `experiments/brace/` 下全部 `.py`/`.sh` 代码（89 个文件）。
- 磁盘同步删除上述目录及 `__pycache__`；唯一有价值的未跟踪数据
  `phase1/eval_results_200/hard_eval_seeds/place_container_plate.json`
  已先拷贝到本目录 `legacy_evidence/`。
- **未删除**：`experiments/brace/` 的 `archive/`、`seeds/`、`datasets/`、
  `records/`、`runs/`、协议 JSON、`BRACE_RW_PLAN.md`、README——BRACE-RW 的
  FAIL 记录按决定保持原样只读。
- 旧代码如需查阅：`git show 8286411:experiments/brace/<file>`。
