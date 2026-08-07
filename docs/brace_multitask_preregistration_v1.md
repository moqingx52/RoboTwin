# BRACE RoboTwin 2.0 多任务预注册 v1

日期：2026-08-07  
状态：**任务、基准和统计设计已冻结；BRACE-v2 方法配方尚待 place 开发实验冻结。**

## 目标

在 RoboTwin 2.0 官方 single-task benchmark 上检验：冻结的 BRACE 方法能否跨任务降低
base capability degradation，同时保持 adaptation 不劣于预算匹配的无 anchor 方法。

该实验首先证明多任务复制，不等同于 A→B sequential transfer。迁移链只有在本实验通过后另行预注册。

## 官方对齐

- Aloha-AgileX embodiment。
- 每任务使用 50 条 `demo_clean` demonstrations。
- Easy 使用 `demo_clean`，Hard 使用 `demo_randomized`，各 100 个 env seeds。
- 本地复现 DP；RDT、Pi0、ACT、DP、DP3 官方数字只作 benchmark context。
- 官方 leaderboard 不能替代同源 Base/SFT/replay 因果对照。

冻结资产：

- `experiments/brace/multitask_protocol.v1.json`
- `experiments/brace/multitask_tasks.v1.json`
- `experiments/brace/leaderboard_snapshot_20260807.json`

## 任务面板

开发/诊断任务不进入主泛化统计：

- `place_container_plate`：唯一允许选择 BRACE-v2 配方的开发任务。
- `dump_bin_bigbin`：既往已见诊断任务，不得用于反向调参。

确认性 held-out tasks：

1. `beat_block_hammer`
2. `click_alarmclock`
3. `handover_mic`
4. `lift_pot`
5. `move_can_pot`
6. `open_laptop`
7. `place_burger_fries`
8. `put_object_cabinet`
9. `shake_bottle`
10. `stack_bowls_three`

任何任务因 replay、训练或评估失败都必须记录为 protocol-feasibility failure，不得从主集合删除。

## 实验矩阵

每个 held-out task 使用 Base 和五个训练 arms：

| Arm | 数据/作用 |
|---|---|
| Base | 本地 50-demo DP，不继续训练 |
| U0 | expert-only continued-training control |
| N1 | matched random chunks，无 anchor |
| B1 | branch-verified chunks，无 anchor |
| B2 | matched random chunks + anchor constraint |
| B3 | branch-verified chunks + anchor constraint |

U0/N1/B1/B2/B3 均跑 training seeds 1–5。2×2 主效应只使用 N1/B1/B2/B3；U0 用于识别额外 optimizer steps 的影响。

## 执行 Gate

当前不得启动 held-out 训练，因为 `multitask_method_freeze.v1.json` 尚不存在。完成 place 上的 BRACE-v2
开发后，使用下列命令冻结实际方法配置：

在任何 place intervention 开发开始前，go/no-go 标准固定为：5/5 paired training seeds 完整，anchor
preservation effect 点估计严格大于 0，且至少 4/5 seeds 为正方向。Source summary 必须使用
`stage=brace_v2_intervention_pilot`、`task=place_container_plate`、`complete=true`、`status=passed`，并在
`preservation_go_no_go` 中记录上述字段和 `passed=true`。未达到标准不得创建 method freeze。

```bash
python experiments/brace/freeze_multitask_method.py \
  --source-run experiments/brace/runs/<PLACE_DEVELOPMENT_RUN> \
  --method-config experiments/brace/<FROZEN_METHOD_CONFIG>.json
```

方法冻结文件必须记录 source `summary.json` 和 method config 的 SHA256。修改任一输入后，所有 held-out
结果均失去该 protocol provenance。

设计校验和候选 seed 生成：

```bash
BRACE_REQUIRE_METHOD_FREEZE=0 bash experiments/brace/run_all.sh multitask-validate
bash experiments/brace/run_all.sh multitask-generate-seeds
```

生成的 seed 文件状态为 `candidate_unvalidated`。Feasibility 仅定义为 expert script/模拟器可解性，不得
查看或使用 base DP 或任何学习策略的结果筛选 seed。必须在 manifest 中记录通过的 evidence 路径及 SHA256、
保持 `learning_policy_performance_consulted=false`，再将互不重叠的 partition 封存为 `status=frozen` 并生成
`.sha256` sidecar，preflight 才会通过。

完整 preflight：

```bash
bash experiments/brace/run_all.sh multitask-preflight \
  --output experiments/brace/runs/<RUN>/preflight.json
```

它检查每任务：官方 50-demo DP checkpoint、50-demo expert zarr、冻结 seed manifest、replay gate 和
BRACE-v2 method freeze。旧的 `*-demo_clean-200-0` checkpoint 不满足本协议。

测试环境说明：`test_preservation_semantics.py` 依赖 DP conda 环境中的 `omegaconf`，应在该环境运行；
基础 Python 环境缺少该依赖时无法收集此测试，不代表 preservation 实现回归。

## 调度与恢复

最终命令由冻结方法 runner 生成到显式 job manifest。模板为
`experiments/brace/multitask_jobs.v1.template.json`。每个 job 声明 argv、依赖、输出 artifact 和 protocol hashes。

```bash
export BRACE_GPU_IDS="0 1 2 3 4 5 6 7"
export BRACE_MULTITASK_JOBS=experiments/brace/runs/<RUN>/jobs.json
export BRACE_MULTITASK_RUN_DIR=experiments/brace/runs/<RUN>
bash experiments/brace/run_all.sh multitask-run

# 中断后复用同一 state.json
bash experiments/brace/run_all.sh multitask-run --resume \
  --state experiments/brace/runs/<RUN>/state.json
```

Scheduler 对每个 physical GPU 最多运行一个 logical job。Train 必须声明 1 process/GPU；simulator/eval
必须声明 3 workers/GPU，并统一设置 `CUDA_VISIBLE_DEVICES`。启动前仍需人工检查 `nvidia-smi` 和容器内现有进程。

## 统计与报告

Primary hierarchy 为 task → training seed → env seed → policy repeat。聚合器按 task 等权做 hierarchical
paired bootstrap，不将 episode 数量当作 task-level 独立样本。

Primary gates：

- H-Preservation：anchor main effect 的 pooled CI lower bound > 0。
- H-Credit：branch verification adaptation main effect 的 pooled CI lower bound > 0。
- H-Pareto：`B3-B1` preservation CI lower bound > 0，且 adaptation CI lower bound > -5 pp。
- H-Generalization：至少 7/10 held-out tasks 的 B3 preservation effect 为正，且没有任务 adaptation < -10 pp。
- Local-DP reproduction：任一任务本地 Base Easy 与冻结 leaderboard DP Easy 的绝对偏差大于 10 pp，
  confirmatory aggregate 即失败；任务仍保留并完整报告，不得删除。

最终 artifact index 必须显式列出全部 10 tasks × (Base + 5 arms × 5 seeds) 的路径和 SHA256。模板为
`multitask_artifact_index.v1.template.json`。生成报告：

```bash
export BRACE_MULTITASK_ARTIFACT_INDEX=experiments/brace/runs/<RUN>/artifact_index.json
export BRACE_MULTITASK_REPORT_OUTPUT=experiments/brace/runs/<RUN>/multitask_summary.json
bash experiments/brace/run_all.sh multitask-report
```

聚合器要求每个 eval artifact 包含完整的 100 Easy、100 Hard 和 100×3 preservation paired work items；
Easy、Hard 和 preservation env seeds 必须分别来自冻结 manifest 的 `confirm_easy`、`confirm_hard` 和
`census_candidate` partition；
矩阵不完整、hash 不符、method freeze 不符或 task 被遗漏时均 fail closed。
