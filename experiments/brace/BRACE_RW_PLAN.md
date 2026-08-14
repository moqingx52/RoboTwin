# BRACE-RW 实验计划（预注册）

**版本**: v1 · **冻结日期**: 2026-08-14 · **分支**: `exp/dp-self-improvement`
**取代**: `brace_v2_derivation.md` 的 E1–E5 单时间步锚路线
（代码已归档至 [`experiments/brace_v2_legacy/`](../brace_v2_legacy/README.md)）

---

## 1. 核心命题（已数学验证）

对冻结策略 π₀，在状态 s 采样动作块 a_i ~ π₀(·|s)，以终局成败
G_i ~ Bernoulli(Q₀(s, a_i)) 加权去噪损失，权重

```
w(G) = (1 + cG) / (1 + cV₀(s)),   V₀(s) = E_a[Q₀(s,a)]
```

则加权 MLE 的不动点是倾斜分布 q_c(a|s) = π₀(a|s)·(1+cQ₀)/(1+cV₀)，
其精确局部改进为

```
Δ_c(s) = E_{q_c}[Q₀] − V₀(s) = c · Var_a[Q₀(s,a)] / (1 + c·V₀(s)) ≥ 0
```

（解析与 Monte Carlo 双重验证，lhs=rhs=0.047545。）关键推论：

- **改进量完全由 Var_a[Q₀] 决定** —— 若同状态下动作间成功率无差异，回报加权无效。
  因此先测方差（E0 门），再决定是否训练。
- 权重有界（≤ 1+c），无 exp(βA) 权重爆炸问题（对照 AWR/DIPOLE）。
- 能力保持改用全扩散轨迹 KL：`D_path ≥ KL(π_teacher‖π_student)`（DPI），
  且 |E_{π_θ}Q − E_{π₀}Q| ≤ √(D_path/2)（Pinsker），取代单时间步锚。

先验文献：AWR (1910.00177)、RWR、Diffusion Policy (2303.04137)、
DPPO (2409.00588)、DIPOLE (2601.00898)、TruDi (2606.15260)、
SIME (2505.01396)、ReGuide (2606.28939)。

## 2. 脚本地图

| 阶段 | 脚本 | 作用 |
|------|------|------|
| E0 采集 | `collect_e0_variance.py` | 同状态 A=8 动作 × R=2 独立续演，1536 回合 |
| E0 分析 | `analyze_e0_variance.py` | σ̂²_Q、Tarone 检验、Δ̂_c、门判定 |
| 数据构建 | `build_rw_dataset.py` | E0 输出 → 带 `data/sample_weight` 的 zarr |
| 轨迹 KL | `trajectory_kl.py` | D_path 度量（probe）+ 可微训练约束 |
| 协议 | `protocol.e0.v1.json` | E0 预注册参数（本文件第 4 节为判定标准） |

训练侧钩子已存在，无需改动：
`RobotImageDataset` 自动读取 `data/sample_weight`；
`robotworkspace.aggregate_training_loss` 池化模式计算
`Σ wᵢ·lossᵢ / Σ wᵢ`（`policy/DP/diffusion_policy/workspace/robotworkspace.py:214`）。

## 3. 实验路径（命令级）

前置：traced 语料 `experiments/brace/rollouts_traced_base200_v2/`（100 环境种子
= 冻结种子文件 `seeds/place_container_plate_base200_line_a_seeds.json` 的
`rollout_train` 分区；32 混合 / 52 全成 / 16 全败，每种子 8 条 rollout，数据在
云端 /workspace），冻结检查点
`policy/DP/checkpoints/{task}-demo_clean-200-0/600.ckpt`。

### 第 0 步 · E0 方差门（先测量，后训练）

```bash
# 云端，8 GPU，约 1536 回合（种子文件/分区已写入协议 seeds_file/seeds_key 字段）
PYTHONPATH=. python experiments/brace/collect_e0_variance.py \
  --protocol experiments/brace/protocol.e0.v1.json \
  --rollout-dir experiments/brace/rollouts_traced_base200_v2 \
  --output-dir experiments/brace/runs/$(date -u +%Y%m%dT%H%M%SZ)_e0_variance

PYTHONPATH=. python experiments/brace/analyze_e0_variance.py \
  --input-dir experiments/brace/runs/<run>_e0_variance \
  --protocol experiments/brace/protocol.e0.v1.json
# → analysis.json: e0_gate_passed true/false
```

E0 同时产出 `chunks/*.npz`（每个分支点 8 条采样动作块及其成败标签），
即训练数据的原材料 —— 门通过后无需再采集。

### 第 1 步 · 构建训练数据（门通过后）

```bash
# N1′ 对照臂（c=0：所有权重=1，同数据同预算）
PYTHONPATH=. python experiments/brace/build_rw_dataset.py \
  --expert-zarr policy/DP/data_phase1_200/place_container_plate-expert_only.zarr \
  --e0-dir experiments/brace/runs/<run>_e0_variance \
  --rollout-dir experiments/brace/rollouts_traced_base200_v2 \
  --output experiments/brace/datasets/<task>_rw_c0.zarr --c 0

# B1′/B3′ 处理臂（c=3）
PYTHONPATH=. python experiments/brace/build_rw_dataset.py \
  ... --output experiments/brace/datasets/<task>_rw_c3.zarr --c 3
```

### 第 2 步 · 五臂训练矩阵

| 臂 | 数据 | c | 轨迹 KL 约束 | 检验角色 |
|----|------|---|--------------|----------|
| U0 | — | — | — | 冻结基线（不训练） |
| N1′ | rw_c0.zarr | 0 | 否 | 同数据无加权对照 |
| B1′ | rw_c3.zarr | 3 | 否 | 加权主效应 |
| B2′ | rw_c0.zarr | 0 | 是 | KL 约束单独效应 |
| B3′ | rw_c3.zarr | 3 | 是 | 加权 + 保持（目标臂） |

五臂共享：优化器、LR/EMA 日程、epoch 数、评测种子集、评测协议。
B2′/B3′ 在训练循环中加 `trajectory_kl.path_kl_training_loss(teacher, student, obs)`
（teacher = 冻结 600.ckpt，n_timesteps=4 随机步的无偏缩放估计）。

**训练启动复用既有 finetune.sh 通路**（v2 pilot 同款超参，hydra 覆盖）：

- `task.dataset.zarr_path=<rw_c*.zarr>`；`training.resume_from_ckpt=<600.ckpt>`；
  `training.normalizer_source=checkpoint`；`training.loss_mode=pooled`；
  LR=5e-5，EPOCHS=10，BATCH=128。
- 批次配比 `dataloader.rollout_per_batch=16` → BatchSampler 强制每批
  112 专家 + 16 rollout chunk（`robotworkspace.py:788` 按 `episode_source`
  分池有放回抽样）。rw zarr 已写 `episode_source`（专家=0 / chunk=1），
  直接兼容。
- `sample_weight` 帧级数组随 ReplayBuffer 载入，pooled 模式做
  `Σw·loss/Σw` 归一化加权 —— **batch 内只有相对权重有意义**，因此
  N1′（全 1）与 B1′（(1+cḠ)/(1+cV̂)）在同一配比下严格可比。
- 唯一新增代码：B2′/B3′ 需在 `robotworkspace` 训练循环中接入
  `path_kl_training_loss`（复用既有 `training.brace_anchor` 开关位挂载，
  teacher 从 `resume_from_ckpt` 冻结副本加载）。

### 第 3 步 · 评测与机制检验

- 主评测：固定种子集上各臂成功率（混合种子域内 + 保持域 Base200 census）。
- Probe：`trajectory_kl.path_kl_terms` 在固定 probe 观测上报告各臂
  D_path 与 Pinsker 上界 √(D_path/2)，对照实际保持域退化量。

## 4. 预注册判定（训练前冻结，不得事后更改）

**E0 门（PASS 才进入第 1 步）**，二者同时成立：

1. 池化 mean Δ̂_{c=3} ≥ 0.02（bootstrap 95% CI 一并报告）；
2. Tarone (1979) 超散布单侧 p ≤ 0.05（确证 Var_a[Q₀] > 0 非估计噪声）。

**主检验（成对，同评测种子）**：

- H1: B1′ − N1′ > 0（加权主效应；McNemar 成对精确检验，单侧 α=0.05）
- H2: B3′ − N1′ > 0（目标臂净收益）
- 保持约束：B3′ 在 Base200 保持域的退化 ≤ Pinsker 界预测；
  B1′（无约束）预期退化更大 —— 这是 KL 约束的存在性证据。

**机制检验**（把理论钉在数据上）：

- Δ_observed(B1′ − N1′) 与 E0 预测 c·σ̂²_Q/(1+cV̂) 同量级且同号。
  若显著改进但与预测量级差 >5×，理论解释失效，结果只作经验记录。

**方向切换（E0 FAIL 时，按序考虑）**：

1. σ̂²_Q ≈ 0 且 V̂ 中等 → 失败源于状态（envseed/早期偏差）而非动作选择：
   转向状态级筛选/数据配比（SIME 式经验筛选），放弃动作级加权。
2. σ̂²_Q ≈ 0 且 V̂ 极低/极高 → 分支点选择无信息量：改分支点选择器
   （更早/更晚的 snapshot），重跑 E0 一次（预算 ≤ 原 E0）。
3. Tarone 显著但 Δ̂_{c=3} < 0.02 → 效应真实但太小：报告功效分析，
   评估提高 A（每态动作数）或 c 的代价后再决定，不默认加大预算。

## 5. 预算与产物

- E0: 1536 回合（≈ 单次 census 的量级）；chunks npz 复用为训练数据。
- 训练: 4 臂 × 既有 screen 预算（≈ 10 epoch 级），不超过 v2 走过的单臂成本。
- 全部输出走既有 immutable-runs 约定：`runs/<UTC>_<stage>/` + `records/` +
  通过后 `promote-run` → `archive/`。

## 6. 谱系

- 数学与文献验证记录：本计划第 1 节结论（2026-08-14 会话，解析 + MC 验证）。
- 旧路线证据不删除：`experiments/brace/archive/`、`runs/LATEST_anchor_*` 保持原样。
- 归档说明：[`experiments/brace_v2_legacy/README.md`](../brace_v2_legacy/README.md)。
