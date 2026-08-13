A观点：
有问题，而且不只是“anchor（锚定）超参数没调好”。我审查的是 `moqingx52/RoboTwin` 的 `exp/dp-self-improvement` 分支，当前 HEAD（分支头）为 [`c690139`](https://github.com/moqingx52/RoboTwin/commit/c690139543b6a4d41f3a8b327473859c51e42b77)。

你的研究问题拆分是成立的，N1/B1/B2/B3 的 (2\times2) 因子设计也基本正确；但目前有几处代码和统计口径会直接改变实验结论。现在最多能判定“当前协议不具备冻结资格”，还不能严格判定“10/10 最终 checkpoint（检查点）位于可行域外”。

## 最严重的问题

| 严重度 | 问题                                                 | 对当前结论的影响                                     |
| --- | -------------------------------------------------- | -------------------------------------------- |
| P0  | 行为保持汇总代码比较错了实验臂                                    | 实际比较 B1−N1，而不是有/无 anchor（锚定）                 |
| P0  | feasibility（可行性）统计整段训练历史，不是最终模型                    | 不能推出 (\theta_{\text{final}}\notin\mathcal F) |
| P0  | branch LCB（分支置信下界）计算不正确                            | 5 个 accepted chunks（接受片段）中至少 3 个被明显过度自信地接受   |
| P1  | branch point（分支点）类型只是错误标签                          | 所谓随机负对照实际上不是随机点                              |
| P1  | 冻结的 branch confirm gate（分支确认门槛）没有执行                | 实际 3 个 seed（种子）、9 个点，却继续做了完整训练               |
| P1  | anchor teacher（锚定教师）用 raw（原始权重），行为评估部署 EMA（指数移动平均） | 内部偏离和实际基线不是同一个策略                             |
| P1  | anchor（锚定）只有大约 16 个固定状态，且对偶惩罚启动过慢                  | 当前失败很可能是机制实现与门槛互相冲突                          |

### 1. 行为保持的代码比较错了实验臂

[`behavioral_preservation_go_no_go()`](https://github.com/moqingx52/RoboTwin/blob/exp/dp-self-improvement/experiments/brace/_aggregate_place_base200_line_a.py#L166-L229) 明确计算的是：

[
P_{\mathrm{B1}}-P_{\mathrm{N1}}
]

但这衡量的是 verification effect（验证片段效应），不是 anchor effect（锚定效应）。

真正的 anchor effect（锚定效应）应该按相同训练 seed（种子）计算：

[
A_i=
\frac{
(P_{\mathrm{B2},i}-P_{\mathrm{N1},i})
+
(P_{\mathrm{B3},i}-P_{\mathrm{B1},i})
}{2}
]

代码目前完全没有在行为保持门槛里使用 B2/B3。也就是说，即使内部 feasibility（可行性）没有失败，现有 joint gate（联合门槛）也可能因为比较错实验臂而产生错误结论。

同时，代码的 `success_rate()` 是把所有 episode（回合）直接求平均；但你的冻结定义要求一个环境 seed（种子）三次全部成功才算保持：

[
I_s(\pi)=\mathbf 1!\left[\sum_{r=1}^3\text{success}_{s,r}=3\right]
]

所以当前行为门槛不仅比较错了实验臂，指标本身也没有实现你描述的 (P_{\mathrm{pres}})。

### 2. “最终检查点不可行”这一结论目前不成立

[`evaluate_constraint_feasibility()`](https://github.com/moqingx52/RoboTwin/blob/exp/dp-self-improvement/experiments/brace/screen_gates.py#L115-L186) 会读取训练日志中的所有：

```text
brace_constraint/base_solved
brace_constraint/boundary
```

然后对整段训练过程求 mean（均值）、P90（90 分位数）和超限比例。

因此实际检验的是：

[
\frac1T\sum_{k=1}^T d_g(\theta_k)
]

而不是最终检查点的：

[
d_g(\theta_T)
]

这两者不是一回事。尤其 primal-dual optimization（原始—对偶优化）本来就允许前期越界、后期由 (\lambda) 拉回来。你们旧的归档数据里也能看到 warmup（预热阶段）偏离很高、训练尾部明显降低的现象。

所以当前正确表述应当是：

> 8/8 有可信日志的 anchor training histories（锚定训练历史）未通过 screen.v1.2 的全轨迹代理门槛；另外 2/10 缺少可信日志。因此 10/10 均不具备冻结资格。

而不能表述为：

> 10/10 最终检查点已被证明满足 (\theta\notin\mathcal F)。

后两条尤其应区分：

* `10/10 gate-ineligible（门槛不合格）`：成立；
* `8/8 measured histories fail（已测训练历史失败）`：成立；
* `10/10 final checkpoints infeasible（最终检查点不可行）`：目前未被证明。

此外，分析器从 `anchor_smoke.identity_epsilon` 读取 (\varepsilon)，而不是从实际训练的 `training.brace_anchor.epsilon` 读取。当前两者碰巧都是 (10^{-4})，但这是一个容易静默失配的代码问题。

### 3. branch LCB（分支置信下界）并不是 (\Delta) 的置信下界

[`summarize_branch_rows()`](https://github.com/moqingx52/RoboTwin/blob/exp/dp-self-improvement/experiments/brace/collect_branches.py#L473-L543) 当前计算：

[
\operatorname{LCB}(\hat p_c)-\hat p_{\mathrm{ctrl}}
]

它只对 candidate（候选组）做 bootstrap（自助法），然后减去 control（对照组）的点估计，没有计算：

[
\operatorname{LCB}(\hat p_c-\hat p_{\mathrm{ctrl}})
]

因此完全忽略了 control（对照组）不确定性，也没有利用相同 continuation seed（续跑种子）的配对结构。

这已经实际影响了 5 个 B1：

* 两个点是 candidate（候选组） (2/3)，control（对照组） (1/9)；
* 一个点是 candidate（候选组） (3/3)，control（对照组） (7/9)。

代码把它们都接受了。即使暂时粗暴地把这些结果当独立 Bernoulli trial（伯努利试验），单侧 Fisher exact test（费舍尔精确检验）分别得到：

* (2/3) 对 (1/9)：(p\approx0.127)，未达到 (\alpha=0.1)；
* (3/3) 对 (7/9)：(p\approx0.545)，远不显著；
* 只有两个 (3/3) 对 (0/9) 的点比较强，(p\approx0.0045)。

真实数据还有 continuation seed（续跑种子）和 failure rollout（失败轨迹）聚类，实际有效样本量通常只会更小。因此目前的 5 个 accepted chunks（接受片段）里，很可能只有 2 个拥有较可靠的证据。

### 4. 三种 branch point（分支点）并没有按名字实现

[`select_branch_points()`](https://github.com/moqingx52/RoboTwin/blob/exp/dp-self-improvement/experiments/brace/collect_branches.py#L77-L134) 的实际流程是：

1. 按成功/失败轨迹的关节距离排序；
2. 取前三个最大 divergence（分歧）点；
3. 按排名依次贴上：

```text
local_divergence_peak
first_persistent_divergence
random_negative_control
```

但代码没有：

* 寻找“第一次持续分歧”；
* 随机抽取 negative control（负对照）；
* 使用传入的 `rng`（随机数生成器）进行分支点采样。

所以 `random_negative_control` 只是“第三大的分歧点”。当前 5 个 B1 中有 2 个来自这种伪随机负对照，而且 [`export_verified_chunks.py`](https://github.com/moqingx52/RoboTwin/blob/exp/dp-self-improvement/experiments/brace/export_verified_chunks.py) 会导出所有 `accepted=true` 的点，不过滤 `point_type`（点类型）。

更深一层的问题是：你们专门选择成功/失败轨迹状态差异最大的时刻，再把失败轨迹的绝对关节动作移植到成功轨迹状态上。于是比较很容易变成：

> 与当前状态匹配的成功动作，对比与当前状态严重不匹配的失败动作。

这会系统性地抬高 candidate（候选组）优势。更干净的 control（对照组）应当从同一个 snapshot state（快照状态）重新采样基础策略动作，或者使用对当前状态重新居中的动作扰动。

### 5. 冻结的 branch confirm gate（分支确认门槛）没有满足

[`screen_protocol.v1.2.json`](https://github.com/moqingx52/RoboTwin/blob/exp/dp-self-improvement/experiments/brace/screen_protocol.v1.2.json#L86-L92) 要求：

* 至少 10 个有效环境 seed（种子）；
* 至少 30 个 branch point（分支点）；
* 单个 seed（种子）的 accepted share（接受占比）不超过 0.2。

实际 Base200 数据是：

* 3 个有效 seed（种子）；
* 9 个 branch point（分支点）；
* 5 个接受点来自 (1/2/2) 三个 seed（种子），最大占比 (2/5=0.4)。

但归档的 `completeness_gate_check.json` 仍给出了 `GO`，随后用这 5 个片段跑了 25 条训练。

如果把它定义成 developmental pilot（开发性预实验），这样做可以；但它不能同时被称为通过了冻结的 branch confirm gate（分支确认门槛）。

### 6. raw teacher（原始权重教师）与 EMA deployment（指数移动平均部署）不一致

训练配置明确设置 teacher reference（教师参考）为 raw model（原始模型），并从 `self.model.state_dict()` 复制教师；但 Base200 census（普查）和行为评估部署的是 EMA model（指数移动平均模型）。

而代码又把当前 EMA（指数移动平均）输出与 raw teacher（原始权重教师）比较：

[
d_{\mathrm{EMA}}=
|f_{\mathrm{EMA},\theta}-f_{\mathrm{raw},\theta_0}|^2
]

这意味着所谓 EMA drift（指数移动平均偏离）可能部分只是基础检查点本身的：

[
f_{\mathrm{EMA},\theta_0}-f_{\mathrm{raw},\theta_0}
]

而不是训练带来的遗忘。

必须统一定义：

* 如果 (\pi_0) 是实际部署的 EMA（指数移动平均）策略，anchor teacher（锚定教师）也应来自 Base200 EMA（指数移动平均）；
* 或者全部改为 raw（原始权重）评估；
* 至少先在 step 0（第零步）报告 raw–EMA baseline gap（原始权重—指数移动平均基线差距）。

### 7. 当前对偶优化与全轨迹门槛在结构上互相冲突

[`BraceDualState`](https://github.com/moqingx52/RoboTwin/blob/exp/dp-self-improvement/policy/DP/diffusion_policy/workspace/robotworkspace.py#L35-L49) 从 (\lambda=0) 开始。`dual_only`（仅对偶）形式意味着：

* 第一步 anchor gradient（锚定梯度）为零；
* 第一步之后的对偶更新看到的还是更新前的约束；
* 只有已经发生偏离后，(\lambda) 才逐渐增大；
* `dual_lr=0.01` 时，若 (d-\varepsilon\approx9\times10^{-4})，单步只增加约 (9\times10^{-6})。

因此它本来就是渐进纠偏机制，并不保证整个优化路径始终位于可行域内。与此同时，feasibility gate（可行性门槛）却把早期所有越界点都计入均值和 P90（90 分位数）。这种组合天然容易失败。

此外，anchor replay set（锚定回放集）默认每组最多 8 条轨迹，每条只取一个固定中段窗口；窗口长度刚好等于训练 horizon（时域长度），所以每组基本只有 8 个可用序列。每个 anchor batch（锚定批次）又正好抽 8 个，等于每一步都在反复约束同一批十几个状态。[构建代码](https://github.com/moqingx52/RoboTwin/blob/exp/dp-self-improvement/experiments/brace/build_anchor_replay_set.py#L99-L166)

这不足以近似：

[
\mathbb E_{x\sim A_g}[\cdots]
]

更像是在 16 个固定状态上的 functional memorization（函数记忆）。

## 哪些部分是合理的

你们不是整个方法都做错了，以下部分质量其实不错：

* 研究问题拆成 credit assignment（信用分配）与 preservation（能力保持）是正确的；
* N1/B1/B2/B3 的因子设计公式正确；
* snapshot restore（快照恢复）、control trace replay（控制轨迹重放）和相同状态分支执行的基础链路比较完整；
* teacher（教师）被冻结、禁止梯度、保存 hash（哈希）并在恢复训练时校验，这部分严谨；
* student/teacher（学生/教师）共享 clean action（干净动作）、noise（噪声）与 timestep（时间步）的实现正确；
* SFT dataset（监督微调数据集）与 anchor replay set（锚定回放集）分开是正确的；
* N1 至少匹配了 env seed（环境种子）、trajectory outcome（轨迹结果）和 chunk index（片段位置），并排除了同一条 B1 轨迹；
* census enrollment（普查入组）和后续 preservation evaluation（保持能力评估）分开，避免直接在同一批重复结果上选择并评估，这个方向正确。

## 当前应该怎样重新下结论

我建议把现有结论改成：

> Base200 条件下，snapshot/replay/branch（快照、重放与分支）工程链路已经跑通，并按当前实现规则导出了 5 个候选片段及 5 个匹配 N1 片段。但当前 branch LCB（分支置信下界）、分支点选择和确认门槛存在统计与实现问题，因此这 5 个片段只能视为开发性候选，不能视为已确认的因果优势片段。
>
> 在 anchor（锚定）侧，8 个具有 seed-matched logs（种子匹配日志）的训练历史均未通过 screen.v1.2 的全训练轨迹代理门槛，另外 2 个缺少可信证据，因此 10 个带 anchor（锚定）的实验臂均不具备方法冻结资格。但该门槛并未直接评估最终 checkpoint（检查点）在固定 held-out probe（留出探针）上的约束值，所以尚不能断言 10 个最终模型全部位于数学可行域之外。

## 下一轮最值得做的事情

先不要立即重跑 25 条训练。现有检查点还能完成三个不需要重训的 forensic analysis（取证分析）：

1. 对每个 epoch-10 checkpoint（第十轮检查点）在固定、留出的 anchor probe set（锚定探针集）上重新计算最终约束，只评估最终权重；同时测量 Base raw–EMA gap（基础原始权重—指数移动平均差距）。

2. 按真正的 3/3 seed-level preservation（种子级能力保持）重新汇总：

[
\text{Anchor effect}_i
======================

\frac{(B2_i-N1_i)+(B3_i-B1_i)}{2}
]

并单独报告两个简单效应 (B2-N1) 和 (B3-B1)。

3. 对 9 个 branch point（分支点）重新做 clustered/paired inference（聚类配对推断）和 multiple-comparison correction（多重比较校正）。按目前数据，我预计可靠 B1 很可能从 5 个缩到 2 个。

最后还有一个统计提醒：5 个训练 seed（种子）中“至少 4 个方向为正”只能作为工程门槛。单侧 exact sign test（精确符号检验）下，4/5 的 (p=0.1875)，并不构成常规意义上的显著性；5/5 才是 (p=0.03125)。因此五个训练 seed（种子）可以做 developmental screen（开发性筛选），但不适合承担很强的统计结论。
B观点：
结论先说：BRACE（分支验证与能力锚定）的完整方法目前确实没有验证成功，但问题不只是“anchor（锚定）阈值太严”。代码和归档里至少有三类独立问题：

1. 当前 anchor（锚定）在行为上没有表现出防遗忘作用；
2. anchor feasibility（锚定可行性）的判定对象和阈值定义不合理；
3. branch verification（分支验证）本身还只是 pilot（先导实验），统计置信度和反事实对照都有漏洞。

所以最准确的结论应该是：

> 当前 branch（分支）基础设施可运行，但尚未形成严格的因果验证证据；当前 dual anchor（对偶锚定）优化器也没有产生可观察的防遗忘收益。BRACE（分支验证与能力锚定）整体尚未被验证，不能只归因于 (\varepsilon=10^{-4}) 太严。

## 一、最严重的 branch verification（分支验证）问题

### 1. 你自己的冻结门没有真正执行

冻结的 `screen.v1.2` 要求：

* 至少 10 个 seed（随机种子）；
* 至少 30 个 branch point（分支点）；
* 单个 seed（随机种子）占比不超过 20%。

但 Base200（基础策略）实际只有：

* 3 个 eligible seed（合格随机种子）；
* 9 个 branch point（分支点）；
* 5 个接受片段；
* 两个 seed（随机种子）分别贡献 2/5，即 40%。

尽管如此，新的 completeness check（完整性检查）仍给出了 `verdict=GO`。这说明 operational completeness（运行完整性）被误当成 scientific validity（科学有效性）。证据见 [Base200 分支完整性检查](https://github.com/moqingx52/RoboTwin/blob/exp/dp-self-improvement/experiments/brace/archive/branches_place_base200_v2/completeness_gate_check.json)、[冻结筛选协议](https://github.com/moqingx52/RoboTwin/blob/exp/dp-self-improvement/experiments/brace/screen_protocol.v1.2.json)。

这里应该直接改成 NO-GO（不通过），而不是“前半部分已验证”。

### 2. 当前 LCB（置信下界）计算是反保守的

实现中：

```python
LCB = bootstrap_lcb(candidate_success) - mean(control_success)
```

只对 candidate（候选组）做 bootstrap（自助采样），control（对照组）的不确定性被忽略。

更严重的是：每个 candidate（候选片段）只有 3 次 continuation（续跑）。如果结果为 3/3 成功，普通 bootstrap（自助采样）每次都只能抽到成功，于是：

[
\operatorname{LCB}(\hat p_c)=1
]

但这不表示真实成功率的 90% 下界是 1。3/3 的单侧精确下界其实远低于 1。归档中大量 `lcb_advantage=1.0` 正是这个退化造成的。相关代码见 [collect_branches.py](https://github.com/moqingx52/RoboTwin/blob/exp/dp-self-improvement/experiments/brace/collect_branches.py)。

正确做法是：

* candidate（候选组）和 control（对照组）共同进入区间估计；
* 使用相同 continuation seed（续跑随机种子）形成配对差值；
* 以 env seed（环境随机种子）或 snapshot（快照）作为 cluster（聚类单位）；
* 使用 cluster bootstrap（聚类自助法）、paired randomization test（配对随机化检验）或 Beta-Binomial model（贝塔二项模型）。

### 3. 当前 control（对照动作）不是真正的同状态反事实

现在的做法是：

* 从成功轨迹恢复状态；
* candidate（候选动作）来自这条成功轨迹；
* control（对照动作）来自同一环境 seed（随机种子）下的失败轨迹、相同 chunk index（片段位置）；
* 把失败轨迹动作移植到成功轨迹状态。

这会混合两个效应：

[
\text{动作是否有价值}
+
\text{动作是否与其原始观测状态匹配}
]

失败轨迹动作是在另一个状态上生成的，搬到成功状态后不适用并不奇怪。因此当前结果更像：

> 成功状态上的原生动作优于从不同状态移植过来的失败动作。

更合适的 control（对照）是从完全相同的 (s_t) 出发：

* 用冻结 Base200（基础策略）更换扩散噪声，重新采样 (K) 个动作片段；
* 或使用状态距离很近的片段；
* 或使用动作幅度、似然、时间位置都匹配的扰动片段。

### 4. “random negative control（随机负对照）”实际上不随机

代码先按成功/失败轨迹的关节差异排序，然后依次给前三个点贴上：

* `local_divergence_peak`
* `first_persistent_divergence`
* `random_negative_control`

第三个只是排序后的第三名，并没有随机采样。这个标签会给读者错误的实验含义。

### 5. 800 条轨迹实际上没有充分利用

当前分支运行只分析了 10 个 pilot seed（先导随机种子），最终只有 3 个 seed（随机种子）同时具有成功和失败轨迹。每个 seed（随机种子）又只取排序后的第一条成功轨迹。

因此“收集了 800 条轨迹”不等于 branch（分支）有效样本量是 800。你应该先扫描现有 800 条数据，找出所有 mixed-outcome seed（成功失败混合随机种子），不一定需要重新收集。

## 二、anchor feasibility（锚定可行性）判定有概念错误

### 1. 这个优化问题不可能是数学上的“不可行”

因为初始化策略就是教师策略：

[
d_g(\theta_0)=0\le\varepsilon
]

所以可行域至少包含 (\theta_0)，即：

[
\theta_0\in\mathcal F
]

因此不能说“anchor mechanism infeasible（锚定机制不可行）”，只能说：

> 当前 stochastic primal-dual solver（随机原始—对偶求解器）没有在给定训练预算内返回满足约束的最终策略。

可能不存在“既显著适应又满足当前约束”的解，但这需要测量 Pareto frontier（帕累托前沿），不能由这 10 个 checkpoint（检查点）直接推出。

### 2. 你测量的是训练路径，不是最终 checkpoint（检查点）

理论定义是：

[
d_g(\theta_{\text{final}})
]

但当前实现把整个训练过程所有 step（训练步）的 constraint（约束值）放在一起计算 mean（均值）、P90（90 分位数）和 violation fraction（超限比例），包括前几百步的 warm-up（预热）瞬态。[screen_gates.py](https://github.com/moqingx52/RoboTwin/blob/exp/dp-self-improvement/experiments/brace/screen_gates.py) 明确这样实现。

于是它回答的是：

> 训练全过程是否一直没有远离教师？

而不是：

> 最终策略是否回到了可行域？

主可行性指标应该只在最终 checkpoint（检查点）上，用冻结的 anchor probe set（锚定探针集）、冻结噪声和冻结 timestep（扩散时间步）重新估计。训练路径数据只能作为优化诊断。

### 3. (10^{-4}) 本来只是 identity tolerance（同一性容差）

源码注释明确写着：

> identity epsilon（同一性阈值）是 implementation smoke tolerance（实现冒烟容差），preservation budget（能力保持预算）应另行校准。

但后续协议又把同一个 (10^{-4}) 用成训练漂移预算。证据见 [锚定损失实现](https://github.com/moqingx52/RoboTwin/blob/exp/dp-self-improvement/policy/DP/diffusion_policy/workspace/robotworkspace.py)。

扩散去噪 MSE（均方误差）的尺度会受到以下因素影响：

* 动作归一化；
* diffusion timestep（扩散时间步）分布；
* 输出维数；
* epsilon prediction（噪声预测）还是 sample prediction（样本预测）；
* 执行动作与非执行 horizon（时间范围）的占比。

所以 (10^{-4}) 没有天然的行为意义。

最好的校准方法是做权重插值：

[
\theta(\alpha)=\theta_0+\alpha(\theta_{\rm SFT}-\theta_0),
\qquad
\alpha\in{0,0.1,0.25,0.5,0.75,1}
]

然后同时测量：

* 最终 (d_g(\theta(\alpha)))；
* Base200（基础策略）行为保持率；
* 新数据适应率。

这样才能找到“内部漂移—行为遗忘”的经验关系，再冻结下一版 (\varepsilon)。

### 4. 训练约束 raw model（原始模型），却同时要求 EMA（指数移动平均）满足同一阈值

EMA（指数移动平均）不是直接优化变量，它会滞后于 raw model（原始模型）。要求整个训练路径中两者都满足同一个极小阈值，比原始约束更强，却没有对应的优化保证。

如果部署使用 EMA（指数移动平均），应当：

* 最终行为主评估使用 EMA（指数移动平均）；
* raw model（原始模型）约束作为训练手段；
* 最终 EMA drift（指数移动平均漂移）作为独立 manipulation check（干预检查）。

### 5. 对偶优化确实太弱，而且有配置没有真正生效

代码中还存在具体问题：

* `lambda_init` 配置没有在实际训练路径中初始化对偶变量；
* `lambda_max` 虽然由更新函数支持，但调用更新时没有传入；
* 使用非默认 `group_weights` 时，训练损失和对偶更新计算的 violation（违反量）不一致；
* A1 最终 `anchor_to_sft_grad_ratio`（锚定梯度与监督微调梯度比）只有约 0.018，即锚定梯度约为 SFT（监督微调）梯度的 1.8%，很难充当“安全绳”。见 [A1 校准结果](https://github.com/moqingx52/RoboTwin/blob/exp/dp-self-improvement/experiments/brace/archive/anchor_calibration_phase3a_20260803/A1/summary.json)。

## 三、行为保持评估也有回归均值问题

Base（基础策略）通过 2/3 成功进入 preservation cohort（保持集合），新策略却要求 3/3 全成功才算保持。更换 policy seed offset（策略随机种子偏移）后，完全没有训练过的 Base fresh（重新评估基础策略）自身也有 7/60、即 11.7% 的“遗忘”。

这说明绝对零遗忘门主要测到了策略随机性和 regression to the mean（均值回归），而不全是参数遗忘。见 [v1.4.2 行为汇总](https://github.com/moqingx52/RoboTwin/blob/exp/dp-self-improvement/experiments/brace/archive/confirmatory_preservation_p1d_v142_20260806/h1_summary.json)。

应改成：

[
\Delta_{\rm pres}
=================

\frac1{|S|}
\sum_s
\left[
\hat p_{\theta}(s)-\hat p_{\theta_0}(s)
\right]
]

并让新策略和 Base（基础策略）使用相同 env seed（环境随机种子）和 policy seed（策略随机种子），形成 common random numbers（公共随机数）配对。每个状态最好不止 3 次重复。

另外，5 个训练 seed（随机种子）里 4/5 同方向不是显著性证据。单侧 sign test（符号检验）中：

* 5/5：(p=1/32=0.03125)；
* 4/5：(p=6/32=0.1875)。

4/5 可以作为 development gate（开发门），不能写成 confirmatory significance（确认性显著）。

## 四、最值得参考的相关工作

| 工作                                                                                               | 对你的直接启示                                                                                                                   |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| [SIME（模态级探索自改进）](https://arxiv.org/html/2505.01396)                                              | 用低成功率状态中的偶然成功做轨迹筛选，再用 IQL（隐式 Q 学习）估计片段价值；这是 branch verification（分支验证）必须超过的直接基线。                                           |
| [ReGuide（引导式扩散策略自改进）](https://arxiv.org/abs/2606.28939)                                          | 对 recovery data（恢复数据）做 matched-data ablation（匹配数据消融），并比较 fine-tuning（微调）和 retraining from scratch（从头重训）；与你的 B1–N1 目标非常接近。 |
| [RoboCat（自改进通用机器人策略）](https://arxiv.org/html/2306.11706)                                         | 把原数据和自生成数据共同重训。你需要加入“从 Base200（基础策略）重新训练旧数据+新片段”这一强基线。                                                                    |
| [CLEAR（持续学习经验回放）](https://arxiv.org/abs/1811.11682)                                              | experience replay（经验回放）加 behavioral cloning（行为克隆）本身就是很强的防遗忘方法。                                                            |
| [ContinualVLA（持续视觉语言动作学习）](https://arxiv.org/html/2605.26820)                                    | 新近实机结果也表明 replay ratio（回放比例）、replay frequency（回放频率）和 action normalization（动作归一化）非常关键。                                     |
| [GEM（梯度情景记忆）](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning) | 直接投影新任务梯度，防止旧记忆损失上升；比从零开始缓慢增长的对偶变量更接近你想要的“每步不破坏旧能力”。                                                                      |

你当前每个 128 样本 batch（批次）里只有 16 个新片段样本，其余主要是旧专家数据，本身已经是非常强的 rehearsal（排练回放）。因此 anchor（锚定）能够额外改善的空间可能很小。必须把“普通旧数据回放”作为正式强基线，而不能把无 anchor（锚定）理解成完全没有保护。

## 五、我建议的最小验证路线

不要立刻重跑完整 25 条训练。按下面顺序更省算力。

1. 修 branch（分支）统计和 gate（门控）。

   * 先利用现有 800 条轨迹，扫描全部成功/失败混合 seed（随机种子）；
   * 至少满足自己冻结的 10 seed（随机种子）/30 point（分支点）；
   * 修复伪随机负对照；
   * 使用同状态策略重采样作为 control（对照）；
   * 对 (\Delta) 做双边不确定性和 seed-level cluster bootstrap（随机种子级聚类自助法）；
   * 现有 5 个片段只能标成 pilot evidence（先导证据）。

2. 做 anchor positive control（锚定阳性对照）。

   * 从 (\theta_0) 人为执行 20–100 步 SFT（监督微调）造成漂移；
   * 停止新数据训练，只运行 anchor（锚定）；
   * 检查它能否把最终 (d_g) 拉回来。

   如果连这个实验都拉不回来，说明是求解器或实现问题，不必跑行为评估。

3. 比较四种稳定机制。

   * 旧专家 replay（回放）；
   * fixed-beta distillation（固定权重蒸馏）；
   * augmented Lagrangian（增广拉格朗日）并使用非零初始 (\lambda)；
   * A-GEM/GEM（平均梯度情景记忆/梯度情景记忆）投影。

   所有方法匹配旧数据量、新数据量、优化步数和反向传播计算量。

4. 重新定义行为端点。

   * Base（基础策略）和新策略使用完全相同的随机数；
   * 使用成功概率差，而不是 2/3 入组后要求 3/3；
   * `boundary`（边界组）先作为 monitor-only（仅监测组）。Base（基础策略）在边界状态本来就不稳定，强行保持它可能是在保存失败行为。

5. 最后才重跑 2×2 因子实验。

   先回答两个独立问题：

   [
   B1-N1>0?
   ]

   [
   \text{稳定机制}-\text{普通回放}>0?
   ]

   只有两者分别成立，再解释 B3（验证片段+稳定机制）的交互效应。

最终判断矩阵应当是：

| 最终内部约束 | 行为保持 | 结论                        |
| ------ | ---- | ------------------------- |
| 失败     | 失败   | 优化器或方法确实失败                |
| 失败     | 通过   | (d_g) 或阈值没有行为意义           |
| 通过     | 失败   | anchor set（锚定集合）或度量保护错了对象 |
| 通过     | 通过   | 才能认为 anchor（锚定）机制有效       |

目前你的结果更接近第一种，但 branch（分支）部分和行为基线仍有设计漏洞。因此我建议先把论文主张降为：

> “同状态反事实分支验证能否提高自生成数据的训练价值”

把防遗忘作为独立模块重新开发。这个方向仍有机会，而且比继续围绕 (10^{-4}) 调对偶学习率更值得做。

