我现在做的是：我把robotwin的env和推理和sft分开了，接入了rlinf，我发现针对一个50成功率的模型，我们只需要测试足够多的轨迹，取成功的再sft就能左脚踩右脚把自己叠到一个足够高的成功率，但是显然这个工作应该有问题，解决这个问题或许能发文章，帮我想一想

## 先给判断

你的现象**不是漏洞，也不是“错误地把成功率刷高了”**。它本质上是一种：

> **Success-Filtered Self-Training / Rejection-Sampling Fine-Tuning / Self-Imitation**

即：

1. 当前策略在环境中采样；
2. 用成功判据过滤轨迹；
3. 把成功轨迹当作新的 demonstrations；
4. 再做 SFT；
5. 重复迭代。

这在数学上接近 **Reward-Weighted Regression、EM policy search 和 Cross-Entropy Method**。近期 SEIL 已经明确采用“在模拟器中运行当前策略、保留成功轨迹、迭代做模仿学习”的路线；SIME 也研究了自采集轨迹的多样性和片段筛选。因此，**“成功轨迹再 SFT”本身已经不足以成为论文贡献**。([arXiv][1])

但你现在的系统非常适合研究一个更重要的问题：

> **为什么 success-only SFT 能提高平均成功率，却可能错误地放大容易样本、错误动作和单一行为模式？如何把轨迹级二值成功信号转化成无偏、细粒度的动作学习信号？**

这个方向有机会形成一篇很扎实的文章。

---

# 一、为什么“左脚踩右脚”真的可以成立

设当前策略是 (\pi_k)，一条轨迹为：

[
\tau=(h_0,a_0,h_1,a_1,\ldots,h_T)
]

其中 (h_t) 是图像、机器人状态和历史信息，最终成功标签为：

[
Y(\tau)\in{0,1}
]

你保留成功轨迹再做 SFT，优化的是：

[
\mathcal L_{\mathrm{success\text{-}SFT}}
========================================

-\mathbb E_{\tau\sim p_{\pi_k}(\tau\mid Y=1)}
\left[
\sum_t\log\pi_\theta(a_t\mid h_t)
\right]
]

成功轨迹分布满足：

[
p_{\pi_k}(\tau\mid Y=1)
=======================

\frac{p_{\pi_k}(\tau)Y(\tau)}
{J(\pi_k)}
]

因此它等价于：

[
\mathcal L
\propto
-\mathbb E_{\tau\sim\pi_k}
\left[
Y(\tau)
\sum_t\log\pi_\theta(a_t\mid h_t)
\right]
]

也就是给成功轨迹权重 1、失败轨迹权重 0 的 reward-weighted behavior cloning。已有理论工作也把 filtered SFT 解释为稀疏奖励 RL 目标的一个下界。([arXiv][2])

更直观地看，在一个给定历史 (h) 上，理想的下一轮策略近似为：

[
\pi_{k+1}(a\mid h)
\approx
p_{\pi_k}(a\mid h,Y=1)
]

根据贝叶斯公式：

[
\pi_{k+1}(a\mid h)
==================

\pi_k(a\mid h)
\frac{Q_k(h,a)}
{V_k(h)}
]

其中：

[
Q_k(h,a)=P(Y=1\mid h,a,\pi_k)
]

[
V_k(h)=P(Y=1\mid h,\pi_k)
]

所以：

* (Q_k(h,a)>V_k(h)) 的动作概率被放大；
* (Q_k(h,a)<V_k(h)) 的动作概率被压低。

这其实是一种**乘法权重策略更新**。重复进行：

[
\pi_{k+n}(a\mid h)
\propto
\pi_k(a\mid h)Q(h,a)^n
]

好的动作模式会不断被放大。

因此，一个 50% 成功率的策略尤其适合这样做：

* 成功轨迹足够多，不缺正样本；
* 又有大量失败，说明策略还有提升空间；
* 策略本身已经覆盖了可行解，只需要把概率质量集中到成功模式上。

所以这个现象是合理的，不是训练 bug。

---

# 二、真正的问题一：成功数据严重偏向“容易初始状态”

这是我认为你最值得首先验证的问题。

设 RoboTwin 的初始配置、物体位置和随机种子为 (z)，原始训练分布为：

[
z\sim\rho(z)
]

策略在该配置上的成功概率为：

[
J_k(z)=P_{\pi_k}(Y=1\mid z)
]

经过成功过滤以后，成功数据中的初始配置分布变成：

[
\rho_k^+(z)
===========

# P(z\mid Y=1)

\frac{\rho(z)J_k(z)}
{\mathbb E_{z\sim\rho}[J_k(z)]}
]

这意味着成功率高的配置会被自动过采样。

例如：

* 容易配置成功率：(0.9)
* 困难配置成功率：(0.1)

即使二者在原始环境中各占一半，经过成功过滤后：

[
\frac{P(z_{\mathrm{easy}}\mid Y=1)}
{P(z_{\mathrm{hard}}\mid Y=1)}
==============================

# \frac{0.9}{0.1}

9
]

你的新 SFT 数据里，容易配置会是困难配置的 9 倍。

于是模型可能表现为：

* 平均成功率快速上升；
* 容易样本接近 100%；
* 困难长尾几乎没改善；
* OOD 和 domain randomization 反而下降；
* 如果评测 seed 与采样 seed 重叠，提升会更加夸张。

这可能制造一种 **self-training illusion**：

> 模型不是学会了更多状态，而是在更熟练地重复自己本来就会的状态。

已有大规模 autonomous imitation learning 研究发现，自动采集成功数据在复杂真实任务上的增益往往有限，而且容易被环境设计、reset、成功检测和初始状态分布等因素制约。([arXiv][3])

---

# 三、真正的问题二：整条成功轨迹不代表每一步都是好动作

你现在给整条轨迹统一标签：

```text
成功轨迹：所有 action chunk 都是正样本
失败轨迹：所有 action chunk 都丢弃
```

但真实情况可能是：

### 成功轨迹中包含坏动作

例如：

```text
正常接近
→ 抓偏
→ 物体滑动
→ 策略偶然恢复
→ 最终成功
```

整轨迹 SFT 会连“抓偏”一起学习。

实际上该动作可能满足：

[
A(h_t,a_t)
==========

Q(h_t,a_t)-V(h_t)<0
]

只是后续动作把它救回来了。

### 失败轨迹中包含大量好动作

例如：

```text
正确接近
→ 正确抓取
→ 正确抬起
→ 最后放置差 2 cm
→ 失败
```

整条失败轨迹被丢弃，但前面 80% 的动作可能都是高质量动作。

所以成功过滤存在两种错误：

[
\boxed{\text{Bad-in-Success}}
]

和

[
\boxed{\text{Good-in-Failure}}
]

近期 VLA 后训练正在集中解决这种 trajectory-level reward 的信用分配问题。Feat2Go 尝试从视觉特征构造过程进度价值；Q-VGM 使用动作敏感 critic 对 flow-matching VLA 进行价值引导；TSIL 则只重放快速成功轨迹，并用 value gap 控制哪些片段值得模仿。([arXiv][4])

因此，单纯“整条成功轨迹 SFT”会越来越难体现创新性。

---

# 四、真正的问题三：策略会发生模式坍缩

假设一个任务有三种有效策略：

```text
模式 A：左手先抓，右手辅助
模式 B：右手先抓，左手辅助
模式 C：双手同步接近
```

当前模型的概率可能是：

[
P(A)=0.5,\quad P(B)=0.3,\quad P(C)=0.2
]

如果第一轮成功数据中 A 偶然更多，SFT 会继续提高 A 的概率。下一轮采样中 A 更多，成功数据中 A 又更多。

最终：

[
P(A)\rightarrow1,\qquad
P(B),P(C)\rightarrow0
]

平均 ID 成功率可能上升，但：

* 动作多样性降低；
* 新场景下缺少替代解；
* 一种抓取方式失效时无法切换；
* `pass@K` 可能不升反降；
* 后续 RL 探索能力下降。

SIME 和 SEIL 都已经把“如何维持自采集数据多样性”视为核心问题：前者做 modal-level exploration 和片段筛选，后者加入 EMA 模型、环境变化和信息量选择器。([arXiv][1])

---

# 五、我最推荐的论文方向

## 题目方向

可以暂时叫：

> **Beyond Success Filtering: Difficulty-Balanced Counterfactual Self-Training for Vision-Language-Action Models**

中文可以理解为：

> **超越成功轨迹过滤：面向 VLA 的难度平衡与反事实片段自训练**

核心不是再提出一个“成功后 SFT”，而是解决两个偏差：

[
\boxed{\text{初始配置选择偏差}}
+
\boxed{\text{轨迹级信用分配偏差}}
]

---

# 六、具体方法：同初始状态分组采样

对于每一个 RoboTwin 初始配置 (z_i)，不要只运行一次，而是用不同 stochastic noise 运行 (K) 次：

[
{\tau_i^1,\tau_i^2,\ldots,\tau_i^K}
]

得到配置级成功率：

[
\hat J(z_i)
===========

\frac{1}{K}
\sum_{j=1}^K Y(\tau_i^j)
]

例如：

```text
seed 001：8/8 成功，p̂=1.00
seed 002：4/8 成功，p̂=0.50
seed 003：1/8 成功，p̂=0.125
seed 004：0/8 成功，p̂=0.00
```

这样你就不再只有一个粗糙的 episode success，而是知道：

* 哪些配置已经完全解决；
* 哪些配置处于能力边界；
* 哪些配置几乎不可能成功；
* 同一配置下哪些动作导致成功、哪些动作导致失败。

这比把所有随机轨迹混在一起强很多。

---

# 七、第一项改进：难度平衡成功 SFT

对成功轨迹赋予配置级权重：

[
w_{\mathrm{cfg}}(z)
===================

\operatorname{clip}
\left(
\frac{1}
{(\hat J(z)+\epsilon)^\alpha},
w_{\min},
w_{\max}
\right)
]

其中：

* (\alpha=0)：退化为普通成功 SFT；
* (\alpha=1)：完全逆成功率校正；
* 实际可以取 (0.3\sim0.7)，避免极端放大。

容易配置：

[
\hat J(z)\approx1
\Rightarrow
w_{\mathrm{cfg}}\text{较小}
]

困难但偶尔成功的配置：

[
\hat J(z)\approx0.1
\Rightarrow
w_{\mathrm{cfg}}\text{较大}
]

这样可以纠正：

[
\rho^+(z)\propto\rho(z)J(z)
]

造成的 easy-seed bias。

对于完全零成功的配置，不要直接给无限权重，因为没有正样本。可以：

* 保留原始 expert demonstration；
* 使用邻近配置 curriculum；
* 增大探索噪声；
* 用检索到的相似成功轨迹引导；
* 暂时不进入纯自训练阶段。

---

# 八、第二项改进：反事实 action-chunk 信用分配

这是更有论文价值的部分。

你的 env、inference、SFT 已经拆开，非常适合做 **branch rollout**。

假设执行到状态 (h_t)，保存模拟器 snapshot。然后从当前 VLA 采样 (M) 个 action chunk：

[
a_t^1,a_t^2,\ldots,a_t^M
]

每次恢复到相同 snapshot：

```text
相同状态 h_t
 ├── action chunk 1 → 当前策略继续执行 → 成功
 ├── action chunk 2 → 当前策略继续执行 → 失败
 ├── action chunk 3 → 当前策略继续执行 → 成功
 └── action chunk 4 → 当前策略继续执行 → 失败
```

估计：

[
\hat Q(h_t,a_t^j)
=================

\frac{1}{N}
\sum_{n=1}^{N}
Y(\tau_{t,j,n})
]

再估计：

[
\hat V(h_t)
===========

\frac{1}{M}
\sum_j\hat Q(h_t,a_t^j)
]

得到 chunk advantage：

[
\hat A(h_t,a_t^j)
=================

\hat Q(h_t,a_t^j)-\hat V(h_t)
]

于是：

* 成功轨迹中 (\hat A<0) 的坏片段不再学习；
* 失败轨迹中 (\hat A>0) 的好片段可以保留；
* 不需要手工设计 grasp reward、distance reward；
* 仍然只依赖最终 binary success verifier；
* 不要求 VLA 提供稳定的 action log-probability。

这对 flow-matching VLA 特别有价值，因为你可以继续使用原生 flow-matching SFT loss，而不必直接做 PPO ratio。

---

# 九、最终训练目标

如果你的 VLA 是 flow matching，标准损失一般类似：

[
\mathcal L_{\mathrm{FM}}
========================

\left|
v_\theta(x_s,s,h)
-u_s(a)
\right|^2
]

可以修改为：

[
\mathcal L_{\mathrm{CST}}
=========================

\sum_{i,t}
w_{\mathrm{cfg}}(z_i)
\cdot
[\hat A_{i,t}]*+^\beta
\cdot
\left|
v*\theta(x_s,s,h_{i,t})
-u_s(a_{i,t})
\right|^2
]

其中：

[
[x]_+=\max(x,0)
]

此外保留原始 expert 数据作为 anchor：

[
\mathcal L
==========

\mathcal L_{\mathrm{CST}}
+
\lambda_{\mathrm{expert}}
\mathcal L_{\mathrm{expert}}
+
\lambda_{\mathrm{KL}}
D_{\mathrm{KL}}(\pi_\theta|\pi_0)
]

三个部分分别解决：

* (\mathcal L_{\mathrm{CST}})：利用自采集经验提高成功率；
* (\mathcal L_{\mathrm{expert}})：防止能力漂移和遗忘；
* KL 或参数约束：防止单轮更新过大。

如果模型是 action-token autoregressive VLA，可以直接使用：

[
-,
w_{\mathrm{cfg}}
[\hat A]*+
\log\pi*\theta(a_t\mid h_t)
]

---

# 十、还可以加入“能力边界主动采样”

不是所有配置都有相同的信息量。

### 已完全解决

[
\hat J(z)\approx1
]

继续采样价值很低。

### 完全不会

[
\hat J(z)\approx0
]

纯自采样几乎得不到成功轨迹。

### 能力边界

[
\hat J(z)\approx0.5
]

同一个配置中同时出现成功和失败，最适合研究动作差异。

可以定义采样优先级：

[
s(z)
====

4\hat J(z)(1-\hat J(z))
+
\lambda_{\mathrm{novel}}N(z)
+
\lambda_{\mathrm{hard}}H(z)
]

第一项在 (J=0.5) 时达到最大。

于是训练数据飞轮不再是：

```text
随机采样 → 过滤成功 → SFT
```

而是：

```text
估计各配置成功率
        ↓
选择能力边界与困难长尾
        ↓
同状态多分支 action chunk 采样
        ↓
估计 chunk advantage
        ↓
难度平衡、片段加权 SFT
        ↓
更新能力地图
```

这会比单纯扩大 rollout 数量更有方法贡献。

---

# 十一、必须防止多样性坍缩

对于同一配置的成功轨迹，可以根据以下特征聚类：

* 左右手主从关系；
* 接近方向；
* grasp pose；
* 接触顺序；
* action chunk embedding；
* 末端轨迹；
* 完成时间。

不要直接保留概率最高的所有成功轨迹，而是每个 mode 保留若干代表：

[
D^+
===

\bigcup_{m=1}^{M}
\operatorname{TopK}(D_m^+)
]

训练采样时对 mode 做均衡，而不是按出现频率采样。

建议同时报告：

* average success；
* `pass@1`；
* `pass@K`；
* action-mode entropy；
* 轨迹 embedding coverage；
* 左右手使用比例；
* grasp/contact pattern 数量。

否则平均成功率升高并不能说明策略真的变强。

---

# 十二、最关键的实验设计

建议先不要马上写复杂算法。先做一组能验证问题是否存在的诊断实验。

## 数据划分

对每个任务固定：

```text
训练配置：1000 个随机 seed
ID held-out：500 个完全未参与采样的 seed
Hard split：基础策略成功率最低的 20% 配置
OOD split：更大物体位置、纹理、光照、物理参数变化
```

训练 seed 与评测 seed 必须完全分开。

对每个训练配置运行 (K=8) 或 (K=16) 次，用来估计配置级成功概率。

## 对比方法

至少比较：

| 方法                      | 作用                     |
| ----------------------- | ---------------------- |
| Initial SFT             | 基础模型                   |
| More original SFT       | 排除只是增加训练步数             |
| Success-only SFT        | 你的强基线                  |
| One-success-per-seed    | 排除容易 seed 数量偏差         |
| Difficulty-weighted SFT | 验证配置偏差                 |
| Fast-success SFT        | 对比时长筛选                 |
| Positive-chunk SFT      | 验证片段信用分配               |
| Proposed full method    | 难度校正 + chunk advantage |
| PPO/GRPO                | RLinf 强基线              |
| Offline value method    | 与 Q/critic 类方法比较       |

Q-VGM 已经在 RoboTwin 2.0 上展示了基于自采集 rollout 和 critic 的 flow-VLA 后训练，因此它会是一个重要的同期对照方向。([arXiv][5])

## 不只看平均成功率

必须报告：

[
\text{Mean SR}
]

[
\text{Worst-20% SR}
]

[
\operatorname{CVaR}_{10%}
]

[
\text{Coverage}
===============

\frac{
|{z:\hat J(z)>\delta}|
}{
|\mathcal Z|
}
]

以及：

* unseen-seed success；
* OOD success；
* 每提升 1% 成功率所需 rollout 数；
* 每个配置的成功率分布；
* 模式熵；
* 轨迹长度；
* 碰撞率和动作 jerk；
* 多轮自训练是否出现退化。

---

# 十三、论文最可能成立的结果故事

一个比较理想的实验故事可能是：

### 发现一：普通成功 SFT 提高平均值，但没有扩大能力覆盖

```text
Initial SFT：
平均 50%，困难集 18%

Success-only SFT：
平均 78%，困难集 20%

说明主要是在重复强化容易配置。
```

### 发现二：成功数据的初始状态分布明显偏移

展示：

[
\rho^+(z)\propto\rho(z)J(z)
]

并画出成功数据中基础难度的直方图，证明成功池主要来自原本成功率高的配置。

### 发现三：轨迹级标签包含大量错误监督

通过 branch rollout 证明：

* 成功轨迹中有一部分 chunk 的 advantage 为负；
* 失败轨迹中有大量 chunk 的 advantage 为正；
* 轨迹越长，错误信用分配越严重。

### 发现四：难度平衡 + 片段筛选提高长尾和 OOD

```text
Proposed：
平均 83%，困难集 54%，OOD 61%
```

具体数值当然需要实验，但**平均成功率 + 困难覆盖 + OOD 同时提高**，会比单纯把 50% 做到 95% 有说服力得多。

---

# 十四、与已有工作的差异必须这样组织

你不能把贡献写成：

> 我们发现成功轨迹可以用于自训练。

因为 SEIL、RoboCat、SIME 和很多 autonomous imitation learning 工作已经覆盖了这一点。([arXiv][1])

你应该写成：

> 现有成功过滤自训练把 episode-level outcome 直接传播给整条轨迹，并隐式地将训练初始状态分布从 (\rho(z)) 改变为 (\rho(z)J(z))。我们系统揭示了由此产生的 difficulty bias、credit leakage 和 mode collapse，并提出基于 matched-reset grouped rollouts 的配置校正与反事实 action-chunk 监督方法。

你的差异点可以是：

1. **理论上刻画 success filtering 的配置选择偏差；**
2. **同一个初始配置进行成功/失败配对，而非跨配置比较；**
3. **通过 simulator snapshot 做 action-chunk 反事实分支；**
4. **同时利用 Good-in-Failure 和过滤 Bad-in-Success；**
5. **完全兼容 flow-matching SFT，不依赖动作 log-probability；**
6. **在相同 rollout 预算下与 PPO/GRPO、success SFT 比较。**

在我查到的公开工作中，**“初始配置 propensity correction + matched-reset counterfactual chunk credit + flow-SFT”这个组合没有被直接完整覆盖**；但近期相关工作非常密集，特别是 Q-VGM、Feat2Go、TSIL 和 SEIL，正式投稿前仍需要继续做系统查新。([arXiv][5])

---

## 我的最终建议

你现在最有价值的不是证明：

> 成功轨迹重复 SFT 可以把 50% 提到很高。

这个结果本身大概率只能成为一个强 baseline。

最值得做的是证明：

> **普通成功过滤提升的是已有成功模式的概率，而不一定扩大策略的能力支持域；其根本原因是容易配置偏采样和轨迹级错误信用分配。**

然后提出：

[
\boxed{
\text{Difficulty-Balanced}
+
\text{Matched-Reset}
+
\text{Counterfactual Chunk Credit}
+
\text{Diversity-Preserving SFT}
}
]

你的 RLinf 解耦架构正好适合这个工作，因为环境可以负责 snapshot 和 branch rollout，推理模块批量生成 action chunk，SFT 模块只需要读取加权后的片段数据。相比直接继续调 PPO，这条路线更容易解释、更稳定，也更容易形成清晰的论文问题和消融实验。

[1]: https://arxiv.org/html/2509.19460v1 "Self-evolved Imitation Learning in Simulated World"
[2]: https://arxiv.org/html/2507.12856v1 "Supervised Fine Tuning on Curated Data is Reinforcement Learning (and can be improved)"
[3]: https://arxiv.org/html/2411.01813v1 "So You Think You Can Scale Up Autonomous Robot Data Collection?"
[4]: https://arxiv.org/html/2605.30795 "Feat2Go: Visual Feature-Grounded Value Estimation for Embodied Reinforcement Learning"
[5]: https://arxiv.org/html/2606.08015v1 "Q-VGM: Q-Guided Value-Gradient Matching for Flow-Matching VLA Policies"

	​

你拟定的这套，我用DP可不可行，因为小的验证的快吧

可以。这里按 **DP = Diffusion Policy** 理解。它非常适合做第一版小验证，甚至比直接上 π₀、OpenVLA-OFT 更适合，因为：

* 模型小、训练快；
* RoboTwin 官方已经提供 DP baseline；
* DP 原生输出连续 action chunk；
* 可以通过不同 diffusion noise 从同一状态采样多种动作；
* 加权 SFT 只需要修改 denoising loss，不需要处理 PPO 的 log-probability。RoboTwin 官方仓库目前明确支持 DP，Diffusion Policy 本身也是基于条件扩散生成连续动作序列。([GitHub][1])

但第一版不要一次把我之前提到的四个点都做完。建议先验证最核心的两个假设：

[
\boxed{\text{成功轨迹存在难度偏采样}}
]

和

[
\boxed{\text{轨迹级成功标签存在错误信用分配}}
]

---

# 一、DP 可以怎样接入你的方案

标准 DP 的训练大致是：

[
a^0_{t:t+H}
\xrightarrow{\text{加噪}}
a^k_{t:t+H}
]

模型根据观测 (o_t)、扩散步 (k)，预测噪声：

[
\hat\epsilon
============

\epsilon_\theta(a^k,k,o_t)
]

标准训练损失：

[
\mathcal L_{\mathrm{DP}}
========================

\left|
\epsilon-\epsilon_\theta(a^k,k,o_t)
\right|^2
]

你只需要给每个 action chunk 加一个权重：

[
\mathcal L_{\mathrm{weighted}}
==============================

w_{i,t}
\left|
\epsilon-\epsilon_\theta(a^k_{i,t},k,o_{i,t})
\right|^2
]

其中 (w_{i,t}) 可以依次做成：

### 普通成功过滤

[
w_{i,t}=Y_i
]

成功轨迹中所有 chunk 权重为 1，失败轨迹全部为 0。

### 难度平衡

[
w_{i,t}
=======

Y_i
\cdot
\frac{1}{(\hat J(z_i)+\epsilon)^\alpha}
]

### chunk 价值加权

[
w_{i,t}
=======

[\hat A(o_{i,t},a_{i,t})]_+
]

因此你几乎不需要改 DP 架构，只改：

```python
loss = mse(pred_noise, target_noise, reduction="none")
loss = loss.mean(dim=action_dims)
loss = (loss * chunk_weight).sum() / (chunk_weight.sum() + 1e-6)
```

这正是 DP 做小规模验证的最大优势。

---

# 二、第一版最小验证不要直接做完整反事实树

完整方案需要：

```text
每个状态
→ 采样 M 个 action chunk
→ 每个 chunk 再 rollout N 次
→ 估计 Q
```

计算量是：

[
O(MNTH)
]

第一版没必要这么重。

建议分成三个逐步实验。

---

# 三、实验 A：先证明 success-only DP 确实存在 easy-seed bias

## 1. 选一个任务

优先选：

* 成功判据明确；
* episode 不太长；
* 基础 DP 成功率约 30%～70%；
* 物体初始位置变化能明显改变难度；
* 不需要复杂语言泛化。

例如 RoboTwin 中较适合的：

* `place_empty_cup`
* `adjust_bottle`
* `place_container_plate`
* `click_bell`

RoboTwin 的 RLinf 接口目前也支持这些任务，并使用连续 action chunks 控制双臂。([rlinf.readthedocs.io][2])

你已经有一个约 50% 成功率的模型，直接用它最好。

---

## 2. 固定初始配置集合

不要每条轨迹都随机生成一个全新的 seed。

准备：

```text
训练配置：100 个 seed
验证配置：100 个未见 seed
困难验证：基础模型成功率最低的 20 个 seed
```

对每个训练 seed 执行 (K=8) 次：

[
\hat J(z_i)
===========

\frac{1}{8}
\sum_{j=1}^{8}Y_{ij}
]

总共只需要：

[
100\times8=800
]

条 rollout。

这已经足以看出成功数据是否向容易配置偏移。

---

## 3. 画两张关键图

第一张：

```text
横轴：每个 seed 的基础成功率 J(z)
纵轴：seed 在成功轨迹池中的出现次数
```

理论上应该近似：

[
N_{\mathrm{success}}(z)
\propto J(z)
]

第二张：

```text
横轴：基础模型的 seed 难度分桶
纵轴：SFT 前后成功率
```

可能看到：

```text
基础成功率 0～20%：几乎不提升
基础成功率 20～40%：少量提升
基础成功率 40～60%：明显提升
基础成功率 60～80%：大幅提升
基础成功率 80～100%：接近满分
```

仅这一实验就能确定你的论文问题是否真实存在。

---

# 四、实验 B：只做难度平衡，不做 critic

对同一批成功轨迹训练两个模型。

## Baseline：Success-only DP

[
w_i=1
]

## 方法：Difficulty-balanced DP

[
w_i
===

\operatorname{clip}
\left(
\frac{1}
{(\hat J(z_i)+0.05)^{0.5}},
1,
4
\right)
]

举例：

| seed 成功率 |     权重 |
| -------: | -----: |
|      1.0 |  约 1.0 |
|      0.5 | 约 1.35 |
|     0.25 | 约 1.83 |
|    0.125 | 约 2.39 |

不建议第一版使用完全逆概率：

[
w=\frac1J
]

因为一次偶然成功的极难 seed 会被赋予过大权重，训练容易不稳定。

还可以使用更简单、更稳的分桶均衡：

```text
easy：J > 0.7
medium：0.3 < J ≤ 0.7
hard：0 < J ≤ 0.3
```

每个 batch 从三类中各采样三分之一。

这一版甚至不需要改 loss，只需要改 sampler。

---

# 五、实验 C：做轻量版 Good-in-Failure

完整 branch rollout 之前，先验证：

> 失败轨迹的前半段是不是也包含有价值的动作？

可以利用 RoboTwin 的任务阶段信息、物体状态或简单规则，把任务拆成阶段。

以放杯子为例：

```text
阶段 0：未接近
阶段 1：末端到达物体附近
阶段 2：成功抓取
阶段 3：成功抬起
阶段 4：到达目标附近
阶段 5：放置成功
```

对于一条最终失败但已经达到阶段 3 的轨迹：

```text
正确接近 → 正确抓取 → 正确抬起 → 放置失败
```

保留它达到最高阶段之前的 chunk：

[
D_{\mathrm{prefix}}
===================

{(o_t,a_t)\mid t\le t_{\max\ progress}}
]

训练三组：

1. 只用成功轨迹；
2. 成功轨迹 + 失败轨迹正确前缀；
3. 成功轨迹 + 随机失败前缀。

如果第 2 组明显好于第 1、3 组，就证明：

[
\boxed{\text{Good-in-Failure 确实存在}}
]

这时再投资 branch rollout 和 value estimation 才值得。

---

# 六、随后再做 DP 版本的反事实 chunk 采样

DP 很适合这一步，因为同一个状态可以通过不同初始噪声生成多个 action chunks：

[
a_t^{(m)}
=========

\operatorname{DP}(o_t,\epsilon_m)
]

其中：

[
\epsilon_m\sim\mathcal N(0,I)
]

在相同状态下采样：

[
a_t^{(1)},a_t^{(2)},\ldots,a_t^{(M)}
]

执行每个候选 chunk，然后继续使用当前 DP rollout。

## 最小参数

第一版可以用：

```text
每条轨迹选择 3 个关键状态
每个状态采样 M=4 个 chunk
每个 chunk 只 rollout 1 次
```

也就是每条原始轨迹增加：

[
3\times4=12
]

次分支，而不是在每一个时间步进行分支。

三个关键状态可以是：

* 抓取前；
* 抬起后；
* 放置前。

---

# 七、DP 的 branch rollout 有一个重要细节

DP 一般预测长度为 (H_{\text{pred}}) 的动作序列，但实际只执行前 (H_{\text{exec}}) 步，然后重新规划。

例如：

```text
prediction horizon = 16
execution horizon = 4
```

做 chunk credit 时，建议把一个候选动作定义为：

[
a_t=a_{t:t+H_{\text{exec}}-1}
]

而不是整段 16 步预测。

原因是：

* 后 12 步实际上没有进入环境；
* 对未执行动作赋予 credit 没有意义；
* (H_{\text{exec}}) 太长会把多个决策混为一个；
* (H_{\text{exec}}) 太短则仿真分支数量过大。

第一版建议：

[
H_{\text{exec}}=4\sim8
]

---

# 八、如何估计 chunk 是否更好

最简单可以使用最终成功：

[
\hat Q(o_t,a_t)=Y
]

但一次 rollout 方差较大。

更稳的方法是加入任务进度：

[
\hat Q(o_t,a_t)
===============

Y+
\lambda
\left(
p_{t+H}-p_t
\right)
]

其中 (p_t) 是任务进度，例如：

* 是否抓住物体；
* 物体高度；
* 物体与目标的距离；
* 当前完成的阶段。

第一版可以用：

[
\lambda=0.1
]

最终成功仍然是主信号，进度只用于区分同样失败的分支。

同一状态的 (M) 个动作归一化：

[
A_m
===

Q_m-\frac1M\sum_{j=1}^M Q_j
]

然后只学习：

[
A_m>0
]

的 chunk。

DP 损失变成：

[
\mathcal L
==========

\sum_m
\operatorname{softplus}
\left(
\frac{A_m}{\tau}
\right)
\mathcal L_{\mathrm{DP}}^{(m)}
]

小验证也可以直接二值化：

```python
chunk_weight = (advantage > 0).float()
```

先不要把算法复杂化。

---

# 九、环境快照可能是工程上最大的坑

RoboTwin 基于仿真环境，但“恢复机械臂关节位置和物体 pose”不一定能百分之百恢复：

* 速度；
* 接触状态；
* 约束求解器内部状态；
* 摩擦接触历史；
* 控制器状态。

所以你要先做一个 determinism test：

1. 在时间 (t) 保存状态；
2. 原轨迹继续运行 50 步；
3. 恢复状态；
4. 输入完全相同动作；
5. 比较物体位置和关节轨迹。

定义误差：

[
e_{\mathrm{replay}}
===================

\frac1T
\sum_t
|x_t^{\mathrm{original}}-x_t^{\mathrm{restore}}|
]

如果恢复不稳定，更稳妥的方案是：

> 从相同初始 seed 重新 reset，然后重放前缀动作，直到分支点。

虽然慢一点，但实验可信度更高。

---

# 十、DP 小验证的推荐规模

## 最小可行版本

```text
任务：1 个
训练 seed：100
每 seed rollout：8
总原始 rollout：800
基础模型：成功率约 50%
```

从中预计获得约 400 条成功轨迹。

训练：

```text
原始 expert 数据：保留
新增成功数据：约 400 条
训练 3 个随机种子
```

比较：

| 模型          | 数据                       |
| ----------- | ------------------------ |
| DP-Base     | 原始 expert                |
| DP-Success  | expert + 所有成功轨迹          |
| DP-Balanced | expert + 难度平衡成功轨迹        |
| DP-Prefix   | expert + 平衡成功轨迹 + 正确失败前缀 |

评估：

```text
ID unseen seeds：100 × 每 seed 3 次
Hard seeds：20 × 每 seed 8 次
OOD：100 条
```

---

# 十一、最重要的指标

不要只报告总成功率。

至少报告：

| 指标                 | 说明                     |
| ------------------ | ---------------------- |
| Mean SR            | 总体成功率                  |
| Hard SR            | 原基础模型最困难 20% seed      |
| Solved coverage    | 有至少一次成功的 seed 比例       |
| Worst-bin SR       | 最困难分桶成功率               |
| Success entropy    | 成功轨迹模式多样性              |
| Rollout efficiency | 每 1000 条 rollout 带来的提升 |

其中最关键的是：

[
\text{Solved Coverage}
======================

\frac{
|{z:\hat J(z)>0}|
}{
|\mathcal Z|
}
]

可能出现：

```text
DP-Base：
Mean SR = 50%
Coverage = 68%

DP-Success：
Mean SR = 78%
Coverage = 70%

DP-Balanced：
Mean SR = 76%
Coverage = 85%
```

这会非常有意思：

* success-only 平均分更高；
* difficulty-balanced 真正扩大了会做的配置范围。

这比单纯比较 78% 和 80% 更像论文结果。

---

# 十二、RLinf 在这个实验里怎么用

你已经把 RoboTwin env、inference、SFT 拆开，就不需要为了 DP 强行使用 PPO 模块。

可以保留：

```text
RLinf Env Workers
    ↓ observations
DP Inference Workers
    ↓ action chunks
Trajectory Buffer
    ↓ success / seed / progress / branch id
DP SFT Worker
```

每条数据建议记录：

```python
{
    "task_id": ...,
    "env_seed": ...,
    "rollout_id": ...,
    "parent_rollout_id": ...,
    "branch_state_id": ...,
    "chunk_start_step": ...,
    "obs": ...,
    "action_chunk": ...,
    "success": ...,
    "max_progress": ...,
    "seed_success_rate": ...,
    "chunk_weight": ...,
}
```

RLinf 在这里主要承担：

* 并行 env rollout；
* DP inference worker 调度；
* 数据回传；
* checkpoint 更新；
* 多轮迭代。

不需要第一版就实现 actor/critic 或 GRPO。RLinf 官方当前的 RoboTwin 示例主要面向 OpenVLA-OFT、π₀/π₀.₅ 等 VLA 的 PPO、GRPO 和 DAgger，但它的 env/rollout/data 解耦模式同样可以作为你的 DP 自训练流水线骨架。([rlinf.readthedocs.io][2])

---

## 最终建议

**用 DP 做，方向是对的。**

而且第一版建议只做：

[
\boxed{
\text{Success-only DP}
\quad\text{vs}\quad
\text{Difficulty-balanced DP}
}
]

先证明成功过滤存在 easy-seed bias。

第二步再加入：

[
\boxed{
\text{Good-in-Failure Prefix}
}
]

只有这两项成立后，再做昂贵的：

[
\boxed{
\text{Matched-State Branch Rollout}
+
\text{Chunk Advantage}
}
]

这样最快，也最容易判断这个研究问题是真贡献，还是只在个别任务上的训练技巧。

[1]: https://github.com/RoboTwin-Platform/RoboTwin "GitHub - RoboTwin-Platform/RoboTwin: RoboTwin 2.0 Offical Repo · GitHub"
[2]: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html "RL with RoboTwin Benchmark — RLinf  documentation"


我感觉你这个实验看起来也不太用rlinf啊，我单开一个robotwin也不是不行

对，你这个判断是对的。

**做第一版验证，完全没必要强行用 RLinf。**
单独启动 RoboTwin，写一个 rollout–筛选–再训练循环，反而更快、更容易排查问题。

你当前要验证的是一个**算法现象**：

[
\text{成功轨迹再 SFT}
\rightarrow
\text{成功率提升是否只是容易样本强化}
]

这个问题与分布式框架无关。论文贡献也不应该建立在“用了 RLinf”上，而应该建立在：

* success filtering 为什么有效；
* 它产生了什么偏差；
* 你的校正方法是否改善长尾、覆盖率或泛化。

---

## 最小实验直接这样做

```text
RoboTwin 单进程/少量并行环境
        ↓
加载 DP checkpoint
        ↓
固定 seed 多次 rollout
        ↓
记录 success、seed、trajectory
        ↓
生成 success-only / balanced 数据集
        ↓
调用原来的 DP SFT 脚本
        ↓
重新评估
```

甚至可以拆成三个普通脚本：

```text
collect_rollouts.py
build_dataset.py
train_dp.py
```

### collect_rollouts.py

```python
for env_seed in train_seeds:
    for rollout_seed in range(num_rollouts):
        env.reset(seed=env_seed)
        trajectory = rollout(policy, env, rollout_seed)
        save({
            "env_seed": env_seed,
            "rollout_seed": rollout_seed,
            "success": trajectory.success,
            "observations": trajectory.observations,
            "actions": trajectory.actions,
        })
```

### build_dataset.py

统计每个环境配置的基础成功率：

[
\hat J(z)
=========

\frac{\text{该 seed 成功次数}}
{\text{该 seed 总测试次数}}
]

然后分别构建：

```text
dataset_success
dataset_seed_balanced
dataset_difficulty_weighted
dataset_success_plus_failure_prefix
```

### train_dp.py

只改 sampler 或 loss weight：

```python
loss_per_sample = diffusion_loss(
    observations,
    actions,
    reduction="none",
)

loss = (
    loss_per_sample * sample_weight
).sum() / sample_weight.sum().clamp_min(1e-6)
```

---

# 为什么单开 RoboTwin 更适合现在

## 1. 实验变量更少

用 RLinf 后，你还要排除：

* worker 间 checkpoint 是否同步；
* rollout 模型和训练模型是否一致；
* 数据是否重复或丢失；
* env reset seed 是否正确传递；
* 异步 rollout 是否来自旧策略；
* 多 worker 是否影响可复现性。

而你现在要研究的恰好是数据分布偏差。框架再引入异步和多 worker，会增加很多混淆变量。

---

## 2. 更容易做严格的 seed 控制

你的实验关键不是吞吐，而是：

> 对同一个环境初始状态反复运行当前策略。

需要严格区分：

```text
env_seed：
决定物体位置、姿态、场景配置

policy_seed：
决定 DP diffusion noise

train_seed：
决定网络训练随机性
```

单独脚本非常容易保证：

```python
env.reset(seed=env_seed)
action = policy.sample(obs, generator=policy_generator)
```

使用 RLinf 后要确保这些 seed 在不同 worker、不同进程间不冲突，第一版没有必要增加这个工作。

---

## 3. DP 验证规模并不需要大集群

例如：

```text
100 个环境 seed
每个 seed 运行 8 次
共 800 条轨迹
```

假设每条 episode 1 分钟，单环境确实慢，但开：

```text
8～16 个 RoboTwin env
```

就已经能明显提速。

不一定需要完整 RLinf；可以直接：

* Python multiprocessing；
* 多个独立 Docker；
* shell 脚本按 GPU 和 seed 分片；
* RoboTwin 自身的并行环境能力。

例如：

```bash
python collect.py --seed-start 0   --seed-end 24  --gpu 0 &
python collect.py --seed-start 25  --seed-end 49  --gpu 1 &
python collect.py --seed-start 50  --seed-end 74  --gpu 2 &
python collect.py --seed-start 75  --seed-end 99  --gpu 3 &
wait
```

这对于验证已经足够。

---

# RLinf 什么时候才真正有价值

当你的工作从“一轮离线自训练”变成下面这种闭环时：

```text
并行采样第 k 轮策略
        ↓
在线计算 success / advantage
        ↓
立即更新模型
        ↓
同步新 checkpoint
        ↓
继续第 k+1 轮采样
```

尤其是满足以下情况时，再用 RLinf：

### 1. 需要很多任务和很多环境

```text
20 个 RoboTwin 任务
× 1000 个 seed
× 每 seed 8 次
```

这时 rollout 数量达到十几万，调度框架才开始明显有价值。

### 2. 需要 branch rollout

如果每条轨迹选择多个状态，每个状态执行多个候选 action chunk：

[
N_{\mathrm{rollout}}
====================

N_{\mathrm{episode}}
\times
N_{\mathrm{branch\ state}}
\times
N_{\mathrm{candidate}}
]

例如：

[
1000\times3\times8=24000
]

分支 rollout 数量增长很快，这时 Env Worker 和 Inference Worker 解耦才真正有意义。

### 3. 要和 PPO/GRPO 做统一对照

如果论文中要比较：

```text
Success-SFT
Difficulty-balanced SFT
Chunk-weighted SFT
PPO
GRPO
```

使用同一套 RLinf rollout 基础设施，可以保证环境、模型、计算预算一致。

### 4. 需要异步训练

例如：

```text
GPU 0～3：持续跑 RoboTwin
GPU 4：DP inference
GPU 5～7：持续 SFT
```

环境不等训练，训练也不等环境，这才是 RLinf 的系统优势。

---

# 我建议你的实施顺序

## 第一阶段：完全不用 RLinf

先做四个模型：

```text
DP-Base
DP-Success
DP-Seed-Balanced
DP-Difficulty-Weighted
```

验证：

1. Success-only 是否真的提升；
2. 提升是否主要来自容易 seed；
3. 平均成功率和 seed coverage 是否背离；
4. 新 seed 和困难 seed 是否提升。

这一步的目的只是判断论文假设是否成立。

---

## 第二阶段：仍然可以不用 RLinf

加入：

```text
失败轨迹正确前缀
```

或者少量关键状态 branch rollout。

先在一个任务上做到方法有效。

---

## 第三阶段：再迁入 RLinf

只有在方法成立后，再把它包装成：

```text
Scalable Self-Improvement for Diffusion Policies
```

用 RLinf 做：

* 多任务；
* 大量并行 rollout；
* 多轮在线迭代；
* 和 RL 算法公平对比；
* 吞吐和扩展性分析。

这样 RLinf 是**扩展与系统验证工具**，而不是核心方法依赖。

---

## 一句话判断

**小验证：单开 RoboTwin 最合理。**

**论文方法：不应依赖 RLinf。**

**规模化、多任务、在线迭代和 PPO/GRPO 对比：再用 RLinf。**

你现在最应该先验证的是：成功率从 50% 提高到 80% 时，究竟是更多 seed 被解决了，还是原来会做的 seed 被做得更稳定了。这个实验用单独 RoboTwin 脚本反而最干净。
