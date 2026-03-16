需要改，但**只改一点点**。

结论是：

* **环境评测流程基本不用改**
* **ACT 的推理主循环基本不用改**
* **数据与 checkpoint 的加载逻辑要小改**
* **如果你把训练时的输入从 RGB 扩成了 “RGB + region branch/RCL”**，那 eval 端必须能构造同样的前向输入，哪怕最后不计算 RCL loss。

## 不需要改的部分

RoboTwin 原生 eval 是这条链路：

1. 用 `eval.sh` 指定任务、测试配置、训练配置、expert_data_num、seed、ckpt
2. 进入 `policy/ACT/eval.py`
3. 调 `imitate_episodes.py --eval`
4. 在环境里 rollout，统计 success rate / return。

你这套里，下面这些通常都**不用动**：

* `eval.sh` 的整体调用方式
* `eval_bc()` 的 rollout 框架
* 环境 `env.reset() / env.step()`
* success rate / return 的统计方式。

因为方案 A 的核心是训练时加 **RCL 正则**，不是把 ACT 变成一个完全不同的 policy family。RCL 是训练约束，eval 时通常不需要再计算训练损失。

## 必须改的部分

### 1. `ACTPolicy` 的 eval forward 要兼容新模型

如果你在训练里把 `ACTPolicy.__call__()` 改成了：

```python
policy(qpos, image, actions=None, is_pad=None, region_mask=None, region_label=None)
```

那 eval 时虽然没有 `actions/is_pad`，也**至少要保证**：

* 新增参数有默认值
* 推理分支在 `actions is None` 时仍然只返回 action
* 不要求一定传 `region_mask` 才能跑。

也就是说，最稳的做法是把 eval forward 设计成：

```python
# train
policy(qpos, image, actions, is_pad, region_mask, region_label)

# eval
policy(qpos, image)
```

这样 `eval_bc()` 几乎不用改。

### 2. 如果你把“mask 也当成模型输入”，eval 也要提供 mask

这里要区分两种接法：

**接法 A：RCL 只是训练附加损失**
这种情况下，eval 不需要 mask，只要 RGB 跑 forward 就行。
这是我更推荐的方式。

**接法 B：mask 进入主干网络，成为推理输入的一部分**
这种情况下，eval 就必须同步生成 mask，否则训练和测试输入分布不一致。
这会让线上评测更复杂。

所以从工程角度，我建议你坚持 **接法 A**：
**mask 只用于训练期的 RCL，不进入 policy 的推理输入。**

这样 eval 几乎不变。

## 我建议你具体改哪几处

### 方案 A 推荐改法：只动 `act_policy.py`

在 `act_policy.py` 里：

* 训练分支：算 `l1 + kl + lambda_rcl * rcl`
* 推理分支：保持原样，只输出 action。

伪代码像这样：

```python
def __call__(self, qpos, image, actions=None, is_pad=None, region_mask=None, region_label=None):
    env_state = None
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    image = normalize(image)

    if actions is None:
        # eval / inference
        a_hat, _, (_, _) = self.model(qpos, image, env_state)
        return a_hat

    # train
    actions = actions[:, :self.model.num_queries]
    is_pad = is_pad[:, :self.model.num_queries]
    a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
    total_kld, *_ = kl_divergence(mu, logvar)
    l1 = ...
    rcl = ...
    return {"l1": l1, "kl": total_kld[0], "rcl": rcl, "loss": l1 + self.kl_weight * total_kld[0] + self.rcl_weight * rcl}
```

这样 `eval_bc()` 里的这句可以继续不变：

```python
all_actions = policy(qpos, curr_image)
```

当前官方 eval 正是这么调用的。

### 你大概率还要改 `imitate_episodes.py` 的 checkpoint config

如果你给模型加了 `rcl_weight` 或新的 backbone/export feature 参数，那 `make_policy()` 使用的 `policy_config` 里也要带上这些字段，否则加载 checkpoint 时配置不完整。当前 `imitate_episodes.py` 是在这里构造 `policy_config` 的。

建议加：

* `rcl_weight`
* `use_rcl`
* `rcl_cam_name` 或 `rcl_cam_id`

但这只影响配置构造，不影响 rollout 框架。

## 环境配置要不要变

要，**但不是因为 eval 逻辑变了，而是因为你要做更有意义的泛化评测**。

RoboTwin 本身就支持把训练配置和测试环境配置分开传；它的 domain randomization 包括 clutter、lighting、background、tabletop height 等，所以你完全可以保留原 eval 流程，只是把 **test config** 设得更难。

我建议保留两套 eval：

### 1. in-distribution eval

训练和测试都用接近的 randomized config。
用途：看收敛和基本成功率。

### 2. harder OOD eval

单独建一个 `demo_roboaugA_eval.yml`，把：

* `clean_background_rate` 更低
* `crazy_random_light_rate` 更高
* clutter 更强。

这样你能真正验证：
**RCL 是否让 policy 更不依赖背景和光照。**


## 最推荐的答案

**最推荐的做法是：eval 流程尽量不改，只让训练时多一个 RCL loss。**

也就是：

* **不改** `eval.sh` 的整体流程
* **不改** `eval_bc()` 的环境 rollout 主体
* **不改** action 预测接口
* **只改** policy 的训练分支
* **可选改** eval config，让测试更难。

一句话总结：

**如果 mask 只用于训练期 RCL，eval 几乎不用改；如果 mask 也进推理，eval 就得同步接入 segmentation 预处理。**

下一步最合适的是把你的实现固定成“训练用 mask，推理不用 mask”这一版。
