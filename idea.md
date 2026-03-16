最小可行版本是：

ACT + 单任务/少任务 + 一张参考框 + DINOv2/GroundingDINO/SAM2 mask 流水线 + 全背景生成 + 1:5 增强 + 简化版 RCL + LIBERO-Plus 评测。

这条路线最有希望在没有官方代码的情况下，复现出 RoboAug 的核心结论：策略对背景、干扰物和光照的鲁棒性显著提高，而且提升不只是来自“多造数据”，还来自“让视觉表示盯住任务区域”。

RoboTwin 本来就把随机背景、桌面 clutter、随机光照、桌面高度这些扰动放在 `task_config/*.yml` 里；`collect_data.py` 会读取这个配置采集数据，`ACT` 的官方流程则是 `collect_data.sh -> policy/ACT/process_data.sh -> policy/ACT/train.sh`。默认 `demo_randomized.yml` 已经开了 `random_background / cluttered_table / random_light / random_table_height`，但 `mesh_segmentation` 和 `actor_segmentation` 还是关的。([GitHub][1])

## 1. 先改哪些文件

我建议你先只动这 6 个位置。

### A. 新建训练配置文件

复制一份 `task_config/demo_randomized.yml`，改名成：

`task_config/demo_roboaugA_train.yml`

建议先填成这样：

```yaml
render_freq: 0
episode_num: 250
use_seed: false
save_freq: 15
embodiment: [aloha-agilex]
language_num: 100

domain_randomization:
  random_background: true
  cluttered_table: true
  clean_background_rate: 0.05
  random_head_camera_dis: 0
  random_table_height: 0.03
  random_light: true
  crazy_random_light_rate: 0.05
  random_embodiment: false

camera:
  head_camera_type: D435
  wrist_camera_type: D435
  collect_head_camera: true
  collect_wrist_camera: true

data_type:
  rgb: true
  third_view: false
  depth: false
  pointcloud: false
  observer: false
  endpose: true
  qpos: true
  mesh_segmentation: false
  actor_segmentation: true

pcd_down_sample_num: 1024
pcd_crop: true
save_path: ./data
clear_cache_freq: 5
collect_data: true
eval_video_log: false
```

这里最关键的是两点：
第一，把 `episode_num` 直接提到 **250**；第二，把 `actor_segmentation: true` 打开。RoboTwin 的配置系统本来就支持这些字段。([robotwin-platform.github.io][2])

### B. 可选再建一个更强的评测配置

再复制一份做：

`task_config/demo_roboaugA_eval.yml`

把训练和评测分开。训练用 `clean_background_rate: 0.05 / crazy_random_light_rate: 0.05`，评测可以更狠一点，例如 `clean_background_rate: 0.0 / crazy_random_light_rate: 0.10`。`eval.sh` 本来就允许“训练配置”和“测试环境配置”分开传。([robotwin-platform.github.io][3])

### C. 改 `envs/camera/camera.py`

这一处我认为**必须改**。原因是 RoboTwin 现在的 `get_segmentation()` 不是存原始 label id，而是先把 segmentation label 映射成伪彩色 RGB 图再写进数据里；这对可视化很好，但对你后面做“精确的 task-relevant binary/multi-class mask”不够友好。([GitHub][4])

你要把这里改成同时保存：

* `actor_segmentation_id`：原始整型 label map
* `actor_segmentation_vis`：原来的彩色可视化图

我建议把 `get_segmentation()` 改成这个思路：

```python
def get_segmentation(self, level="actor"):
    def _get_segmentation(camera, level="actor"):
        seg_labels = camera.get_picture("Segmentation")  # [H, W, 4]
        if level == "mesh":
            label_id = seg_labels[..., 0].astype(np.uint16)
        elif level == "actor":
            label_id = seg_labels[..., 1].astype(np.uint16)

        colormap = sorted(set(ImageColor.colormap.values()))
        color_palette = np.array([ImageColor.getrgb(color) for color in colormap], dtype=np.uint8)
        label_vis = color_palette[label_id % len(color_palette)]

        return {
            f"{level}_segmentation_id": label_id,
            f"{level}_segmentation_vis": label_vis,
        }
```

这样改完以后，`Base_Task.get_obs()` 不用再动，因为它本来就是把 camera 返回的 dict 直接 merge 进 `observation/<camera>/...`。([GitHub][5])

### D. 改 `policy/ACT/process_data.py`

这个文件现在只做了三件事：读原始 HDF5、解码三路 RGB、resize 到 `640x480`、然后写成 ACT 训练格式；它**没有处理 segmentation**。它最后还会把 `SIM_TASK_CONFIGS.json` 里的 camera names 固定成 `["cam_high", "cam_right_wrist", "cam_left_wrist"]`。([GitHub][6])

所以这里要改成：

1. `load_hdf5()` 额外读 `actor_segmentation_id`
2. `data_transform()` 里把 head camera 的 mask 一起 resize 后写入新数据集
3. 最终输出不只是 `observations/images/*`，还要输出 `observations/masks/*`

我建议**只给 `cam_high` 做 RCL mask**，先别给左右 wrist camera 上 RCL。原因是 RoboAug 的 ACT 版本本身就是“第三视角 RGB + 机器人状态”，而 RoboTwin 当前 ACT 输入虽然有三路图，但 clutter/background/light 的主要干扰都最集中出现在 head camera。先把 `cam_high` 做对，训练最稳。([arXiv][7])

### E. 改 `policy/ACT/utils.py`

`EpisodicDataset.__getitem__()` 现在返回的是：

`image_data, qpos_data, action_data, is_pad`

而且 `load_data()` 会自动做 **80/20 train-val 划分**。([GitHub][8])

你要把它改成：

`image_data, qpos_data, action_data, is_pad, region_mask, region_label`

这里的 `region_mask` 我建议是：

* shape = `[K, H, W]`
* `K` 是 task-relevant semantic classes 数量

单任务先做 2～4 类就够：

* `robot`
* `manipulated_object`
* `goal/receptacle`
* `task_relevant_articulation`（例如 drawer handle / lid，按任务需要）

### F. 改 `policy/ACT/act_policy.py`，必要时再改 `policy/ACT/detr/models/detr_vae.py`

当前 `ACTPolicy.__call__()` 的训练 loss 只有：

`loss = l1 + kl * kl_weight`。([GitHub][9])

你要改成：

`loss = l1 + kl * kl_weight + lambda_rcl * rcl`

RCL 最方便的接法，是在 `act_policy.py` 里额外拿到 backbone feature map，基于 `cam_high` 的 `region_mask` 做 masked average pooling，得到 region embedding，然后做 supervised contrastive loss。

如果你发现 `act_policy.py` 拿不到中间视觉特征，那就在 `policy/ACT/detr/models/detr_vae.py` 里加一个辅助函数，比如：

```python
def extract_visual_feat(self, image, cam_id=0):
    features, pos = self.backbones[0](image[:, cam_id])
    features = features[0]
    feat = self.input_proj(features)
    return feat
```

这样 `ACTPolicy` 在训练时可以调 `self.model.extract_visual_feat(image, cam_id=0)`。当前 DETR-VAE 结构里，backbone 和 `input_proj` 都已经在模型内部了，这样改最顺。([GitHub][10])

---

## 2. 按 RoboAug 需求，到底生成多少条“视频/轨迹”？

我建议你在 **RoboTwin 里直接采 250 条训练轨迹/episodes**，而不是先采 50 条再离线造 200 条视频。

原因很简单：

* RoboAug 的结论是**增强比例 1:3 到 1:8 都有效，1:5 最合适**。([arXiv][7])
* RoboAug 真实机器人实验通常是 **50 条原始轨迹扩到 250 条**。([arXiv][7])
* 但你这里选的是**方案 A**，也就是不用 RoboAug 的生成式换背景，而用 RoboTwin 原生 randomization，所以最自然的等效实现就是：**直接采到 250 条已经随机化的 HDF5 episodes**。训练 ACT 用的是 HDF5 轨迹，不是 mp4。`process_data.sh` 也是按 trajectory 数量 `expert_data_num` 来处理，不是按视频数。([robotwin-platform.github.io][3])

所以我的建议是：

### 你实际要准备三套数据

1. `demo_clean`：**50 条**
   这是你的对照组，验证“没 randomization、没 RCL”的基线。([GitHub][11])

2. `demo_roboaugA_train`：**250 条**
   这是主训练集，等效于 RoboAug 的 1:5 数据规模。([arXiv][7])

3. `demo_randomized` 或 `demo_roboaugA_eval`：评测环境
   用来测泛化，不参与训练。RoboTwin 官方 ACT 评测就是这么分开的。([robotwin-platform.github.io][3])

### 采集命令

```bash
bash collect_data.sh <task_name> demo_clean 0
bash collect_data.sh <task_name> demo_roboaugA_train 0
```

---

## 3. 数据应该怎么处理

### 第一步：先把分割信息真的存进原始 episode

现在 `collect_data.py` 会加载 `task_config/<name>.yml`，然后按配置跑 seed search 和 replay collecting；`Base_Task.get_obs()` 只有在 `mesh_segmentation` / `actor_segmentation` 为真时才会把 segmentation 写进 `observation`。([GitHub][1])

所以：

* 训练配置里开 `actor_segmentation: true`
* collector 不用改入口脚本
* 只要改 `camera.py`，把 raw id 一并存下来

### 第二步：在 `process_data.py` 里把 segmentation 变成 task mask

这里建议你不要直接拿整张 `actor_segmentation_id` 喂训练，而是离线变成**语义 mask**。

我建议在预处理中生成：

* `observations/masks/cam_high`，shape `[T, K, H, W]`
* `observations/mask_labels`，记录第 `k` 个通道代表什么类

其中 `K` 建议先设成：

* `0 = robot`
* `1 = target object`
* `2 = receptacle / goal object`
* `3 = task articulation`（有需要再开）

然后：

* RGB resize 用双线性 / OpenCV 默认
* mask resize 必须用 `INTER_NEAREST`

因为现在 `process_data.py` 会把 RGB 都 resize 到 `640x480`，mask 必须跟着同尺度走，不然 feature pooling 会错位。([GitHub][6])

### 第三步：只对 `cam_high` 做 RCL

原因不是 wrist camera 没用，而是你第一版最需要的是**把 head camera 上的背景/光照/桌面扰动鲁棒性拉起来**。RoboAug 的 ACT 实验也是第三视角主导；而 RoboTwin 当前 ACT 三路图像都在训练里，已经足够复杂了。第一版只对 `cam_high` 产生 region embedding，通常最稳。([arXiv][7])

---

## 4. ACT 训练函数具体怎么改

### 4.1 `process_data.py`

把输出从现在的：

* `action`
* `observations/qpos`
* `observations/images/cam_high`
* `observations/images/cam_right_wrist`
* `observations/images/cam_left_wrist`

扩展成：

* `observations/masks/cam_high`
* `observations/region_labels`

当前官方 `process_data.py` 只处理三路 RGB、qpos、action。([GitHub][6])

### 4.2 `utils.py`

把 dataloader 返回值改成：

```python
return image_data, qpos_data, action_data, is_pad, region_mask, region_label
```

其中：

* `region_mask` 取 `start_ts` 对应那一帧
* `region_label` 用于 supervised contrastive loss 的 class label

### 4.3 `imitate_episodes.py`

现在 `forward_pass(data, policy)` 只解包 4 个张量。你要改成：

```python
def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad, region_mask, region_label = data
    ...
    return policy(qpos_data, image_data, action_data, is_pad, region_mask, region_label)
```

训练循环本身不用大改，还是在 `train_bc()` 里拿 `forward_dict["loss"]` 反传。当前 train/eval 主循环已经很清楚，改这层最省事。([GitHub][12])

### 4.4 `act_policy.py`

我建议你把 `ACTPolicy.__call__()` 改成这种接口：

```python
def __call__(self, qpos, image, actions=None, is_pad=None, region_mask=None, region_label=None):
```

然后训练时多算一个 `rcl`：

```python
loss_dict["l1"] = l1
loss_dict["kl"] = total_kld[0]
loss_dict["rcl"] = rcl
loss_dict["loss"] = loss_dict["l1"] + self.kl_weight * loss_dict["kl"] + self.rcl_weight * loss_dict["rcl"]
```

当前原版只有 `l1 + kl`，所以这一层就是你真正把 RoboAug 接进 RoboTwin ACT 的核心位置。([GitHub][9])

### 4.5 `detr/models/detr_vae.py`

这里加一个视觉特征导出函数，最实用。

RCL 不要从最终 action query 上做，**从 backbone 输出的空间 feature map 做**。这样你可以：

1. 用 `cam_high` 得到 `feat_map: [B, C, Hf, Wf]`
2. 把 `region_mask` resize 到 `[Hf, Wf]`
3. masked average pooling 得到 region embedding
4. `F.normalize`
5. 送进 supervised contrastive loss

一个很实用的写法是：

```python
feat = self.model.extract_visual_feat(image, cam_id=0)   # B,C,Hf,Wf
mask = F.interpolate(region_mask.float(), size=feat.shape[-2:], mode="nearest")
region_feat = (feat.unsqueeze(1) * mask.unsqueeze(2)).sum(dim=[-1, -2]) / (mask.sum(dim=[-1,-2]).unsqueeze(-1) + 1e-6)
region_feat = F.normalize(region_feat, dim=-1)
```

### 4.6 RCL 本身怎么定义

RoboAug 的 region-contrastive 本质是 supervised contrastive：**同语义类为正样本，不同语义类为负样本**，温度参数表里给的是 **0.07**。([arXiv][7])

你在单任务里可以这样定：

* 正样本：不同 batch 中、同一类 region，例如“target object”
* 负样本：不同类 region，例如“robot” vs “target object”，或 “target object” vs “goal object”

先别做太复杂的 hard negative mining，第一版普通 SupCon 就够了。

---

## 5. 训练超参数怎么调

这里有一个很关键的点：**RoboTwin 原生 ACT 和 RoboAug 论文的 ACT 配方不一样。**

RoboTwin 当前 `train.sh` 默认是：

* `batch_size=8`
* `num_epochs=6000`
* `lr=1e-5`
* `hidden_dim=512`
* `chunk_size=50`。([GitHub][13])

而 RoboAug 论文里 ACT 那一列是：

* `batch size=24`
* `lr=1e-4`
* `vision encoder=ResNet50`
* `training step=50K`
* `temperature=0.07`。([arXiv][7])

所以我的建议是分两阶段：

### 第一阶段：最小改动跑通

先保持 RoboTwin ACT 主体不变：

* 先不换 ResNet50
* 先保留 `lr=1e-5`
* 只加 `RCL`
* `expert_data_num=250`

这一步目标是确认：

1. 数据能正确处理
2. mask 对齐没问题
3. loss 能降
4. 泛化比原生 randomized ACT 更好

### 第二阶段：往 RoboAug 靠

等第一阶段稳定后，再做两件事：

1. 把 `imitate_episodes.py` 里的

```python
backbone = "resnet18"
```

改成

```python
backbone = "resnet50"
```

2. 把 `train.sh` 的学习率提高到 `3e-5 ~ 1e-4` 做 sweep

### 训练轮数怎么从 6000 改

RoboTwin 的 `load_data()` 是 **80/20 split**，训练步数近似：

`steps_per_epoch = ceil(0.8 * expert_data_num / batch_size)`。([GitHub][8])

所以如果你现在用：

* `expert_data_num = 250`
* `batch_size = 8`

那每个 epoch 大约 **25 steps**。
如果你想接近 RoboAug 的 **50K training steps**，那就应该把：

`num_epochs ≈ 50000 / 25 = 2000`

所以我建议你把 `train.sh` 改成：

* `batch_size=8`
* `num_epochs=2000`
* `save_freq=500`

而不是继续用默认 6000。默认 6000 对 250 条数据会训练得太久。


参考：https://github.com/RoboTwin-Platform/RoboTwin
参考：https://arxiv.org/abs/2602.14032
