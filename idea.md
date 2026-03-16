ACT + 单任务/少任务 + 一张参考框 + DINOv2/GroundingDINO/SAM2 mask 流水线 + 全背景生成 + 1:5 增强 + 简化版 RCL + LIBERO-Plus 评测。
复现：RoboAug 的核心结论——策略对背景、干扰物和光照的鲁棒性显著提高，而且提升不只是来自“多造数据”，还来自“让视觉表示盯住任务区域”。

RoboTwin 本来就把随机背景、桌面 clutter、随机光照、桌面高度这些扰动放在 `task_config/*.yml` 里；`collect_data.py` 会读取这个配置采集数据，`ACT` 的官方流程则是 `collect_data.sh -> policy/ACT/process_data.sh -> policy/ACT/train.sh`。默认 `demo_randomized.yml` 已经开了 `random_background / cluttered_table / random_light / random_table_height`，但 `mesh_segmentation` 和 `actor_segmentation` 还是关的。([GitHub][1])

## 1. 改动文件


### A. 训练配置


`task_config/demo_roboaugA_train.yml`


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
([robotwin-platform.github.io])

### B. 评测配置增强

`task_config/demo_roboaugA_eval.yml`
例如 `clean_background_rate: 0.0 / crazy_random_light_rate: 0.10`

### C. 修改了 `envs/camera/camera.py`

实现task-relevant binary/multi-class mask

同时保存：

* `actor_segmentation_id`：原始整型 label map
* `actor_segmentation_vis`：原来的彩色可视化图

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


### D. 改 `policy/ACT/process_data.py`

这个文件是读原始 HDF5、解码三路 RGB、resize 到 `640x480`、然后写成 ACT 训练格式；最后把 `SIM_TASK_CONFIGS.json` 里的 camera names 固定成 `["cam_high", "cam_right_wrist", "cam_left_wrist"]`。
改成：

1. `load_hdf5()` 额外读 `actor_segmentation_id`
2. `data_transform()` 里把 head camera 的 mask 一起 resize 后写入新数据集
3. 最终输出不只是 `observations/images/*`，还要输出 `observations/masks/*`

只给 `cam_high` 做 RCL mask，左右 wrist camera 上不上 RCL。原因是 RoboAug 的 ACT 版本本身就是“第三视角 RGB + 机器人状态”，而 RoboTwin 当前 ACT 输入虽然有三路图，但 clutter/background/light 的主要干扰都最集中出现在 head camera。

### E. 改 `policy/ACT/utils.py`

`EpisodicDataset.__getitem__()` 现在返回的是：

`image_data, qpos_data, action_data, is_pad`

而且 `load_data()` 会自动做 **80/20 train-val 划分**。

改成：

`image_data, qpos_data, action_data, is_pad, region_mask, region_label`

 `region_mask`实现 ：

* shape = `[K, H, W]`
* `K` 是 task-relevant semantic classes 数量

单任务先做 2～4 类：

* `robot`
* `manipulated_object`
* `goal/receptacle`
* `task_relevant_articulation`（例如 drawer handle / lid，按任务需要）

### F. 改 `policy/ACT/act_policy.py`，必要时再改 `policy/ACT/detr/models/detr_vae.py`

改成：

`loss = l1 + kl * kl_weight + lambda_rcl * rcl`

在 `act_policy.py` 里额外拿到 backbone feature map，基于 `cam_high` 的 `region_mask` 做 masked average pooling，得到 region embedding，然后做 supervised contrastive loss。

在 `policy/ACT/detr/models/detr_vae.py` 里加一个辅助函数：

```python
def extract_visual_feat(self, image, cam_id=0):
    features, pos = self.backbones[0](image[:, cam_id])
    features = features[0]
    feat = self.input_proj(features)
    return feat
```

这样 `ACTPolicy` 在训练时可以调 `self.model.extract_visual_feat(image, cam_id=0)`。当前 DETR-VAE 结构里，backbone 和 `input_proj` 都已经在模型内部了



参考：https://github.com/RoboTwin-Platform/RoboTwin
参考：https://arxiv.org/abs/2602.14032
