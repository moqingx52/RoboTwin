if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hashlib
import hydra
import torch
import torch.nn as nn
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy

import tqdm, random
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class BraceDualState(nn.Module):
    """Checkpointable non-negative multipliers for BRACE protection groups."""

    def __init__(self, groups):
        super().__init__()
        self.groups = tuple(groups)
        self.register_buffer("values", torch.zeros(len(self.groups), dtype=torch.float32))

    def update(self, constraints, epsilons, lr):
        with torch.no_grad():
            for idx, group in enumerate(self.groups):
                if group in constraints:
                    self.values[idx].add_(lr * (constraints[group].detach() - epsilons[group])).clamp_(min=0.0)

    def as_dict(self):
        return {group: float(self.values[idx].detach().cpu()) for idx, group in enumerate(self.groups)}


def module_sha256(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _index_obs(obs, indices):
    return {key: value.index_select(0, indices) for key, value in obs.items()}


def compute_brace_anchor_loss(student, teacher, batch, cfg, dual_state):
    """Functional denoiser constraint with shared teacher action, noise and t."""
    source = batch.get("sample_source")
    if source is None:
        raise ValueError("BRACE anchor requires batch['sample_source'] group labels")
    group_sources = {"base_solved": 0, "boundary": 1}
    configured = OmegaConf.select(cfg, "training.brace_anchor.groups", default=group_sources)
    group_sources = {str(key): int(value) for key, value in dict(configured).items()}
    max_per_group = int(OmegaConf.select(cfg, "training.brace_anchor.samples_per_group", default=8))
    epsilon_cfg = OmegaConf.select(cfg, "training.brace_anchor.epsilon", default=1e-4)
    if isinstance(epsilon_cfg, (float, int)):
        epsilons = {group: float(epsilon_cfg) for group in group_sources}
    else:
        epsilons = {group: float(epsilon_cfg[group]) for group in group_sources}

    constraints = {}
    weighted = student.model.weight.new_zeros(()) if hasattr(student.model, "weight") else next(student.parameters()).new_zeros(())
    student_was_training = student.training
    student.eval()  # fixes crop selection; gradients remain enabled
    teacher.eval()
    try:
        for dual_idx, (group, source_id) in enumerate(group_sources.items()):
            indices = torch.nonzero(source == source_id, as_tuple=False).flatten()[:max_per_group]
            if indices.numel() == 0:
                continue
            obs = _index_obs(batch["obs"], indices)
            with torch.no_grad():
                clean_action = teacher.predict_action(obs)["action_pred"]
                noise = torch.randn_like(clean_action)
                timesteps = torch.randint(
                    0,
                    teacher.noise_scheduler.config.num_train_timesteps,
                    (clean_action.shape[0],),
                    device=clean_action.device,
                ).long()
                noisy_action = teacher.make_noisy_action(clean_action, noise, timesteps)
                teacher_pred = teacher.denoise_action(obs, noisy_action, timesteps)
            student_pred = student.denoise_action(obs, noisy_action, timesteps)
            constraint = torch.mean((student_pred - teacher_pred.detach()) ** 2)
            constraints[group] = constraint
            weighted = weighted + dual_state.values[dual_idx] * (constraint - epsilons[group])
    finally:
        student.train(student_was_training)
    return weighted, constraints, epsilons


def _masked_mean(values, mask, weights=None):
    if not bool(mask.any()):
        return None
    selected = values[mask]
    if weights is not None:
        selected_weights = weights[mask]
        return (selected * selected_weights).sum() / selected_weights.sum().clamp_min(1e-6)
    return selected.mean()


def aggregate_training_loss(model, batch, cfg):
    loss_mode = OmegaConf.select(cfg, "training.loss_mode", default="pooled")
    per_sample_loss = model.compute_loss(batch, per_sample=True)
    sample_weight = batch.get("sample_weight")
    if sample_weight is not None:
        if sample_weight.ndim > 1:
            sample_weight = sample_weight.float().mean(dim=tuple(range(1, sample_weight.ndim)))
        sample_weight = sample_weight.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)

    if loss_mode == "pooled":
        if sample_weight is not None:
            loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
        else:
            loss = per_sample_loss.mean()
        return loss, {}

    if loss_mode != "source_separated":
        raise ValueError(f"Unsupported training.loss_mode: {loss_mode}")

    sample_source = batch.get("sample_source")
    if sample_source is None:
        raise ValueError("training.loss_mode=source_separated requires batch['sample_source'].")

    expert_mask = sample_source == 0
    rollout_mask = sample_source == 1
    prefix_mask = sample_source == 2
    expert_loss = _masked_mean(per_sample_loss, expert_mask, sample_weight)
    rollout_loss = _masked_mean(per_sample_loss, rollout_mask, sample_weight)
    prefix_loss = _masked_mean(per_sample_loss, prefix_mask, sample_weight)

    lambda_expert = float(OmegaConf.select(cfg, "training.lambda_expert", default=1.0))
    lambda_rollout = float(OmegaConf.select(cfg, "training.lambda_rollout", default=0.0))
    lambda_prefix = float(OmegaConf.select(cfg, "training.lambda_prefix", default=0.0))

    if expert_loss is None and rollout_loss is None and prefix_loss is None:
        raise ValueError("source_separated loss received an empty batch.")
    if expert_loss is None:
        raise ValueError("source_separated loss requires expert samples in the batch.")
    if rollout_loss is None and lambda_rollout > 0:
        raise ValueError("source_separated loss requires rollout samples when lambda_rollout > 0.")
    if prefix_loss is None and lambda_prefix > 0:
        raise ValueError("source_separated loss requires prefix samples when lambda_prefix > 0.")

    loss = lambda_expert * expert_loss
    if rollout_loss is not None:
        loss = loss + lambda_rollout * rollout_loss
    if prefix_loss is not None:
        loss = loss + lambda_prefix * prefix_loss

    aux = {
        "expert_count": int(expert_mask.sum().item()),
        "rollout_count": int(rollout_mask.sum().item()),
        "prefix_count": int(prefix_mask.sum().item()),
        "expert_loss": float(expert_loss.detach().item()),
        "rollout_loss": float(rollout_loss.detach().item()) if rollout_loss is not None else None,
        "prefix_loss": float(prefix_loss.detach().item()) if prefix_loss is not None else None,
        "weighted_expert_contrib": float((lambda_expert * expert_loss).detach().item()),
        "weighted_rollout_contrib": (
            float((lambda_rollout * rollout_loss).detach().item()) if rollout_loss is not None else 0.0
        ),
        "weighted_prefix_contrib": (
            float((lambda_prefix * prefix_loss).detach().item()) if prefix_loss is not None else 0.0
        ),
        "lambda_expert": lambda_expert,
        "lambda_rollout": lambda_rollout,
        "lambda_prefix": lambda_prefix,
    }
    return loss, aux


def apply_normalizer_from_config(model, ema_model, dataset, cfg):
    source = OmegaConf.select(cfg, "training.normalizer_source", default="dataset")
    if source == "dataset":
        normalizer = dataset.get_normalizer()
        model.set_normalizer(normalizer)
        if ema_model is not None:
            ema_model.set_normalizer(normalizer)
        return "dataset"
    if source == "checkpoint":
        return "checkpoint"
    raise ValueError(f"Unsupported training.normalizer_source: {source}")


class RobotWorkspace(BaseWorkspace):
    include_keys = [
        "global_step",
        "epoch",
        "brace_teacher_sha256",
        "teacher_checkpoint_path",
        "teacher_reference",
    ]
    exclude_keys = ("brace_teacher",)

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        self.brace_teacher = None
        self.brace_dual_state = None
        self.brace_teacher_sha256 = None
        self.teacher_checkpoint_path = None
        self.teacher_reference = "raw"
        if bool(OmegaConf.select(cfg, "training.brace_anchor.enabled", default=False)):
            groups = OmegaConf.select(
                cfg,
                "training.brace_anchor.groups",
                default={"base_solved": 0, "boundary": 1},
            )
            self.brace_teacher = copy.deepcopy(self.model)
            self.brace_teacher.requires_grad_(False)
            self.brace_dual_state = BraceDualState(dict(groups).keys())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def _rebuild_brace_teacher_from_checkpoint(self) -> None:
        if self.brace_teacher is None:
            return
        if not self.teacher_checkpoint_path:
            raise RuntimeError("BRACE resume missing teacher_checkpoint_path")
        teacher_path = pathlib.Path(self.teacher_checkpoint_path)
        if not teacher_path.is_file():
            raise RuntimeError(f"BRACE teacher checkpoint missing: {teacher_path}")
        payload = torch.load(teacher_path.open("rb"), pickle_module=dill, map_location="cpu")
        state_dicts = payload.get("state_dicts", {})
        if "brace_teacher" in state_dicts:
            self.brace_teacher.load_state_dict(state_dicts["brace_teacher"])
        elif "model" in state_dicts:
            self.brace_teacher.load_state_dict(state_dicts["model"])
        else:
            raise RuntimeError(f"BRACE teacher checkpoint has no model weights: {teacher_path}")
        actual_teacher_hash = module_sha256(self.brace_teacher)
        if self.brace_teacher_sha256 and self.brace_teacher_sha256 != actual_teacher_hash:
            raise RuntimeError(
                "Frozen BRACE teacher hash changed across resume: "
                f"stored={self.brace_teacher_sha256}, actual={actual_teacher_hash}"
            )

    def load_checkpoint(self, path=None, tag="latest", exclude_keys=None, include_keys=None, **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)
        payload = torch.load(path.open("rb"), pickle_module=dill, **kwargs)
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys, **kwargs)
        if self.brace_teacher is not None and "brace_teacher" not in payload.get("state_dicts", {}):
            if self.teacher_checkpoint_path:
                self._rebuild_brace_teacher_from_checkpoint()
        return payload

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        seed = cfg.training.seed
        head_camera_type = cfg.head_camera_type

        # resume training
        resume_training_ckpt = OmegaConf.select(cfg, "training.resume_training_ckpt", default=None)
        if resume_training_ckpt:
            resume_training_ckpt = pathlib.Path(resume_training_ckpt)
            print(f"Resuming full training state from {resume_training_ckpt}")
            self.load_checkpoint(path=resume_training_ckpt)
            try:
                self.epoch = int(resume_training_ckpt.stem)
            except ValueError as exc:
                raise ValueError(
                    f"Training resume checkpoint must have a numeric filename: {resume_training_ckpt}"
                ) from exc
        elif cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        resume_from_ckpt = OmegaConf.select(cfg, "training.resume_from_ckpt", default=None)
        if resume_from_ckpt and not resume_training_ckpt:
            print(f"Loading model weights from {resume_from_ckpt}")
            self.load_checkpoint(
                path=resume_from_ckpt,
                exclude_keys=("optimizer", ),
                include_keys=(),
            )
            self.global_step = 0
            self.epoch = 0

        anchor_enabled = bool(OmegaConf.select(cfg, "training.brace_anchor.enabled", default=False))
        if anchor_enabled:
            if not resume_training_ckpt:
                self.teacher_checkpoint_path = str(pathlib.Path(resume_from_ckpt).resolve())
                self.teacher_reference = "raw"
                self.brace_teacher.load_state_dict(self.model.state_dict())
                self.brace_teacher_sha256 = module_sha256(self.brace_teacher)
                self.brace_dual_state.values.zero_()
            else:
                self._rebuild_brace_teacher_from_checkpoint()

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = create_dataloader(dataset, **cfg.dataloader)
        normalizer_source = apply_normalizer_from_config(
            self.model,
            self.ema_model if cfg.training.use_ema else None,
            dataset,
            cfg,
        )
        print(f"Using normalizer_source={normalizer_source}")

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) //
            cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            # EMAModel is a lightweight helper rather than a workspace member;
            # reconstruct its schedule position from the checkpointed step.
            ema.optimization_step = int(self.global_step)
            ema.decay = ema.get_decay(ema.optimization_step)

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        env_runner = None

        # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"),
                                             **cfg.checkpoint.topk)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        if self.brace_teacher is not None:
            self.brace_teacher.to(device)
            self.brace_teacher.eval()
            self.brace_teacher.requires_grad_(False)
        if self.brace_dual_state is not None:
            self.brace_dual_state.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        log_source_loss_every = int(OmegaConf.select(cfg, "training.log_source_loss_every", default=0) or 0)
        log_source_grad_norm_every = int(
            OmegaConf.select(cfg, "training.log_source_grad_norm_every", default=0) or 0
        )

        with JsonLogger(log_path) as json_logger:
            stop_after_epoch = OmegaConf.select(
                cfg,
                "training.stop_after_epoch",
                default=cfg.training.num_epochs,
            )
            if stop_after_epoch is None:
                stop_after_epoch = cfg.training.num_epochs
            stop_after_epoch = int(stop_after_epoch)
            if not self.epoch <= stop_after_epoch <= cfg.training.num_epochs:
                raise ValueError(
                    f"Expected current epoch <= stop_after_epoch <= num_epochs, got "
                    f"{self.epoch} <= {stop_after_epoch} <= {cfg.training.num_epochs}"
                )
            for local_epoch_idx in range(self.epoch, stop_after_epoch):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        # compute loss
                        raw_loss, loss_aux = aggregate_training_loss(self.model, batch, cfg)
                        anchor_constraints = {}
                        anchor_epsilons = {}
                        anchor_term = raw_loss.new_zeros(())
                        if anchor_enabled:
                            anchor_term, anchor_constraints, anchor_epsilons = compute_brace_anchor_loss(
                                self.model,
                                self.brace_teacher,
                                batch,
                                cfg,
                                self.brace_dual_state,
                            )
                        total_loss = raw_loss + anchor_term
                        loss = total_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if (self.global_step % cfg.training.gradient_accumulate_every == 0):
                            if (
                                log_source_grad_norm_every > 0
                                and loss_aux
                                and (self.global_step % log_source_grad_norm_every == 0)
                            ):
                                total_norm = 0.0
                                for param in self.model.parameters():
                                    if param.grad is not None:
                                        total_norm += float(param.grad.data.norm(2).item() ** 2)
                                loss_aux["grad_norm_total"] = float(total_norm ** 0.5)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            if anchor_enabled:
                                dual_lr = float(OmegaConf.select(cfg, "training.brace_anchor.dual_lr"))
                                self.brace_dual_state.update(anchor_constraints, anchor_epsilons, dual_lr)

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "total_loss": float(total_loss.detach().item()),
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                            "normalizer_source": normalizer_source,
                        }
                        if anchor_enabled:
                            step_log["brace_teacher_sha256"] = self.brace_teacher_sha256
                            for group, value in anchor_constraints.items():
                                step_log[f"brace_constraint/{group}"] = float(value.detach().item())
                            for group, value in self.brace_dual_state.as_dict().items():
                                step_log[f"brace_dual/{group}"] = value
                        if loss_aux and (
                            log_source_loss_every > 0 and self.global_step % log_source_loss_every == 0
                        ):
                            step_log.update(loss_aux)

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps
                                is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                                val_dataloader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps
                                        is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if (
                    ((self.epoch + 1) % cfg.training.checkpoint_every) == 0
                    or (self.epoch + 1) == stop_after_epoch
                ):
                    # checkpointing
                    save_name = OmegaConf.select(cfg, "training.checkpoint_name", default=None)
                    if not save_name:
                        save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                    self.save_checkpoint(f"checkpoints/{save_name}-{seed}/{self.epoch + 1}.ckpt")  # TODO

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


class BatchSampler:

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
        num_batches: int = None,
        sample_sources: np.ndarray = None,
        expert_ratio: float = None,
        rollout_per_batch: int = None,
        prefix_per_batch: int = None,
        sample_groups: np.ndarray = None,
        group_stratified_rollout: bool = False,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.natural_num_batch = data_size // batch_size
        if self.natural_num_batch == 0 and num_batches is not None:
            raise ValueError(
                f"Cannot request fixed batches: dataset has {data_size} samples, "
                f"fewer than batch_size={batch_size}."
            )
        self.num_batch = int(num_batches) if num_batches is not None else self.natural_num_batch
        if self.num_batch < 0:
            raise ValueError(f"num_batches must be positive, got {self.num_batch}.")
        self.discard = data_size - batch_size * self.natural_num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None
        self.expert_ratio = expert_ratio
        self.rollout_per_batch = rollout_per_batch
        self.prefix_per_batch = prefix_per_batch or 0
        self.group_stratified_rollout = group_stratified_rollout
        self.expert_indices = None
        self.rollout_indices = None
        self.prefix_indices = None
        self.rollout_group_ids = None
        self._rollout_group_cursor = 0

        if rollout_per_batch is not None:
            if sample_sources is None or len(sample_sources) != data_size:
                raise ValueError("rollout_per_batch requires sample_sources aligned with dataset indices.")
            self.expert_indices = np.flatnonzero(np.asarray(sample_sources) == 0)
            self.rollout_indices = np.flatnonzero(np.asarray(sample_sources) == 1)
            self.prefix_indices = np.flatnonzero(np.asarray(sample_sources) == 2)
            self.expert_per_batch = batch_size - int(rollout_per_batch) - int(self.prefix_per_batch)
            if self.expert_per_batch < 0:
                raise ValueError("rollout_per_batch + prefix_per_batch exceeds batch_size.")
            if self.expert_per_batch and len(self.expert_indices) == 0:
                raise ValueError("Requested expert samples but the dataset has no expert indices.")
            if rollout_per_batch and len(self.rollout_indices) == 0:
                raise ValueError("Requested rollout samples but the dataset has no rollout indices.")
            if self.prefix_per_batch and len(self.prefix_indices) == 0:
                raise ValueError("Requested prefix samples but the dataset has no prefix indices.")
            if group_stratified_rollout and sample_groups is not None:
                self.rollout_group_ids = np.asarray(sample_groups)[self.rollout_indices]
            print(
                "Using fixed source budget batches: "
                f"expert={self.expert_per_batch}, rollout={rollout_per_batch}, "
                f"prefix={self.prefix_per_batch}, group_stratified_rollout={group_stratified_rollout}"
            )
        elif expert_ratio is not None:
            if not 0.0 <= float(expert_ratio) <= 1.0:
                raise ValueError(f"expert_ratio must be in [0, 1], got {expert_ratio}.")
            if sample_sources is None or len(sample_sources) != data_size:
                raise ValueError("expert_ratio requires one sample source label per dataset index.")
            self.expert_indices = np.flatnonzero(np.asarray(sample_sources) == 0)
            self.rollout_indices = np.flatnonzero(np.asarray(sample_sources) == 1)
            expert_per_batch = int(round(batch_size * float(expert_ratio)))
            rollout_per_batch = batch_size - expert_per_batch
            if expert_per_batch and len(self.expert_indices) == 0:
                raise ValueError("Requested expert samples but the dataset has no expert indices.")
            if rollout_per_batch and len(self.rollout_indices) == 0:
                raise ValueError("Requested rollout samples but the dataset has no rollout indices.")
            self.expert_per_batch = expert_per_batch
            self.rollout_per_batch = rollout_per_batch
            print(
                "Using source-aware batches: "
                f"expert={self.expert_per_batch}, rollout={self.rollout_per_batch}, "
                f"expert_pool={len(self.expert_indices)}, rollout_pool={len(self.rollout_indices)}"
            )

    def _sample_stratified_rollout(self, rng, size: int):
        if self.rollout_group_ids is None or len(self.rollout_indices) == 0:
            return rng.choice(
                self.rollout_indices,
                size=size,
                replace=len(self.rollout_indices) < size,
            )
        unique_groups = np.unique(self.rollout_group_ids)
        picks = []
        group_cursor = self._rollout_group_cursor
        while len(picks) < size:
            group = unique_groups[group_cursor % len(unique_groups)]
            group_cursor += 1
            pool = self.rollout_indices[self.rollout_group_ids == group]
            if len(pool) == 0:
                continue
            picks.append(rng.choice(pool))
        self._rollout_group_cursor = group_cursor
        return np.asarray(picks, dtype=np.int64)

    def __iter__(self):
        if self.rollout_per_batch is not None and self.expert_indices is not None:
            rng = self.rng if self.rng is not None else np.random.default_rng(0)
            for _ in range(self.num_batch):
                parts = []
                if self.expert_per_batch:
                    parts.append(
                        rng.choice(
                            self.expert_indices,
                            size=self.expert_per_batch,
                            replace=len(self.expert_indices) < self.expert_per_batch,
                        )
                    )
                if self.rollout_per_batch:
                    if self.group_stratified_rollout:
                        parts.append(self._sample_stratified_rollout(rng, self.rollout_per_batch))
                    else:
                        parts.append(
                            rng.choice(
                                self.rollout_indices,
                                size=self.rollout_per_batch,
                                replace=len(self.rollout_indices) < self.rollout_per_batch,
                            )
                        )
                if self.prefix_per_batch:
                    parts.append(
                        rng.choice(
                            self.prefix_indices,
                            size=self.prefix_per_batch,
                            replace=len(self.prefix_indices) < self.prefix_per_batch,
                        )
                    )
                batch = np.concatenate(parts).astype(np.int64, copy=False)
                if self.shuffle:
                    rng.shuffle(batch)
                yield batch
            return

        yielded = 0
        while yielded < self.num_batch:
            if self.shuffle:
                indices = self.rng.permutation(self.data_size)
            else:
                indices = np.arange(self.data_size)
            if self.natural_num_batch == 0:
                return
            if self.discard > 0:
                indices = indices[:-self.discard]
            indices = indices.reshape(self.natural_num_batch, self.batch_size)
            for batch in indices:
                if yielded >= self.num_batch:
                    return
                yield batch
                yielded += 1

    def __len__(self):
        return self.num_batch


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
    num_batches: int = None,
    expert_ratio: float = None,
    rollout_per_batch: int = None,
    prefix_per_batch: int = None,
    group_stratified_rollout: bool = False,
):
    batch_sampler = BatchSampler(
        len(dataset),
        batch_size,
        shuffle=shuffle,
        seed=seed,
        drop_last=True,
        num_batches=num_batches,
        sample_sources=getattr(dataset, "sample_sources", None),
        expert_ratio=expert_ratio,
        rollout_per_batch=rollout_per_batch,
        prefix_per_batch=prefix_per_batch,
        sample_groups=getattr(dataset, "sample_groups", None),
        group_stratified_rollout=group_stratified_rollout,
    )

    def collate(x):
        assert len(x) == 1
        return x[0]

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    return dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
