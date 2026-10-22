import argparse
import csv
import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from torch.optim import lr_scheduler
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef
from pathlib import Path
import numpy as np

from dataset_moe_phys import PhysicsMoEDataset, collate_fn_moe_phys
from model_softmoe_phys import PhysicsMoE, normalize_density_to_electrons
from physics_utils import PhysicsUtils
from training_protocol import stage_is_enabled

torch.set_float32_matmul_precision('high')
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


MODEL_SCALE_PRESETS = {
    "base": {
        "backbone_channels": 64,
        "guide_hidden_channels": 32,
        "expert_hidden_channels": 32,
        "bottleneck_modes": (4, 4, 4),
        "norm_groups": 8,
        "backbone_channel_multipliers": (1, 2, 4, 8),
        "encoder_block_counts": (1, 1, 1, 1),
        "decoder_block_counts": (1, 1, 1),
        "shared_expert_depth": 2,
    },
    "large": {
        "backbone_channels": 96,
        "guide_hidden_channels": 64,
        "expert_hidden_channels": 64,
        "bottleneck_modes": (6, 6, 6),
        "norm_groups": 8,
        "backbone_channel_multipliers": (1, 2, 4, 8),
        "encoder_block_counts": (1, 1, 1, 1),
        "decoder_block_counts": (1, 1, 1),
        "shared_expert_depth": 2,
    },
    "xlarge": {
        "backbone_channels": 128,
        "guide_hidden_channels": 96,
        "expert_hidden_channels": 96,
        "bottleneck_modes": (8, 8, 8),
        "norm_groups": 16,
        "backbone_channel_multipliers": (1, 2, 4, 8),
        "encoder_block_counts": (1, 1, 1, 1),
        "decoder_block_counts": (1, 1, 1),
        "shared_expert_depth": 2,
    },
    "xxlarge": {
        "backbone_channels": 160,
        "guide_hidden_channels": 128,
        "expert_hidden_channels": 128,
        "bottleneck_modes": (10, 10, 10),
        "norm_groups": 16,
        "backbone_channel_multipliers": (1, 2, 4, 8),
        "encoder_block_counts": (1, 1, 1, 1),
        "decoder_block_counts": (1, 1, 1),
        "shared_expert_depth": 2,
    },
    "large_2m": {
        "backbone_channels": 96,
        "guide_hidden_channels": 64,
        "expert_hidden_channels": 64,
        "bottleneck_modes": (6, 6, 5),
        "norm_groups": 8,
        "backbone_channel_multipliers": (1, 2, 4, 8),
        "encoder_block_counts": (1, 2, 2, 2),
        "decoder_block_counts": (2, 2, 1),
        "shared_expert_depth": 2,
    },
    "xlarge_2m": {
        "backbone_channels": 128,
        "guide_hidden_channels": 96,
        "expert_hidden_channels": 96,
        "bottleneck_modes": (8, 8, 5),
        "norm_groups": 16,
        "backbone_channel_multipliers": (1, 2, 4, 8),
        "encoder_block_counts": (1, 2, 2, 2),
        "decoder_block_counts": (2, 2, 1),
        "shared_expert_depth": 2,
    },
    "xlarge_safe_2m": {
        "backbone_channels": 112,
        "guide_hidden_channels": 80,
        "expert_hidden_channels": 80,
        "bottleneck_modes": (6, 6, 5),
        "norm_groups": 16,
        "backbone_channel_multipliers": (1, 2, 4, 8),
        "encoder_block_counts": (1, 2, 2, 2),
        "decoder_block_counts": (2, 2, 1),
        "shared_expert_depth": 2,
    },
    "xlarge_mid_2m": {
        "backbone_channels": 140,
        "guide_hidden_channels": 104,
        "expert_hidden_channels": 104,
        "bottleneck_modes": (6, 6, 5),
        "norm_groups": 16,
        "backbone_channel_multipliers": (1, 2, 4, 8),
        "encoder_block_counts": (1, 2, 2, 2),
        "decoder_block_counts": (2, 2, 1),
        "shared_expert_depth": 2,
    },
    "wide_2m": {
        "backbone_channels": 160,
        "guide_hidden_channels": 128,
        "expert_hidden_channels": 96,
        "bottleneck_modes": (8, 8, 5),
        "norm_groups": 16,
        "backbone_channel_multipliers": (1, 2, 4, 8),
        "encoder_block_counts": (1, 2, 2, 2),
        "decoder_block_counts": (2, 2, 1),
        "shared_expert_depth": 2,
    },
    "wide_lite_2m": {
        "backbone_channels": 160,
        "guide_hidden_channels": 128,
        "expert_hidden_channels": 96,
        "bottleneck_modes": (6, 6, 5),
        "norm_groups": 16,
        "backbone_channel_multipliers": (1, 2, 4, 8),
        "encoder_block_counts": (1, 2, 2, 2),
        "decoder_block_counts": (2, 2, 1),
        "shared_expert_depth": 2,
    },
}


def parse_limit_batches(value):
    if isinstance(value, (int, float)):
        return value
    text = str(value)
    if "." in text:
        return float(text)
    return int(text)


def count_parameters(module: nn.Module):
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return total, trainable


def format_param_count(num_params: int):
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.3f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.3f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    return str(num_params)


def resolve_model_scale_config(args):
    preset = MODEL_SCALE_PRESETS[args.model_scale].copy()
    preset["backbone_channels"] = args.backbone_channels or preset["backbone_channels"]
    preset["guide_hidden_channels"] = args.guide_hidden_channels or preset["guide_hidden_channels"]
    preset["expert_hidden_channels"] = args.expert_hidden_channels or preset["expert_hidden_channels"]
    preset["norm_groups"] = args.norm_groups or preset["norm_groups"]
    if args.bottleneck_modes is not None:
        preset["bottleneck_modes"] = tuple(args.bottleneck_modes)
    if args.backbone_channel_multipliers is not None:
        preset["backbone_channel_multipliers"] = tuple(args.backbone_channel_multipliers)
    if args.encoder_block_counts is not None:
        preset["encoder_block_counts"] = tuple(args.encoder_block_counts)
    if args.decoder_block_counts is not None:
        preset["decoder_block_counts"] = tuple(args.decoder_block_counts)
    if args.shared_expert_depth > 0:
        preset["shared_expert_depth"] = args.shared_expert_depth
    return preset


def resolve_train_val_sizes(dataset_size: int, val_split: float, val_size: int):
    if val_size > 0:
        n_val = min(val_size, max(1, dataset_size - 1))
    else:
        n_val = int(dataset_size * val_split)
        n_val = min(max(n_val, 1), max(1, dataset_size - 1))
    n_train = dataset_size - n_val
    return n_train, n_val


def is_rank_zero_process():
    rank = os.environ.get('RANK')
    if rank is None:
        rank = os.environ.get('LOCAL_RANK')
    if rank is None:
        return True
    return str(rank) == '0'


class PeriodicStatusLogger(pl.Callback):
    def __init__(self, log_every_n_steps: int = 200):
        super().__init__()
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self._last_time = None
        self._last_step = None

    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self._last_time = time.time()
        self._last_step = trainer.global_step
        total_target = trainer.max_steps if trainer.max_steps and trainer.max_steps > 0 else trainer.num_training_batches
        print(f"[TrainStart] epoch={trainer.current_epoch} global_step={trainer.global_step} total_target_steps={total_target}", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        global_step = trainer.global_step
        if (global_step <= 0 or global_step % self.log_every_n_steps != 0
                or global_step == self._last_step):
            return
        now = time.time()
        if self._last_time is None or self._last_step is None:
            self._last_time = now
            self._last_step = global_step
            return
        elapsed = max(now - self._last_time, 1e-6)
        step_delta = max(global_step - self._last_step, 1)
        sec_per_step = elapsed / step_delta
        steps_per_sec = 1.0 / sec_per_step
        max_steps = trainer.max_steps if trainer.max_steps and trainer.max_steps > 0 else None
        eta_msg = 'unknown'
        if max_steps is not None and max_steps > global_step:
            remaining = max_steps - global_step
            eta_seconds = remaining * sec_per_step
            eta_hours = eta_seconds / 3600.0
            eta_msg = f"{eta_hours:.2f}h"
        epoch_total = trainer.num_training_batches
        epoch_progress = f"{batch_idx + 1}/{epoch_total}" if epoch_total else f"{batch_idx + 1}/?"
        lr = None
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0].get('lr')
        metrics = trainer.callback_metrics
        train_mae = metrics.get('train/mae')
        train_loss = metrics.get('train/loss')
        mae_msg = f"{float(train_mae):.8f}" if train_mae is not None else 'n/a'
        loss_msg = f"{float(train_loss):.8f}" if train_loss is not None else 'n/a'
        lr_msg = f"{lr:.3e}" if lr is not None else 'n/a'
        print(
            f"[TrainStatus] epoch={trainer.current_epoch} batch={epoch_progress} global_step={global_step} "
            f"sec_per_step={sec_per_step:.3f} steps_per_sec={steps_per_sec:.3f} eta={eta_msg} "
            f"lr={lr_msg} train_loss={loss_msg} train_mae={mae_msg}",
            flush=True,
        )
        self._last_time = now
        self._last_step = global_step

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        metrics = trainer.callback_metrics
        train_mae = metrics.get('train/mae')
        train_loss = metrics.get('train/loss')
        mae_msg = f"{float(train_mae):.8f}" if train_mae is not None else 'n/a'
        loss_msg = f"{float(train_loss):.8f}" if train_loss is not None else 'n/a'
        print(
            f"[EpochEnd] epoch={trainer.current_epoch} global_step={trainer.global_step} train_loss={loss_msg} train_mae={mae_msg}",
            flush=True,
        )


class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, min_steps: int = 0, min_epochs: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.min_steps = max(0, int(min_steps))
        self.min_epochs = max(0, int(min_epochs))
        self._reported_delay = False
        self._allow_early_stop = True

    def _run_early_stopping_check(self, trainer):
        delayed = trainer.global_step < self.min_steps or trainer.current_epoch < self.min_epochs
        if delayed:
            if trainer.is_global_zero and self.verbose and not self._reported_delay:
                print(
                    "[EarlyStopping] delayed until "
                    f"global_step>={self.min_steps} and epoch>={self.min_epochs}; "
                    f"current global_step={trainer.global_step}, epoch={trainer.current_epoch}",
                    flush=True,
                )
                self._reported_delay = True
        if self.monitor not in trainer.callback_metrics:
            if trainer.is_global_zero and self.verbose:
                available = ", ".join(sorted(str(k) for k in trainer.callback_metrics.keys()))
                print(
                    f"[EarlyStopping] monitor `{self.monitor}` not available yet; "
                    f"available metrics: {available}",
                    flush=True,
                )
            return
        # Track validation checks before the minimum step; delay only stopping.
        self._allow_early_stop = not delayed
        return super()._run_early_stopping_check(trainer)

    def _evaluate_stopping_criteria(self, current):
        should_stop, reason = super()._evaluate_stopping_criteria(current)
        if should_stop and not self._allow_early_stop:
            return False, None
        return should_stop, reason


class FitSummaryLogger(pl.Callback):
    def __init__(self):
        super().__init__()
        self.fit_start_time = None

    def on_fit_start(self, trainer, pl_module):
        self.fit_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_fit_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        print(
            f"[FitEnd] stopped={trainer.should_stop} epoch={trainer.current_epoch} "
            f"global_step={trainer.global_step} "
            f"wall_seconds={time.time() - self.fit_start_time:.3f}",
            flush=True,
        )
        if torch.cuda.is_available():
            print(
                "[FitResources] "
                f"peak_allocated_bytes={torch.cuda.max_memory_allocated()} "
                f"peak_reserved_bytes={torch.cuda.max_memory_reserved()}",
                flush=True,
            )
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                print(
                    f"[CheckpointSummary] dir={callback.dirpath} "
                    f"best_model_path={callback.best_model_path or 'n/a'} "
                    f"best_model_score={callback.best_model_score}",
                    flush=True,
                )


class SampleExposureCounter(pl.Callback):
    """Count samples actually fed to every DDP rank, including short batches."""

    def __init__(self, output_path: Path):
        super().__init__()
        self.output_path = Path(output_path)
        self.local_samples = 0
        self.local_batches = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch is None or 'sample_id' not in batch:
            raise RuntimeError('Cannot audit exposure count without sample IDs')
        self.local_samples += len(batch['sample_id'])
        self.local_batches += 1

    def on_fit_end(self, trainer, pl_module):
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        # NCCL collectives require CUDA tensors, including at on_fit_end.
        device = (torch.device('cuda', torch.cuda.current_device())
                  if distributed and torch.distributed.get_backend() == 'nccl'
                  else pl_module.device)
        totals = torch.tensor([self.local_samples, self.local_batches],
                              device=device, dtype=torch.long)
        if distributed:
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        if trainer.is_global_zero:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(json.dumps({
                'actual_samples_seen': int(totals[0].item()),
                'rank_batch_events': int(totals[1].item()),
                'optimizer_steps': int(trainer.global_step),
            }, indent=2, sort_keys=True))


class FixedEpochBudget(pl.Callback):
    """Stop after a matched number of fully validated epochs."""

    def __init__(self, epochs: int):
        super().__init__()
        self.epochs = max(1, int(epochs))

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if trainer.current_epoch + 1 >= self.epochs:
            trainer.should_stop = True
            if trainer.is_global_zero:
                print(
                    f"[FixedEpochBudget] completed {self.epochs} validated epochs",
                    flush=True,
                )


def load_weights_only(module: nn.Module, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    # The grid buffer must match the requested grid after loading weights.
    if getattr(getattr(module, 'hparams', None), 'core_geometry', 'legacy') == 'voxel_center':
        from reviewer_experiments.core_precision.geometry import centered_grid
        module.grid_buffer.copy_(centered_grid(module.hparams.grid_res,
                                               tuple(module.hparams.grid_bounds),
                                               module.grid_buffer.device))
    print(f"Loaded weights from {checkpoint_path}")
    if missing:
        print(f"Missing keys while loading: {missing}")
    if unexpected:
        print(f"Unexpected keys while loading: {unexpected}")


def freeze_softmoe_routing(module: nn.Module):
    frozen = []
    for name, param in module.named_parameters():
        if name.startswith("model.router.") or name.startswith("model.guide_predictor_log10."):
            param.requires_grad_(False)
            frozen.append(name)
    print(f"Frozen routing-related params: {len(frozen)} tensors")


def build_sample_weight_map(csv_path: str, metric_name: str):
    weight_map = {}
    metric_values = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            metric_raw = row.get(metric_name, "")
            if not sample_id or metric_raw == "":
                continue
            try:
                metric_val = float(metric_raw)
            except ValueError:
                continue
            weight_map[sample_id] = metric_val
            metric_values.append(metric_val)
    if not metric_values:
        return {}, None
    metric_values.sort()
    median = metric_values[len(metric_values) // 2]
    return weight_map, median


def build_weighted_sampler(train_subset, full_dataset, csv_path: str, metric_name: str, scale: float, power: float, max_weight: float):
    metric_map, median = build_sample_weight_map(csv_path, metric_name)
    if not metric_map or median is None or median <= 0:
        return None

    weights = []
    for original_idx in train_subset.indices:
        sample_id = str(full_dataset.indices[original_idx])
        metric_val = metric_map.get(sample_id)
        if metric_val is None:
            weights.append(1.0)
            continue
        hardness = max(0.0, metric_val / median - 1.0)
        weight = 1.0 + scale * (hardness ** power)
        weights.append(min(max_weight, weight))

    weight_tensor = torch.tensor(weights, dtype=torch.double)
    print(
        f"Using weighted sampler from {csv_path} "
        f"(metric={metric_name}, median={median:.6g}, max_weight={float(weight_tensor.max()):.3f})"
    )
    return WeightedRandomSampler(weight_tensor, num_samples=len(weight_tensor), replacement=True)

class MoEPhysModule(pl.LightningModule):
    def __init__(self,
                 atom_channels, pe_channels, num_experts, initial_bins,
                 grid_res=64, grid_bounds=(-10.0, 10.0),
                 lr=1e-4,
                 lambda_global=1.0,
                 lambda_l1=0.0,
                 lambda_guide=1e-5,
                 constraint_start_epoch=-1,
                 constraint_start_step=-1,
                 guide_loss_mask_threshold=1e-10,
                 routing_backbone_grad_scale=1.0,
                 threshold_lr_scale=1.0,
                 threshold_freeze_epochs=0,
                 threshold_freeze_steps=0,
                 hard_routing_epoch=-1,
                 hard_routing_step=-1,
                 warmup_epochs=0,
                 warmup_steps=0,
                 schedule_total_steps=0,
                 min_lr_ratio=0.0,
                 backbone_channels=64,
                 backbone_channel_multipliers=(1, 2, 4, 8),
                 bottleneck_modes=(4, 4, 4),
                 norm_groups=8,
                 encoder_block_counts=(1, 1, 1, 1),
                 decoder_block_counts=(1, 1, 1),
                 guide_hidden_channels=32,
                 expert_hidden_channels=32,
                 shared_expert_depth=2,
                 use_zero_optimizer=False,
                 use_activation_checkpointing=False,
                 optimizer_type='adamw',
                 prediction_space='cubic',
                 magnitude_weight_power=0.0,
                 bottleneck_type='hybrid',
                 routing_space='log',
                 expert_output_mode='interval_sigmoid',
                 fusion_mode='soft',
                 nominal_training_mode='none',
                 electron_count_loss_weight=0.0,
                 voxel_volume_bohr3=0.064,
                 core_geometry='legacy',
                 core_l1_weight=0.0,
                 atom_local_residual=False):
        super().__init__()
        self.save_hyperparameters()

        self.model = PhysicsMoE(
            atom_channels,
            pe_channels,
            num_experts=num_experts,
            initial_bins=initial_bins,
            n_channels=backbone_channels,
            backbone_channel_multipliers=backbone_channel_multipliers,
            bottleneck_modes=bottleneck_modes,
            norm_groups=norm_groups,
            encoder_block_counts=encoder_block_counts,
            decoder_block_counts=decoder_block_counts,
            guide_hidden_channels=guide_hidden_channels,
            expert_hidden_channels=expert_hidden_channels,
            shared_expert_depth=shared_expert_depth,
            routing_backbone_grad_scale=routing_backbone_grad_scale,
            use_activation_checkpointing=use_activation_checkpointing,
            prediction_space=prediction_space,
            bottleneck_type=bottleneck_type,
            routing_space=routing_space,
            expert_output_mode=expert_output_mode,
            fusion_mode=fusion_mode,
        )

        if core_geometry not in {'legacy', 'voxel_center'}:
            raise ValueError(f'Unknown core geometry: {core_geometry}')
        if core_l1_weight < 0:
            raise ValueError('core_l1_weight must be nonnegative')
        if core_geometry == 'voxel_center':
            from reviewer_experiments.core_precision.geometry import centered_grid
            self.grid_coords = centered_grid(grid_res, grid_bounds, torch.device('cpu'))
            from reviewer_experiments.core_precision.geometry import centered_pe
            self.register_buffer('centered_pe_buffer', centered_pe(grid_res))
        else:
            self.grid_coords = PhysicsUtils.get_grid_coords(grid_res, grid_bounds, device='cpu')
        self.register_buffer('grid_buffer', self.grid_coords)
        if atom_local_residual:
            from reviewer_experiments.core_precision.atom_local import AtomLocalResidual
            self.atom_local_branch = AtomLocalResidual(bounds=grid_bounds)
        else:
            self.atom_local_branch = None

        self.val_mae = MeanAbsoluteError()
        self.val_nominal_mae = MeanAbsoluteError()
        self.val_nominal_rmse = MeanSquaredError(squared=False)
        self.val_nominal_pearson = PearsonCorrCoef()
        self.val_mse = MeanSquaredError()
        self.val_pearson = PearsonCorrCoef()
        self.train_mae = MeanAbsoluteError()
        self._thresholds_frozen = False
        self._hard_routing_enabled = False
        # Freeze before DDP registers parameters.
        if threshold_freeze_epochs >= 999 and threshold_freeze_steps <= 0:
            self._set_threshold_trainable(False)

    def forward(
        self,
        atom_grid,
        pe_grid,
        v_ext,
        z_spike,
        target_integral=None,
        nominal_electrons=None,
        use_constraint=True,
        atom_coords=None,
        atom_z=None,
    ):
        outputs = self.model(
            atom_grid,
            pe_grid,
            v_ext,
            z_spike,
            target_integral,
            use_constraint,
        )
        if self.atom_local_branch is not None:
            if atom_coords is None or atom_z is None:
                raise ValueError('Atom-local residual requires atom coordinates and elements')
            outputs = list(outputs)
            correction = self.atom_local_branch(atom_coords, atom_z, outputs[2].shape[-1])
            outputs[2] = torch.clamp(outputs[2] + correction, min=0.)
            outputs = tuple(outputs)
        if (
            getattr(self.hparams, "nominal_training_mode", "none")
            == "hard"
            and nominal_electrons is not None
        ):
            outputs = list(outputs)
            outputs[2] = normalize_density_to_electrons(
                outputs[2],
                nominal_electrons,
                self.hparams.voxel_volume_bohr3,
            )
            outputs = tuple(outputs)
        return outputs

    def _set_threshold_trainable(self, trainable: bool):
        if hasattr(self.model, "router") and hasattr(self.model.router, "log_gaps"):
            self.model.router.log_gaps.requires_grad_(trainable)
            self._thresholds_frozen = not trainable

    def on_fit_start(self):
        if self.hparams.threshold_freeze_epochs > 0 or self.hparams.threshold_freeze_steps > 0:
            self._set_threshold_trainable(False)
        self.model.use_hard_routing = False

    def on_train_epoch_start(self):
        hard_epoch = self.hparams.hard_routing_epoch
        hard_step = int(getattr(self.hparams, 'hard_routing_step', -1))
        if hard_step >= 0:
            should_harden = self.global_step >= hard_step
        else:
            should_harden = hard_epoch >= 0 and self.current_epoch >= hard_epoch
        if should_harden and not self._hard_routing_enabled:
            self.model.use_hard_routing = True
            self._hard_routing_enabled = True
            self._set_threshold_trainable(False)

        freeze_steps = int(getattr(self.hparams, 'threshold_freeze_steps', 0))
        freeze_by_step = freeze_steps > 0 and self.global_step < freeze_steps
        should_freeze = (
            should_harden
            or freeze_by_step
            or self.current_epoch < self.hparams.threshold_freeze_epochs
        )
        if should_freeze and not self._thresholds_frozen:
            self._set_threshold_trainable(False)
        elif (not should_freeze) and self._thresholds_frozen:
            self._set_threshold_trainable(True)

    def _step(self, batch, stage):
        atom_grid = batch['atom_grid']
        pe_grid = batch['pe_grid']
        atom_coords = batch['atom_coords']
        atom_z = batch['atom_z']
        if getattr(self.hparams, 'core_geometry', 'legacy') == 'voxel_center':
            pe_grid = self.centered_pe_buffer[0]
        targets_raw = batch['target_density']
        nominal_electrons = batch['target_electrons']
        nominal_training_mode = str(
            getattr(self.hparams, 'nominal_training_mode', 'none')
        )
        if nominal_training_mode in {'hard', 'soft'}:
            targets = normalize_density_to_electrons(
                targets_raw,
                nominal_electrons,
                self.hparams.voxel_volume_bohr3,
            )
        else:
            targets = targets_raw

        with torch.no_grad():
            v_ext = PhysicsUtils.build_vext(atom_coords, atom_z, self.grid_buffer)
        z_spike = torch.sum(atom_grid, dim=1, keepdim=True)

        use_constraint = stage_is_enabled(
            current_epoch=int(self.current_epoch),
            global_step=int(self.global_step),
            start_epoch=int(
                getattr(self.hparams, 'constraint_start_epoch', -1)
            ),
            start_step=int(
                getattr(self.hparams, 'constraint_start_step', -1)
            ),
        )

        routing_weights, thresholds_log10, final_density, guide_field_log10 = self(
            atom_grid, pe_grid, v_ext, z_spike,
            target_integral=batch['target_integral'],
            nominal_electrons=nominal_electrons,
            use_constraint=use_constraint,
            atom_coords=atom_coords,
            atom_z=atom_z,
        )

        epsilon_cubic = 1e-8
        epsilon_log = 1e-35

        pred_final_cubic = torch.pow(final_density + epsilon_cubic, 1.0/3.0)
        target_cubic = torch.pow(targets + epsilon_cubic, 1.0/3.0)
        target_log10 = torch.log10(targets + epsilon_log)

        if getattr(self.hparams, 'prediction_space', 'cubic') == 'raw':
            global_error = torch.square(final_density - targets)
        else:
            global_error = torch.square(pred_final_cubic - target_cubic)
        weight_power = float(getattr(self.hparams, 'magnitude_weight_power', 0.0))
        if weight_power > 0:
            weights = torch.pow(targets + 1e-8, weight_power)
            weights = weights / (weights.mean().detach() + 1e-8)
            loss_global = torch.mean(global_error * weights)
        else:
            loss_global = torch.mean(global_error)
        loss_l1 = F.l1_loss(final_density, targets)

        guide_mask = targets > self.hparams.guide_loss_mask_threshold
        if torch.any(guide_mask):
            loss_guide = F.mse_loss(guide_field_log10[guide_mask], target_log10[guide_mask])
        else:
            loss_guide = torch.zeros((), device=targets.device, dtype=targets.dtype)

        predicted_electrons = (
            final_density.sum(dim=(1, 2, 3, 4))
            * float(self.hparams.voxel_volume_bohr3)
        )
        electron_log_ratio = torch.log(
            (predicted_electrons + 1e-8)
            / torch.clamp(nominal_electrons, min=1.0)
        )
        loss_electron = torch.mean(electron_log_ratio.square())
        electron_weight = (
            float(self.hparams.electron_count_loss_weight)
            if nominal_training_mode == 'soft'
            else 0.0
        )
        loss = (self.hparams.lambda_global * loss_global +
                self.hparams.lambda_l1 * loss_l1 +
                self.hparams.lambda_guide * loss_guide +
                electron_weight * loss_electron)
        core_l1_weight = float(getattr(self.hparams, 'core_l1_weight', 0.0))
        if core_l1_weight:
            from reviewer_experiments.core_precision.atom_local import atom_core_weight
            core_weight = atom_core_weight(atom_coords, atom_z, self.grid_buffer[0])
            loss_core = (core_weight * (final_density - targets).abs()).mean()
            loss = loss + core_l1_weight * loss_core
            self.log(f'{stage}/loss_core_l1', loss_core, batch_size=atom_grid.shape[0], sync_dist=True)

        batch_size = atom_grid.shape[0]
        self.log(f'{stage}/loss', loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log(f'{stage}/loss_global', loss_global, batch_size=batch_size, sync_dist=True)
        self.log(f'{stage}/loss_l1', loss_l1, batch_size=batch_size, sync_dist=True)
        self.log(f'{stage}/loss_guide', loss_guide, batch_size=batch_size, sync_dist=True)
        self.log(
            f'{stage}/loss_electron',
            loss_electron,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f'{stage}/electron_count_abs',
            torch.mean(torch.abs(predicted_electrons - nominal_electrons)),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f'{stage}/mae_vs_raw_target',
            F.l1_loss(final_density, targets_raw),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f'{stage}/guide_mask_ratio',
            guide_mask.float().mean(),
            batch_size=batch_size,
            sync_dist=True,
        )

        if stage == 'train' and self.global_step % 200 == 0 and self.trainer.is_global_zero:
            t_real = torch.pow(10.0, thresholds_log10)
            t_vals = [f"{t.item():.1e}" for t in t_real]
            print(f"\n[Step {self.global_step}] Learned Router Thresholds: {t_vals}")
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            print(
                f"[Step {self.global_step}] use_constraint={use_constraint} "
                f"use_hard_routing={self.model.use_hard_routing} lr={current_lr:.3e}",
                flush=True,
            )

        final_flat = final_density.flatten()
        target_flat = targets.flatten()

        if stage == 'train':
            self.train_mae(final_flat, target_flat)
            self.log('train/mae', self.train_mae, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        elif stage == 'val':
            self.val_mae(final_flat, target_flat)
            self.val_mse(final_flat, target_flat)
            self.val_pearson(final_flat, target_flat)
            nominal_density = normalize_density_to_electrons(
                final_density.float(),
                nominal_electrons.float(),
                self.hparams.voxel_volume_bohr3,
                eps=1e-12,
            )
            nominal_flat = nominal_density.flatten()
            self.val_nominal_mae(nominal_flat, targets_raw.flatten())
            self.val_nominal_rmse(nominal_flat, targets_raw.flatten())
            self.val_nominal_pearson(nominal_flat, targets_raw.flatten())

            self.log('val/mae', self.val_mae, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log('val_mae', self.val_mae, batch_size=batch_size, sync_dist=True)
            self.log('val/mse', self.val_mse, batch_size=batch_size, sync_dist=True)
            self.log('val/pearson', self.val_pearson, batch_size=batch_size, sync_dist=True)
            self.log('val/nominal_mae', self.val_nominal_mae, batch_size=batch_size, sync_dist=True)
            self.log('val_nominal_mae', self.val_nominal_mae, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log('val/nominal_rmse', self.val_nominal_rmse, batch_size=batch_size, sync_dist=True)
            self.log('val/nominal_pearson', self.val_nominal_pearson, batch_size=batch_size, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx): return self._step(batch, 'train')
    def validation_step(self, batch, batch_idx): return self._step(batch, 'val')

    def configure_optimizers(self):
        threshold_param = self.model.router.log_gaps
        threshold_param_id = id(threshold_param)

        main_params = [
            p for p in self.parameters()
            if id(p) != threshold_param_id
        ]
        threshold_lr = self.hparams.lr * self.hparams.threshold_lr_scale
        param_groups = [
            {"params": main_params, "lr": self.hparams.lr},
            {"params": [threshold_param], "lr": threshold_lr},
        ]

        optimizer_type = str(getattr(self.hparams, "optimizer_type", "adamw")).lower()
        if optimizer_type == "adafactor":
            from transformers.optimization import Adafactor
            opt = Adafactor(
                param_groups,
                lr=self.hparams.lr,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
            )
        elif self.hparams.use_zero_optimizer:
            from torch.distributed.optim import ZeroRedundancyOptimizer
            opt = ZeroRedundancyOptimizer(
                param_groups,
                optimizer_class=torch.optim.AdamW,
                lr=self.hparams.lr,
                foreach=False,
            )
        else:
            opt = torch.optim.AdamW(param_groups, lr=self.hparams.lr, foreach=False)
        eta_min = float(self.hparams.lr) * float(self.hparams.min_lr_ratio)
        max_steps = int(getattr(self.trainer, "max_steps", -1))
        if max_steps and max_steps > 0:
            schedule_steps = int(getattr(self.hparams, 'schedule_total_steps', 0)) or max_steps
            if schedule_steps < max_steps:
                raise ValueError('Schedule horizon must not precede the training cutoff')
            warmup_steps = int(min(self.hparams.warmup_steps, max(0, schedule_steps - 1)))
            if warmup_steps > 0:
                warmup = lr_scheduler.LinearLR(
                    opt,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                cosine = lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=max(1, schedule_steps - warmup_steps),
                    eta_min=eta_min,
                )
                sch = lr_scheduler.SequentialLR(
                    opt,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_steps],
                )
            else:
                sch = lr_scheduler.CosineAnnealingLR(opt, T_max=schedule_steps, eta_min=eta_min)
            interval = "step"
        else:
            max_epochs = int(self.trainer.max_epochs)
            warmup_epochs = int(min(self.hparams.warmup_epochs, max(0, max_epochs - 1)))
            if warmup_epochs > 0:
                warmup = lr_scheduler.LinearLR(
                    opt,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine = lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=max(1, max_epochs - warmup_epochs),
                    eta_min=eta_min,
                )
                sch = lr_scheduler.SequentialLR(
                    opt,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs],
                )
            else:
                sch = lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=eta_min)
            interval = "epoch"
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": interval,
                "frequency": 1,
            },
        }


def get_args():
    parser = argparse.ArgumentParser(description="Train Hybrid-Space Learnable MoE Model")
    parser.add_argument('--devices', type=str, default='0', help="GPU IDs to use (e.g. '0' or '0,1')")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--raw_data_root', type=str, default="data/raw_data_cube")
    parser.add_argument('--cache_dir', type=str, default="data/preprocessed_cache")
    parser.add_argument('--log_dir', type=str, default="logs_soft_moe_hybrid")
    parser.add_argument('--checkpoint_root', type=str, default="checkpoints_hybrid")
    parser.add_argument('--experiment_name', type=str, default="soft_moe_hybrid_v1")

    parser.add_argument('--grid_res', type=int, default=64)
    parser.add_argument('--grid_bounds', type=float, nargs=2, default=[-10.0, 10.0])
    parser.add_argument('--num_pe_freqs', type=int, default=4)
    parser.add_argument('--model_scale', type=str, choices=sorted(MODEL_SCALE_PRESETS.keys()), default='base')
    parser.add_argument('--backbone_channels', type=int, default=0)
    parser.add_argument('--guide_hidden_channels', type=int, default=0)
    parser.add_argument('--expert_hidden_channels', type=int, default=0)
    parser.add_argument('--bottleneck_modes', type=int, nargs=3, default=None)
    parser.add_argument('--norm_groups', type=int, default=0)
    parser.add_argument('--backbone_channel_multipliers', type=int, nargs=4, default=None)
    parser.add_argument('--encoder_block_counts', type=int, nargs=4, default=None)
    parser.add_argument('--decoder_block_counts', type=int, nargs=3, default=None)
    parser.add_argument('--shared_expert_depth', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--max_steps', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of prefetched batches per DataLoader worker')
    parser.add_argument('--pin_memory', action='store_true', help='Enable DataLoader pin_memory')
    parser.add_argument('--disable_persistent_workers', action='store_true', help='Disable DataLoader persistent workers for stability')
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--schedule_total_steps', type=int, default=0)
    parser.add_argument('--min_lr_ratio', type=float, default=0.0)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--val_size', type=int, default=0)
    parser.add_argument(
        '--sample_manifest',
        type=str,
        default='',
        help=(
            'Optional JSON manifest whose samples order defines the dataset. '
            'Used to make reviewer train/validation splits identical to baselines.'
        ),
    )
    parser.add_argument('--train_id_file', type=str, default='', help='Numpy integer training-ID array')
    parser.add_argument('--val_id_file', type=str, default='', help='Numpy integer validation-ID array')
    parser.add_argument('--strict_data', action='store_true', help='Fail instead of silently dropping unreadable samples')
    parser.add_argument('--checkpoint_metric', choices=['val_mae', 'val_nominal_mae'], default='val_mae')
    parser.add_argument('--save_raw_best', action='store_true')
    parser.add_argument('--save_final_checkpoint', action='store_true', help='After a fixed-step fit, write and verify an explicit final weights-only checkpoint')
    parser.add_argument('--record_sample_exposures', action='store_true', help='Write actual cross-rank sample exposure count after training')
    parser.add_argument('--limit_train_batches', type=parse_limit_batches, default=1.0)
    parser.add_argument('--limit_val_batches', type=parse_limit_batches, default=1.0)
    parser.add_argument('--skip_lr_find', action='store_true')
    parser.add_argument('--logger_type', type=str, choices=['tensorboard', 'csv', 'none'], default='tensorboard')
    parser.add_argument('--use_zero_optimizer', action='store_true', help='Shard optimizer states across DDP ranks with ZeroRedundancyOptimizer')
    parser.add_argument('--use_activation_checkpointing', action='store_true', help='Enable activation checkpointing in the backbone and expert trunk to reduce memory')
    parser.add_argument('--optimizer_type', type=str, default='adamw', choices=['adamw', 'adafactor'], help='Optimizer backend')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--init_from_checkpoint', type=str, default='')
    parser.add_argument('--resume_from_checkpoint', type=str, default='', help='Resume full trainer state from a Lightning checkpoint')
    parser.add_argument(
        '--stop_after_epochs',
        type=int,
        default=0,
        help='Stop after this many fully validated epochs without changing the LR scheduler horizon; 0 disables.',
    )
    parser.add_argument('--freeze_routing', action='store_true')
    parser.add_argument('--val_check_interval', type=parse_limit_batches, default=1.0)
    parser.add_argument('--validation_by_steps', action='store_true', help='Allow step-based validation intervals longer than one epoch')
    parser.add_argument('--early_stopping_patience', type=int, default=0)
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0)
    parser.add_argument('--early_stopping_min_steps', type=int, default=0, help='Do not allow early stopping before this many optimizer steps')
    parser.add_argument('--early_stopping_min_epochs', type=int, default=0, help='Do not allow early stopping before this many epochs')
    parser.add_argument('--use_swa', action='store_true')
    parser.add_argument('--swa_lrs', type=float, default=0.0)
    parser.add_argument('--swa_epoch_start', type=float, default=0.8)
    parser.add_argument('--save_top_k', type=int, default=1)
    parser.add_argument('--status_log_interval', type=int, default=200, help='Print a clean rank0 training status line every N global steps')
    parser.add_argument('--disable_validation', action='store_true', help='Disable validation during training for maximum stability')
    parser.add_argument('--sample_weight_csv', type=str, default='')
    parser.add_argument('--sample_weight_metric', type=str, default='mae')
    parser.add_argument('--sample_weight_scale', type=float, default=1.0)
    parser.add_argument('--sample_weight_power', type=float, default=1.0)
    parser.add_argument('--sample_weight_max', type=float, default=2.0)

    parser.add_argument('--lambda_global', type=float, default=1.0)
    parser.add_argument('--lambda_l1', type=float, default=0.0)
    parser.add_argument('--lambda_guide', type=float, default=1e-5)
    parser.add_argument(
        '--constraint_start_epoch',
        type=int,
        default=-1,
        help=(
            'Enable the in-model electron-count constraint at this epoch; '
            'negative values disable it.'
        ),
    )
    parser.add_argument('--constraint_start_step', type=int, default=-1, help='Enable electron-count constraint after this many optimizer steps; negative values keep epoch-based behavior')
    parser.add_argument('--guide_loss_mask_threshold', type=float, default=1e-10)
    parser.add_argument('--routing_backbone_grad_scale', type=float, default=1.0)
    parser.add_argument('--threshold_lr_scale', type=float, default=0.1)
    parser.add_argument('--threshold_freeze_epochs', type=int, default=0)
    parser.add_argument('--threshold_freeze_steps', type=int, default=0, help='Keep router thresholds frozen until this many optimizer steps; 0 disables step-based freezing')
    parser.add_argument('--hard_routing_epoch', type=int, default=-1)
    parser.add_argument('--hard_routing_step', type=int, default=-1, help='Enable hard routing after this many optimizer steps; negative values keep epoch-based behavior')
    parser.add_argument('--bins_mode', type=str, choices=['paper_default', 'decade_exact'], default='decade_exact')
    parser.add_argument(
        '--expert_count',
        type=int,
        default=0,
        help='Override expert count; 1 creates the cube-root single-head control.',
    )
    parser.add_argument(
        '--prediction_space',
        choices=['cubic', 'raw'],
        default='cubic',
        help='Regression space for the prediction head.',
    )
    parser.add_argument(
        '--magnitude_weight_power',
        type=float,
        default=0.0,
        help='If positive, weight voxel loss by (target+eps)^power.',
    )
    parser.add_argument(
        '--bottleneck_type',
        choices=['hybrid', 'cnn'],
        default='hybrid',
        help='Use the full FNO-CNN bottleneck or the CNN-only E8 control.',
    )
    parser.add_argument(
        '--routing_space',
        choices=['log', 'linear'],
        default='log',
        help='Coordinate system used for guide/threshold routing comparisons.',
    )
    parser.add_argument(
        '--expert_output_mode',
        choices=['interval_sigmoid', 'interval_unbounded', 'global_sigmoid'],
        default='interval_sigmoid',
        help=(
            'Expert output parameterization: Full bounded interval remapping, '
            'threshold-scaled unbounded output, or globally bounded output '
            'without expert-specific intervals.'
        ),
    )
    parser.add_argument(
        '--fusion_mode',
        choices=['soft', 'top1_st'],
        default='soft',
        help='Soft weighted expert fusion or straight-through top-1 fusion.',
    )
    parser.add_argument(
        '--nominal_training_mode',
        choices=['none', 'hard', 'soft'],
        default='none',
        help=(
            'Train against a target normalized to sum(Z): hard also applies a '
            'differentiable nominal-electron normalization to predictions; '
            'soft instead adds a squared log-ratio electron-count loss.'
        ),
    )
    parser.add_argument(
        '--electron_count_loss_weight',
        type=float,
        default=0.01,
        help='Weight of the squared log electron-count ratio in soft mode.',
    )
    parser.add_argument(
        '--voxel_volume_bohr3',
        type=float,
        default=0.064,
        help='Physical integration weight of each cached raw-density sample.',
    )
    parser.add_argument('--decade_min_exp', type=int, default=-34)
    parser.add_argument('--core_geometry', choices=['legacy', 'voxel_center'], default='legacy')
    parser.add_argument('--core_l1_weight', type=float, default=0.0)
    parser.add_argument('--atom_local_residual', action='store_true')
    parser.add_argument('--target_override_dir', type=str, default='')
    parser.add_argument('--packed_data_root', type=str, default='',
                        help='Immutable packed F3 training shards for the 2M-pool rerun')
    parser.add_argument('--packed_val_labels', type=str, default='',
                        help='Frozen 2,066-molecule trilinear validation labels')
    parser.add_argument('--save_every_n_optimizer_steps', type=int, default=0,
                        help='Retain fixed-step candidate checkpoints for source-CUBE selection')
    parser.add_argument('--fixed_data_order', action='store_true')
    parser.add_argument('--decade_max_exp', type=int, default=1)

    return parser.parse_args()


def build_initial_bins(args):
    if args.expert_count > 0:
        if args.expert_count == 1:
            return [0.0, np.inf]
        exponents = np.linspace(
            args.decade_min_exp,
            args.decade_max_exp,
            args.expert_count - 1,
        )
        return [0.0] + [10.0 ** float(exp) for exp in exponents] + [np.inf]
    if args.bins_mode == 'paper_default':
        return [0, 1e-34, 1e-30, 1e-26, 1e-22, 1e-18, 1e-14, 1e-10, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, np.inf]

    exponents = list(range(args.decade_min_exp, args.decade_max_exp + 1))
    return [0.0] + [10.0 ** exp for exp in exponents] + [np.inf]

def main():
    args = get_args()
    if args.packed_data_root or args.packed_val_labels or args.target_override_dir:
        raise ValueError('Packed data and target overrides belong to later experiments, not this paper release')
    if args.core_geometry != 'legacy' or args.core_l1_weight or args.atom_local_residual:
        raise ValueError('Core-precision extensions belong to later experiments, not this paper release')
    pl.seed_everything(args.seed)
    model_scale_config = resolve_model_scale_config(args)

    INITIAL_BINS = build_initial_bins(args)
    NUM_EXPERTS = len(INITIAL_BINS) - 1
    GRID_BOUNDS = tuple(args.grid_bounds)

    is_main_process = is_rank_zero_process()

    if is_main_process:
        print(f"Initializing dataset (Using {NUM_EXPERTS} Hybrid-Space Experts)...")

    sample_ids = None
    manifest_train_size = None
    if bool(args.train_id_file) != bool(args.val_id_file):
        raise ValueError('--train_id_file and --val_id_file must be provided together')
    if args.sample_manifest and args.train_id_file:
        raise ValueError('Use either --sample_manifest or Numpy ID files')
    if args.train_id_file:
        train_ids = np.load(args.train_id_file, mmap_mode='r', allow_pickle=False)
        val_ids = np.load(args.val_id_file, mmap_mode='r', allow_pickle=False)
        if train_ids.ndim != 1 or val_ids.ndim != 1 or train_ids.dtype.kind != 'S' or val_ids.dtype.kind != 'S':
            raise ValueError('ID files must contain one-dimensional fixed-width byte-string arrays')
        if not len(train_ids) or not len(val_ids):
            raise ValueError('Both ID files must be nonempty')
        sample_ids = np.concatenate((train_ids, val_ids))
        manifest_train_size = len(train_ids)
        if len(np.unique(sample_ids)) != len(sample_ids):
            raise ValueError('Train and validation IDs must be unique and disjoint')
        if is_main_process:
            print(f'Using immutable ID files: train={len(train_ids)}, val={len(val_ids)}')
    elif args.sample_manifest:
        with Path(args.sample_manifest).open(encoding='utf-8') as handle:
            sample_manifest = json.load(handle)
        manifest_train_rows = sample_manifest.get('train_samples')
        manifest_eval_rows = sample_manifest.get('eval_samples')
        if manifest_train_rows is not None and manifest_eval_rows is not None:
            sample_ids = [
                str(row['sample_id'])
                for row in (manifest_train_rows + manifest_eval_rows)
            ]
            manifest_train_size = len(manifest_train_rows)
        else:
            sample_ids = [
                str(row['sample_id'])
                for row in sample_manifest.get(
                    'samples', sample_manifest.get('eval_samples', [])
                )
            ]
        if not sample_ids:
            raise ValueError(f"No sample IDs found in {args.sample_manifest}")
        if is_main_process:
            print(
                f"Using explicit sample manifest {args.sample_manifest} "
                f"with {len(sample_ids)} ordered IDs"
            )

    full_dataset = PhysicsMoEDataset(
        cache_dir=Path(args.cache_dir),
        raw_data_root=Path(args.raw_data_root),
        grid_res=args.grid_res,
        grid_bounds=GRID_BOUNDS,
        num_pe_freqs=args.num_pe_freqs,
        class_bins=INITIAL_BINS,
        sample_ids=sample_ids,
        strict=args.strict_data,
    )

    NUM_ATOM_TYPES = full_dataset.num_atom_channels
    NUM_PE_CHANNELS = 3 * 2 * args.num_pe_freqs

    n_total = len(full_dataset)
    if manifest_train_size is not None:
        n_train = manifest_train_size
        n_val = n_total - n_train
        train_set = torch.utils.data.Subset(full_dataset, range(0, n_train))
        val_set = torch.utils.data.Subset(full_dataset, range(n_train, n_total))
    else:
        n_train, n_val = resolve_train_val_sizes(
            n_total, args.val_split, args.val_size
        )
        train_set, val_set = random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )

    use_persistent_workers = args.num_workers > 0 and (not args.disable_persistent_workers)
    use_pin_memory = bool(args.pin_memory)
    if args.prefetch_factor < 1:
        raise ValueError('prefetch_factor must be at least 1')
    loader_worker_options = {
        'num_workers': args.num_workers,
        'pin_memory': use_pin_memory,
        'persistent_workers': use_persistent_workers,
    }
    if args.num_workers > 0:
        loader_worker_options['prefetch_factor'] = args.prefetch_factor
    train_sampler = None
    if args.sample_weight_csv:
        train_sampler = build_weighted_sampler(
            train_set,
            full_dataset,
            args.sample_weight_csv,
            args.sample_weight_metric,
            args.sample_weight_scale,
            args.sample_weight_power,
            args.sample_weight_max,
        )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_worker_options,
        collate_fn=collate_fn_moe_phys,
        generator=torch.Generator().manual_seed(args.seed) if args.fixed_data_order else None,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn_moe_phys, **loader_worker_options)

    if is_main_process:
        print("Initializing Hybrid-Space MoE model...")
    model = MoEPhysModule(
        NUM_ATOM_TYPES,
        NUM_PE_CHANNELS,
        grid_res=args.grid_res,
        grid_bounds=GRID_BOUNDS,
        num_experts=NUM_EXPERTS,
        initial_bins=INITIAL_BINS,
        lr=args.learning_rate,
        lambda_global=args.lambda_global,
        lambda_l1=args.lambda_l1,
        lambda_guide=args.lambda_guide,
        constraint_start_epoch=args.constraint_start_epoch,
        constraint_start_step=args.constraint_start_step,
        guide_loss_mask_threshold=args.guide_loss_mask_threshold,
        routing_backbone_grad_scale=args.routing_backbone_grad_scale,
        threshold_lr_scale=args.threshold_lr_scale,
        threshold_freeze_epochs=args.threshold_freeze_epochs,
        threshold_freeze_steps=args.threshold_freeze_steps,
        hard_routing_epoch=args.hard_routing_epoch,
        hard_routing_step=args.hard_routing_step,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
        schedule_total_steps=args.schedule_total_steps,
        min_lr_ratio=args.min_lr_ratio,
        backbone_channels=model_scale_config["backbone_channels"],
        backbone_channel_multipliers=model_scale_config["backbone_channel_multipliers"],
        bottleneck_modes=model_scale_config["bottleneck_modes"],
        norm_groups=model_scale_config["norm_groups"],
        encoder_block_counts=model_scale_config["encoder_block_counts"],
        decoder_block_counts=model_scale_config["decoder_block_counts"],
        guide_hidden_channels=model_scale_config["guide_hidden_channels"],
        expert_hidden_channels=model_scale_config["expert_hidden_channels"],
        shared_expert_depth=model_scale_config["shared_expert_depth"],
        prediction_space=args.prediction_space,
        magnitude_weight_power=args.magnitude_weight_power,
        bottleneck_type=args.bottleneck_type,
        routing_space=args.routing_space,
        expert_output_mode=args.expert_output_mode,
        fusion_mode=args.fusion_mode,
        nominal_training_mode=args.nominal_training_mode,
        electron_count_loss_weight=args.electron_count_loss_weight,
        voxel_volume_bohr3=args.voxel_volume_bohr3,
        core_geometry=args.core_geometry,
        core_l1_weight=args.core_l1_weight,
        atom_local_residual=args.atom_local_residual,
    )
    if is_main_process:
        total_params, trainable_params = count_parameters(model)
        print(
            "Model scale config:",
            {
                "model_scale": args.model_scale,
                **model_scale_config,
            },
        )
        print(
            f"Parameter count: total={format_param_count(total_params)} "
            f"({total_params:,}), trainable={format_param_count(trainable_params)} "
            f"({trainable_params:,})"
        )
    if args.init_from_checkpoint:
        load_weights_only(model, args.init_from_checkpoint)
    if args.freeze_routing:
        freeze_softmoe_routing(model)

    if args.logger_type == 'tensorboard':
        logger = TensorBoardLogger(args.log_dir, name=args.experiment_name)
    elif args.logger_type == 'csv':
        logger = CSVLogger(args.log_dir, name=args.experiment_name)
    else:
        logger = False
    if args.disable_validation or float(args.limit_val_batches) == 0:
        train_ckpt_every = int(args.val_check_interval) if isinstance(args.val_check_interval, (int, float)) and float(args.val_check_interval) >= 1 else 5000
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(args.checkpoint_root) / args.experiment_name,
            filename='step-{step:08d}',
            auto_insert_metric_name=False,
            every_n_train_steps=max(1, train_ckpt_every),
            save_top_k=1,
            save_last=True,
            monitor=None,
        )
        callbacks = [checkpoint_callback, PeriodicStatusLogger(args.status_log_interval), FitSummaryLogger()]
        if args.early_stopping_patience > 0:
            callbacks.append(
                DelayedEarlyStopping(
                    monitor='train/mae',
                    mode='min',
                    patience=args.early_stopping_patience,
                    min_delta=args.early_stopping_min_delta,
                    check_on_train_epoch_end=True,
                    min_steps=args.early_stopping_min_steps,
                    min_epochs=args.early_stopping_min_epochs,
                    verbose=True,
                )
            )
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor=args.checkpoint_metric,
            dirpath=Path(args.checkpoint_root) / args.experiment_name,
            filename='best-step={step:08d}-epoch={epoch:02d}',
            auto_insert_metric_name=False,
            save_top_k=args.save_top_k,
            mode='min',
            save_last=True,
        )
        # Lightning interprets val_check_interval=1.0 as once per epoch.
        callbacks = [checkpoint_callback, PeriodicStatusLogger(args.status_log_interval), FitSummaryLogger()]
        if args.save_raw_best and args.checkpoint_metric != 'val_mae':
            callbacks.append(ModelCheckpoint(
                monitor='val_mae',
                dirpath=Path(args.checkpoint_root) / args.experiment_name,
                filename='raw-best-step={step:08d}-epoch={epoch:02d}',
                auto_insert_metric_name=False,
                save_top_k=1,
                mode='min',
            ))
        if args.early_stopping_patience > 0:
            callbacks.append(
                DelayedEarlyStopping(
                    monitor=args.checkpoint_metric,
                    mode='min',
                    patience=args.early_stopping_patience,
                    min_delta=args.early_stopping_min_delta,
                    min_steps=args.early_stopping_min_steps,
                    min_epochs=args.early_stopping_min_epochs,
                    verbose=True,
                )
            )
    if args.stop_after_epochs > 0:
        callbacks.append(FixedEpochBudget(args.stop_after_epochs))
    if args.record_sample_exposures:
        exposure_path = Path(args.checkpoint_root) / args.experiment_name / 'sample_exposures.json'
        callbacks.append(SampleExposureCounter(exposure_path))
    if args.save_every_n_optimizer_steps > 0:
        callbacks.append(ModelCheckpoint(
            dirpath=Path(args.checkpoint_root) / args.experiment_name,
            filename='candidate-step={step:08d}',
            auto_insert_metric_name=False,
            save_top_k=-1,
            every_n_train_steps=args.save_every_n_optimizer_steps,
            save_on_train_epoch_end=False,
        ))
    if args.use_swa:
        swa_lrs = args.swa_lrs if args.swa_lrs > 0 else args.learning_rate * 0.5
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=swa_lrs,
                swa_epoch_start=args.swa_epoch_start,
            )
        )

    from pytorch_lightning.strategies import DDPStrategy

    if isinstance(args.devices, str):
        normalized_devices = [int(x.strip()) for x in args.devices.split(',') if x.strip()]
    elif isinstance(args.devices, int):
        normalized_devices = [args.devices]
    else:
        normalized_devices = args.devices

    num_devices = len(normalized_devices) if isinstance(normalized_devices, (list, tuple)) else normalized_devices

    if num_devices > 1:
        from pytorch_lightning.strategies import DDPStrategy
        dynamic_unused = (
            0 < args.threshold_freeze_epochs < 999
            or args.threshold_freeze_steps > 0
            or args.hard_routing_epoch >= 0
            or args.hard_routing_step >= 0
        )
        strategy = DDPStrategy(find_unused_parameters=dynamic_unused)
    else:
        strategy = "auto"

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=normalized_devices,
        strategy=strategy,
        max_epochs=args.max_epochs if args.max_steps <= 0 else -1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        logger=logger,
        callbacks=callbacks,
        precision=args.precision,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=(0 if args.disable_validation else args.limit_val_batches),
        num_sanity_val_steps=(0 if args.disable_validation else 2),
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None if args.validation_by_steps else 1,
        enable_progress_bar=False,
        log_every_n_steps=max(1, args.status_log_interval),
    )

    if num_devices == 1 and not args.skip_lr_find:
        print("Single GPU detected. Running LR Finder...")
        tuner = trainer.tuner
        lr_finder = tuner.lr_find(model, train_dataloaders=train_loader)
        new_lr = lr_finder.suggestion()
        print(f"Suggested LR: {new_lr}")
        model.hparams.lr = new_lr
    elif num_devices == 1 and args.skip_lr_find:
        print("Single GPU detected. Skipping automatic LR Finder.")
    else:
        if is_main_process:
            print(f"\n[Warning] Multi-GPU ({normalized_devices}) detected. Skipping automatic LR Finder.")
            print(f"Using provided learning rate: {args.learning_rate}\n")

    ckpt_path = args.resume_from_checkpoint or None
    if is_main_process:
        if ckpt_path:
            print(f"Starting training from checkpoint: {ckpt_path}")
        else:
            print("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=(None if args.disable_validation else val_loader),
        ckpt_path=ckpt_path,
    )
    if args.save_final_checkpoint:
        if args.max_steps <= 0 or trainer.global_step != args.max_steps:
            raise RuntimeError(
                f'Final checkpoint requires exactly {args.max_steps} optimizer steps; '
                f'observed {trainer.global_step}'
            )
        final_path = Path(args.checkpoint_root) / args.experiment_name / f'final_step{args.max_steps}.ckpt'
        trainer.save_checkpoint(str(final_path), weights_only=True)
        if is_main_process:
            print(f'Saved exact-step weights-only checkpoint: {final_path}', flush=True)

if __name__ == "__main__":
    main()
