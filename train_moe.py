import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassJaccardIndex
from model import UNet3D 
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset_classifier import VoxelizedMoEDataset, collate_fn_skip_error
from train_classifier import calculate_class_weights_from_dataset # 假设这个函数不再需要，直接使用固定的权重
import argparse # <--- 1. 导入 argparse

torch.set_float32_matmul_precision('high')
from torch.optim import lr_scheduler 
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef
import torch.nn.functional as F

# ----------------- 模型定义 (与之前完全相同) -----------------
def angle_difference(angle1, angle2):
        """计算两个角度之间的最短距离，范围在 [-pi, pi]"""
        diff = angle1 - angle2
        return torch.remainder(diff + torch.pi, 2 * torch.pi) - torch.pi

class MixtureOfExpertsModel(pl.LightningModule):
    def __init__(self, in_channels_atom, in_channels_pe, num_classes, class_bins,
                 learning_rate=1e-4, lambda_reg=0.1, class_weights=None,
                 lambda_freq_amp=0.01, lambda_freq_phase=0.05):
        super().__init__()
        # 使用 save_hyperparameters() 保存所有传入的参数
        self.save_hyperparameters()
        if not isinstance(class_bins, torch.Tensor):
            class_bins = torch.tensor(class_bins, dtype=torch.float32)
        min_vals = torch.zeros_like(class_bins)
        min_vals[1:] = class_bins[:-1]
        max_vals = class_bins
        
        self.register_buffer("min_vals", min_vals)
        self.register_buffer("max_vals", max_vals)

        self.unet_backbone = UNet3D(
            in_channels_atom=in_channels_atom,
            in_channels_pe=in_channels_pe,
            out_channels=64,
            n_channels=64
        )
        self.gating_head = nn.Conv3d(in_channels=64, out_channels=num_classes, kernel_size=1)

        def create_expert_head():
            return nn.Sequential(
                nn.Conv3d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(32, 1, kernel_size=1),
                nn.Sigmoid()
            )
        self.experts = nn.ModuleList([create_expert_head() for _ in range(num_classes)])
        
        self.cls_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.reg_criterion = nn.MSELoss(reduction='none')
        self.freq_loss_fn = nn.MSELoss()
        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes, average='macro')
        self.val_mse = MeanSquaredError()
        self.val_mae= MeanAbsoluteError()
        self.val_pearson = PearsonCorrCoef()

    def forward(self, atom_grid, pe_grid):
        features = self.unet_backbone(atom_grid, pe_grid)
        logits = self.gating_head(features)
        
        expert_outputs_normalized = [expert(features) for expert in self.experts]
        expert_outputs_normalized = torch.stack(expert_outputs_normalized, dim=1).squeeze(2)
        
        min_vals = self.min_vals.view(1, -1, 1, 1, 1)
        max_vals = self.max_vals.view(1, -1, 1, 1, 1)
        max_vals = torch.nan_to_num(max_vals, posinf=1000.0) 
        expert_outputs_rescaled = min_vals + expert_outputs_normalized * (max_vals - min_vals)
        return logits, expert_outputs_rescaled

    def _calculate_loss(self, logits, expert_outputs, target_labels, target_densities):
        loss_cls = self.cls_criterion(logits, target_labels)
        labels_reshaped = target_labels.unsqueeze(1)
        chosen_expert_preds = torch.gather(expert_outputs, 1, labels_reshaped.long()).squeeze(1)

        # --- 新的 MSLE 回归损失 ---
        # 为防止 log(0) 导致 -inf，我们给 pred 和 target 都加上一个很小的 epsilon
        # 使用 1e-6 就足够了，比 +1 更适合你数据的尺度
        epsilon = 1
        log_pred = torch.log(chosen_expert_preds + epsilon)
        log_target = torch.log(target_densities + epsilon)
        
        # 计算每个体素的平方对数误差
        per_voxel_loss_reg = (log_pred - log_target) ** 2

        # --- 应用掩码，和之前一样 ---
        mask = (target_labels > 0).float()
        loss_reg = (per_voxel_loss_reg * mask).sum() / mask.sum().clamp(min=1e-6)

        # --- 总损失计算，和之前一样 ---
        supervised_loss = loss_cls + self.hparams.lambda_reg * loss_reg
        
        loss_amp = torch.tensor(0.0, device=self.device)
        loss_phase = torch.tensor(0.0, device=self.device)
        
        if self.hparams.lambda_freq_amp > 0 or self.hparams.lambda_freq_phase > 0:
            pred_freq = torch.fft.rfftn(chosen_expert_preds, dim=(1,2,3))
            target_freq = torch.fft.rfftn(target_densities, dim=(1,2,3))
            
            if self.hparams.lambda_freq_amp > 0:
                pred_amp = torch.abs(pred_freq)
                target_amp = torch.abs(target_freq)
                loss_amp = self.freq_loss_fn(torch.log(pred_amp + 1e-8), torch.log(target_amp + 1e-8))

            if self.hparams.lambda_freq_phase > 0:
                pred_phase = torch.angle(pred_freq)
                target_phase = torch.angle(target_freq)
                phase_diff = angle_difference(pred_phase, target_phase)
                loss_phase = torch.mean(phase_diff ** 2)
        
        total_loss = (supervised_loss + 
                      self.hparams.lambda_freq_amp * loss_amp +
                      self.hparams.lambda_freq_phase * loss_phase)
                      
        return total_loss, loss_cls, loss_reg, loss_amp, loss_phase

    # training_step, validation_step, on_validation_epoch_end 保持不变...
    def training_step(self, batch, batch_idx):
        atom_grids, _, pe_grids, target_labels, target_densities, _ = batch
        if atom_grids.nelement() == 0: return None
        pe_grids = pe_grids.to(torch.float32)
        logits, expert_outputs = self.forward(atom_grids, pe_grids)
        total_loss, loss_cls, loss_reg, loss_amp, loss_phase = self._calculate_loss(
            logits, expert_outputs, target_labels, target_densities
        )
        if batch_idx % 50 == 0: 
            print(f"\nEpoch {self.current_epoch}: total_loss={total_loss.item():.4f}, loss_cls={loss_cls.item():.4f}, loss_reg={loss_reg.item():.4f}, loss_amp={loss_amp.item():.4f}, loss_phase={loss_phase.item():.4f}")
        
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_cls', loss_cls, on_step=True, on_epoch=True)
        self.log('train/loss_reg', loss_reg, on_step=True, on_epoch=True)
        self.log('train/loss_amp', loss_amp, on_step=True, on_epoch=True)
        self.log('train/loss_phase', loss_phase, on_step=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        atom_grids, _, pe_grids, target_labels, target_densities, _ = batch
        if atom_grids.nelement() == 0: return
        pe_grids = pe_grids.to(torch.float32)
        logits, expert_outputs = self.forward(atom_grids, pe_grids)
        total_loss, loss_cls, loss_reg, loss_amp, loss_phase = self._calculate_loss(
            logits, expert_outputs, target_labels, target_densities
        )
        preds_labels = torch.argmax(logits, dim=1)
        self.val_iou.update(preds_labels, target_labels)
        chosen_expert_preds = torch.gather(expert_outputs, 1, preds_labels.unsqueeze(1)).squeeze(1)
        preds_flat = chosen_expert_preds.flatten(); targets_flat = target_densities.flatten()
        self.val_mse.update(preds_flat, targets_flat) 
        self.val_mae.update(preds_flat, targets_flat)
        self.val_pearson.update(preds_flat, targets_flat)
        self.log('val/total_loss', total_loss, on_epoch=True)
        self.log('val/loss_cls', loss_cls, on_epoch=True)
        self.log('val/loss_reg', loss_reg, on_epoch=True)
        self.log('val/loss_amp', loss_amp, on_epoch=True)
        self.log('val/loss_phase', loss_phase, on_epoch=True)

    def on_validation_epoch_end(self):
        epoch_iou = self.val_iou.compute()
        epoch_mse = self.val_mse.compute()
        epoch_mae = self.val_mae.compute()
        epoch_pearson = self.val_pearson.compute()
        self.log('val/epoch_iou_macro', epoch_iou, prog_bar=True)
        
        self.log('val/epoch_mse', epoch_mse, prog_bar=True)
        self.log('val/epoch_mae', epoch_mae, prog_bar=True)
        self.log('val_epoch_mae', epoch_mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/epoch_pearson', epoch_pearson, prog_bar=True)
        print(f"\nEpoch {self.current_epoch} Val Results:")
        print(f"  - IoU    : {epoch_iou:.4f}")
        print(f"  - MSE    : {epoch_mse:.4e}")
        print(f"  - MAE    : {epoch_mae:.4e}")
        print(f"  - Pearson: {epoch_pearson:.4f}") 
        self.val_iou.reset()
        self.val_mse.reset()
        self.val_mae.reset()
        self.val_pearson.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1e-6 
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

# <--- 2. 创建一个函数来设置和解析所有命令行参数 --->
def get_args():
    parser = argparse.ArgumentParser(description="Train a Mixture of Experts model for electron density prediction.")

    # --- 路径和名称参数 ---
    path_group = parser.add_argument_group('Paths and Names')
    path_group.add_argument('--raw_data_root', type=str, default="./raw_data_cube", help="Path to the raw data.")
    path_group.add_argument('--cache_dir', type=str, default="./preprocessed_cache", help="Directory to store preprocessed data.")
    path_group.add_argument('--log_dir', type=str, default="logs", help="Directory for TensorBoard logs.")
    path_group.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Directory to save model checkpoints.")
    path_group.add_argument('--experiment_name', type=str, default="density_moe_v1", help="Name for this specific experiment run.")

    # --- 数据和模型参数 ---
    model_group = parser.add_argument_group('Data and Model')
    model_group.add_argument('--grid_res', type=int, default=64, help="Grid resolution.")
    model_group.add_argument('--grid_bounds', type=float, nargs=2, default=[-10.0, 10.0], help="Grid boundaries as two floats (min max).")
    model_group.add_argument('--num_pe_freqs', type=int, default=4, help="Number of positional encoding frequencies.")

    # --- 训练超参数 ---
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--batch_size', type=int, default=8, help="Batch size for training.")
    train_group.add_argument('--max_epochs', type=int, default=30, help="Maximum number of training epochs.")
    train_group.add_argument('--num_workers', type=int, default=12, help="Number of workers for DataLoader.")
    train_group.add_argument('--learning_rate', type=float, default=1.5e-4, help="Initial learning rate.")
    train_group.add_argument('--val_split', type=float, default=0.1, help="Fraction of data to use for validation.")
    train_group.add_argument('--gradient_clip_val', type=float, default=1.0, help="Gradient clipping value.")
    train_group.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    train_group.add_argument('--find_lr', type=bool, default=True, help="If set, run PyTorch Lightning's LR finder.")

    # --- 损失函数权重 ---
    loss_group = parser.add_argument_group('Loss Weights')
    loss_group.add_argument('--lambda_reg', type=float, default=1.0, help="Weight for the regression loss.")
    loss_group.add_argument('--lambda_freq_amp', type=float, default=0.1, help="Weight for the frequency amplitude loss.")
    loss_group.add_argument('--lambda_freq_phase', type=float, default=0.1, help="Weight for the frequency phase loss.")
    
    return parser.parse_args()

def main():
    # <--- 3. 在 main 函数开头解析参数 --->
    args = get_args()
    print(args)
    pl.seed_everything(args.seed)

    # --- 使用 args 中的值代替硬编码的值 ---
    GRID_BOUNDS = tuple(args.grid_bounds)
    CLASS_BINS = [0, 1e-34,1e-30,1e-26,1e-22,1e-18,1e-14, 1e-10, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, np.inf]
    NUM_CLASSES = len(CLASS_BINS)

    print("Initializing dataset...")
    full_dataset = VoxelizedMoEDataset(
        cache_dir=Path(args.cache_dir),
        raw_data_root_for_scan=Path(args.raw_data_root),
        grid_res=args.grid_res,
        grid_bounds=GRID_BOUNDS,
        num_pe_freqs=args.num_pe_freqs,
        class_bins=CLASS_BINS,
        augment_on_the_fly=True
    )
    
    NUM_ATOM_TYPES = full_dataset.num_atom_channels
    NUM_PE_CHANNELS = 3 * 2 * args.num_pe_freqs
    
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    #class_weights = calculate_class_weights_from_dataset(train_dataset, NUM_CLASSES)
    class_weights = torch.tensor([1.2289e-03, 7.6671e+00, 3.5563e+00, 1.4089e+00, 6.1142e-01, 2.7651e-01,
        1.2692e-01, 5.8464e-02, 3.8853e-02, 3.3855e-02, 5.5618e-02, 5.6813e-02,
        6.1824e-02, 4.6262e-02])
    
    print("Setting up DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=True)
    
    print("Initializing model...")
    model = MixtureOfExpertsModel(
        in_channels_atom=NUM_ATOM_TYPES,
        in_channels_pe=NUM_PE_CHANNELS,
        num_classes=NUM_CLASSES,
        class_bins=CLASS_BINS,
        learning_rate=args.learning_rate,
        lambda_reg=args.lambda_reg,
        class_weights=class_weights,
        lambda_freq_amp=args.lambda_freq_amp,
        lambda_freq_phase=args.lambda_freq_phase
    )

    # --- 配置 Logger 和 Checkpoint 回调 ---
    logger = TensorBoardLogger(args.log_dir, name=args.experiment_name)
    checkpoint_callback = ModelCheckpoint(
            monitor='val_epoch_mae',
            dirpath=Path(args.checkpoint_dir) / args.experiment_name, # 将检查点保存在实验专属子目录中
            filename='best-model-{epoch:02d}-mae={val_epoch_mae:.4f}', 
            save_top_k=1, 
            mode='min' # mae 越小越好
        )
        
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=1, 
        max_epochs=args.max_epochs, 
        logger=logger, 
        callbacks=[checkpoint_callback], 
        gradient_clip_val=args.gradient_clip_val
    )

    if args.find_lr:
        print("Running LR Finder...")
        tuner = trainer.tuner
        lr_finder = tuner.lr_find(model, train_dataloaders=train_loader)
        new_lr = lr_finder.suggestion()
        print(f"Suggested LR: {new_lr}")
        model.hparams.learning_rate = new_lr

    print(f"Starting training for experiment: {args.experiment_name}")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("MoE 模型训练完成！")

if __name__ == "__main__":
    main()
'''nohup python train_moe.py \
  --experiment_name "moe_fly_bs16-wophase" \
  --max_epochs 50 \
  --batch_size 8 \
  --learning_rate 2e-4 \
  --lambda_freq_amp 0.1 \
  --lambda_freq_phase 0.0 \
  --lambda_reg 1.0 \
  > logs/ablation_wophase.log 2>&1 &
  '''