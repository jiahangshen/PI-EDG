import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import lr_scheduler
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef
from pathlib import Path
import numpy as np
# 导入自定义模块
from dataset_moe_phys import PhysicsMoEDataset, collate_fn_moe_phys
from model_moe_phys import PhysicsMoE
from physics_utils import PhysicsUtils # 包含 build_vext

torch.set_float32_matmul_precision('high')


# --- 1. MoE 专用硬约束层 ---
class MoEHardConstraintLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # MoE 输出已经是 ReLU+Sigmoid 处理过的正数，不需要额外的 Softplus
        # 也不需要立方根变换，因为 MoE 专家直接预测 Real Density (或其缩放版)

    def forward(self, pred_density, target_integral):
        """
        pred_density: [B, 1, D, H, W] (MoE 融合后的密度)
        target_integral: [B] (真实积分值)
        """
        # 1. 计算当前预测积分 (Raw Sum)
        # [B, 1, D, H, W] -> [B, 1, 1, 1, 1]
        current_sum = torch.sum(pred_density, dim=(1, 2, 3, 4), keepdim=True) + 1e-8
        
        # 2. 准备目标
        # [B] -> [B, 1, 1, 1, 1]
        target_sum = target_integral.view(-1, 1, 1, 1, 1)
        
        # 3. 计算缩放因子
        scale = target_sum / current_sum
        
        # 4. 应用缩放
        final_density = pred_density * scale
        
        return final_density



class MoEPhysModule(pl.LightningModule):
    def __init__(self, 
                 atom_channels, pe_channels, num_classes, class_bins,
                 grid_res=64, grid_bounds=(-10.0, 10.0),
                 lr=1e-4, 
                 lambda_cls=1.0, 
                 lambda_expert=1.0, 
                 lambda_global=10.0,
                 class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. 准备输入通道
        # Atom + PE + Vext(1) + Zspike(1)
        # 2. 初始化模型
        self.model = PhysicsMoE(atom_channels,pe_channels, num_classes=num_classes, class_bins=class_bins ,n_channels=64)
        
        # 3. 辅助工具
        self.grid_coords = PhysicsUtils.get_grid_coords(grid_res, grid_bounds, device='cpu')
        self.register_buffer('grid_buffer', self.grid_coords)
        if class_weights is not None:
            # 如果是 Tensor，确保它被 register_buffer 这样会自动移动到 GPU
            self.register_buffer('class_weights', class_weights)
            self.cls_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.cls_loss = torch.nn.CrossEntropyLoss()
        
        # --- 增加指标 ---
        # 验证集指标
        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.val_pearson = PearsonCorrCoef()
        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes, average='macro')
        
        # 训练集指标 (可选，用于监控过拟合)
        self.train_mae = MeanAbsoluteError()
    def forward(self, atom_grid, pe_grid, v_ext, z_spike, target_integral=None, use_constraint=True):
        return self.model(atom_grid, pe_grid, v_ext, z_spike, target_integral, use_constraint)

    def _step(self, batch, stage):
        # 1. 解包 & 构建输入
        atom_grid = batch['atom_grid']
        pe_grid = batch['pe_grid']
        atom_coords = batch['atom_coords']
        atom_z = batch['atom_z']
        
        # 动态构建物理场
        with torch.no_grad():
            v_ext = PhysicsUtils.build_vext(atom_coords, atom_z, self.grid_buffer)
        z_spike = torch.sum(atom_grid, dim=1, keepdim=True)
        
        # 拼接所有输入: [B, C_all, D, H, W]
        
        # 2. 前向传播
        # 训练前5轮 Bypass，之后开启约束
        use_constraint = (self.current_epoch >= 50)
        
        logits, expert_out, final_density = self.model(
            atom_grid, pe_grid, v_ext, z_spike,
            total_electrons=batch['target_integral'], 
            use_constraint=use_constraint
        )
        
        # 3. 计算损失
        targets = batch['target_density']
        labels = batch['target_labels']
        
        # A. 分类 Loss
        loss_cls = self.cls_loss(logits, labels)
        gathered_expert_cubic = torch.gather(expert_out, 1, labels.unsqueeze(1))
        
        # 将 Target 转换为 n^(1/3)
        targets_cubic = torch.pow(targets, 1.0/3.0)
        
        # 计算 MSE (直接比较立方根，不需要再取 Log，因为立方根已经压缩过了)
        # 或者如果非要 Log，两边都 Log
        # 推荐：直接 MSE(pred_cubic, target_cubic)
        per_voxel_loss = (gathered_expert_cubic - targets_cubic) ** 2
        
        mask = (labels > 0).float()
        loss_expert = (per_voxel_loss * mask).sum() / (mask.sum() + 1e-6)
        
        # C. 全局物理 Loss (Cubic MSE on Final Density)
        # final_density 是物理空间的 n (已归一化)
        # targets 是物理空间的 n
        
        # 将两者都转换到立方根空间
        epsilon = 1e-8 # 防止导数爆炸
        pred_final_cubic = torch.pow(final_density + epsilon, 1.0/3.0)
        target_final_cubic = torch.pow(targets + epsilon, 1.0/3.0)
        
        # 直接计算 MSE
        loss_global = F.mse_loss(pred_final_cubic, target_final_cubic)
        
        
        # 总 Loss
        loss = (self.hparams.lambda_cls * loss_cls + 
                self.hparams.lambda_expert * loss_expert + 
                self.hparams.lambda_global * loss_global)
        
        # --- 4. 全面日志记录 (Logging) ---
        batch_size = atom_grid.shape[0]
        
        # 基础 Loss
        self.log(f'{stage}/loss', loss, prog_bar=True, batch_size=batch_size)
        self.log(f'{stage}/loss_cls', loss_cls, batch_size=batch_size)
        self.log(f'{stage}/loss_expert', loss_expert, batch_size=batch_size)
        self.log(f'{stage}/loss_global', loss_global, prog_bar=True, batch_size=batch_size)
        
        # 物理指标 (MAE)
        final_flat = final_density.flatten()
        target_flat = targets.flatten()
        
        if stage == 'train':
            self.train_mae(final_flat, target_flat)
            self.log('train/mae', self.train_mae, on_step=False, on_epoch=True, batch_size=batch_size)
            
        elif stage == 'val':
            # 更新各项指标
            self.val_mae(final_flat, target_flat)
            self.val_mse(final_flat, target_flat)
            self.val_pearson(final_flat, target_flat)
            
            # 计算分类 IoU
            preds_labels = torch.argmax(logits, dim=1)
            self.val_iou(preds_labels, labels)
            
            # 记录
            self.log('val/mae', self.val_mae, prog_bar=True, batch_size=batch_size)
            self.log('val/mse', self.val_mse, batch_size=batch_size)
            self.log('val/pearson', self.val_pearson, batch_size=batch_size)
            self.log('val/iou', self.val_iou, batch_size=batch_size)
            
        return loss

    def training_step(self, batch, batch_idx): return self._step(batch, 'train')
    def validation_step(self, batch, batch_idx): return self._step(batch, 'val')
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sch = lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return [opt], [sch]

# ... (get_args, main 函数与之前类似，只需实例化上面的 Class) ...
# 请确保 main 函数中正确传递 BINS 和 Weights
# --- 3. 参数解析与主函数 ---
def get_args():
    parser = argparse.ArgumentParser(description="Train Physics-Informed MoE Model")

    # 路径
    parser.add_argument('--raw_data_root', type=str, default="./raw_data_cube")
    parser.add_argument('--cache_dir', type=str, default="./preprocessed_cache")
    parser.add_argument('--log_dir', type=str, default="logs_moe_phys")
    parser.add_argument('--experiment_name', type=str, default="moe_phys_v1")

    # 模型
    parser.add_argument('--grid_res', type=int, default=64)
    parser.add_argument('--grid_bounds', type=float, nargs=2, default=[-10.0, 10.0])
    parser.add_argument('--num_pe_freqs', type=int, default=4)

    # 训练
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--val_split', type=float, default=0.1)
    
    # 权重
    parser.add_argument('--lambda_cls', type=float, default=1.0, help="Classification loss weight")
    parser.add_argument('--lambda_expert', type=float, default=1.0, help="Expert regression loss weight")
    parser.add_argument('--lambda_global', type=float, default=10.0, help="Global physics loss weight (Main target)")

    return parser.parse_args()

def main():
    args = get_args()
    pl.seed_everything(42)

    # MoE Bins (保持不变)
    CLASS_BINS = [0, 1e-34, 1e-30, 1e-26, 1e-22, 1e-18, 1e-14, 1e-10, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, np.inf]
    NUM_CLASSES = len(CLASS_BINS)
    GRID_BOUNDS = tuple(args.grid_bounds)

    print("Initializing dataset...")
    # 注意：这里调用的是您修改过 __getitem__ (返回 integral) 的 Dataset
    full_dataset = PhysicsMoEDataset(
        cache_dir=Path(args.cache_dir),
        raw_data_root=Path(args.raw_data_root),
        grid_res=args.grid_res,
        grid_bounds=GRID_BOUNDS,
        num_pe_freqs=args.num_pe_freqs,
        class_bins=CLASS_BINS,
    )
    
    NUM_ATOM_TYPES = full_dataset.num_atom_channels
    NUM_PE_CHANNELS = 3 * 2 * args.num_pe_freqs
    
    # Split
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    # Class Weights (Hardcoded for stability)
    class_weights = torch.tensor([1.2e-03, 7.6, 3.5, 1.4, 0.6, 0.27, 0.12, 0.058, 0.038, 0.033, 0.055, 0.056, 0.061, 0.046])

    print("Setting up DataLoaders...")
    # 注意：这里使用的是修改过支持 integral 的 collate_fn_skip_error
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_moe_phys, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_moe_phys, persistent_workers=True)
    
    print("Initializing Physics-Informed MoE model...")
    model = MoEPhysModule(
        NUM_ATOM_TYPES,
        NUM_PE_CHANNELS,
        num_classes=NUM_CLASSES,
        class_bins=CLASS_BINS,
        lr=args.learning_rate,
        lambda_cls=args.lambda_cls,
        lambda_expert=args.lambda_expert,
        lambda_global=args.lambda_global,
        class_weights=class_weights
    )

    # Logger & Checkpoint
    logger = TensorBoardLogger(args.log_dir, name=args.experiment_name)
    checkpoint_callback = ModelCheckpoint(
        monitor='val/mae',
        dirpath=Path("checkpoints") / args.experiment_name,
        filename='best-{epoch:02d}-{val/mae:.4f}', 
        save_top_k=1, 
        mode='min'
    )
    
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=1, 
        max_epochs=args.max_epochs, 
        logger=logger, 
        callbacks=[checkpoint_callback], 
        gradient_clip_val=1.0
    )
    print("Running LR Finder...")
    tuner = trainer.tuner
    lr_finder = tuner.lr_find(model, train_dataloaders=train_loader)
    new_lr = lr_finder.suggestion()
    print(f"Suggested LR: {new_lr}")
    model.hparams.learning_rate = new_lr
    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()