import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import lr_scheduler
from pathlib import Path
import numpy as np
from tqdm import tqdm
#nohup python train_hybrid_loss.py >>trainhybrid.log 2>&1 &
# =============================================================================
# 假设这些是你项目中的其他模块
# 请确保文件名和路径正确
# =============================================================================

# 从 model.py 导入你的 UNet3D 和基础模块
from model import UNet3D 

# 从 dataset.py 或 dataset_classifier.py 导入数据集和辅助函数
from dataset_classifier import VoxelizedMoEDataset 

# 从 pe_utils.py 导入
from pe_utils import create_coordinate_grid, positional_encoding

# =============================================================================
#  辅助函数
# =============================================================================

def collate_fn_fourier(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        # 匹配__getitem__的返回数量
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
    
    # 用一个下划线 `_` 作为占位符来接收并忽略不用的 target_labels
    atom_grids,_, pe_grids,_, target_densities,_  = zip(*batch)
    
    # 只返回我们需要的3个
    return (torch.stack(atom_grids), torch.stack(pe_grids), torch.stack(target_densities))

# =============================================================================
#  核心：混合损失的 LightningModule
# =============================================================================

class HybridLossDensityModel(pl.LightningModule):
    def __init__(self, in_channels_atom, in_channels_pe,
                 learning_rate=1e-4, lambda_freq_amp=0.1, lambda_freq_phase=0.5,
                 unet_out_channels=1, unet_internal_channels=64):
        super().__init__()
        self.save_hyperparameters()

        # 1. 核心模型：一个标准的空间域 UNet (使用您原来的后期融合版本)
        self.unet = UNet3D(
            in_channels_atom=in_channels_atom,
            in_channels_pe=in_channels_pe,
            out_channels=unet_out_channels,
            n_channels=unet_internal_channels
        )
        
        # 2. 损失函数
        self.spatial_loss_fn = nn.L1Loss() # 空间域用 MAE
        self.freq_loss_fn = nn.L1Loss()    # 频域也用 MAE

        # 3. 验证指标
        from torchmetrics.regression import MeanAbsoluteError, PearsonCorrCoef, MeanSquaredError
        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.val_pearson = PearsonCorrCoef()

    @staticmethod
    def angle_difference(angle1, angle2):
        """计算两个角度之间的最短距离，范围在 [-pi, pi]"""
        diff = angle1 - angle2
        return torch.remainder(diff + torch.pi, 2 * torch.pi) - torch.pi

    def forward(self, atom_grid, pe_grid):
        # 模型现在非常简单：直接从 UNet 输出空间密度
        # UNet的输出已经是 [B, 1, D, H, W]，所以 squeeze(1)
        return self.unet(atom_grid, pe_grid).squeeze(1)

    def _calculate_loss(self, pred_density, target_density):
        # a. 计算空间域损失 (主要损失)
        loss_spatial = self.spatial_loss_fn(pred_density, target_density)
        
        # b. 计算频域损失 (正则化)
        loss_amp = torch.tensor(0.0, device=self.device)
        loss_phase = torch.tensor(0.0, device=self.device)
        
        # 只有当权重>0时才计算，以节省算力
        if self.hparams.lambda_freq_amp > 0 or self.hparams.lambda_freq_phase > 0:
            # 确保 dim 参数与数据形状匹配, PyTorch 通常是 (N, C, D, H, W)
            # 我们的 pred_density 是 (N, D, H, W), 所以 dim=(1,2,3)
            pred_freq = torch.fft.rfftn(pred_density, dim=(1,2,3))
            target_freq = torch.fft.rfftn(target_density, dim=(1,2,3))
            
            # 振幅损失 (对数尺度)
            if self.hparams.lambda_freq_amp > 0:
                pred_amp = torch.abs(pred_freq)
                target_amp = torch.abs(target_freq)
                loss_amp = self.freq_loss_fn(torch.log(pred_amp + 1e-8), torch.log(target_amp + 1e-8))

            # 相位损失 (使用角度差异)
            if self.hparams.lambda_freq_phase > 0:
                pred_phase = torch.angle(pred_freq)
                target_phase = torch.angle(target_freq)
                phase_diff = self.angle_difference(pred_phase, target_phase)
                loss_phase = torch.mean(torch.abs(phase_diff))
        
        # c. 总损失
        total_loss = (loss_spatial + 
                      self.hparams.lambda_freq_amp * loss_amp +
                      self.hparams.lambda_freq_phase * loss_phase)
                      
        return total_loss, loss_spatial, loss_amp, loss_phase

    def training_step(self, batch, batch_idx):
        atom_grids, pe_grids, target_densities = batch
        if atom_grids.nelement() == 0: return None
        
        pe_grids = pe_grids.to(torch.float32)
        
        pred_density = self.forward(atom_grids, pe_grids)
        
        total_loss, loss_spatial, loss_amp, loss_phase = self._calculate_loss(
            pred_density, target_densities
        )
        
        # 日志记录
        if batch_idx % 50 == 0:
            print(f"\nEpoch {self.current_epoch}, Step {batch_idx}: total_loss={total_loss.item():.4f}, loss_spatial={loss_spatial.item():.4f}, loss_amp={loss_amp.item():.4f}, loss_phase={loss_phase.item():.4f}")
        
        self.log_dict({
            'train/total_loss': total_loss,
            'train/loss_spatial': loss_spatial,
            'train/loss_amp': loss_amp,
            'train/loss_phase': loss_phase
        }, on_step=True, on_epoch=True, prog_bar=True, batch_size=atom_grids.size(0))
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        atom_grids, pe_grids, target_densities = batch
        if atom_grids.nelement() == 0: return
        
        pe_grids = pe_grids.to(torch.float32)
        pred_density = self.forward(atom_grids, pe_grids)
        
        total_loss, loss_spatial, loss_amp, loss_phase = self._calculate_loss(
            pred_density, target_densities
        )
        
        # 更新指标
        preds_flat = pred_density.flatten()
        targets_flat = target_densities.flatten()
        self.val_mae.update(preds_flat, targets_flat)
        self.val_mse.update(preds_flat, targets_flat)
        if preds_flat.std() > 1e-6 and targets_flat.std() > 1e-6:
            self.val_pearson.update(preds_flat, targets_flat)

        self.log_dict({
            'val/total_loss': total_loss,
            'val/loss_spatial': loss_spatial,
            'val/loss_amp': loss_amp,
            'val/loss_phase': loss_phase
        }, on_epoch=True, batch_size=atom_grids.size(0))

    def on_validation_epoch_end(self):
        epoch_mae = self.val_mae.compute()
        epoch_mse = self.val_mse.compute()
        epoch_pearson = self.val_pearson.compute()

        self.log('val/epoch_mae', epoch_mae, prog_bar=True)
        self.log('val/epoch_mse', epoch_mse)
        self.log('val/epoch_pearson', epoch_pearson, prog_bar=True)

        print(f"\nEpoch {self.current_epoch} Val Results:")
        print(f"  - MAE    : {epoch_mae:.4e}")
        print(f"  - MSE    : {epoch_mse:.4e}")
        print(f"  - Pearson: {epoch_pearson:.4f}")

        self.val_mae.reset()
        self.val_mse.reset()
        self.val_pearson.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        # 确保 trainer 对象已经被关联
        if self.trainer is None:
            return optimizer # 在 lr_find 等阶段，trainer 可能不存在

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1e-6 
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

# =============================================================================
#  主训练流程
# =============================================================================

def main():
    # --- 超参数 ---
    RAW_DATA_ROOT = Path("./raw_data_cube") 
    CACHE_DIR = Path("./preprocessed_cache")
    GRID_RESOLUTION = 64
    GRID_BOUNDS = (-10.0, 10.0)
    NUM_PE_FREQS = 4
    BATCH_SIZE = 8
    LEARNING_RATE = 1.5e-4
    NUM_EPOCHS = 50
    VAL_SPLIT_RATIO = 0.1
    NUM_WORKERS = 12

    # 新的混合损失权重
    LAMBDA_FREQ_AMP = 0.1
    LAMBDA_FREQ_PHASE = 0.5 # 给相位稍高的权重
    
    # 确保 Pytorch Lightning 种子
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    # --- 数据集设置 ---
    # !! 重要 !!
    # 确保您的 VoxelizedMoEDataset 的 __getitem__ 现在返回:
    # return atom_grid.float(), pe_grid, target_density_grid.float()
    
    full_dataset = VoxelizedMoEDataset(
        cache_dir=CACHE_DIR,
        raw_data_root_for_scan=RAW_DATA_ROOT,
        grid_res=GRID_RESOLUTION,
        grid_bounds=GRID_BOUNDS,
        num_pe_freqs=NUM_PE_FREQS,
        class_bins=[] # 不再需要分类
    )
    
    NUM_ATOM_TYPES = full_dataset.num_atom_channels
    NUM_PE_CHANNELS = 3 * 2 * NUM_PE_FREQS if NUM_PE_FREQS > 0 else 0

    n_total = len(full_dataset)
    n_val = int(n_total * VAL_SPLIT_RATIO)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True, 
                              collate_fn=collate_fn_fourier, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True, 
                            collate_fn=collate_fn_fourier, persistent_workers=True)

    # --- 模型实例化 ---
    model = HybridLossDensityModel(
        in_channels_atom=NUM_ATOM_TYPES,
        in_channels_pe=NUM_PE_CHANNELS,
        learning_rate=LEARNING_RATE,
        lambda_freq_amp=LAMBDA_FREQ_AMP,
        lambda_freq_phase=LAMBDA_FREQ_PHASE
    )
    
    # --- 训练器设置 ---
    logger = TensorBoardLogger("logs", name="hybrid_loss_density_model")
    checkpoint_callback = ModelCheckpoint(
            monitor='val/epoch_mae', # 现在我们可以监控皮尔逊系数了
            dirpath='checkpoints_hybrid', 
            filename='best-hybrid-model-{epoch:02d}-pearson={val/epoch_mae:.4f}', 
            save_top_k=3, 
            mode='max' # 皮尔逊系数越大越好
        )
        
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=1, 
        max_epochs=NUM_EPOCHS, 
        logger=logger, 
        callbacks=[checkpoint_callback], 
        gradient_clip_val=1.0
    )

    # (可选) 重新运行学习率查找
    tuner = trainer.tuner.lr_find(model, train_dataloaders=train_loader)
    new_lr = tuner.suggestion()
    print(f"找到建议的学习率: {new_lr}")
    model.hparams.learning_rate = new_lr

    print("开始训练混合损失模型...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("混合损失模型训练完成！")

if __name__ == "__main__":
    main()