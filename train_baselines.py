import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import lr_scheduler
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef
from pathlib import Path
import numpy as np

# 导入您的自定义数据集模块
from dataset_moe_phys import PhysicsMoEDataset, collate_fn_moe_phys

torch.set_float32_matmul_precision('high')

# =========================================================================
# 1. 纯数据驱动架构定义库 (Architecture Zoo)
# =========================================================================

# --- A. 现代大核 CNN: ConvNeXt-3D ---
class LayerNorm3D(nn.Module):
    """支持 channels_first (N, C, D, H, W) 的 LayerNorm"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class ConvNeXtBlock3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 1. 7x7x7 大核 Depthwise 卷积
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = LayerNorm3D(dim, eps=1e-6)
        # 2. Inverted Bottleneck (膨胀 4 倍)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1) 

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return input + x

class ConvNeXtUNet3D(nn.Module):
    """结合了 U-Net 的宏观结构和 ConvNeXt 的微观 Block 的强力基线"""
    def __init__(self, in_channels, out_channels=1, base_dim=32):
        super().__init__()
        # Encoder
        self.stem = nn.Conv3d(in_channels, base_dim, kernel_size=4, stride=4) # 快速下采样减少显存
        self.enc1 = ConvNeXtBlock3D(base_dim)
        self.down1 = nn.Conv3d(base_dim, base_dim*2, kernel_size=2, stride=2)
        self.enc2 = ConvNeXtBlock3D(base_dim*2)
        self.down2 = nn.Conv3d(base_dim*2, base_dim*4, kernel_size=2, stride=2)
        self.enc3 = ConvNeXtBlock3D(base_dim*4)
        
        # Bottleneck
        self.bottle = ConvNeXtBlock3D(base_dim*4)
        
        # Decoder
        self.up2 = nn.ConvTranspose3d(base_dim*4, base_dim*2, kernel_size=2, stride=2)
        self.dec2 = ConvNeXtBlock3D(base_dim*2)
        self.up1 = nn.ConvTranspose3d(base_dim*2, base_dim, kernel_size=2, stride=2)
        self.dec1 = ConvNeXtBlock3D(base_dim)
        
        # 恢复到原分辨率 64x64x64
        self.up_stem = nn.ConvTranspose3d(base_dim, base_dim, kernel_size=4, stride=4)
        
        # Prediction Head
        self.head = nn.Conv3d(base_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x_stem = self.stem(x)
        e1 = self.enc1(x_stem)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        # Bottleneck
        b = self.bottle(e3)
        # Decoder (含 Skip Connection)
        d2 = self.dec2(self.up2(b) + e2)
        d1 = self.dec1(self.up1(d2) + e1)
        # Head
        out = self.up_stem(d1)
        return self.head(out)


# --- B. 现代 Transformer: Swin-UNETR ---
def get_swin_unetr(in_channels, grid_res):
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError:
        raise ImportError("Please install MONAI to use SwinUNETR: pip install monai")
    return SwinUNETR(
        img_size=(grid_res, grid_res, grid_res),
        in_channels=in_channels,
        out_channels=1, 
        feature_size=24, 
        use_checkpoint=True,
    )

# --- C. 经典 CNN: Vanilla U-Net ---
def get_classic_unet(in_channels_atom, in_channels_pe):
    from model import UNet3D 
    class ClassicUNetBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            # 正常初始化您的 UNet3D，不改变其原本的通道逻辑
            self.unet = UNet3D(in_channels_atom=in_channels_atom, in_channels_pe=in_channels_pe, out_channels=64, n_channels=64)
            self.regression_head = nn.Conv3d(64, 1, kernel_size=1)
            
        def forward(self, atom_grid, pe_grid):
            # 正常传入两个参数，让 UNet3D 内部按原计划处理 PE
            features = self.unet(atom_grid, pe_grid)
            return self.regression_head(features)
            
    return ClassicUNetBaseline()


# =========================================================================
# 2. 定义 Baseline 专用的 Lightning Module (无物理知识)
# =========================================================================

class NaiveBaselineModule(pl.LightningModule):
    def __init__(self, model_type, in_channels_atom, in_channels_pe, grid_res=64, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model_type = model_type
        total_in_channels = in_channels_atom + in_channels_pe
        
        # 初始化选定的网络架构
        if model_type == "swin_unetr":
            self.model = get_swin_unetr(total_in_channels, grid_res)
        elif model_type == "convnext_3d":
            self.model = ConvNeXtUNet3D(in_channels=total_in_channels, out_channels=1)
        elif model_type == "classic_unet":
            self.model = get_classic_unet(in_channels_atom, in_channels_pe)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.val_pearson = PearsonCorrCoef()

    def forward(self, atom_grid, pe_grid):
        if self.model_type == "classic_unet":
            # Classic UNet 需要两个独立的输入
            raw_pred = self.model(atom_grid, pe_grid)
        else:
            # Swin-UNETR 和 ConvNeXt 需要拼接后的单一输入
            if pe_grid is not None:
                x = torch.cat([atom_grid, pe_grid], dim=1)
            else:
                x = atom_grid
            raw_pred = self.model(x)
            
        # 激活函数：确保电子密度为正数
        return F.softplus(raw_pred)

    def _step(self, batch, stage):
        atom_grid = batch['atom_grid']
        pe_grid = batch['pe_grid']
        targets = batch['target_density']
        
        # 1. 模型直接端到端预测 (传入两个参数)
        preds = self(atom_grid, pe_grid).squeeze(1) # [B, 1, D, H, W] -> [B, D, H, W]
        
        # 2. Naive 损失：直接在原始空间上算 MSE
        loss = F.mse_loss(preds, targets)
        
        batch_size = atom_grid.shape[0]
        self.log(f'{stage}/loss', loss, prog_bar=True, batch_size=batch_size)
        
        if stage == 'val':
            preds_flat = preds.flatten()
            targets_flat = targets.flatten()
            self.val_mae.update(preds_flat, targets_flat)
            self.val_mse.update(preds_flat, targets_flat)
            self.val_pearson.update(preds_flat, targets_flat)
            self.log('val/mae', self.val_mae, prog_bar=True, batch_size=batch_size)
            self.log('val/mse', self.val_mse, batch_size=batch_size)
            self.log('val/pearson', self.val_pearson, batch_size=batch_size)
            
        return loss
    def training_step(self, batch, batch_idx): return self._step(batch, 'train')
    def validation_step(self, batch, batch_idx): return self._step(batch, 'val')
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sch = lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return [opt], [sch]


# =========================================================================
# 3. 参数解析与训练启动
# =========================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Train Pure Data-Driven Baselines")

    # --- 选择 Baseline 模型 ---
    parser.add_argument('--model_type', type=str, default="convnext_3d", 
                        choices=["swin_unetr", "convnext_3d", "classic_unet"], 
                        help="Choose the baseline architecture: swin_unetr (Transformer), convnext_3d (Modern CNN), classic_unet (Classic CNN)")

    parser.add_argument('--raw_data_root', type=str, default="./raw_data_cube")
    parser.add_argument('--cache_dir', type=str, default="./preprocessed_cache")
    parser.add_argument('--log_dir', type=str, default="logs_baselines")
    parser.add_argument('--experiment_name', type=str, default="baseline_run")

    parser.add_argument('--grid_res', type=int, default=64)
    parser.add_argument('--grid_bounds', type=float, nargs=2, default=[-10.0, 10.0])
    parser.add_argument('--num_pe_freqs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.1)

    return parser.parse_args()

def main():
    args = get_args()
    pl.seed_everything(42)

    CLASS_BINS = [0, 1e-34, 1e-30, 1e-26, 1e-22, 1e-18, 1e-14, 1e-10, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, np.inf]
    GRID_BOUNDS = tuple(args.grid_bounds)

    print(f"[{args.model_type.upper()}] Initializing dataset...")
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
    
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    print("Setting up DataLoaders...")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_moe_phys, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_moe_phys, persistent_workers=True)
    
    print(f">>> Initializing Pure Data-Driven Baseline: {args.model_type} <<<")
    model = NaiveBaselineModule(
        model_type=args.model_type,
        in_channels_atom=NUM_ATOM_TYPES,
        in_channels_pe=NUM_PE_CHANNELS,
        grid_res=args.grid_res,
        lr=args.learning_rate
    )

    full_experiment_name = f"{args.experiment_name}_{args.model_type}"
    logger = TensorBoardLogger(args.log_dir, name=full_experiment_name)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val/mae',
        dirpath=Path("checkpoints_baselines") / full_experiment_name,
        filename='best-{epoch:02d}-{val/mae:.4f}', 
        save_top_k=1, 
        mode='min' 
    )
    
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=[1], 
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
    model.hparams.lr = new_lr 
    
    print(f"Starting training for baseline: {args.model_type}...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()