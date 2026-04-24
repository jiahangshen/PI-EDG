import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import lr_scheduler
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef
from pathlib import Path

# --- 导入我们新写的专属 Dataset ---
from dataset_nature_baseline import NatureBaselineDataset, collate_fn_nature
# 导入 Nature 模型 (即前文复刻的 SRResNet3D)
from nature_resnet import SRResNet3D 

torch.set_float32_matmul_precision('high')

class NatureBaselineModule(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # 实例化 SRResNet3D (输入通道为1，因为SAD是单通道标量场)
        # 【关键修改】：显式设置 upscale_factor=1
        # 这样模型内部会使用 nn.Identity() 跳过上采样层，保持 64x64x64 输出
        self.model = SRResNet3D(
            in_channels=1, 
            out_channels=1, 
            num_features=64, 
            num_res_blocks=16, 
            upscale_factor=1  # <--- 必须改为 1
        )
        
        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.val_pearson = PearsonCorrCoef()

    def forward(self, sad_grid, target_electrons):
        return self.model(sad_grid, target_electrons)

    def _step(self, batch, stage):
        # --- 极其干净的数据流 ---
        x_sad = batch['sad_grid'] 
        targets = batch['target_density']
        
        if 'target_integral' in batch:
            target_electrons = batch['target_integral']
            print(target_electrons)
            # 如果 Dataset 没给，我们当场算一下 target 的积分作为 electron number
        target_electrons = torch.sum(targets, dim=(1, 2, 3, 4)) 
        print(target_electrons)
        # 1. 预测 (现在传入两个参数)
        preds = self(x_sad, target_electrons).squeeze(1) 
        
        # 2. 损失函数 (最原始的均方误差)
        loss = F.mse_loss(preds, targets)
        
        batch_size = x_sad.shape[0]
        self.log(f'{stage}/loss', loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        
        if stage == 'val':
            preds_flat = preds.flatten()
            targets_flat = targets.flatten()
            self.val_mae.update(preds_flat, targets_flat)
            self.val_mse.update(preds_flat, targets_flat)
            self.val_pearson.update(preds_flat, targets_flat)
            
            self.log('val/mae', self.val_mae, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log('val/mse', self.val_mse, batch_size=batch_size, sync_dist=True)
            self.log('val/pearson', self.val_pearson, batch_size=batch_size, sync_dist=True)
            
        return loss

    def training_step(self, batch, batch_idx): return self._step(batch, 'train')
    def validation_step(self, batch, batch_idx): return self._step(batch, 'val')
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sch = lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return [opt], [sch]

# =========================================================================
# 参数解析与启动
# =========================================================================
def get_args():
    parser = argparse.ArgumentParser(description="Train Nature Comm 2025 Baseline (SAD Input)")
    parser.add_argument('--devices', type=int, default=2, help="Number of GPUs")
    parser.add_argument('--raw_data_root', type=str, default="./raw_data_cube")
    parser.add_argument('--cache_dir', type=str, default="./preprocessed_cache")
    parser.add_argument('--log_dir', type=str, default="logs_baselines")
    parser.add_argument('--experiment_name', type=str, default="nature_baseline_sad_v1")
    parser.add_argument('--sad_dir', type=str, default="./preprocessed_sad", help="Directory containing precomputed SAD grids")
    parser.add_argument('--grid_res', type=int, default=64)
    parser.add_argument('--grid_bounds', type=float, nargs=2, default=[-10.0, 10.0])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.1)

    return parser.parse_args()

def main():
    args = get_args()
    
    # 【非常关键】固定随机种子，确保数据集划分与您的 MoE 训练完全一致！
    pl.seed_everything(42)

    is_main_process = (torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized()

    if is_main_process:
        print("\n=======================================================")
        print("Initializing Dedicated Dataset for Nature SAD Baseline...")
        print("=======================================================\n")
        
    full_dataset = NatureBaselineDataset(
        raw_data_root=Path(args.raw_data_root),
        cache_dir=Path(args.cache_dir),
        sad_dir=Path(args.sad_dir), # === 传给 Dataset ===
        grid_res=args.grid_res,
        grid_bounds=tuple(args.grid_bounds)
    )
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    
    # 使用和您的主干网络完全一样的切分方式
    train_set, val_set = random_split(full_dataset,[n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    if is_main_process:
        print("Setting up DataLoaders...")
        
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_nature, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_nature, persistent_workers=True)
    
    if is_main_process:
        print("Initializing 3D SRResNet Baseline Model...")
        
    model = NatureBaselineModule(lr=args.learning_rate)

    logger = TensorBoardLogger(args.log_dir, name=args.experiment_name)
    checkpoint_callback = ModelCheckpoint(
        monitor='val/mae',
        dirpath=Path("checkpoints_baselines") / args.experiment_name,
        filename='best-{epoch:02d}-{val/mae:.4f}', 
        save_top_k=1, 
        mode='min'
    )
    
    from pytorch_lightning.strategies import DDPStrategy
    strategy = DDPStrategy(find_unused_parameters=False) if args.devices > 1 else "auto"
    
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=args.devices, 
        strategy=strategy,
        max_epochs=args.max_epochs, 
        logger=logger, 
        callbacks=[checkpoint_callback], 
        gradient_clip_val=1.0,
        precision='16'  # <--- 增加这一行
    )
    
    if args.devices == 1:
        print("Running LR Finder...")
        tuner = trainer.tuner
        lr_finder = tuner.lr_find(model, train_dataloaders=train_loader)
        new_lr = lr_finder.suggestion()
        print(f"Suggested LR: {new_lr}")
        model.hparams.lr = new_lr
    
    if is_main_process:
        print("Starting training Nature SAD Baseline...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()