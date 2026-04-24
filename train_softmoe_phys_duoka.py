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

# 导入自定义模块
from dataset_moe_phys import PhysicsMoEDataset, collate_fn_moe_phys
from model_softmoe_phys import PhysicsMoE 
from physics_utils import PhysicsUtils 

torch.set_float32_matmul_precision('high')

class MoEPhysModule(pl.LightningModule):
    def __init__(self, 
                 atom_channels, pe_channels, num_experts, initial_bins,
                 grid_res=64, grid_bounds=(-10.0, 10.0),
                 lr=1e-4, 
                 lambda_global=1.0,   
                 lambda_guide=0.5):   
        super().__init__()
        self.save_hyperparameters()
        
        self.model = PhysicsMoE(atom_channels, pe_channels, num_experts=num_experts, initial_bins=initial_bins, n_channels=64)
        
        self.grid_coords = PhysicsUtils.get_grid_coords(grid_res, grid_bounds, device='cpu')
        self.register_buffer('grid_buffer', self.grid_coords)
        
        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.val_pearson = PearsonCorrCoef()
        self.train_mae = MeanAbsoluteError()

    def forward(self, atom_grid, pe_grid, v_ext, z_spike, target_integral=None, use_constraint=True):
        return self.model(atom_grid, pe_grid, v_ext, z_spike, target_integral, use_constraint)

    def _step(self, batch, stage):
        atom_grid = batch['atom_grid']
        pe_grid = batch['pe_grid']
        atom_coords = batch['atom_coords']
        atom_z = batch['atom_z']
        targets = batch['target_density'] 
        
        with torch.no_grad():
            v_ext = PhysicsUtils.build_vext(atom_coords, atom_z, self.grid_buffer)
        z_spike = torch.sum(atom_grid, dim=1, keepdim=True)
        
        use_constraint = (self.current_epoch >= 50)
        
        # --- 1. 前向传播 ---
        # 注意：这里的 guide_field_log10 已经是模型内部 detach 过的安全版本
        routing_weights, thresholds_log10, final_density, guide_field_log10 = self.model(
            atom_grid, pe_grid, v_ext, z_spike,
            total_electrons=batch['target_integral'], 
            use_constraint=use_constraint
        )
        
        # --- 2. 双重空间的目标转换 ---
        epsilon_cubic = 1e-8
        epsilon_log = 1e-35 
        
        pred_final_cubic = torch.pow(final_density + epsilon_cubic, 1.0/3.0)
        target_cubic = torch.pow(targets + epsilon_cubic, 1.0/3.0)
        target_log10 = torch.log10(targets + epsilon_log)
        
        # --- 3. 损失计算 ---
        loss_global = F.mse_loss(pred_final_cubic, target_cubic)
        
        # 这里的梯度只会更新 guide_predictor，绝对不会回流到 backbone
        loss_guide = F.mse_loss(guide_field_log10, target_log10)
        
        loss = (self.hparams.lambda_global * loss_global + 
                self.hparams.lambda_guide * loss_guide)
        
        # --- 4. 日志记录 ---
        batch_size = atom_grid.shape[0]
        self.log(f'{stage}/loss', loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log(f'{stage}/loss_global', loss_global, batch_size=batch_size, sync_dist=True)
        self.log(f'{stage}/loss_guide', loss_guide, batch_size=batch_size, sync_dist=True)
        
        # 主进程打印阈值
        if stage == 'train' and self.global_step % 200 == 0 and self.trainer.is_global_zero:
            t_real = torch.pow(10.0, thresholds_log10)
            t_vals =[f"{t.item():.1e}" for t in t_real]
            print(f"\n[Step {self.global_step}] Learned Router Thresholds: {t_vals}")
        
        final_flat = final_density.flatten()
        target_flat = targets.flatten()
        
        if stage == 'train':
            self.train_mae(final_flat, target_flat)
            self.log('train/mae', self.train_mae, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        elif stage == 'val':
            self.val_mae(final_flat, target_flat)
            self.val_mse(final_flat, target_flat)
            self.val_pearson(final_flat, target_flat)
            
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


def get_args():
    parser = argparse.ArgumentParser(description="Train Hybrid-Space Learnable MoE Model")
    parser.add_argument('--devices', type=int, default=2, help="Number of GPUs")
    parser.add_argument('--raw_data_root', type=str, default="./raw_data_cube")
    parser.add_argument('--cache_dir', type=str, default="./preprocessed_cache")
    parser.add_argument('--log_dir', type=str, default="logs_moe_hybrid")
    parser.add_argument('--experiment_name', type=str, default="moe_hybrid_v1")

    parser.add_argument('--grid_res', type=int, default=64)
    parser.add_argument('--grid_bounds', type=float, nargs=2, default=[-10.0, 10.0])
    parser.add_argument('--num_pe_freqs', type=int, default=4)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--val_split', type=float, default=0.1)
    
    parser.add_argument('--lambda_global', type=float, default=1.0)
    parser.add_argument('--lambda_guide', type=float, default=0.001)

    return parser.parse_args()

def main():
    args = get_args()
    pl.seed_everything(42)

    INITIAL_BINS =[0, 1e-34, 1e-30, 1e-26, 1e-22, 1e-18, 1e-14, 1e-10, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, np.inf]
    NUM_EXPERTS = len(INITIAL_BINS) - 1 
    GRID_BOUNDS = tuple(args.grid_bounds)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0
    else:
        is_main_process = True

    if is_main_process:
        print(f"Initializing dataset (Using {NUM_EXPERTS} Hybrid-Space Experts)...")
        
    full_dataset = PhysicsMoEDataset(
        cache_dir=Path(args.cache_dir),
        raw_data_root=Path(args.raw_data_root),
        grid_res=args.grid_res,
        grid_bounds=GRID_BOUNDS,
        num_pe_freqs=args.num_pe_freqs,
        class_bins=INITIAL_BINS, 
    )
    
    NUM_ATOM_TYPES = full_dataset.num_atom_channels
    NUM_PE_CHANNELS = 3 * 2 * args.num_pe_freqs
    
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_set, val_set = random_split(full_dataset,[n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_moe_phys, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_moe_phys, persistent_workers=True)
    
    if is_main_process:
        print("Initializing Hybrid-Space MoE model...")
    model = MoEPhysModule(
        NUM_ATOM_TYPES,
        NUM_PE_CHANNELS,
        num_experts=NUM_EXPERTS,
        initial_bins=INITIAL_BINS, 
        lr=args.learning_rate,
        lambda_global=args.lambda_global,
        lambda_guide=args.lambda_guide
    )

    logger = TensorBoardLogger(args.log_dir, name=args.experiment_name)
    checkpoint_callback = ModelCheckpoint(
        monitor='val/mae',
        dirpath=Path("checkpoints_hybrid") / args.experiment_name,
        filename='best-{epoch:02d}-{val/mae:.4f}', 
        save_top_k=1, 
        mode='min'
    )
    
    from pytorch_lightning.strategies import DDPStrategy

    if args.devices > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = "auto"
        
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=args.devices,     
        strategy=strategy,        
        max_epochs=args.max_epochs, 
        logger=logger, 
        callbacks=[checkpoint_callback], 
        gradient_clip_val=1.0
    )
    
    if args.devices == 1:
        print("Single GPU detected. Running LR Finder...")
        tuner = trainer.tuner
        lr_finder = tuner.lr_find(model, train_dataloaders=train_loader)
        new_lr = lr_finder.suggestion()
        print(f"Suggested LR: {new_lr}")
        model.hparams.lr = new_lr
    else:
        if is_main_process:
            print(f"\n[Warning] Multi-GPU ({args.devices}) detected. Skipping automatic LR Finder.")
            print(f"Using provided learning rate: {args.learning_rate}\n")
    
    if is_main_process:
        print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()