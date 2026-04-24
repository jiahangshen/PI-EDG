import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, SpearmanCorrCoef
from dataset import VoxelizedMoleculeCachedDataset, collate_fn_skip_error
from model import UNet3D 

torch.set_float32_matmul_precision('high')



class ChargeConservationLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(ChargeConservationLoss, self).__init__(); self.epsilon = epsilon
    def forward(self, prediction, target):
        pred_sum = torch.sum(prediction); target_sum = torch.sum(target)
        squared_difference = (pred_sum - target_sum) ** 2
        return squared_difference / (target_sum ** 2 + self.epsilon)

class DensityModel(pl.LightningModule):
    def __init__(self, in_channels_atom, in_channels_pe, out_channels,
                 learning_rate=1e-4, alpha=1.0, gamma=0.01,
                 eval_threshold=0.01, w_phy=0.25, w_mae=0.25,
                 w_pearson=0.25, w_spearman=0.25):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = UNet3D(
            in_channels_atom=in_channels_atom,
            in_channels_pe=in_channels_pe,
            out_channels=out_channels,
            n_channels=64
        )
        
        self.mse_loss = nn.MSELoss()
        self.charge_loss = ChargeConservationLoss()
        self.current_gamma = self.hparams.gamma
        self.epsilon=0.1
        self.validation_step_outputs = []

    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()

    def on_train_epoch_start(self):
        if self.current_epoch > 0 and self.current_epoch % 3 == 0:
            decay_rate = 0.1
            self.current_gamma *= decay_rate
            min_gamma = 1e-4
            self.current_gamma = max(self.current_gamma, min_gamma)
            print(f"\nEpoch {self.current_epoch}: Annealing GAMMA_CHARGE. New value: {self.current_gamma}\n")

    def forward(self, atom_grid, pe_grid):
        return self.model(atom_grid, pe_grid)

    def _calculate_hybrid_loss(self, log_predictions, log_targets):
        loss_mse = self.mse_loss(log_predictions, log_targets)
        pred_real_space = torch.pow(log_predictions, 3.0) # 新代码
        target_real_space = torch.pow(log_targets, 3.0)
        loss_charge = self.charge_loss(pred_real_space, target_real_space)
        # --------------------
        
        total_loss = self.hparams.alpha * loss_mse + self.hparams.gamma * loss_charge
        return total_loss, loss_mse, loss_charge
        
        
    def training_step(self, batch, batch_idx):
        atom_grids, pe_grids, targets = batch
        pe_grids = pe_grids.to(torch.float32)
        targets = targets.to(torch.float32)
        if atom_grids.nelement() == 0: return None        
        predictions = self.forward(atom_grids, pe_grids)
        loss, loss_mse, loss_charge = self._calculate_hybrid_loss(predictions, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mse_loss', loss_mse, on_step=True, on_epoch=True, logger=True)
        self.log('train_charge_loss', loss_charge, on_step=True, on_epoch=True, logger=True)
        self.log('current_gamma', self.current_gamma, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        atom_grids, pe_grids, targets = batch
        pe_grids = pe_grids.to(torch.float32)
        targets = targets.to(torch.float32)
        if atom_grids.nelement() == 0: 
            return 
        
        predictions = self.forward(atom_grids, pe_grids)
        loss, _, _ = self._calculate_hybrid_loss(predictions, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        output = {"preds": predictions.flatten().cpu(), "targets": targets.flatten().cpu()}
        self.validation_step_outputs.append(output)

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        
        if not outputs:
            print("Validation epoch produced no outputs.")
            return

        all_preds_log = torch.cat([out['preds'] for out in outputs])
        all_targets_log = torch.cat([out['targets'] for out in outputs])

        # --- 核心修改：将预测和目标从对数空间转换回真实物理空间 ---
        all_preds = torch.pow(all_preds_log, 3.0) # 新代码
        all_targets = torch.pow(all_targets_log, 3.0)
        # -----------------------------------------------------------

        # 现在，所有后续的指标计算都在真实物理空间中进行，结果将具有正确的物理意义。
        epoch_mae = torch.mean(torch.abs(all_preds - all_targets))

        vx = all_preds - torch.mean(all_preds)
        vy = all_targets - torch.mean(all_targets)
        denominator = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
        epoch_pearson = torch.sum(vx * vy) / (denominator + 1e-6) 

        # Spearman 的计算仍然是正确的，因为它基于秩
        preds_rank = torch.argsort(torch.argsort(all_preds)).float()
        targets_rank = torch.argsort(torch.argsort(all_targets)).float()
        vx_rank = preds_rank - torch.mean(preds_rank)
        vy_rank = targets_rank - torch.mean(targets_rank)
        denominator_rank = torch.sqrt(torch.sum(vx_rank ** 2)) * torch.sqrt(torch.sum(vy_rank ** 2))
        epoch_spearman = torch.sum(vx_rank * vy_rank) / (denominator_rank + 1e-6)

        self.log('val_epoch_mae', epoch_mae, prog_bar=True)
        self.log('val_epoch_pearson_pct', epoch_pearson * 100, prog_bar=True)
        self.log('val_epoch_spearman_pct', epoch_spearman * 100, prog_bar=True)
        
        # 这里的求和现在是总电荷，计算结果正确
        pred_sum = torch.sum(all_preds); target_sum = torch.sum(all_targets)
        phy_err = torch.abs(pred_sum - target_sum) / (target_sum + 1e-6)
        
        # 所有 score 的计算也都基于正确的物理量了
        score_phy = 1.0 / (1.0 + phy_err)
        score_mae = 1.0 / (1.0 + epoch_mae)
        score_pearson = epoch_pearson.clamp(min=0)
        score_spearman = epoch_spearman.clamp(min=0)
        overall_score = (self.hparams.w_phy * score_phy + self.hparams.w_mae * score_mae +
                         self.hparams.w_pearson * score_pearson + self.hparams.w_spearman * score_spearman)
        self.log('phy_err', phy_err, prog_bar=True)

        self.log('val_overall_score', overall_score, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_overall_score"}}

def main():
    RAW_DATA_ROOT = Path("./raw_data_cube") 
    CACHE_DIR = Path("./preprocessed_cache")
    GRID_RESOLUTION = 64
    GRID_BOUNDS = (-10.0, 10.0)
    NUM_PE_FREQS = 4
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 100
    VAL_SPLIT_RATIO = 0.1
    NUM_WORKERS = 13
    ALPHA_MSE = 1.0
    GAMMA_CHARGE = 0.1
    W_PHY=0.8; W_MAE=0.2; W_PEARSON=0; W_SPEARMAN=0
    print("正在初始化缓存数据集 (含PE)...")
    try:
        full_dataset = VoxelizedMoleculeCachedDataset(
            cache_dir=CACHE_DIR,
            raw_data_root_for_scan=RAW_DATA_ROOT,
            grid_res=GRID_RESOLUTION,
            grid_bounds=GRID_BOUNDS,
            num_pe_freqs=NUM_PE_FREQS
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"初始化数据集失败: {e}"); return
        
    NUM_ATOM_TYPES = full_dataset.num_atom_channels
    NUM_PE_CHANNELS = 3 * 2 * NUM_PE_FREQS
    print(f"原子通道数: {NUM_ATOM_TYPES}")
    print(f"PE 通道数: {NUM_PE_CHANNELS}")
    n_total = len(full_dataset); n_val = int(n_total * VAL_SPLIT_RATIO); n_train = n_total - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    print(f"总样本数: {n_total}, 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    validation_sample_names = [full_dataset.indices[i] for i in val_dataset.indices]
    
    print("验证集样本的索引名称 (前20个):")
    print(validation_sample_names[:20])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False, collate_fn=collate_fn_skip_error, persistent_workers=True if NUM_WORKERS > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False, collate_fn=collate_fn_skip_error, persistent_workers=True if NUM_WORKERS > 0 else False)
    model = DensityModel(
        in_channels_atom=NUM_ATOM_TYPES,
        in_channels_pe=NUM_PE_CHANNELS,
        out_channels=1, 
        learning_rate=LEARNING_RATE,
        alpha=ALPHA_MSE, gamma=GAMMA_CHARGE,
        w_phy=W_PHY, w_mae=W_MAE, w_pearson=W_PEARSON, w_spearman=W_SPEARMAN
    )
    
    logger = TensorBoardLogger("logs", name="density_prediction_late_fusion")
    
    checkpoint_callback = ModelCheckpoint(monitor='val_overall_score', dirpath='checkpoints_late_fusion', filename='best-model-{epoch:02d}-score={val_overall_score:.4f}', save_top_k=1, mode='max')
    early_stop_callback = EarlyStopping(monitor="val_overall_score", patience=20, verbose=True, mode='max')
    
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=NUM_EPOCHS, logger=logger, callbacks=[checkpoint_callback, early_stop_callback], gradient_clip_val=1.0, profiler='simple',auto_lr_find=True )
    
    print("开始训练模型 (高斯模糊 + 后期融合 PE)...")
    '''print("开始自动寻找最佳学习率...")
    lr_finder = trainer.tuner.lr_find(model, train_dataloaders=train_loader)
    new_lr = lr_finder.suggestion()
    print(f"找到建议的学习率: {new_lr}")
    model.hparams.learning_rate = new_lr
    model.lr = new_lr 

    fig = lr_finder.plot(suggest=True)
    fig.savefig('lr_finder_plot.png')
    print("学习率查找曲线图已保存为 'lr_finder_plot.png'")
    print(f"使用新的学习率 {model.hparams.learning_rate:.6f} 开始训练模型...")'''
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("训练完成！")
    if hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
        print(f"最好的模型保存在: {checkpoint_callback.best_model_path}")
    else:
        print("未能保存最佳模型。")

if __name__ == "__main__":
    pl.seed_everything(42)
    main()