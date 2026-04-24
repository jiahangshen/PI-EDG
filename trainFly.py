import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassJaccardIndex
from model import UNet3D ,UNetsingle3D
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset_classifier import VoxelizedMoEDataset, collate_fn_skip_error
from train_classifier import calculate_class_weights_from_dataset 
torch.set_float32_matmul_precision('high')
from torch.optim import lr_scheduler 
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef
import torch.nn.functional as F
# file: train_fourier.py (建议新建一个文件)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import lr_scheduler

# 确保导入您修改后的、后期融合PE的UNet
from model import UNet3D 
# 导入所需的验证指标
from torchmetrics.regression import MeanAbsoluteError, PearsonCorrCoef, MeanSquaredError


# in your LightningModule's __init__
# (这个新头会替换掉你原来的 self.phase_head)
def angle_difference(angle1, angle2):
        # 计算两个角度之间的最短距离，范围在 [-pi, pi]
    diff = angle1 - angle2
    return torch.atan2(torch.sin(diff), torch.cos(diff))
class ImplicitPhaseHead(nn.Module):
    def __init__(self, in_channels, n_freqs=10, hidden_dim=256, n_layers=4):
        super().__init__()
        self.feature_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.n_freqs = n_freqs
        self.pos_encoder_dim = 3 * 2 * n_freqs

        layers = []
        current_dim = self.pos_encoder_dim + hidden_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Tanh())
        
        self.phase_mlp = nn.Sequential(*layers)
        
    def positional_encoding(self, coords):
        freqs = 2.0 ** torch.arange(self.n_freqs, device=coords.device)
        scaled_coords = coords.unsqueeze(-1) * freqs
        sines = torch.sin(scaled_coords)
        cosines = torch.cos(scaled_coords)
        encoded = torch.cat([sines, cosines], dim=-1)
        return encoded.flatten(start_dim=-2)

    def forward(self, spatial_features, freq_grid):
        # spatial_features: [B, C, D_s, H_s, W_s]
        # freq_grid: [B, D_f, H_f, W_f, 3]
        
        B = spatial_features.size(0)
        # 显式地从 freq_grid 获取维度
        D_f, H_f, W_f = freq_grid.shape[1], freq_grid.shape[2], freq_grid.shape[3]
        
        # a. 提取全局条件向量
        global_cond = self.global_pool(spatial_features).view(B, -1)
        global_cond = self.feature_mlp(global_cond) # [B, hidden_dim]
        
        # b. 对频率坐标进行位置编码
        encoded_freqs = self.positional_encoding(freq_grid) # [B, D_f, H_f, W_f, pos_encoder_dim]
        
        # c. 将全局条件向量广播并与编码后的频率坐标拼接
        # 确保 expand 的维度顺序与 freq_grid/encoded_freqs 一致
        global_cond_expanded = global_cond.view(B, 1, 1, 1, -1).expand(-1, D_f, H_f, W_f, -1)
        
        # --- 调试打印 ---
        # print(f"encoded_freqs shape: {encoded_freqs.shape}")
        # print(f"global_cond_expanded shape: {global_cond_expanded.shape}")
        
        mlp_input = torch.cat([encoded_freqs, global_cond_expanded], dim=-1)
        
        # d. 通过主MLP得到相位谱
        phase_logits = self.phase_mlp(mlp_input) # [B, D_f, H_f, W_f, 1]
        
        # squeeze(-1) to remove the last dim, permute to [B, D, H, W] convention
        return phase_logits.squeeze(-1) # [B, D_f, H_f, W_f]

class GlobalDecoderHead(nn.Module):
    def __init__(self, in_channels, spatial_dim=64, out_freq_shape=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_freq_shape = out_freq_shape # e.g., (64, 64, 33)
        
        # 1. 全局信息聚合
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # 2. MLP 用于处理全局特征向量
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
        
        # 3. 解码器/上采样器，将全局向量“广播”回频域谱的形状
        # 我们可以使用转置卷积来实现
        # 我们需要从一个小的空间尺寸开始，逐步上采样
        self.decoder = nn.Sequential(
            # 从 1x1x1 -> 4x4x4 (近似)
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            # 4x4x4 -> 8x8x8
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 8x8x8 -> 16x16x16
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 16x16x16 -> 32x32x32
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 32x32x32 -> 64x64x64 (近似)
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        # x: [B, C, H, W, D] (空间特征)
        
        # 1. 聚合为全局向量
        global_feat = self.global_pool(x) # [B, C, 1, 1, 1]
        global_feat = global_feat.view(x.size(0), -1) # [B, C]
        
        # 2. 通过MLP处理
        processed_feat = self.mlp(global_feat) # [B, 512]
        
        # 3. 准备解码
        processed_feat = processed_feat.view(x.size(0), 512, 1, 1, 1) # [B, 512, 1, 1, 1]
        
        # 4. 解码为频域谱形状
        # 输出的形状可能需要裁剪或填充以精确匹配 out_freq_shape
        output_logits = self.decoder(processed_feat)
        
        # 裁剪到正确的最终尺寸
        output_logits = output_logits[..., :self.out_freq_shape[0], :self.out_freq_shape[1], :self.out_freq_shape[2]]
        
        return output_logits
class FourierDensityModel(pl.LightningModule):
    def __init__(self, in_channels_atom, in_channels_pe,
                 learning_rate=1e-4, lambda_amp=1.0, lambda_phase=50.0, lambda_spatial=0.1,
                 unet_out_channels=64, unet_internal_channels=64):
        super().__init__()
        # 保存所有超参数，方便后续加载和调试
        self.save_hyperparameters()

        # 1. 空间特征提取器: 使用您原来的后期融合UNet
        self.unet_backbone = UNet3D(
            in_channels_atom=in_channels_atom,
            in_channels_pe=in_channels_pe,
            out_channels=unet_out_channels,
            n_channels=unet_internal_channels
        )

        # 2. 振幅预测头
        spatial_dim = 64 
        freq_dim = 64 # 我们希望预测的频域谱尺寸
        freq_D_out = freq_dim // 2 + 1
        self.amp_head = GlobalDecoderHead(unet_out_channels, spatial_dim, (spatial_dim, spatial_dim, freq_D_out))
        self.phase_head = ImplicitPhaseHead(in_channels=unet_out_channels)
        # 4. 损失函数
        self.mae_loss = nn.L1Loss()
        
        # 5. 验证指标
        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.val_pearson = PearsonCorrCoef()

    def forward(self, atom_grid, pe_grid):
        # a. 在空间域提取高级特征
        # 假设 unet_backbone 输入 [B, C, D, H, W], 输出 [B, C_feat, D_s, H_s, W_s]
        spatial_features = self.unet_backbone(atom_grid, pe_grid)
        
        # b. 振幅头
        pred_amp_logits = self.amp_head(spatial_features) # Shape: [B, 1, D_s, H_s, W_s]
        
        # c. --- 新的相位预测流程 ---
        
        # 1. 创建频率坐标网格，严格遵循 (D, H, W) 约定
        B, C, D, H, W = atom_grid.shape
        device = atom_grid.device
        
        freq_coords_d = torch.fft.fftfreq(D, d=1.0, device=device)
        freq_coords_h = torch.fft.fftfreq(H, d=1.0, device=device)
        freq_coords_w = torch.fft.rfftfreq(W, d=1.0, device=device)
        W_f = freq_coords_w.shape[0]

        # 使用 'ij' 索引，输入顺序 (d, h, w) 确保输出的 grid 张量
        # 形状为 (D, H, W_f)
        grid_d, grid_h, grid_w = torch.meshgrid(freq_coords_d, freq_coords_h, freq_coords_w, indexing="ij")
        
        # 堆叠后，freq_grid_base 的形状是 (D, H, W_f, 3)
        freq_grid_base = torch.stack((grid_d, grid_h, grid_w), dim=-1)
        
        # 扩展到 batch size, 最终形状 [B, D, H, W_f, 3]
        freq_grid = freq_grid_base.unsqueeze(0).expand(B, -1, -1, -1, -1)
        
        # 2. 调用新的 phase_head
        pred_phase_logits = self.phase_head(spatial_features, freq_grid) # 应输出 [B, D, H, W_f]

        # d. 调整尺寸和范围
        # 关键修正：确保 pred_amp_logits 与 pred_phase_logits 在空间维度上匹配
        # unet 的输出 spatial_features 可能尺寸变小了，我们需要将 freq 相关的预测对齐到那个尺寸
        
        # 首先，将 amp_logits 的尺寸裁剪到与 phase_logits 一致
        # 我们假设 amp_head 不改变空间尺寸，所以 amp_logits 和 spatial_features 尺寸相同
        D_s, H_s, W_s = spatial_features.shape[2:]
        
        # 我们需要一个与 spatial_features 尺寸匹配的 phase_logits。
        # 但 ImplicitPhaseHead 输出的是全尺寸的。我们需要对输入 freq_grid 进行下采样。
        # 这是一个更复杂的问题。让我们做一个更简单的修正：假设 UNet 不改变尺寸。
        
        # 让我们检查 amp_head 的输出尺寸
        # pred_amp_logits shape: [B, 1, D, H, W] (假设 UNet out_channels = in_channels, 且无下采样)
        
        # --- 让我们回到最根本的错误点 ---
        # 错误信息: Expected size 64 but got size 33 for tensor number 1
        # 这几乎总是意味着 torch.meshgrid 的输出顺序与 PyTorch 其他部分的约定不符
        # PyTorch 的 grid_sample, interpolate 等函数期望 grid 的形状是 (N, H, W, D, 3)
        # 让我们尝试 'xy' 索引
        
        # --- 重写第1步，使用 'xy' 索引并调整顺序 ---
        grid_h, grid_w, grid_d = torch.meshgrid(freq_coords_h, freq_coords_w, freq_coords_d, indexing="xy")
        # 'xy' 索引下，输出形状为 (len(w), len(h), len(d)) -> (33, 64, 64)
        
        # 我们需要 (D, H, W_f)，所以我们需要调整 meshgrid 的输入顺序和 permute
        grid_d, grid_h, grid_w = torch.meshgrid(freq_coords_d, freq_coords_h, freq_coords_w, indexing="ij")
        # grid_d.shape -> (64, 64, 33)
        
        # 这表明 freq_grid 的形状是 (64, 64, 33, 3)
        # H, W, D_f
        # 而我们的约定是 (D, H, W_f) -> (64, 64, 33)
        
        # 好吧，让我们放弃复杂的 meshgrid 调试，用最笨但最可靠的方法
        
        # --- 最终、最稳健的 freq_grid 创建方式 ---
        D_f, H_f, W_f = D, H, W // 2 + 1
        d_coords = torch.linspace(-0.5, 0.5, D_f, device=device)
        h_coords = torch.linspace(-0.5, 0.5, H_f, device=device)
        w_coords = torch.linspace(0, 0.5, W_f, device=device)
        
        # 创建一个 [D, H, W, 3] 的空张量
        freq_grid = torch.empty(D_f, H_f, W_f, 3, device=device)
        
        # 逐个填充坐标
        freq_grid[..., 0] = d_coords.view(-1, 1, 1)
        freq_grid[..., 1] = h_coords.view(1, -1, 1)
        freq_grid[..., 2] = w_coords.view(1, 1, -1)
        
        freq_grid = freq_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
        # 现在 freq_grid 的形状绝对是 [B, D_f, H_f, W_f, 3]

        # 2. 调用新的 phase_head
        pred_phase_logits = self.phase_head(spatial_features, freq_grid) # 输出 [B, D_f, H_f, W_f]
        
        # 3. 调整尺寸和范围
        pred_amp = F.relu(pred_amp_logits.squeeze(1)) # [B, D_s, H_s, W_s]
        
        # 关键：如果 unet 缩小了尺寸，我们需要将 amp 和 phase 对齐
        if pred_amp.shape != pred_phase_logits.shape:
             # 将尺寸较小的 amp 上采样到与 phase 一致
             pred_amp = F.interpolate(pred_amp.unsqueeze(1), size=pred_phase_logits.shape[1:], mode='trilinear', align_corners=False).squeeze(1)

        pred_phase = pred_phase_logits * torch.pi

        # e. 合成并进行逆变换
        pred_freq_domain = torch.polar(pred_amp, pred_phase)
        pred_density = torch.fft.irfftn(pred_freq_domain, s=(D, H, W), dim=(1,2,3))
        
        return pred_amp, pred_phase, pred_density

    def _calculate_loss(self, pred_amp, pred_phase, pred_density, target_density):
        # a. 将目标密度转换到频域
        target_freq = torch.fft.rfftn(target_density, dim=(1,2,3))
        target_amp = torch.abs(target_freq)
        target_phase = torch.angle(target_freq)
        
        # b. 计算振幅损失 (使用对数尺度以稳定训练)
        epsilon = 1e-8
        loss_amp = self.mae_loss(torch.log(pred_amp + epsilon), torch.log(target_amp + epsilon))
        # c. 计算相位损失        
        phase_diff = angle_difference(pred_phase, target_phase)
        loss_phase = torch.mean(torch.abs(phase_diff))
        
        # d. (可选) 计算空间域损失，作为辅助监督
        loss_spatial = self.mae_loss(pred_density, target_density)
        
        # e. 加权求和得到总损失
        total_loss = (self.hparams.lambda_amp * loss_amp + 
                      self.hparams.lambda_phase * loss_phase +
                      self.hparams.lambda_spatial * loss_spatial)
                      
        return total_loss, loss_amp, loss_phase, loss_spatial

    def training_step(self, batch, batch_idx):
        # 解包简化的 batch
        atom_grids,_, pe_grids,_, target_densities,_ = batch
        if atom_grids.nelement() == 0: return None
        
        # 确保 pe_grid 是 float32
        pe_grids = pe_grids.to(torch.float32)
        
        # 前向传播
        pred_amp, pred_phase, pred_density = self.forward(atom_grids, pe_grids)
        
        # 计算损失
        total_loss, loss_amp, loss_phase, loss_spatial = self._calculate_loss(
            pred_amp, pred_phase, pred_density, target_densities
        )
        
        # 日志记录
        if batch_idx % 50 == 0: 
            print(f"\nEpoch {self.current_epoch}, Step {batch_idx}: total_loss={total_loss.item():.4f}, loss_amp={loss_amp.item():.4f}, loss_phase={loss_phase.item():.4f}, loss_spatial={loss_spatial.item():.4f}")
        
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_amp', loss_amp, on_step=True, on_epoch=True)
        self.log('train/loss_phase', loss_phase, on_step=True, on_epoch=True)
        self.log('train/loss_spatial', loss_spatial, on_step=True, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        atom_grids,_, pe_grids,_, target_densities,_ = batch

        if atom_grids.nelement() == 0: return

        pe_grids = pe_grids.to(torch.float32)
        
        # 前向传播
        pred_amp, pred_phase, pred_density = self.forward(atom_grids, pe_grids)
        
        # 计算损失以进行监控
        total_loss, loss_amp, loss_phase, loss_spatial = self._calculate_loss(
            pred_amp, pred_phase, pred_density, target_densities
        )
        
        # 更新验证指标 (在空间域进行评估)
        preds_flat = pred_density.flatten()
        targets_flat = target_densities.flatten()
        self.val_mae.update(preds_flat, targets_flat)
        self.val_mse.update(preds_flat, targets_flat)
        # 只有当 preds_flat 和 targets_flat 的标准差不为0时，Pearson才能计算
        if preds_flat.std() > 1e-6 and targets_flat.std() > 1e-6:
            self.val_pearson.update(preds_flat, targets_flat)

        self.log('val/total_loss', total_loss, on_epoch=True)
        self.log('val/loss_amp', loss_amp, on_epoch=True)
        self.log('val/loss_phase', loss_phase, on_epoch=True)
        self.log('val/loss_spatial', loss_spatial, on_epoch=True)

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
        # 您可以复用之前的学习率调度器
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches, # 确保 trainer 已被关联
            eta_min=1e-6 
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
def main():
    # ... (所有参数定义不变) ...
    # 新增或调整的超参数
    RAW_DATA_ROOT = Path("./raw_data_cube") 
    CACHE_DIR = Path("./preprocessed_cache")
    GRID_RESOLUTION = 64
    GRID_BOUNDS = (-10.0, 10.0)
    NUM_PE_FREQS = 4
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-5
    LAMBDA_REG = 0.1
    LAMBDA_ALIGN = 0.5
    NUM_EPOCHS = 50
    VAL_SPLIT_RATIO = 0.1
    NUM_WORKERS = 12
    CLASS_BINS = [0, 1e-14, 1e-10, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, np.inf]
    NUM_CLASSES = len(CLASS_BINS)
    LAMBDA_FREQ = 0.01
    UNET_INTERNAL_CHANNELS = 64 # 控制两个UNet的宽度/大小

    # Dataset 和 DataLoader 的实例化完全不变！
    full_dataset = VoxelizedMoEDataset(
        cache_dir=CACHE_DIR,
        raw_data_root_for_scan=RAW_DATA_ROOT,
        grid_res=GRID_RESOLUTION,
        grid_bounds=GRID_BOUNDS,
        num_pe_freqs=NUM_PE_FREQS,
        class_bins=CLASS_BINS,
        augment_on_the_fly=True
    )
    NUM_ATOM_TYPES = full_dataset.num_atom_channels
    NUM_PE_CHANNELS = 3 * 2 * NUM_PE_FREQS
    n_total = len(full_dataset); n_val = int(n_total * VAL_SPLIT_RATIO); n_train = n_total - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    class_weights = torch.tensor([0.0261, 1.4925, 1.2437, 0.8265, 0.7202, 1.1831, 1.2085, 1.3151, 0.9841])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=True)


    # 关键：实例化新的模型
    model = FourierDensityModel(
        # in_channels_atom 和 in_channels_pe 实际上不再直接用于UNet
        # 但为了兼容 Dataset 和其他可能的用途，我们先按原样传入
        in_channels_atom=NUM_ATOM_TYPES,
        in_channels_pe=NUM_PE_CHANNELS,
        #num_classes=NUM_CLASSES,
        #class_bins=CLASS_BINS,
        learning_rate=LEARNING_RATE,
        #lambda_reg=LAMBDA_REG,
        #class_weights=class_weights,
        #lambda_freq=LAMBDA_FREQ,
        #unet_internal_channels=UNET_INTERNAL_CHANNELS
    )
    logger = TensorBoardLogger("logs", name="density_fly_moe_fixed")
    # Trainer 和 .fit() 调用完全不变
    # 确保 checkpoint monitor 的 key 和 mode 正确
    checkpoint_callback = ModelCheckpoint(
            monitor='val/epoch_mae', # 监控总损失
            dirpath='checkpoints_freq_moe', 
            filename='best-freq-moe-model-{epoch:02d}-loss={val/epoch_mae:.4f}', 
            save_top_k=1, 
            mode='min' # 模式改为'min'因为我们监控的是loss
        )
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=NUM_EPOCHS, logger=logger, callbacks=[checkpoint_callback], gradient_clip_val=1.0)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("MoE 模型训练完成！")
if __name__ == "__main__":
    pl.seed_everything(42)
    main()