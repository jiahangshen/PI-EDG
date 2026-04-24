import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pathlib import Path

# --- 导入自定义模块 ---
# 确保 model_phys.py, physics_utils.py, dataset_phys.py 在同一目录下
from model_phys import PIUNet3D
from physics_utils import EnergyLoss, PhysicsUtils
from dataset_phys import PhysicsDataset, collate_fn_physics

# 设置矩阵乘法精度以加速训练 (A100/3090/4090 等显卡推荐)
torch.set_float32_matmul_precision('high')
import matplotlib.pyplot as plt


def visualize_vext_check(atom_coords, atom_z, grid_coords, 
                         save_path="vext_alignment_check.png", 
                         slice_dim=2): # slice_dim=2 表示沿 Z 轴切片 (XY平面)
    """
    可视化检查 V_ext 和原子位置是否对齐。
    
    Args:
        atom_coords: [B, N, 3]
        atom_z: [B, N]
        grid_coords: [1, D, H, W, 3] 或 [B, D, H, W, 3]
    """
    # 1. 计算 V_ext (使用您修正后的 build_vext)
    # 注意：为了画图好看，这里 epsilon 可以给稍微大一点，或者做截断
    vext_tensor = PhysicsUtils.build_vext(atom_coords, atom_z, grid_coords, epsilon=0.1)
    
    # 2. 取第一个样本，并转换到 CPU
    # vext_tensor shape: [B, 1, D, H, W]
    # 我们取 Batch 0, Channel 0
    vext_data = vext_tensor[0, 0].detach().cpu().numpy() 
    
    # 3. 选择切片索引 (取中心)
    mid_idx = vext_data.shape[slice_dim] // 2
    
    # 4. 提取 2D 切片
    if slice_dim == 0:   # 切 X 轴 -> YZ 平面
        slice_img = vext_data[mid_idx, :, :]
        xlabel, ylabel = 'Z', 'Y'
        atom_x_idx, atom_y_idx = 2, 1 # 对应 atom_coords 的 indices
    elif slice_dim == 1: # 切 Y 轴 -> XZ 平面
        slice_img = vext_data[:, mid_idx, :]
        xlabel, ylabel = 'Z', 'X'
        atom_x_idx, atom_y_idx = 2, 0
    else:                # 切 Z 轴 -> XY 平面 (默认)
        slice_img = vext_data[:, :, mid_idx] # 注意：PyTorch 通常是 [D, H, W] -> [Z, Y, X]
        # 这里的切片取决于您的 grid 是 [Z, Y, X] 还是 [X, Y, Z]
        # 根据您之前的 meshgrid(indexing='ij')，通常 dim 0 是 Z, dim 1 是 Y, dim 2 是 X
        # 所以 vext[mid, :, :] 切出来的是 YX 平面
        slice_img = vext_data[mid_idx, :, :] 
        xlabel, ylabel = 'X', 'Y' # 对应 grid 的最后两维
        atom_x_idx, atom_y_idx = 0, 1
    
    # 5. 开始绘图
    plt.figure(figsize=(10, 8))
    
    # --- 技巧：使用 'jet' 或 'turbo' 色图，能清晰展示从 0 到 负无穷 的变化 ---
    # vmin/vmax 用于截断极值，防止原子核中心的 -10000 把整个图对比度毁了
    # 一般 vext 在原子附近是 -10 到 -50 左右，背景是 0
    plt.imshow(slice_img, cmap='jet', origin='lower', vmin=-20, vmax=0)
    plt.colorbar(label='Potential (Hartree)')
    
    # 6. 标出原子的“理论位置” (验证对齐的核心！)
    # 获取网格范围和步长 (假设是 -10 到 10)
    # 注意：这里需要您传入 grid_bounds，或者假设它是 -10 到 10
    # 为了通用，这里简单反推一下：
    res = vext_data.shape[0]
    # 假设您的 grid_coords 覆盖了整个范围，我们取其 min 和 max
    # grid_coords[0, :, :, :, 0] 是 X 坐标场
    grid_x_min = grid_coords[0, ..., 0].min().item()
    grid_x_max = grid_coords[0, ..., 0].max().item()
    step = (grid_x_max - grid_x_min) / (res - 1)
    
    # 筛选出位于切片平面附近的原子
    batch_atoms = atom_coords[0].detach().cpu().numpy() # [N, 3]
    slice_z_center = grid_coords[0, mid_idx, mid_idx, mid_idx, slice_dim].item() # 切片对应的物理坐标
    
    for i in range(batch_atoms.shape[0]):
        atom_pos = batch_atoms[i]
        # 如果原子的垂向坐标距离切片平面很近 (< 1 个体素)，就画出来
        dist_to_plane = abs(atom_pos[slice_dim] - slice_z_center)
        
        if dist_to_plane < step:
            # 计算原子在图上的像素坐标
            # Pixel = (Coord - Min) / Step
            pixel_x = (atom_pos[atom_x_idx] - grid_x_min) / step
            pixel_y = (atom_pos[atom_y_idx] - grid_x_min) / step # 假设 grid 是正方体
            
            plt.scatter(pixel_x, pixel_y, c='white', marker='x', s=100, linewidth=2, label='True Atom Pos')

    plt.title(f'V_ext Slice Check (Dim {slice_dim}, Idx {mid_idx})\nWhite X should match Blue/Black Holes')
    plt.xlabel(f'{xlabel} Index')
    plt.ylabel(f'{ylabel} Index')
    # 只显示一次 label
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend([handles[0]], [labels[0]])
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Check image saved to {save_path}")
    plt.close()

# 使用示例 (在您的训练脚本中)
# visualize_vext_check(batch['atom_coords'], batch['atom_z'], self.grid_coords)
class PhysicsDensityModule(pl.LightningModule):
    def __init__(self, 
                 atom_channels: int, 
                 grid_res: int, 
                 grid_bounds: tuple,
                 lr: float = 1e-4,
                 # 下面两个参数作为默认值
                 lambda_reg: float = 10.0, 
                 lambda_phys: float = 0.001):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = PIUNet3D(
            in_channels_atom=atom_channels, 
            n_channels=64,
            grid_res=grid_res,
            grid_bounds=grid_bounds
        )
        
        # 初始化 Loss Function
        self.loss_fn = EnergyLoss(grid_res, grid_bounds)
        
        grid_coords = PhysicsUtils.get_grid_coords(grid_res, grid_bounds, device='cpu')
        self.register_buffer('grid_coords', grid_coords)
    def _get_dynamic_weights(self):
        """
        根据当前 Epoch 动态计算损失权重 (Warm-up 策略)
        """
        current_epoch = self.current_epoch
        
        # --- 策略配置 ---
        warmup_epochs = 50
        
        if current_epoch < warmup_epochs:
            # 阶段 1: 纯监督学习，快速学习大概形状
            # Log-MSE 的数值通常较大，所以 reg 权重可以设小一点
            curr_reg = 10.0 
            curr_phys = 0.0 # 暂时关闭物理约束，避免初期梯度混乱
        else:
            # 阶段 2: 引入物理约束，稍微放松监督约束
            curr_reg = 1.0
            # 物理能量通常很大 (-1000左右)，且 Log-MSE 约为 1-10
            # 这里的权重是为了平衡梯度量级
            curr_phys = 0.0
            
        return curr_reg, curr_phys
    def forward(self, atom_grid, v_ext, z_spike, total_electrons):
        """
        前向传播包装器
        """
        return self.model(atom_grid, v_ext, z_spike, total_electrons)

    def _common_step(self, batch, stage='train'):
        # 1. 解包数据
        atom_grid = batch['atom_grid']
        atom_coords = batch['atom_coords']
        atom_z = batch['atom_z']
        target_density = batch['target_density']
        total_electrons = batch['total_electrons']
        target_integral = batch['target_integral'] # <---以此替代 total_electrons
        # 构建物理场
        v_ext = PhysicsUtils.build_vext(atom_coords, atom_z, self.grid_coords)
        z_spike = torch.sum(atom_grid, dim=1, keepdim=True) 
        bypass_constraint = (self.current_epoch < 100)
        #visualize_vext_check(batch['atom_coords'], batch['atom_z'], self.grid_coords)

        # 将 use_constraint 参数传给模型
        # 注意：如果是验证集(val)，通常即使在初期我们也想看带约束的效果，
        # 但为了 Loss 的一致性，建议验证集也保持同步，或者验证集始终开启。
        # 这里为了稳定，我们让它跟训练保持一致。
        use_constraint = not bypass_constraint
        
        final_density, _ = self.model(
            atom_grid, v_ext, z_spike, total_electrons=target_integral, 
            use_constraint=use_constraint # <--- 传入开关
        )
        # 模型预测

        # --- 修改点 2: 获取动态权重并传入 Loss 函数 ---
        curr_reg, curr_phys = self._get_dynamic_weights()
        
        # 计算损失
        loss, loss_log_mse, e_total, e_ne, e_h = self.loss_fn(
            final_density, 
            target_density, 
            v_ext,
            curr_lambda_reg=curr_reg,   # 传入动态权重
            curr_lambda_phys=curr_phys  # 传入动态权重
        )

        # --- 修改点 3: 增强日志记录 ---
        batch_size = atom_grid.shape[0]
        self.log(f'{stage}/loss', loss, batch_size=batch_size, prog_bar=True)
        
        # 记录当前的权重，方便 verify 策略是否生效
        if stage == 'train':
            self.log('weights/reg', curr_reg)
            self.log('weights/phys', curr_phys)
            
        # 记录 Log-MSE
        self.log(f'{stage}/log_mse', loss_log_mse, batch_size=batch_size, prog_bar=True)
        
        # 记录物理能量 (包括 E_ne)
        self.log(f'{stage}/energy_total', torch.mean(e_total), batch_size=batch_size)
        self.log(f'{stage}/energy_ne', torch.mean(e_ne), batch_size=batch_size) # <--- 新增
        self.log(f'{stage}/energy_H', torch.mean(e_h), batch_size=batch_size)
        
        # 记录 MAE (直观指标)
        mae = torch.mean(torch.abs(final_density - target_density))
        self.log(f'{stage}/mae', mae, batch_size=batch_size, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, stage='val')

    def configure_optimizers(self):
        # 使用 AdamW 优化器和 Cosine 学习率调度
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }

def get_args():
    parser = argparse.ArgumentParser(description="Train Physics-Informed Density Model")
    
    # --- 路径 ---
    parser.add_argument('--raw_data', type=str, default="./raw_data_cube", help="Path to raw data")
    parser.add_argument('--cache_dir', type=str, default="./preprocessed_cache", help="Path to cache")
    parser.add_argument('--log_dir', type=str, default="./logs_phys", help="Log directory")
    
    # --- 网格设置 ---
    parser.add_argument('--grid_res', type=int, default=64, help="Grid resolution")
    parser.add_argument('--grid_range', type=float, default=10.0, help="Grid bounds (+/-)")
    
    # --- 训练超参 ---
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # --- 物理权重 ---
    parser.add_argument('--lambda_reg', type=float, default=1.0, 
                        help="Weight for Density Regression (MSE). Start high.")
    parser.add_argument('--lambda_phys', type=float, default=0.0, 
                        help="Weight for Total Energy Loss. Start low to avoid instability.")
    
    return parser.parse_args()

def main():
    args = get_args()
    pl.seed_everything(42)
    
    # 1. 数据集准备
    grid_bounds = (-args.grid_range, args.grid_range)
    
    full_dataset = PhysicsDataset(
        cache_dir=Path(args.cache_dir),
        raw_data_root_for_scan=Path(args.raw_data),
        grid_res=args.grid_res,
        grid_bounds=grid_bounds
    )
    
    # 划分训练集/验证集
    n_total = len(full_dataset)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])
    
    print(f"Training on {len(train_set)} samples, Validating on {len(val_set)} samples.")
    
    # DataLoader (注意使用 collate_fn_physics)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, 
        collate_fn=collate_fn_physics
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True, 
        collate_fn=collate_fn_physics
    )
    
    # 2. 模型初始化
    model = PhysicsDensityModule(
        atom_channels=full_dataset.num_atom_channels,
        grid_res=args.grid_res,
        grid_bounds=grid_bounds,
        lr=args.lr,
        lambda_reg=args.lambda_reg,
        lambda_phys=args.lambda_phys,
        # 如果显存足够大，可以不需要改动，默认即为 64
        # 如果显存爆了，在这里改回 32
    )
    
    # 3. 回调和 Logger
    logger = TensorBoardLogger(args.log_dir, name="phys_unet_v1")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.log_dir) / "checkpoints",
        filename='best-{epoch:02d}-{val/mae:.4f}',
        monitor='val/mae',
        mode='min',
        save_top_k=1
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # 4. Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=0.5, # 物理计算容易产生大梯度，强烈建议裁剪
        log_every_n_steps=10
    )
    print("Running LR Finder...")
    tuner = trainer.tuner
    lr_finder = tuner.lr_find(model, train_dataloaders=train_loader)
    new_lr = lr_finder.suggestion()
    print(f"Suggested LR: {new_lr}")
    model.hparams.learning_rate = new_lr
    # 5. 开始训练
    print("Starting Physics-Informed Training...")
    print(f"Loss Weights -> Regression: {args.lambda_reg}, Physics: {args.lambda_phys}")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()