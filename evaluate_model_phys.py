import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef

# --- 导入自定义模块 ---
# 确保 train_phys_moe.py, dataset_moe_phys.py, model_moe_phys.py, physics_utils.py 在同一目录
from train_phys_moe import MoEPhysModule 
from dataset_moe_phys import PhysicsMoEDataset, collate_fn_moe_phys
from physics_utils import PhysicsUtils

# --- 1. 配置参数 ---
# !!! 请修改为您实际的 Checkpoint 路径 !!!
CHECKPOINT_PATH = Path("checkpoints_hybrid/moe_hybrid_v1/best-epoch=23-val/mae=0.0035.ckpt") 

# --- 数据集参数 (与训练一致) ---
RAW_DATA_ROOT = Path("./raw_data_cube") 
CACHE_DIR = Path("./preprocessed_cache")
GRID_RESOLUTION = 64
GRID_BOUNDS = (-10.0, 10.0)
NUM_PE_FREQS = 4
# MoE Bins (必须与训练时完全一致!)
CLASS_BINS = [0, 1e-34, 1e-30, 1e-26, 1e-22, 1e-18, 1e-14, 1e-10, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, np.inf]
NUM_CLASSES = len(CLASS_BINS)

BATCH_SIZE = 8 
VAL_SPLIT_RATIO = 0.1
NUM_WORKERS = 8

def evaluate():
    """
    加载 Physics-Informed MoE 模型，并在验证集上进行详细评估。
    """
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. 加载模型 ---
    print(f"Loading Physics-Informed MoE model from: {CHECKPOINT_PATH}")
    try:
        # 注意：Lightning 会自动从 checkpoint 加载 hyperparameters
        model = MoEPhysModule.load_from_checkpoint(CHECKPOINT_PATH)
        # 强制覆盖 grid_buffer 设备 (有时 load_from_checkpoint 不会自动移动 buffer)
        model.grid_buffer = model.grid_buffer.to(device) 
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- 3. 准备数据集 ---
    print("Preparing validation dataset...")
    
    # 使用 PhysicsMoEDataset
    try:
        full_dataset = PhysicsMoEDataset(
            cache_dir=CACHE_DIR,
            raw_data_root=RAW_DATA_ROOT,
            grid_res=GRID_RESOLUTION,
            grid_bounds=GRID_BOUNDS,
            num_pe_freqs=NUM_PE_FREQS,
            class_bins=CLASS_BINS
        )
    except Exception as e:
        print(f"数据集初始化失败: {e}")
        return

    n_total = len(full_dataset)
    n_val = int(n_total * VAL_SPLIT_RATIO)
    n_train = n_total - n_val
    _, val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    # 使用 collate_fn_moe_phys
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn_moe_phys
    )
    print(f"Validation dataset prepared with {len(val_dataset)} samples.")
    
    all_preds = []
    all_targets = []

    # --- 4. 执行推理并收集结果 ---
    print("Running predictions on the validation set...")
    
    # 获取 Grid Buffer 用于计算 v_ext
    grid_buffer = model.grid_buffer.to(device)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Predicting"):
            if batch is None: continue
            
            # 解包并移动数据
            atom_grid = batch['atom_grid'].float().to(device)
            pe_grid = batch['pe_grid'].float().to(device)
            atom_coords = batch['atom_coords'].to(device)
            atom_z = batch['atom_z'].to(device)
            target_density = batch['target_density'].float().to(device)
            target_integral = batch['target_integral'].float().to(device)

            # --- 关键：动态构建物理场 & 拼接输入 (与 train_phys_moe.py 一致) ---
            v_ext = PhysicsUtils.build_vext(atom_coords, atom_z, grid_buffer)
            z_spike = torch.sum(atom_grid, dim=1, keepdim=True)
            
            # [B, C_all, D, H, W]
            inputs = torch.cat([atom_grid, pe_grid, v_ext, z_spike], dim=1)

            # 前向传播
            # use_constraint=True 启用硬约束
            # target_integral 传入真实的 Raw Sum
            _, _, final_density = model(atom_grid, pe_grid, v_ext, z_spike,target_integral=target_integral, use_constraint=False)
            
            all_preds.append(final_density.cpu())
            all_targets.append(target_density.cpu())

    print("Prediction collection finished. Calculating metrics...")
    
    all_preds_tensor = torch.cat(all_preds, dim=0).flatten()
    all_targets_tensor = torch.cat(all_targets, dim=0).flatten()
    
    del all_preds, all_targets

    # --- 5. 总体评估指标 ---
    mse = MeanSquaredError()(all_preds_tensor, all_targets_tensor)
    mae = MeanAbsoluteError()(all_preds_tensor, all_targets_tensor)
    pearson = PearsonCorrCoef()(all_preds_tensor, all_targets_tensor)
    rmse = torch.sqrt(mse)
    
    # MAPE (Target > 1e-34)
    epsilon_mape = 1e-34
    valid_mask_mape = all_targets_tensor > epsilon_mape
    if valid_mask_mape.sum() > 0:
        mape = torch.mean(torch.abs((all_preds_tensor[valid_mask_mape] - all_targets_tensor[valid_mask_mape]) / all_targets_tensor[valid_mask_mape])) * 100
    else:
        mape = torch.tensor(float('nan'))

    print("\n--- Overall Metrics ---")
    print(f"  - MSE       : {mse.item():.4e}")
    print(f"  - RMSE      : {rmse.item():.4e}")
    print(f"  - MAE       : {mae.item():.4e}")
    print(f"  - Pearson   : {pearson.item():.4f}")
    print(f"  - MAPE(%)   : {mape.item():.2f}  (calculated on targets > {epsilon_mape:.0e})")

    # --- 6. 分箱详细评估 ---
    print("\n--- MAE, MAPE & Log10 Error by Density Magnitude ---")
    targets_np = all_targets_tensor.numpy() 
    preds_np = all_preds_tensor.numpy()
    
    abs_errors = np.abs(preds_np - targets_np)
    sq_errors = np.square(preds_np - targets_np)
    
    # 物理密度分箱 (与训练时的 Bins 保持一致，或者使用更细的 bins 用于展示)
    # 这里使用训练时的 CLASS_BINS，或者是您上面定义的 report_bins
    # 为了展示效果，我们可以用您原来的 report_bins
    report_bins = [0, 1e-14, 1e-10, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, np.inf]
    
    bin_indices = np.digitize(targets_np, bins=report_bins)
    
    results = []
    total_points = targets_np.size
    
    for i in range(1, len(report_bins)):
        mask = (bin_indices == i)
        points_in_bin = np.sum(mask)
        
        if points_in_bin == 0: continue
            
        mae_in_bin = np.mean(abs_errors[mask])
        rmse_in_bin = np.sqrt(np.mean(sq_errors[mask]))
        
        targets_in_bin = targets_np[mask]
        preds_in_bin = preds_np[mask]
        
        # Safe masks for MAPE/LogError
        safe_threshold = 1e-12
        safe_mask = (targets_in_bin > safe_threshold) & (preds_in_bin > safe_threshold)
        
        # MAPE
        if np.sum(safe_mask) == 0:
            mape_in_bin_str = "N/A"
        else:
            mape_vals = np.abs((preds_in_bin[safe_mask] - targets_in_bin[safe_mask]) / targets_in_bin[safe_mask])
            mape_in_bin = np.mean(mape_vals) * 100
            mape_in_bin_str = f"{mape_in_bin:.2f}"
            
        # MALog10E
        if np.sum(safe_mask) == 0:
            ma_log10_e_str = "N/A"
        else:
            log10_error = np.abs(np.log10(preds_in_bin[safe_mask]) - np.log10(targets_in_bin[safe_mask]))
            ma_log10_e = np.mean(log10_error)
            ma_log10_e_str = f"{ma_log10_e:.3f}"

        percentage = (points_in_bin / total_points) * 100
        
        range_label = f"({report_bins[i-1]:.1e}, {report_bins[i]:.1e}]"
        if report_bins[i-1] == 0:
            range_label = f"[{0}, {report_bins[i]:.1e}]"
            
        results.append({
            "Density Range": range_label,
            "% of Total": f"{percentage:.2f}%",
            "MAE": f"{mae_in_bin:.3e}",
            "RMSE": f"{rmse_in_bin:.3e}",
            "MAPE(%)": mape_in_bin_str,
            "MALog10E": ma_log10_e_str
        })
        
    df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    df = df[["Density Range", "% of Total", "MAE", "RMSE", "MAPE(%)", "MALog10E"]]
    print(df.to_string(index=False))

if __name__ == "__main__":
    evaluate()