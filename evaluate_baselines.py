import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef

# 导入基线模型的定义
from train_baselines import NaiveBaselineModule
from dataset_moe_phys import PhysicsMoEDataset, collate_fn_moe_phys

# =========================================================================
# 1. 评估配置
# =========================================================================
# !!! 关键：在这里填入您训练出来的 Baseline 的 checkpoint 路径 !!!
CHECKPOINT_PATH = Path("checkpoints_baselines/run_1_classic_unet/best-epoch=03-val/mae=0.0072.ckpt") 

# 数据集参数 (必须与训练时一致)
RAW_DATA_ROOT = Path("./raw_data_cube") 
CACHE_DIR = Path("./preprocessed_cache")
GRID_RESOLUTION = 64
GRID_BOUNDS = (-10.0, 10.0)
NUM_PE_FREQS = 4
BATCH_SIZE = 8 
VAL_SPLIT_RATIO = 0.1
NUM_WORKERS = 12

# Bins 仅用于数据集初始化，评估时不用它
CLASS_BINS = [0, 1e-34, 1e-30, 1e-26, 1e-22, 1e-18, 1e-14, 1e-10, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, np.inf]

def evaluate():
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 加载基线模型 ---
    print(f"Loading Baseline Model from: {CHECKPOINT_PATH}")
    # NaiveBaselineModule 会自动从 ckpt 中读取它原来是 swin 还是 unet
    model = NaiveBaselineModule.load_from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH, 
        map_location=device,
        strict=False
    )
    model.to(device)
    model.eval()
    print(f"Model ({model.model_type}) loaded successfully.")

    # --- 2. 准备验证集 ---
    print("Preparing validation dataset...")
    full_dataset = PhysicsMoEDataset(
        cache_dir=CACHE_DIR,
        raw_data_root=RAW_DATA_ROOT,
        grid_res=GRID_RESOLUTION,
        grid_bounds=GRID_BOUNDS,
        num_pe_freqs=NUM_PE_FREQS,
        class_bins=CLASS_BINS,
    )
    n_total = len(full_dataset)
    n_val = int(n_total * VAL_SPLIT_RATIO)
    n_train = n_total - n_val
    _, val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn_moe_phys)
    
    all_preds = []
    all_targets = []

    # --- 3. 运行推理 ---
    print("Running predictions on the validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Predicting"):
            atom_grid = batch['atom_grid'].to(device)
            pe_grid = batch['pe_grid']
            targets = batch['target_density'].to(device)
            
            # 构建输入 (无物理先验)
            if pe_grid is not None:
                pe_grid = pe_grid.to(device, torch.float32)
                if model.model_type == "classic_unet":
                    preds = model.model(atom_grid, pe_grid)
                else:
                    x = torch.cat([atom_grid, pe_grid], dim=1)
                    preds = model.model(x)
            else:
                preds = model.model(atom_grid)
                
            preds = torch.nn.functional.softplus(preds).squeeze(1)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # --- 4. 汇总与整体指标计算 ---
    print("Calculating metrics...")
    all_preds_tensor = torch.cat(all_preds, dim=0).flatten()
    all_targets_tensor = torch.cat(all_targets, dim=0).flatten()
    del all_preds, all_targets
    
    mse = MeanSquaredError()(all_preds_tensor, all_targets_tensor)
    mae = MeanAbsoluteError()(all_preds_tensor, all_targets_tensor)
    pearson = PearsonCorrCoef()(all_preds_tensor, all_targets_tensor)
    rmse = torch.sqrt(mse)
    
    print("\n--- Overall Metrics (Baseline) ---")
    print(f"  - MSE       : {mse.item():.4e}")
    print(f"  - RMSE      : {rmse.item():.4e}")
    print(f"  - MAE       : {mae.item():.4e}")
    print(f"  - Pearson   : {pearson.item():.4f}")

    # --- 5. 核心：分箱计算 MALog10E ---
    print("\n--- MAE, MAPE & Log10 Error by Density Magnitude ---")
    targets_np = all_targets_tensor.numpy() 
    preds_np = all_preds_tensor.numpy()
    abs_errors = np.abs(preds_np - targets_np)
    sq_errors = np.square(preds_np - targets_np)
    
    report_bins = [0, 1e-34, 1e-10, 1e-7, 1e-5, 1e-3, 1e-2, np.inf]
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
        
        safe_mask = (targets_in_bin > 1e-35) & (preds_in_bin > 1e-35)
        
        # MAPE
        if np.sum(safe_mask) == 0: mape_in_bin_str = "N/A"
        else:
            mape_vals = np.abs((preds_in_bin[safe_mask] - targets_in_bin[safe_mask]) / targets_in_bin[safe_mask])
            mape_in_bin_str = f"{np.mean(mape_vals) * 100:.2f}"
            
        # MALog10E (关键指标)
        if np.sum(safe_mask) == 0: ma_log10_e_str = "N/A"
        else:
            log10_error = np.abs(np.log10(preds_in_bin[safe_mask]) - np.log10(targets_in_bin[safe_mask]))
            ma_log10_e_str = f"{np.mean(log10_error):.3f}"

        percentage = (points_in_bin / total_points) * 100
        range_label = f"({report_bins[i-1]:.1e}, {report_bins[i]:.1e}]" if report_bins[i-1] != 0 else f"[{0}, {report_bins[i]:.1e}]"
            
        results.append({
            "Density Range": range_label,
            "% of Total": f"{percentage:.2f}%",
            "MAE": f"{mae_in_bin:.3e}",
            "RMSE": f"{rmse_in_bin:.3e}",
            "MAPE(%)": mape_in_bin_str,
            "MALog10E": ma_log10_e_str
        })
        
    df = pd.DataFrame(results)
    df = df[["Density Range", "% of Total", "MAE", "RMSE", "MAPE(%)", "MALog10E"]]
    print(df.to_string(index=False))

if __name__ == "__main__":
    evaluate()