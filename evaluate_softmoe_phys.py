import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef

# 导入您的自定义模块 (确保这些文件在同级目录或 Python 路径中)
from dataset_moe_phys import PhysicsMoEDataset, collate_fn_moe_phys
from physics_utils import PhysicsUtils
from train_softmoe_phys_duoka import MoEPhysModule # 导入包含 Lightning 逻辑的类

# =========================================================================
# 1. 评估配置
# =========================================================================

# !!! 关键：将这里替换为您训练出的最好模型的 .ckpt 文件路径 !!!
CHECKPOINT_PATH = Path("checkpoints_hybrid/moe_hybrid_v1/duoka-best-epoch=21-val/mae=0.0035.ckpt")

# --- 数据集参数 (必须与训练时严格一致) ---
RAW_DATA_ROOT = Path("./raw_data_cube") 
CACHE_DIR = Path("./preprocessed_cache")
GRID_RESOLUTION = 64
GRID_BOUNDS = (-10.0, 10.0)
NUM_PE_FREQS = 4
BATCH_SIZE = 8 
VAL_SPLIT_RATIO = 0.1
NUM_WORKERS = 12

# INITIAL_BINS 仅用于初始化 Dataset，以保证数据划分和特征维度一致
INITIAL_BINS = [0, 1e-34, 1e-30, 1e-26, 1e-22, 1e-18, 1e-14, 1e-10, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, np.inf]

def evaluate():
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 加载训练好的混合空间 MoE 模型 ---
    print(f"Loading Learnable MoE Model from: {CHECKPOINT_PATH}")
    try:
        model = MoEPhysModule.load_from_checkpoint(
            checkpoint_path=CHECKPOINT_PATH, 
            map_location=device,
            strict=False # 允许少许不匹配
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please verify that CHECKPOINT_PATH is correct and points to a model trained with MoEPhysModule.")
        return
        
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 打印模型当前学习到的阈值 (非常有助于理解模型到底做了什么)
    with torch.no_grad():
        # 获取路由器当前的 log10 阈值
        log10_thresholds = model.model.router.get_thresholds_log10()
        # 转换为真实的物理线性尺度
        real_thresholds = torch.pow(10.0, log10_thresholds)
        t_vals = [f"{t.item():.1e}" for t in real_thresholds]
        print(f"\n[Model Insights] Final Learned Router Thresholds: {t_vals}\n")

    # --- 2. 准备验证集 ---
    print("Preparing validation dataset...")
    full_dataset = PhysicsMoEDataset(
        cache_dir=CACHE_DIR,
        raw_data_root=RAW_DATA_ROOT,
        grid_res=GRID_RESOLUTION,
        grid_bounds=GRID_BOUNDS,
        num_pe_freqs=NUM_PE_FREQS,
        class_bins=INITIAL_BINS,
    )
    n_total = len(full_dataset)
    n_val = int(n_total * VAL_SPLIT_RATIO)
    n_train = n_total - n_val
    _, val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn_moe_phys)
    print(f"Validation dataset prepared with {len(val_dataset)} samples.")
    
    all_preds = []
    all_targets = []
    
    # 提前生成评估用的 grid_buffer (用于计算 v_ext)
    grid_buffer = PhysicsUtils.get_grid_coords(GRID_RESOLUTION, GRID_BOUNDS, device=device)

    # --- 3. 运行推理 ---
    print("Running predictions on the validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Predicting"):
            atom_grid = batch['atom_grid'].to(device)
            pe_grid = batch['pe_grid'].to(device, torch.float32) if batch['pe_grid'] is not None else None
            atom_coords = batch['atom_coords'].to(device)
            atom_z = batch['atom_z'].to(device)
            target_integral = batch['target_integral'].to(device)
            targets = batch['target_density'].to(device)
            
            # 动态计算所需的物理输入
            v_ext = PhysicsUtils.build_vext(atom_coords, atom_z, grid_buffer)
            z_spike = torch.sum(atom_grid, dim=1, keepdim=True)
            
            # 调用模型前向传播
            # 新模型返回: routing_weights, thresholds_log10, final_density, guide_field_log10
            # 评估时启用物理硬约束 (use_constraint=True)
            _, _, final_density, _ = model(
                atom_grid, pe_grid, v_ext, z_spike, 
                target_integral=target_integral, 
                use_constraint=True
            )
            
            # 提取我们需要评估的预测密度 [B, 1, D, H, W] -> [B, D, H, W]
            preds = final_density.squeeze(1)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # --- 4. 汇总与整体指标计算 ---
    print("Prediction collection finished. Calculating metrics...")
    all_preds_tensor = torch.cat(all_preds, dim=0).flatten()
    all_targets_tensor = torch.cat(all_targets, dim=0).flatten()
    del all_preds, all_targets
    
    mse = MeanSquaredError()(all_preds_tensor, all_targets_tensor)
    mae = MeanAbsoluteError()(all_preds_tensor, all_targets_tensor)
    pearson = PearsonCorrCoef()(all_preds_tensor, all_targets_tensor)
    rmse = torch.sqrt(mse)
    
    print("\n=======================================================")
    print("--- Overall Metrics (Learnable MoE) ---")
    print(f"  - MSE       : {mse.item():.4e}")
    print(f"  - RMSE      : {rmse.item():.4e}")
    print(f"  - MAE       : {mae.item():.4e}")
    print(f"  - Pearson   : {pearson.item():.4f}")
    print("=======================================================")

    # --- 5. 分箱计算 (MALog10E 和 MAPE) ---
    print("\n--- MAE, MAPE & Log10 Error by Density Magnitude ---")
    targets_np = all_targets_tensor.numpy() 
    preds_np = all_preds_tensor.numpy()
    
    abs_errors = np.abs(preds_np - targets_np)
    sq_errors = np.square(preds_np - targets_np)
    
    # 定义用于生成评估报告的分箱边界
    # 这里我们使用一个涵盖大范围对数量级的标准分箱，以便清晰地看出趋势
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
        
        # 定义计算对数和相对误差的“安全掩码” (排除了真空噪声)
        safe_mask = (targets_in_bin > 1e-35) & (preds_in_bin > 1e-35)
        
        # 计算 MAPE (相对百分比误差)
        if np.sum(safe_mask) == 0: 
            mape_in_bin_str = "N/A"
        else:
            mape_vals = np.abs((preds_in_bin[safe_mask] - targets_in_bin[safe_mask]) / targets_in_bin[safe_mask])
            mape_in_bin_str = f"{np.mean(mape_vals) * 100:.2f}"
            
        # 计算 MALog10E (平均绝对数量级误差) - 这是评估 30 级跨度最重要的指标！
        if np.sum(safe_mask) == 0: 
            ma_log10_e_str = "N/A"
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
    print("=======================================================\n")

if __name__ == "__main__":
    evaluate()