import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from train_moe import MixtureOfExpertsModel
from dataset_classifier import VoxelizedMoEDataset, collate_fn_skip_error


CHECKPOINT_PATH = Path("./checkpoints_moe/best-moe-model-epoch=43-iou=val\epoch_iou_macro=0.0000.ckpt") # <-- 关键：更新为你最好的模型路径

RAW_DATA_ROOT = Path("./raw_data_cube") 
CACHE_DIR = Path("./preprocessed_cache")
GRID_RESOLUTION = 64
GRID_BOUNDS = (-10.0, 10.0)
NUM_PE_FREQS = 4
BATCH_SIZE = 8 
VAL_SPLIT_RATIO = 0.1
NUM_WORKERS = 12

CLASS_BINS = [0, 1e-14, 1e-10, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, np.inf]
NUM_CLASSES = len(CLASS_BINS) 


def evaluate():
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {CHECKPOINT_PATH}")
    model = MixtureOfExpertsModel.load_from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH, 
        map_location=device,
        strict=False
    )
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    full_dataset = VoxelizedMoEDataset(
        cache_dir=CACHE_DIR,
        raw_data_root_for_scan=RAW_DATA_ROOT,
        grid_res=GRID_RESOLUTION,
        grid_bounds=GRID_BOUNDS,
        num_pe_freqs=NUM_PE_FREQS,
        class_bins=CLASS_BINS
    )
    n_total = len(full_dataset)
    n_val = int(n_total * VAL_SPLIT_RATIO)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn_skip_error)
    print(f"Validation dataset prepared with {len(val_dataset)} samples.")
    all_preds = []
    all_targets = []

    print("Running predictions on the validation set (on GPU)...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            atom_grids, pe_grids, _, target_densities = batch
            if atom_grids.nelement() == 0: continue
            
            atom_grids = atom_grids.to(device)
            pe_grids = pe_grids.to(device, torch.float32)
            target_densities = target_densities.to(device)
            
            logits, expert_outputs = model(atom_grids, pe_grids)
            preds_labels = torch.argmax(logits, dim=1)
            chosen_expert_preds = torch.gather(expert_outputs, 1, preds_labels.unsqueeze(1)).squeeze(1)

            all_preds.append(chosen_expert_preds.cpu())
            all_targets.append(target_densities.cpu())

        all_preds_tensor = torch.cat(all_preds, dim=0).flatten()
        all_targets_tensor = torch.cat(all_targets, dim=0).flatten()
        print("Prediction collection finished")
        del all_preds
        del all_targets

        mse_metric = MeanSquaredError()
        mae_metric = MeanAbsoluteError()
        pearson_metric = PearsonCorrCoef()
        spearman_metric = SpearmanCorrCoef()

        mse = mse_metric(all_preds_tensor, all_targets_tensor)
        mae = mae_metric(all_preds_tensor, all_targets_tensor)
        pearson = pearson_metric(all_preds_tensor, all_targets_tensor)
        spearman = spearman_metric(all_preds_tensor, all_targets_tensor)
        rmse = torch.sqrt(mse)

        print(f"  - MSE       : {mse.item():.4e}")
        print(f"  - RMSE      : {rmse.item():.4e}")
        print(f"  - MAE       : {mae.item():.4e}")
        print(f"  - Pearson   : {pearson.item():.4f}")
        print(f"  - Spearman  : {spearman.item():.4f}")

        print("\n--- MAE & RMSE by Density Magnitude ---")
        print("Moving data to CPU for magnitude analysis...")
        targets_np = all_targets_tensor.cpu().numpy() 
        preds_np = all_preds_tensor.cpu().numpy()
    
    
    abs_errors = np.abs(preds_np - targets_np)
    sq_errors = np.square(preds_np - targets_np)
    

    report_bins = [0, 1e-10, 1e-7, 1e-5, 1e-3, 1e-2, np.inf]
    bin_indices = np.digitize(targets_np, bins=report_bins)
    
    results = []
    total_points = targets_np.size
    
    for i in range(1, len(report_bins)):
        mask = (bin_indices == i)
        points_in_bin = np.sum(mask)
        
        if points_in_bin == 0: continue
            
        mae_in_bin = np.mean(abs_errors[mask])
        rmse_in_bin = np.sqrt(np.mean(sq_errors[mask]))
        
        percentage = (points_in_bin / total_points) * 100
        
        range_label = f"({report_bins[i-1]:.1e}, {report_bins[i]:.1e}]"
        if report_bins[i-1] == 0:
            range_label = f"[{0}, {report_bins[i]:.1e}]"
            
        results.append({
            "Density Range": range_label,
            "Points Count": f"{points_in_bin:,}",
            "% of Total": f"{percentage:.4f}%",
            "MAE": f"{mae_in_bin:.4e}",
            "RMSE": f"{rmse_in_bin:.4e}"
        })
        
    df = pd.DataFrame(results)
    print(df.to_string(index=False))


if __name__ == "__main__":
    from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, PearsonCorrCoef, SpearmanCorrCoef
    evaluate()