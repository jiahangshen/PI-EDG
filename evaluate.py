"""Evaluate a PI-EDG checkpoint on a frozen list of cached molecule IDs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dataset_moe_phys import PhysicsMoEDataset, load_sample_ids
from model_softmoe_phys import normalize_density_to_electrons
from physics_utils import PhysicsUtils
from train import load_model_weights


def metrics(pred, target):
    diff = pred - target
    x = pred.ravel().astype(np.float64)
    y = target.ravel().astype(np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])
    return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff * diff))), pearson


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--sample-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--voxel-volume-bohr3", type=float, default=0.064)
    args = parser.parse_args()
    ids = load_sample_ids(args.sample_ids)
    device = torch.device(args.device)
    module = load_model_weights(args.checkpoint, device)
    grid_res = int(module.hparams.grid_res)
    grid_bounds = tuple(module.hparams.grid_bounds)
    dataset = PhysicsMoEDataset(
        cache_dir=args.cache_dir,
        grid_res=grid_res, grid_bounds=grid_bounds,
        sample_ids=ids, num_pe_freqs=module.hparams.num_pe_freqs,
    )
    grid = PhysicsUtils.get_grid_coords(grid_res, grid_bounds, device=device)
    rows = []
    with torch.inference_mode():
        for index, sample_id in enumerate(ids):
            item = dataset[index]
            atom_grid = item["atom_grid"][None].to(device)
            pe_grid = item["pe_grid"].to(device)
            atom_coords = item["atom_coords"][None].to(device)
            atom_z = item["atom_z"][None].to(device)
            v_ext = PhysicsUtils.build_vext(atom_coords, atom_z, grid)
            _, _, raw, _ = module(
                atom_grid, pe_grid, v_ext, atom_grid.sum(dim=1, keepdim=True),
            )
            nominal = normalize_density_to_electrons(
                raw, item["target_electrons"][None].to(device),
                args.voxel_volume_bohr3,
            )
            target = item["target_density"][0].numpy()
            raw_np = raw[0, 0].float().cpu().numpy()
            nominal_np = nominal[0, 0].float().cpu().numpy()
            raw_mae, raw_rmse, raw_pearson = metrics(raw_np, target)
            norm_mae, norm_rmse, norm_pearson = metrics(nominal_np, target)
            rows.append({
                "sample_id": sample_id,
                "raw_mae": raw_mae, "raw_rmse": raw_rmse, "raw_pearson": raw_pearson,
                "normalized_mae": norm_mae, "normalized_rmse": norm_rmse,
                "normalized_pearson": norm_pearson,
                "raw_electron_error": float(abs(raw_np.sum() * args.voxel_volume_bohr3 - item["target_electrons"].item())),
                "normalized_electron_error": float(abs(nominal_np.sum() * args.voxel_volume_bohr3 - item["target_electrons"].item())),
            })
            if (index + 1) % 100 == 0:
                print(f"Evaluated {index + 1}/{len(ids)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_molecule.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"n_molecules": len(rows), "metric_aggregation": "mean of per-molecule values",
               "reference": "raw cached DFT grid", "prediction_normalization": "clip then scale to sum(Z)"}
    for key in rows[0]:
        if key != "sample_id":
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            summary[key] = float(finite.mean()) if finite.size else None
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
