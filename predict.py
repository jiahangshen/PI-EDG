"""Generate a 64^3 PI-EDG density grid from one neutral molecular geometry."""

import argparse
from pathlib import Path

import numpy as np
import torch
from ase.io import read as ase_read

from dataset_moe_phys import create_pe_grid
from dataset_phys import voxelize_structure
from model_softmoe_phys import normalize_density_to_electrons
from physics_utils import PhysicsUtils
from train_softmoe_phys_duoka import MoEPhysModule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True, help="SDF or XYZ geometry")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--voxel-volume-bohr3", type=float, default=0.064,
                        help="Integration volume used by the EDBench source-grid protocol")
    args = parser.parse_args()

    device = torch.device(args.device)
    module = MoEPhysModule.load_from_checkpoint(str(args.checkpoint), map_location=device)
    module.eval().to(device)
    module.model.use_hard_routing = False
    grid_res = int(module.hparams.grid_res)
    grid_bounds = tuple(module.hparams.grid_bounds)
    atom_map = {1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5,
                14: 6, 15: 7, 16: 8, 17: 9, 32: 10, 34: 11, 35: 12}

    atoms = ase_read(str(args.structure))
    atom_coords = atoms.get_positions().astype(np.float32)
    atomic_numbers = atoms.get_atomic_numbers().astype(np.int16)
    unsupported = sorted(set(map(int, atomic_numbers)) - set(atom_map))
    if unsupported:
        raise ValueError(f"Unsupported atomic numbers: {unsupported}")
    centroid = atom_coords.mean(axis=0)
    centered = atom_coords - centroid
    atom_grid = voxelize_structure(centered, atomic_numbers, grid_res, grid_bounds, atom_map)[None].to(device)
    pe_grid = create_pe_grid(grid_res, 4)[0].to(device)
    coords = torch.from_numpy(centered)[None].to(device)
    charges = torch.from_numpy(atomic_numbers.astype(np.int64))[None].to(device)
    grid = PhysicsUtils.get_grid_coords(grid_res, grid_bounds, device=device)
    with torch.inference_mode():
        v_ext = PhysicsUtils.build_vext(coords, charges, grid)
        z_spike = atom_grid.sum(dim=1, keepdim=True)
        _, _, raw, _ = module(
            atom_grid, pe_grid, v_ext, z_spike,
            target_integral=None, use_constraint=False,
        )
        normalized = normalize_density_to_electrons(
            raw, charges.sum(dim=1).float(), args.voxel_volume_bohr3
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        raw_density_zyx=raw[0, 0].float().cpu().numpy(),
        normalized_density_zyx=normalized[0, 0].float().cpu().numpy(),
        atom_coords_angstrom=centered,
        original_centroid_angstrom=centroid,
        atomic_numbers=atomic_numbers,
        grid_bounds_angstrom=np.asarray(grid_bounds),
        voxel_volume_bohr3=np.float32(args.voxel_volume_bohr3),
    )
    print(f"Saved raw and neutral-electron-normalized grids to {args.output}")


if __name__ == "__main__":
    main()
