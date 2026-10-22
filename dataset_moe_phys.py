"""Cached EDBench dataset for the PI-EDG voxel model."""

import errno
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset_phys import ATOM_MAP, voxelize_structure


def load_sample_ids(path):
    path = Path(path)
    if path.suffix == ".npy":
        values = np.load(path, allow_pickle=False)
        if values.ndim != 1:
            raise ValueError("ID array must be one dimensional")
        ids = [value.decode("ascii") if isinstance(value, bytes) else str(value) for value in values]
    else:
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Sample IDs must be nonempty and unique")
    return ids


def open_cache_readonly(path):
    if hasattr(os, "O_NOATIME"):
        try:
            return os.fdopen(os.open(path, os.O_RDONLY | os.O_NOATIME), "rb")
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.EINVAL, errno.EOPNOTSUPP):
                raise
    return open(path, "rb")


def create_pe_grid(grid_res, num_freqs=4):
    coords = torch.linspace(-1, 1, grid_res)
    gz, gy, gx = torch.meshgrid(coords, coords, coords, indexing="ij")
    grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0)
    freqs = 2.0 ** torch.arange(num_freqs)
    scaled = grid.unsqueeze(-1) * freqs * torch.pi
    pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
    return pe.flatten(-2).permute(0, 4, 1, 2, 3).float()


class PhysicsMoEDataset(Dataset):
    def __init__(self, cache_dir, sample_ids, grid_res=64, grid_bounds=(-10.0, 10.0), num_pe_freqs=4):
        self.cache_dir = Path(cache_dir)
        self.grid_res = int(grid_res)
        self.grid_bounds = tuple(grid_bounds)
        self.indices = [value.decode("ascii") if isinstance(value, bytes) else str(value) for value in sample_ids]
        if not self.indices or len(self.indices) != len(set(self.indices)):
            raise ValueError("Sample IDs must be nonempty and unique")
        self.atom_map = ATOM_MAP
        self.num_atom_channels = len(ATOM_MAP)
        self.pe_grid = create_pe_grid(self.grid_res, num_pe_freqs)[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_id = self.indices[idx]
        path = self.cache_dir / f"{sample_id}.npz"
        try:
            with open_cache_readonly(path) as source, np.load(source, allow_pickle=False) as data:
                atom_coords = np.asarray(data["atom_coords"], dtype=np.float32)
                atom_z = np.asarray(data["atom_z_numbers"], dtype=np.int16)
                target_np = np.asarray(data["target_density"], dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(f"Cannot load sample {sample_id} from {path}") from exc
        if target_np.shape == (self.grid_res, self.grid_res, self.grid_res):
            target_np = target_np[None]
        if target_np.shape != (1, self.grid_res, self.grid_res, self.grid_res):
            raise ValueError(f"Invalid target shape for sample {sample_id}: {target_np.shape}")
        if atom_coords.shape != (len(atom_z), 3) or not np.isfinite(atom_coords).all():
            raise ValueError(f"Invalid atom coordinates for sample {sample_id}")

        atom_grid = voxelize_structure(atom_coords, atom_z, self.grid_res, self.grid_bounds, ATOM_MAP)
        return {
            "sample_id": sample_id,
            "atom_grid": atom_grid,
            "pe_grid": self.pe_grid,
            "target_density": torch.from_numpy(target_np),
            "target_electrons": torch.tensor(float(atom_z.sum()), dtype=torch.float32),
            "atom_coords": torch.from_numpy(atom_coords),
            "atom_z": torch.from_numpy(atom_z.astype(np.int64)),
        }


def collate_fn_moe_phys(batch):
    max_atoms = max(item["atom_coords"].shape[0] for item in batch)
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "atom_grid": torch.stack([item["atom_grid"] for item in batch]),
        "pe_grid": batch[0]["pe_grid"],
        "target_density": torch.stack([item["target_density"] for item in batch]),
        "target_electrons": torch.stack([item["target_electrons"] for item in batch]),
        "atom_coords": torch.stack([F.pad(item["atom_coords"], (0, 0, 0, max_atoms - len(item["atom_z"]))) for item in batch]),
        "atom_z": torch.stack([F.pad(item["atom_z"], (0, max_atoms - len(item["atom_z"]))) for item in batch]),
    }
