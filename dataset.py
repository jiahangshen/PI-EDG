"""Cached EDBench dataset and geometry voxelization for PI-EDG."""

import errno
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ATOM_MAP = {
    1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 14: 6,
    15: 7, 16: 8, 17: 9, 32: 10, 34: 11, 35: 12,
}


def voxelize_structure(
    atom_coords: np.ndarray,
    atom_z_numbers: np.ndarray,
    grid_res: int,
    grid_bounds: tuple,
    atom_map: dict = ATOM_MAP,
    sigma: float = 0.1,
) -> torch.Tensor:
    grid_min, grid_max = grid_bounds
    num_channels = len(atom_map)
    voxel_grid = torch.zeros((num_channels, grid_res, grid_res, grid_res), dtype=torch.float32)

    voxel_size = (grid_max - grid_min) / grid_res
    if voxel_size <= 0:
        raise ValueError("grid_bounds must define a positive box")

    coords_1d = torch.linspace(grid_min + voxel_size / 2, grid_max - voxel_size / 2, grid_res)
    gaussian_radius = max(1, int(np.ceil((4.0 * sigma) / voxel_size)))
    denom = 2.0 * sigma**2

    for atom_pos_np, atom_z in zip(atom_coords, atom_z_numbers):
        atom_z = int(atom_z)
        if atom_z not in atom_map:
            raise ValueError(f"Unsupported atomic number: {atom_z}")

        channel_idx = atom_map[atom_z]
        atom_pos = torch.as_tensor(atom_pos_np, dtype=torch.float32)
        center_idx = torch.round((atom_pos - grid_min) / voxel_size - 0.5).to(torch.long)
        x0 = max(0, int(center_idx[0]) - gaussian_radius)
        x1 = min(grid_res, int(center_idx[0]) + gaussian_radius + 1)
        y0 = max(0, int(center_idx[1]) - gaussian_radius)
        y1 = min(grid_res, int(center_idx[1]) + gaussian_radius + 1)
        z0 = max(0, int(center_idx[2]) - gaussian_radius)
        z1 = min(grid_res, int(center_idx[2]) + gaussian_radius + 1)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            continue

        local_z, local_y, local_x = coords_1d[z0:z1], coords_1d[y0:y1], coords_1d[x0:x1]
        grid_z, grid_y, grid_x = torch.meshgrid(local_z, local_y, local_x, indexing="ij")
        dx = grid_x - atom_pos[0]
        dy = grid_y - atom_pos[1]
        dz = grid_z - atom_pos[2]
        gaussian_blob = torch.exp(-(dx * dx + dy * dy + dz * dz) / denom)

        current = voxel_grid[channel_idx, z0:z1, y0:y1, x0:x1]
        voxel_grid[channel_idx, z0:z1, y0:y1, x0:x1] = torch.maximum(current, gaussian_blob)

    return voxel_grid


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
