import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import errno
import os
import numpy as np

from dataset_phys import voxelize_structure, voxelize_raw_density_from_points


def open_cache_readonly(path):
    """Open a cache file without updating access time where supported."""
    if hasattr(os, 'O_NOATIME'):
        try:
            return os.fdopen(os.open(path, os.O_RDONLY | os.O_NOATIME), 'rb')
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.EINVAL, errno.EOPNOTSUPP):
                raise
    return open(path, 'rb')


def classify_density(density_grid, bins):
    bins_tensor = torch.tensor(bins, device=density_grid.device)
    comparison = density_grid.unsqueeze(-1) > bins_tensor
    class_grid = torch.sum(comparison, dim=-1).squeeze(0)
    return class_grid.long()


def create_pe_grid(grid_res, num_freqs=4):
    coords = torch.linspace(-1, 1, grid_res)
    gz, gy, gx = torch.meshgrid(coords, coords, coords, indexing='ij')
    grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0)
    freqs = 2.0 ** torch.arange(num_freqs)
    scaled = grid.unsqueeze(-1) * freqs * torch.pi
    pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
    return pe.flatten(-2).permute(0, 4, 1, 2, 3).float()


class PhysicsMoEDataset(Dataset):
    def __init__(
        self,
        cache_dir,
        raw_data_root,
        grid_res=64,
        grid_bounds=(-10.0, 10.0),
        num_pe_freqs=4,
        class_bins=None,
        sad_dir=None,
        sample_ids=None,
        strict=False,
    ):
        self.cache_dir = Path(cache_dir)
        self.raw_data_root = Path(raw_data_root)
        self.sad_dir = None
        self.grid_res = grid_res
        self.grid_bounds = grid_bounds
        self.class_bins = class_bins
        self.strict = strict
        self.indices = (
            sample_ids if isinstance(sample_ids, np.ndarray)
            else [str(sample_id) for sample_id in sample_ids]
            if sample_ids is not None
            else sorted(f.stem for f in self.cache_dir.glob("*.npz"))
        )

        self.atom_map = {1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 14: 6, 15: 7, 16: 8, 17: 9, 32: 10, 34: 11, 35: 12}
        self.num_atom_channels = len(self.atom_map)
        self.pe_grid = self._create_pe_grid(num_pe_freqs)

    def _create_pe_grid(self, num_freqs):
        return create_pe_grid(self.grid_res, num_freqs)

    def __len__(self):
        return len(self.indices)

    def _load_sad_grid(self, sample_id):
        return None

    def __getitem__(self, idx):
        try:
            sample_key = self.indices[idx]
            sample_id = sample_key.decode('ascii') if isinstance(sample_key, bytes) else str(sample_key)
            with open_cache_readonly(self.cache_dir / f"{sample_id}.npz") as source, np.load(source) as data:
                atom_coords = data['atom_coords'].astype(np.float32, copy=False)
                atom_z = data['atom_z_numbers'].astype(np.int16, copy=False)

                if 'target_density' in data.files:
                    target_density_np = data['target_density'].astype(np.float32, copy=False)
                    if target_density_np.ndim == 3:
                        target_density = torch.from_numpy(target_density_np).unsqueeze(0)
                    else:
                        target_density = torch.from_numpy(target_density_np)
                else:
                    dens_coords = data['density_coords'].astype(np.float32, copy=False)
                    dens_vals = data['density_values'].astype(np.float32, copy=False)
                    centroid = atom_coords.mean(axis=0)
                    atom_coords = atom_coords - centroid
                    dens_coords = dens_coords - centroid
                    target_density = voxelize_raw_density_from_points(dens_coords, dens_vals, self.grid_res, self.grid_bounds)

            atom_grid = voxelize_structure(atom_coords, atom_z, self.grid_res, self.grid_bounds, self.atom_map)
            target_labels = classify_density(target_density, self.class_bins)
            target_integral = torch.sum(target_density)
            target_electrons = torch.tensor(float(atom_z.sum()), dtype=torch.float32)
            sad_grid = self._load_sad_grid(sample_id)

            return {
                'sample_id': sample_id,
                'atom_grid': atom_grid,
                'pe_grid': self.pe_grid.squeeze(0),
                'sad_grid': sad_grid,
                'target_density': target_density,
                'target_labels': target_labels,
                'target_integral': target_integral,
                'target_electrons': target_electrons,
                'atom_coords': torch.from_numpy(atom_coords).float(),
                'atom_z': torch.from_numpy(atom_z).long(),
            }
        except Exception as exc:
            if self.strict:
                raise RuntimeError(
                    f"Cannot load sample {sample_id} from {self.cache_dir}"
                ) from exc
            return None


def collate_fn_moe_phys(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    max_atoms = max([b['atom_coords'].shape[0] for b in batch])
    batched_coords, batched_z = [], []

    for b in batch:
        diff = max_atoms - b['atom_coords'].shape[0]
        batched_coords.append(F.pad(b['atom_coords'], (0, 0, 0, diff)))
        batched_z.append(F.pad(b['atom_z'], (0, diff)))

    sad_grids = None
    if all(b.get('sad_grid') is not None for b in batch):
        sad_grids = torch.stack([b['sad_grid'] for b in batch])

    return {
        'sample_id': [b['sample_id'] for b in batch],
        'atom_grid': torch.stack([b['atom_grid'] for b in batch]),
        'pe_grid': batch[0]['pe_grid'],
        'sad_grid': sad_grids,
        'target_density': torch.stack([b['target_density'] for b in batch]),
        'target_labels': torch.stack([b['target_labels'] for b in batch]),
        'target_integral': torch.stack([b['target_integral'] for b in batch]),
        'target_electrons': torch.stack([b['target_electrons'] for b in batch]),
        'atom_coords': torch.stack(batched_coords),
        'atom_z': torch.stack(batched_z),
    }
