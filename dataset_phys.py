import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

try:
    from ase.io import read as ase_read
except Exception:
    ase_read = None


ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "Br": 35,
    "I": 53,
}


def read_xyz_atomic_numbers(xyz_path: Path):
    with xyz_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    num_atoms = int(lines[0])
    atom_lines = lines[2 : 2 + num_atoms]
    atomic_numbers = []
    for line in atom_lines:
        symbol = line.split()[0]
        if symbol not in ATOMIC_NUMBERS:
            raise KeyError(f"Unsupported element symbol in {xyz_path}: {symbol}")
        atomic_numbers.append(ATOMIC_NUMBERS[symbol])
    return atomic_numbers

def voxelize_structure(
    atom_coords: np.ndarray,
    atom_z_numbers: np.ndarray,
    grid_res: int,
    grid_bounds: tuple,
    atom_map: dict,
    sigma: float = 0.1
) -> torch.Tensor:
    grid_min, grid_max = grid_bounds
    num_channels = len(atom_map)
    voxel_grid = torch.zeros((num_channels, grid_res, grid_res, grid_res), dtype=torch.float32)

    voxel_size = (grid_max - grid_min) / grid_res
    if voxel_size <= 0:
        return voxel_grid

    coords_1d = torch.linspace(grid_min + voxel_size / 2, grid_max - voxel_size / 2, grid_res)
    gaussian_radius = max(1, int(np.ceil((4.0 * sigma) / voxel_size)))
    denom = 2.0 * (sigma ** 2)

    for i in range(atom_coords.shape[0]):
        atom_z = int(atom_z_numbers[i])
        if atom_z not in atom_map:
            continue

        channel_idx = atom_map[atom_z]
        atom_pos = torch.from_numpy(atom_coords[i]).float()

        center_idx = torch.round((atom_pos - grid_min) / voxel_size - 0.5).to(torch.long)
        x0 = max(0, int(center_idx[0]) - gaussian_radius)
        x1 = min(grid_res, int(center_idx[0]) + gaussian_radius + 1)
        y0 = max(0, int(center_idx[1]) - gaussian_radius)
        y1 = min(grid_res, int(center_idx[1]) + gaussian_radius + 1)
        z0 = max(0, int(center_idx[2]) - gaussian_radius)
        z1 = min(grid_res, int(center_idx[2]) + gaussian_radius + 1)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            continue

        local_x = coords_1d[x0:x1]
        local_y = coords_1d[y0:y1]
        local_z = coords_1d[z0:z1]
        grid_z, grid_y, grid_x = torch.meshgrid(local_z, local_y, local_x, indexing="ij")
        dx = grid_x - atom_pos[0]
        dy = grid_y - atom_pos[1]
        dz = grid_z - atom_pos[2]
        gaussian_blob = torch.exp(-(dx * dx + dy * dy + dz * dz) / denom)

        current = voxel_grid[channel_idx, z0:z1, y0:y1, x0:x1]
        voxel_grid[channel_idx, z0:z1, y0:y1, x0:x1] = torch.maximum(current, gaussian_blob)

    return voxel_grid

def voxelize_raw_density_from_points(
    aligned_density_coords: np.ndarray,
    density_values: np.ndarray,
    grid_res: int,
    grid_bounds: tuple
) -> torch.Tensor:
    grid_min, grid_max = grid_bounds
    target_voxel_grid = torch.zeros((grid_res, grid_res, grid_res), dtype=torch.float32)
    target_voxel_size = (grid_max - grid_min) / grid_res

    if target_voxel_size <= 0:
        return target_voxel_grid.unsqueeze(0)

    target_all_indices = ((aligned_density_coords - grid_min) / target_voxel_size).astype(int)

    valid_mask = np.all((target_all_indices >= 0) & (target_all_indices < grid_res), axis=1)
    valid_target_indices = target_all_indices[valid_mask]
    valid_density_values = density_values[valid_mask]

    if len(valid_target_indices) > 0:
        vz, vy, vx = valid_target_indices[:, 2], valid_target_indices[:, 1], valid_target_indices[:, 0]
        target_voxel_grid.index_put_(
            (torch.from_numpy(vz), torch.from_numpy(vy), torch.from_numpy(vx)),
            torch.from_numpy(valid_density_values).float(),
            accumulate=True
        )

    return target_voxel_grid.unsqueeze(0)

class PhysicsDataset(Dataset):
    def __init__(self,
                 cache_dir: Path,
                 raw_data_root_for_scan: Path,
                 grid_res: int = 64,
                 grid_bounds: tuple = (-10.0, 10.0)):
        self.cache_dir = cache_dir
        self.grid_res = grid_res
        self.grid_bounds = grid_bounds

        print(f"正在扫描缓存目录 {cache_dir}...")
        self.indices = sorted([f.stem for f in self.cache_dir.glob("*.npz")])
        if not self.indices:
            raise FileNotFoundError(f"在 {cache_dir} 中找不到 .npz 文件。")
        print(f"找到 {len(self.indices)} 个缓存文件。")

        print("正在扫描原子类型...")
        self.atom_map = self._discover_elements(raw_data_root_for_scan, self.indices)
        self.num_atom_channels = len(self.atom_map)
        print(f"原子映射: {self.atom_map}")

    def _discover_elements(self, raw_data_root, indices_to_scan):
        unique_z_values = set()
        num_scan_samples = len(indices_to_scan)
        for i in tqdm(range(num_scan_samples), desc="Scanning Elements"):
            index = indices_to_scan[i]
            sample_dir = raw_data_root / index
            try:
                if sample_dir.exists():
                    structure_path = next(sample_dir.glob("*.sdf"), None)
                    structure_format = "sdf"
                    if structure_path is None:
                        structure_path = next(sample_dir.glob("*.xyz"), None)
                        structure_format = None
                    if structure_path is not None:
                        if structure_path.suffix == ".xyz" and ase_read is None:
                            unique_z_values.update(read_xyz_atomic_numbers(structure_path))
                        else:
                            atoms = ase_read(str(structure_path), format=structure_format)
                            unique_z_values.update(atoms.get_atomic_numbers())
                        continue

                cache_data = np.load(self.cache_dir / f"{index}.npz")
                unique_z_values.update(cache_data["atom_z_numbers"].tolist())
            except Exception:
                pass

        sorted_unique_z = sorted(list(unique_z_values))
        return {z: i for i, z in enumerate(sorted_unique_z)}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        index = self.indices[idx]
        npz_file_path = self.cache_dir / f"{index}.npz"

        try:
            cache_data = np.load(npz_file_path)
            atom_coords = cache_data['atom_coords']
            atom_z_numbers = cache_data['atom_z_numbers']
            density_coords = cache_data['density_coords']
            density_values = cache_data['density_values']

            centroid = np.mean(atom_coords, axis=0)
            centered_atom_coords = atom_coords - centroid
            centered_density_coords = density_coords - centroid

            total_electrons = float(np.sum(atom_z_numbers))

            atom_grid = voxelize_structure(
                centered_atom_coords, atom_z_numbers,
                self.grid_res, self.grid_bounds, self.atom_map
            )

            target_density = voxelize_raw_density_from_points(
                centered_density_coords, density_values,
                self.grid_res, self.grid_bounds
            )
            target_integral = torch.sum(target_density)
            return {
                'atom_grid': atom_grid,
                'atom_coords': torch.from_numpy(centered_atom_coords).float(),
                'atom_z': torch.from_numpy(atom_z_numbers).float(),
                'target_density': target_density,
                'total_electrons': torch.tensor(total_electrons, dtype=torch.float32),
                'target_integral': target_integral,
                'index': index
            }

        except Exception as e:
            print(f"\nError loading {index}: {e}")
            return None

def collate_fn_physics(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    max_atoms = max([b['atom_coords'].shape[0] for b in batch])

    batched_atom_grids = []
    batched_atom_coords = []
    batched_atom_z = []
    batched_target_densities = []
    batched_total_electrons = []
    batched_target_integral = []
    for b in batch:
        batched_atom_grids.append(b['atom_grid'])
        batched_target_densities.append(b['target_density'])
        batched_total_electrons.append(b['total_electrons'])
        batched_target_integral.append(b['target_integral'])
        coords = b['atom_coords']
        z = b['atom_z']
        n_atoms = coords.shape[0]
        diff = max_atoms - n_atoms

        if diff > 0:
            coords_padded = F.pad(coords, (0, 0, 0, diff), value=0.0)
            z_padded = F.pad(z, (0, diff), value=0.0)
        else:
            coords_padded = coords
            z_padded = z

        batched_atom_coords.append(coords_padded)
        batched_atom_z.append(z_padded)

    return {
        'atom_grid': torch.stack(batched_atom_grids),
        'atom_coords': torch.stack(batched_atom_coords),
        'atom_z': torch.stack(batched_atom_z),
        'target_density': torch.stack(batched_target_densities),
        'total_electrons': torch.stack(batched_total_electrons),
        'target_integral': torch.stack(batched_target_integral)
    }
