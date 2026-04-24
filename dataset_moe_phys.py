import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
from ase.io import read as ase_read
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

# =========================================================================
# 1. 辅助函数
# =========================================================================

def voxelize_structure(atom_coords, atom_z_numbers, grid_res, grid_bounds, atom_map, sigma=0.2):
    """生成原子高斯场 (Input)"""
    grid_min, grid_max = grid_bounds
    num_channels = len(atom_map)
    voxel_grid = torch.zeros((num_channels, grid_res, grid_res, grid_res), dtype=torch.float32)
    voxel_size = (grid_max - grid_min) / grid_res
    
    coords_1d = torch.linspace(grid_min + voxel_size/2, grid_max - voxel_size/2, grid_res)
    grid_z, grid_y, grid_x = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    voxel_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)

    for i in range(atom_coords.shape[0]):
        atom_z = int(atom_z_numbers[i])
        if atom_z in atom_map:
            channel_idx = atom_map[atom_z]
            atom_pos = torch.from_numpy(atom_coords[i]).float()
            distances_sq = torch.sum((voxel_coords - atom_pos) ** 2, dim=-1)
            gaussian_blob = torch.exp(-distances_sq / (2 * (sigma ** 2)))
            voxel_grid[channel_idx] = torch.maximum(voxel_grid[channel_idx], gaussian_blob)
    return voxel_grid

def voxelize_raw_density(coords, values, grid_res, grid_bounds):
    """生成真实密度场 (Target) - 使用累加法"""
    grid_min, grid_max = grid_bounds
    target_grid = torch.zeros((grid_res, grid_res, grid_res), dtype=torch.float32)
    voxel_size = (grid_max - grid_min) / grid_res
    
    indices = ((coords - grid_min) / voxel_size).astype(int)
    valid = np.all((indices >= 0) & (indices < grid_res), axis=1)
    
    indices = indices[valid]
    values = values[valid]
    
    if len(indices) > 0:
        target_grid.index_put_(
            (torch.from_numpy(indices[:,2]), torch.from_numpy(indices[:,1]), torch.from_numpy(indices[:,0])),
            torch.from_numpy(values).float(), accumulate=True
        )
    return target_grid.unsqueeze(0)

def classify_density(density_grid, bins):
    """生成分类标签"""
    bins_tensor = torch.tensor(bins, device=density_grid.device)
    # 广播比较: density [1, D, H, W] vs bins [K]
    # 结果: [1, D, H, W, K]
    comparison = density_grid.unsqueeze(-1) > bins_tensor
    # Sum 得到类别索引 (0 到 K-1)
    class_grid = torch.sum(comparison, dim=-1).squeeze(0)
    return class_grid.long()

# =========================================================================
# 2. Dataset
# =========================================================================
# dataset_moe_phys.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
from ase.io import read as ase_read
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

# 复用之前的 voxelize 函数
from dataset_phys import voxelize_structure, voxelize_raw_density_from_points

def classify_density(density_grid, bins):
    bins_tensor = torch.tensor(bins, device=density_grid.device)
    comparison = density_grid.unsqueeze(-1) > bins_tensor
    class_grid = torch.sum(comparison, dim=-1).squeeze(0)
    return class_grid.long()

class PhysicsMoEDataset(Dataset):
    def __init__(self, cache_dir, raw_data_root, grid_res=64, grid_bounds=(-10.0, 10.0), 
                 num_pe_freqs=4, class_bins=None):
        self.cache_dir = Path(cache_dir)
        self.grid_res = grid_res
        self.grid_bounds = grid_bounds
        self.class_bins = class_bins
        self.indices = sorted([f.stem for f in self.cache_dir.glob("*.npz")])
        
        # 扫描原子类型 (简化版，假设您已经知道映射)
        # 实际使用时建议用完整的 scan 逻辑
        self.atom_map = {1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 14: 6, 15: 7, 16: 8, 17: 9, 32: 10, 34: 11, 35: 12} 
        self.num_atom_channels = len(self.atom_map)
        
        # 预计算 PE
        self.pe_grid = self._create_pe_grid(num_pe_freqs)

    def _create_pe_grid(self, num_freqs):
        coords = torch.linspace(-1, 1, self.grid_res)
        gz, gy, gx = torch.meshgrid(coords, coords, coords, indexing='ij')
        grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0)
        freqs = 2.0 ** torch.arange(num_freqs)
        scaled = grid.unsqueeze(-1) * freqs * torch.pi
        pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return pe.flatten(-2).permute(0, 4, 1, 2, 3).float()

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        try:
            data = np.load(self.cache_dir / f"{self.indices[idx]}.npz")
            atom_coords = data['atom_coords']
            atom_z = data['atom_z_numbers']
            dens_coords = data['density_coords']
            dens_vals = data['density_values']
            
            centroid = atom_coords.mean(axis=0)
            atom_coords -= centroid
            dens_coords -= centroid
            
            # 1. 准备原子网格
            atom_grid = voxelize_structure(atom_coords, atom_z, self.grid_res, self.grid_bounds, self.atom_map)
            
            # 2. 准备真实密度
            target_density = voxelize_raw_density_from_points(dens_coords, dens_vals, self.grid_res, self.grid_bounds)
            
            # 3. 准备分类标签
            target_labels = classify_density(target_density, self.class_bins)
            
            # 4. 准备积分
            target_integral = torch.sum(target_density)
            
            # 返回字典！
            return {
                'atom_grid': atom_grid,
                'pe_grid': self.pe_grid.squeeze(0),
                'target_density': target_density,
                'target_labels': target_labels,
                'target_integral': target_integral,
                'atom_coords': torch.from_numpy(atom_coords).float(), # 必须返回这个！
                'atom_z': torch.from_numpy(atom_z).float(),           # 必须返回这个！
            }
        except Exception: return None

def collate_fn_moe_phys(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    # Pad Atom Coords
    max_atoms = max([b['atom_coords'].shape[0] for b in batch])
    batched_coords, batched_z = [], []
    
    for b in batch:
        diff = max_atoms - b['atom_coords'].shape[0]
        batched_coords.append(F.pad(b['atom_coords'], (0,0,0,diff)))
        batched_z.append(F.pad(b['atom_z'], (0,diff)))

    return {
        'atom_grid': torch.stack([b['atom_grid'] for b in batch]),
        'pe_grid': torch.stack([b['pe_grid'] for b in batch]),
        'target_density': torch.stack([b['target_density'] for b in batch]),
        'target_labels': torch.stack([b['target_labels'] for b in batch]),
        'target_integral': torch.stack([b['target_integral'] for b in batch]),
        'atom_coords': torch.stack(batched_coords),
        'atom_z': torch.stack(batched_z),
    }