import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
from ase.io import read as ase_read
import torch.nn.functional as F

# =========================================================================
# 1. 辅助函数：体素化核心逻辑
# =========================================================================

def voxelize_structure(
    atom_coords: np.ndarray,
    atom_z_numbers: np.ndarray,
    grid_res: int,
    grid_bounds: tuple,
    atom_map: dict,
    sigma: float = 0.1  # 高斯模糊的宽度，控制输入特征的稀疏度
) -> torch.Tensor:
    """
    将原子坐标转换为多通道的高斯模糊网格（作为模型输入）。
    """
    grid_min, grid_max = grid_bounds
    num_channels = len(atom_map)
    voxel_grid = torch.zeros((num_channels, grid_res, grid_res, grid_res), dtype=torch.float32)
    
    voxel_size = (grid_max - grid_min) / grid_res
    if voxel_size <= 0: return voxel_grid

    # 创建网格坐标
    coords_1d = torch.linspace(grid_min + voxel_size/2, grid_max - voxel_size/2, grid_res)
    grid_z, grid_y, grid_x = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    voxel_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1) # [res, res, res, 3]

    # 遍历每个原子，添加高斯球
    for i in range(atom_coords.shape[0]):
        atom_z = int(atom_z_numbers[i])
        if atom_z in atom_map:
            channel_idx = atom_map[atom_z]
            atom_pos = torch.from_numpy(atom_coords[i]).float()
            
            # 简单的距离计算和高斯映射 (只计算原子附近的局部区域会更快，这里为了简单做全局计算)
            # 优化提示：实际生产中可只计算原子周围 box 内的体素
            distances_sq = torch.sum((voxel_coords - atom_pos) ** 2, dim=-1)
            gaussian_blob = torch.exp(-distances_sq / (2 * (sigma ** 2)))
            
            # 取最大值叠加（保留最强的信号）
            voxel_grid[channel_idx] = torch.maximum(voxel_grid[channel_idx], gaussian_blob)
            
    return voxel_grid

def voxelize_raw_density_from_points(
    aligned_density_coords: np.ndarray,
    density_values: np.ndarray,
    grid_res: int, 
    grid_bounds: tuple
) -> torch.Tensor:
    """
    将密度点云体素化为物理密度网格（作为 Ground Truth）。
    **关键**：这里不进行 Log 变换，因为物理损失函数需要真实的电子数密度。
    """
    grid_min, grid_max = grid_bounds
    target_voxel_grid = torch.zeros((grid_res, grid_res, grid_res), dtype=torch.float32)
    target_voxel_size = (grid_max - grid_min) / grid_res

    if target_voxel_size <= 0:
        return target_voxel_grid.unsqueeze(0)

    # 将坐标映射到网格索引
    target_all_indices = ((aligned_density_coords - grid_min) / target_voxel_size).astype(int)
    
    # 过滤出在网格范围内的点
    valid_mask = np.all((target_all_indices >= 0) & (target_all_indices < grid_res), axis=1)
    valid_target_indices = target_all_indices[valid_mask]
    valid_density_values = density_values[valid_mask]

    # 累加密度值 (index_put_)
    if len(valid_target_indices) > 0:
        vz, vy, vx = valid_target_indices[:, 2], valid_target_indices[:, 1], valid_target_indices[:, 0]
        target_voxel_grid.index_put_(
            (torch.from_numpy(vz), torch.from_numpy(vy), torch.from_numpy(vx)),
            torch.from_numpy(valid_density_values).float(),
            accumulate=True
        )
    
    # 增加通道维度 [1, res, res, res]
    return target_voxel_grid.unsqueeze(0)

# =========================================================================
# 2. Dataset 类
# =========================================================================

class PhysicsDataset(Dataset):
    def __init__(self, 
                 cache_dir: Path,
                 raw_data_root_for_scan: Path,
                 grid_res: int = 64, 
                 grid_bounds: tuple = (-10.0, 10.0)):
        """
        物理知情 Dataset。
        不再返回分类标签，而是返回用于物理计算的原始数据。
        """
        self.cache_dir = cache_dir
        self.grid_res = grid_res
        self.grid_bounds = grid_bounds
        
        # 1. 扫描缓存文件
        print(f"正在扫描缓存目录 {cache_dir}...")
        self.indices = sorted([f.stem for f in self.cache_dir.glob("*.npz")])
        if not self.indices: 
            raise FileNotFoundError(f"在 {cache_dir} 中找不到 .npz 文件。")
        print(f"找到 {len(self.indices)} 个缓存文件。")

        # 2. 确定原子类型映射 (Atom Map)
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
                 sdf_file_path = next(sample_dir.glob("*.sdf"))
                 atoms = ase_read(str(sdf_file_path), format='sdf')
                 unique_z_values.update(atoms.get_atomic_numbers())
             except Exception: pass
        
        sorted_unique_z = sorted(list(unique_z_values))
        return {z: i for i, z in enumerate(sorted_unique_z)}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        index = self.indices[idx]
        npz_file_path = self.cache_dir / f"{index}.npz"
        
        try:
            # --- 1. 加载数据 ---
            cache_data = np.load(npz_file_path)
            atom_coords = cache_data['atom_coords']     # [N_atoms, 3]
            atom_z_numbers = cache_data['atom_z_numbers'] # [N_atoms]
            density_coords = cache_data['density_coords'] # [M_points, 3]
            density_values = cache_data['density_values'] # [M_points]
            
            # --- 2. 几何中心化 ---
            # 物理计算对外势 V_ext 的位置非常敏感，必须确保 atom 和 density 共享同一个中心
            centroid = np.mean(atom_coords, axis=0)
            centered_atom_coords = atom_coords - centroid
            centered_density_coords = density_coords - centroid
            
            # --- 3. 计算物理量 ---
            # 总电子数 N (硬约束目标)
            total_electrons = float(np.sum(atom_z_numbers))
            
            # --- 4. 体素化输入特征 (Input Features) ---
            # 模型看到的“化学环境”
            atom_grid = voxelize_structure(
                centered_atom_coords, atom_z_numbers,
                self.grid_res, self.grid_bounds, self.atom_map
            )
            
            # --- 5. 体素化 Ground Truth (Target) ---
            # 真实的物理密度，用于计算 Loss
            target_density = voxelize_raw_density_from_points(
                centered_density_coords, density_values,
                self.grid_res, self.grid_bounds
            )
            target_integral = torch.sum(target_density)
            # --- 6. 返回字典 ---
            # 我们返回原子的原始坐标(centered)，以便在 GPU 上精确计算 V_ext
            return {
                'atom_grid': atom_grid,                   # [C, D, H, W]
                'atom_coords': torch.from_numpy(centered_atom_coords).float(), # [N_atoms, 3]
                'atom_z': torch.from_numpy(atom_z_numbers).float(),            # [N_atoms]
                'target_density': target_density,         # [1, D, H, W]
                'total_electrons': torch.tensor(total_electrons, dtype=torch.float32), # Scalar
                'target_integral': target_integral, # 真实值 (1174.6) <--- 新增
                'index': index
            }

        except Exception as e:
            print(f"\nError loading {index}: {e}")
            return None

# =========================================================================
# 3. Collate Function (处理变长原子坐标)
# =========================================================================

def collate_fn_physics(batch):
    """
    自定义 collate 函数，用于处理 atom_coords 和 atom_z 的变长问题。
    会对原子坐标进行 Padding (用 0 填充)。
    """
    # 1. 过滤 None (加载失败的样本)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    # 2. 获取 batch 中最大的原子数量
    max_atoms = max([b['atom_coords'].shape[0] for b in batch])
    
    batched_atom_grids = []
    batched_atom_coords = []
    batched_atom_z = []
    batched_target_densities = []
    batched_total_electrons = []
    batched_target_integral=[]
    for b in batch:
        # 常规堆叠
        batched_atom_grids.append(b['atom_grid'])
        batched_target_densities.append(b['target_density'])
        batched_total_electrons.append(b['total_electrons'])
        batched_target_integral.append(b['target_integral'])
        # 变长数据 Padding
        coords = b['atom_coords']
        z = b['atom_z']
        n_atoms = coords.shape[0]
        diff = max_atoms - n_atoms
        
        # Pad 坐标 [N, 3] -> [Max, 3] (在第0维填充 diff 行)
        if diff > 0:
            coords_padded = F.pad(coords, (0, 0, 0, diff), value=0.0)
            z_padded = F.pad(z, (0, diff), value=0.0) # Z=0 代表无原子，不会产生电势
        else:
            coords_padded = coords
            z_padded = z
            
        batched_atom_coords.append(coords_padded)
        batched_atom_z.append(z_padded)
    
    # 3. 转换为 Tensor Batch
    return {
        'atom_grid': torch.stack(batched_atom_grids),          # [B, C, D, H, W]
        'atom_coords': torch.stack(batched_atom_coords),       # [B, Max_Atoms, 3]
        'atom_z': torch.stack(batched_atom_z),                 # [B, Max_Atoms]
        'target_density': torch.stack(batched_target_densities), # [B, 1, D, H, W]
        'total_electrons': torch.stack(batched_total_electrons), # [B]
        'target_integral':torch.stack(batched_target_integral)   # [B]
    }