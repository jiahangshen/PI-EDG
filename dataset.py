import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from ase.io import read as ase_read
from tqdm import tqdm

# 从 pe_utils.py 导入 (确保 pe_utils.py 存在且包含最终修正版函数)
from pe_utils import create_coordinate_grid, positional_encoding
def create_distance_weight_grid(
    atom_coords: np.ndarray,
    grid_res: int,
    grid_bounds: tuple,
    beta: float = 1.0, # 控制权重衰减速度的超参数
    min_weight: float = 0.1 # 给予背景区域的最小权重
) -> torch.Tensor:
    """
    根据体素到最近原子的距离，生成一个权重网格。
    权重由一个指数衰减函数决定: w = (1-min_w) * exp(-beta * d) + min_w
    """
    grid_min, grid_max = grid_bounds
    
    # 预先计算体素中心坐标网格 (与 voxelize_structure 中的逻辑相同)
    voxel_size = (grid_max - grid_min) / grid_res
    coords_1d = torch.linspace(grid_min + voxel_size/2, grid_max - voxel_size/2, grid_res)
    grid_z, grid_y, grid_x = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    voxel_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1) # shape: [res, res, res, 3]

    if atom_coords.shape[0] == 0:
        # 如果没有原子，所有权重都为最小值
        return torch.full((1, grid_res, grid_res, grid_res), min_weight)

    # 将原子坐标转换为 torch.Tensor
    atom_pos = torch.from_numpy(atom_coords).float() # shape: [num_atoms, 3]
    
    # 使用广播机制高效计算每个体素到所有原子的距离
    # voxel_coords: [res, res, res, 1, 3]
    # atom_pos:     [1,   1,   1, num_atoms, 3]
    distances_sq = torch.sum((voxel_coords.unsqueeze(-2) - atom_pos) ** 2, dim=-1)
    
    # 找到每个体素到最近原子的距离
    min_distances_sq = torch.min(distances_sq, dim=-1).values
    min_distances = torch.sqrt(min_distances_sq)
    
    # 使用指数衰减函数计算权重
    weights = (1.0 - min_weight) * torch.exp(-beta * min_distances) + min_weight
    
    # 增加一个通道维度，并转换为 float16
    return weights.unsqueeze(0)
# --- 核心处理函数 ---
def voxelize_structure(
    atom_coords: np.ndarray,
    atom_z_numbers: np.ndarray,
    grid_res: int,
    grid_bounds: tuple,
    atom_map: dict,
    sigma: float = 0.8
) -> torch.Tensor:
    """高斯模糊版本的 voxelize_structure"""
    grid_min, grid_max = grid_bounds
    num_channels = len(atom_map)
    voxel_grid = torch.zeros((num_channels, grid_res, grid_res, grid_res), dtype=torch.float32)
    voxel_size = (grid_max - grid_min) / grid_res
    if voxel_size <= 0: return voxel_grid
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

def voxelize_from_points(
    aligned_density_coords: np.ndarray,
    density_values: np.ndarray,
    grid_res: int, grid_bounds: tuple, epsilon = 0.1
) -> torch.Tensor:
    """从点云体素化密度 (积分/求和)"""
    grid_min_ang, grid_max_ang = grid_bounds
    target_voxel_grid = torch.zeros((grid_res, grid_res, grid_res), dtype=torch.float32)
    target_voxel_size_ang = (grid_max_ang - grid_min_ang) / grid_res
    if target_voxel_size_ang <= 0: return target_voxel_grid.unsqueeze(0)
    target_all_indices = ((aligned_density_coords - grid_min_ang) / target_voxel_size_ang).astype(int)
    valid_mask = np.all((target_all_indices >= 0) & (target_all_indices < grid_res), axis=1)
    valid_target_indices = target_all_indices[valid_mask]
    valid_density_values = density_values[valid_mask]
    if len(valid_target_indices) > 0:
        vz, vy, vx = valid_target_indices[:, 2], valid_target_indices[:, 1], valid_target_indices[:, 0]
        target_voxel_grid.index_put_( (torch.from_numpy(vz), torch.from_numpy(vy), torch.from_numpy(vx)), torch.from_numpy(valid_density_values).float(), accumulate=True )
      # 定义你的 epsilon
    # log_transformed_grid = torch.log1p(target_voxel_grid) # 原来的代码
    transformed_grid = torch.pow(target_voxel_grid, 1.0/3.0) # 新代码
    return transformed_grid.unsqueeze(0).to(torch.float16)

# --- VoxelizedMoleculeCachedDataset 类 (包含 PE) ---
class VoxelizedMoleculeCachedDataset(Dataset):
    def __init__(self, cache_dir: Path,
                 raw_data_root_for_scan: Path,
                 grid_res: int, grid_bounds: tuple,
                 num_pe_freqs: int): # <-- 重新加入 num_pe_freqs
        self.cache_dir = cache_dir
        self.grid_res = grid_res
        self.grid_bounds = grid_bounds
        self.num_pe_freqs = num_pe_freqs # <-- 保存 PE 频率

        print(f"正在扫描缓存目录 {cache_dir}...")
        self.indices = sorted([f.stem for f in self.cache_dir.glob("*.npz")])
        if not self.indices: raise FileNotFoundError(f"在 {cache_dir} 中找不到 .npz 文件。")
        print(f"找到 {len(self.indices)} 个缓存文件。")

        print("正在动态扫描原子类型...")
        self.atom_map = self._discover_elements(raw_data_root_for_scan, self.indices)
        self.num_atom_channels = len(self.atom_map)

        # --- 重新加入：预计算位置编码 ---
        if self.num_pe_freqs > 0:
            print("正在预计算位置编码网格...")
            coord_grid = create_coordinate_grid(self.grid_res)
            pe_grid = positional_encoding(coord_grid, self.num_pe_freqs)
            # 保存为 float16 以节省内存
            self.pe_grid = pe_grid.permute(3, 0, 1, 2)
            print(f"位置编码网格准备就绪，通道数: {self.pe_grid.shape[0]}")
        else:
            self.pe_grid = None # 如果不使用PE，则为None
        # ------------------------------------

    def _discover_elements(self, raw_data_root, indices_to_scan):
        # ... (这个函数保持不变) ...
        unique_z_values = set()
        num_scan_samples = min(200, len(indices_to_scan))
        print(f"正在扫描前 {num_scan_samples} 个样本的原始SDF文件以确定原子类型...")
        for i in tqdm(range(num_scan_samples)):
             index = indices_to_scan[i]
             sample_dir = raw_data_root / index
             try:
                 sdf_file_path = next(sample_dir.glob("*.sdf"))
                 atoms = ase_read(str(sdf_file_path), format='sdf')
                 unique_z_values.update(atoms.get_atomic_numbers())
             except Exception: pass
        if not unique_z_values: raise RuntimeError("无法从样本SDF文件中发现任何原子类型！")
        sorted_unique_z = sorted(list(unique_z_values)); atom_map = {z: i for i, z in enumerate(sorted_unique_z)}
        print(f"发现 {len(atom_map)} 种原子类型: {atom_map}"); return atom_map

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        index = self.indices[idx]
        npz_file_path = self.cache_dir / f"{index}.npz"
        try:
            cache_data = np.load(npz_file_path)
            # 使用正确的键名加载（假设您使用了无对齐版本生成缓存）
            atom_coords = cache_data['atom_coords']
            atom_z_numbers = cache_data['atom_z_numbers']
            density_coords = cache_data['density_coords']
            density_values = cache_data['density_values']
            
            # 中心化
            centroid = atom_coords.mean(axis=0)
            centered_atom_coords = atom_coords - centroid
            centered_density_coords = density_coords - centroid
            
            # 体素化结构 (高斯模糊)
            atom_grid = voxelize_structure(
                centered_atom_coords, atom_z_numbers,
                self.grid_res, self.grid_bounds, self.atom_map
            )
            
            # 体素化密度
            target_grid = voxelize_from_points(
                centered_density_coords, density_values,
                self.grid_res, self.grid_bounds
            )
            
            # --- 核心修改：返回三个张量 ---
            if self.pe_grid is not None:
                return atom_grid, self.pe_grid, target_grid
            else: # 如果不使用 PE，则返回一个占位符或引发错误，取决于模型如何处理
                  # 这里我们返回 atom_grid 两次，但模型应该只使用第一个
                  # 或者更好的做法是在 __init__ 中就确定好
                  # 为简单起见，如果不用PE，则不应该调用这个Dataset？
                  # 或者模型应该能处理 pe_grid 为 None 的情况？
                  # --> 最好的方式还是确保模型和数据生成方式一致
                  raise ValueError("Dataset 初始化时 num_pe_freqs > 0，但 pe_grid 未生成。")
                  # 或者根据模型设计返回： return atom_grid, torch.empty(0), target_grid

        except Exception as e:
            print(f"\n处理缓存文件 {npz_file_path} 时发生错误: {e}")
            return None, None, None # 返回三个 None

# collate_fn_skip_error 函数需要修改以处理可能的三元组 None
def collate_fn_skip_error(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        # 返回一个空的元组，每个元素都是空tensor，以匹配期望的输出数量
        return (torch.empty(0), torch.empty(0), torch.empty(0),
                torch.empty(0), torch.empty(0), torch.empty(0))
    
    # 按照新的 __getitem__ 返回顺序解包
    atom_grids_orig, atom_grids_rot, pe_grids, target_labels, target_densities, rot_matrices = zip(*batch)
    
    return (torch.stack(atom_grids_orig), torch.stack(atom_grids_rot),
            torch.stack(pe_grids), torch.stack(target_labels),
            torch.stack(target_densities), torch.stack(rot_matrices))
