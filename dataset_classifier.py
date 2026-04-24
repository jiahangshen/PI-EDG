# file: dataset_classifier.py
from pe_utils import create_coordinate_grid, positional_encoding
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
from ase.io import read as ase_read
from dataset import voxelize_structure
from dataset import collate_fn_skip_error
from scipy.spatial.transform import Rotation
import torch.nn.functional as F

# --- 你可以从旧的 dataset.py 中复制这些辅助函数 ---
# pe_utils.py (create_coordinate_grid, positional_encoding)
# voxelize_structure
# ...

# -----------------------------------------------------
#               核心：新的数据处理函数
# -----------------------------------------------------
# 在你定义 voxelize_from_points 的同一个文件中

def voxelize_raw_density_from_points(
    aligned_density_coords: np.ndarray,
    density_values: np.ndarray,
    grid_res: int, 
    grid_bounds: tuple
) -> torch.Tensor:
    """
    从点云体素化原始密度值 (求和)，不进行任何数值变换。
    返回一个 [res, res, res] 的 float32 Tensor。
    """
    grid_min_ang, grid_max_ang = grid_bounds
    target_voxel_grid = torch.zeros((grid_res, grid_res, grid_res), dtype=torch.float32)
    target_voxel_size_ang = (grid_max_ang - grid_min_ang) / grid_res

    if target_voxel_size_ang <= 0:
        return target_voxel_grid # 返回 [res, res, res]

    target_all_indices = ((aligned_density_coords - grid_min_ang) / target_voxel_size_ang).astype(int)
    
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
    
    return target_voxel_grid
def classify_density_grid(density_grid: torch.Tensor, bins: list) -> torch.LongTensor:
    """
    将浮点数的密度网格转换为整数类别的标签网格。
    
    Args:
        density_grid: 包含真实电子密度的张量，shape: [1, res, res, res]
        bins: 定义数量级区间的列表，例如 [1e-5, 1e-3, 1.0]

    Returns:
        一个 torch.LongTensor，其中每个元素是其对应的类别索引。
        类别 0: ( -inf, bins[0] ]
        类别 1: ( bins[0], bins[1] ]
        ...
    """
    # 确保 bins 是一个 tensor
    bins_tensor = torch.tensor(bins, dtype=density_grid.dtype, device=density_grid.device)
    
    # unsqueeze 将 bins 变成列向量，以便广播
    # [1, 1, 1, num_bins]
    # density_grid is [1, res, res, res, 1]
    # 广播后，每个体素都会与所有 bin 边界进行比较
    # 结果是一个布尔张量，shape: [1, res, res, res, num_bins]
    comparison = density_grid.unsqueeze(-1) > bins_tensor
    
    # torch.sum(..., dim=-1) 会计算每个体素值大于多少个 bin 边界
    # 这恰好就是它的类别索引
    # 例如，值 > 0 个边界 -> 类别 0
    # 值 > 1 个边界 -> 类别 1
    class_grid = torch.sum(comparison, dim=-1).squeeze(0) # 移除通道维度
    
    return class_grid.long() # 损失函数需要 LongTensor

def voxelize_and_classify_from_points(
    aligned_density_coords: np.ndarray,
    density_values: np.ndarray,
    grid_res: int, 
    grid_bounds: tuple,
    bins: list
) -> torch.LongTensor:
    """
    从点云体素化密度，然后直接转换为分类标签。
    """
    grid_min_ang, grid_max_ang = grid_bounds
    target_voxel_grid = torch.zeros((grid_res, grid_res, grid_res), dtype=torch.float32)
    # ... (从你旧的 dataset.py 中复制完整的体素化逻辑) ...
    # 确保 target_voxel_grid 被正确填充
    target_voxel_size_ang = (grid_max_ang - grid_min_ang) / grid_res
    if target_voxel_size_ang <= 0: return torch.zeros((grid_res, grid_res, grid_res), dtype=torch.long)
    target_all_indices = ((aligned_density_coords - grid_min_ang) / target_voxel_size_ang).astype(int)
    valid_mask = np.all((target_all_indices >= 0) & (target_all_indices < grid_res), axis=1)
    valid_target_indices = target_all_indices[valid_mask]
    valid_density_values = density_values[valid_mask]
    if len(valid_target_indices) > 0:
        vz, vy, vx = valid_target_indices[:, 2], valid_target_indices[:, 1], valid_target_indices[:, 0]
        target_voxel_grid.index_put_( (torch.from_numpy(vz), torch.from_numpy(vy), torch.from_numpy(vx)), torch.from_numpy(valid_density_values).float(), accumulate=True )
    
    # 现在调用分类函数
    class_labels = classify_density_grid(target_voxel_grid.unsqueeze(0), bins)
    return class_labels # 返回的是 LongTensor, shape: [res, res, res]

# -----------------------------------------------------
#                   新的 Dataset 类
# -----------------------------------------------------
class VoxelizedClassifierDataset(Dataset):
    def __init__(self, cache_dir: Path,
                 raw_data_root_for_scan: Path,
                 grid_res: int, grid_bounds: tuple,
                 num_pe_freqs: int,
                 class_bins: list): # <-- 新增：传入分类边界
        self.cache_dir = cache_dir
        self.grid_res = grid_res
        self.grid_bounds = grid_bounds
        self.num_pe_freqs = num_pe_freqs
        self.class_bins = class_bins # <-- 保存分类边界
        
        # ... (从你旧的 VoxelizedMoleculeCachedDataset 复制 __init__ 的其余部分)
        # 包括扫描缓存、发现原子、预计算 PE 等
        print(f"正在扫描缓存目录 {cache_dir}...")
        self.indices = sorted([f.stem for f in self.cache_dir.glob("*.npz")])
        if not self.indices: raise FileNotFoundError(f"在 {cache_dir} 中找不到 .npz 文件。")
        print(f"找到 {len(self.indices)} 个缓存文件。")

        print("正在动态扫描原子类型...")
        # 假设 _discover_elements 方法也被复制过来了
        self.atom_map = self._discover_elements(raw_data_root_for_scan, self.indices)
        self.num_atom_channels = len(self.atom_map)
        
        # 假设 pe_utils 被导入了
        if self.num_pe_freqs > 0:
            print("正在预计算位置编码网格...")
            coord_grid = create_coordinate_grid(self.grid_res)
            pe_grid = positional_encoding(coord_grid, self.num_pe_freqs)
            self.pe_grid = pe_grid.permute(3, 0, 1, 2).to(torch.float16)
        else:
            self.pe_grid = None
        # ...

    def _discover_elements(self, raw_data_root, indices_to_scan):
        # ... (这个函数保持不变, 从旧文件复制) ...
        unique_z_values = set()
        num_scan_samples = min(200, len(indices_to_scan))
        for i in tqdm(range(num_scan_samples)):
             index = indices_to_scan[i]
             sample_dir = raw_data_root / index
             try:
                 sdf_file_path = next(sample_dir.glob("*.sdf"))
                 atoms = ase_read(str(sdf_file_path), format='sdf')
                 unique_z_values.update(atoms.get_atomic_numbers())
             except Exception: pass
        sorted_unique_z = sorted(list(unique_z_values)); atom_map = {z: i for i, z in enumerate(sorted_unique_z)}
        return atom_map


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
            
            centroid = atom_coords.mean(axis=0)
            centered_atom_coords = atom_coords - centroid
            centered_density_coords = density_coords - centroid
            
            # 假设 voxelize_structure 已被定义
 # 假设它在 dataset.py
            atom_grid = voxelize_structure(
                centered_atom_coords, atom_z_numbers,
                self.grid_res, self.grid_bounds, self.atom_map
            )
            
            # 关键修改：调用新的体素化+分类函数
            target_labels = voxelize_and_classify_from_points(
                centered_density_coords, density_values,
                self.grid_res, self.grid_bounds,
                self.class_bins # <-- 使用 self.class_bins
            )
            
            if self.pe_grid is not None:
                return atom_grid, self.pe_grid, target_labels # 返回的是标签
            else:
                raise ValueError("PE grid not generated.")

        except Exception as e:
            print(f"\n处理缓存文件 {npz_file_path} 时发生错误: {e}")
            return None, None, None
            
# collate_fn 应该和之前的一样，可以复用
class VoxelizedMoEDataset(VoxelizedClassifierDataset):
    def __init__(self, *args, augment_on_the_fly=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment_on_the_fly = augment_on_the_fly
        print(f"在线数据增强 (旋转) 已 {'启用' if augment_on_the_fly else '禁用'}.")
    """
    一个继承自原分类数据集的MoE数据集。
    主要区别在于 __getitem__ 会额外返回原始的密度值。
    """
    def __getitem__(self, idx: int):
        index = self.indices[idx]
        npz_file_path = self.cache_dir / f"{index}.npz"
        try:
            # --- 步骤 1 & 2: 加载和预处理 NumPy 数据 ---
            cache_data = np.load(npz_file_path)
            atom_coords = cache_data['atom_coords']
            atom_z_numbers = cache_data['atom_z_numbers']
            density_coords = cache_data['density_coords']
            density_values = cache_data['density_values']
            
            centroid = np.mean(atom_coords, axis=0)
            centered_atom_coords = atom_coords - centroid
            centered_density_coords = density_coords - centroid
            
            # --- 新增：如果启用增强，则生成旋转矩阵 ---
            if self.augment_on_the_fly:
                # 生成随机旋转矩阵
                random_rotation = Rotation.random()
                rot_matrix = random_rotation.as_matrix().astype(np.float32)
                
                # 对原子坐标进行旋转
                rotated_atom_coords = centered_atom_coords @ rot_matrix.T
            else:
                # 如果不旋转，则使用原始坐标
                rotated_atom_coords = centered_atom_coords
                rot_matrix = np.eye(3, dtype=np.float32) # 单位矩阵

            # --- 步骤 3: 体素化 ---
            # a) 创建原始原子网格
            atom_grid_orig = voxelize_structure(
                centered_atom_coords, atom_z_numbers,
                self.grid_res, self.grid_bounds, self.atom_map
            )
            # b) 创建旋转后的原子网格
            atom_grid_rot = voxelize_structure(
                rotated_atom_coords, atom_z_numbers,
                self.grid_res, self.grid_bounds, self.atom_map
            )

            # c) 创建原始密度值网格 (这个只需要一份，因为它是我们的目标)
            target_density_grid = voxelize_raw_density_from_points(
                centered_density_coords,
                density_values,
                self.grid_res,
                self.grid_bounds
            )
            
            # d) 创建类别标签网格 (也只需要一份)
            target_labels = classify_density_grid(target_density_grid.unsqueeze(0), self.class_bins)
            
            # --- 步骤 4: 准备返回数据 ---
            pe_grid = self.pe_grid
            if pe_grid is None:
                raise ValueError("PE grid not generated.")
            
            # 返回原始原子网格、旋转后的原子网格、PE网格、目标标签、目标密度和旋转矩阵
            return (atom_grid_orig.float(), atom_grid_rot.float(), pe_grid,
                    target_labels.long(), target_density_grid.float(),
                    torch.from_numpy(rot_matrix))

        except Exception as e:
            print(f"\n处理缓存文件 {npz_file_path} 时发生严重错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None,None,None