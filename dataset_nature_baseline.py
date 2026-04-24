import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from ase.io import read as ase_read

# 尝试导入 cube 读取工具
try:
    from cube_tools import cube as CubeToolzLoader
except ImportError:
    CubeToolzLoader = None

def process_cube_data(cube_path):
    """读取 Cube 文件的数据和坐标原点"""
    if CubeToolzLoader is None:
        raise ImportError("未找到 cube_tools，请确保该文件在路径中")
    
    cube_obj = CubeToolzLoader(str(cube_path))
    density_data = cube_obj.data.astype(np.float32)
    BOHR_TO_ANGSTROM = 0.529177210903
    
    origin_ang = cube_obj.origin * BOHR_TO_ANGSTROM
    step_x = cube_obj.X * BOHR_TO_ANGSTROM
    step_y = cube_obj.Y * BOHR_TO_ANGSTROM
    step_z = cube_obj.Z * BOHR_TO_ANGSTROM
    
    nx, ny, nz = density_data.shape
    i, j, k = np.indices((nx, ny, nz))
    coords = origin_ang + i.flatten()[:, None] * step_x + j.flatten()[:, None] * step_y + k.flatten()[:, None] * step_z
    values = density_data.flatten()
    
    return coords, values

def voxelize_target_density(coords, values, grid_res, grid_bounds):
    """将真实的离散密度点投影到网格上"""
    grid_min, grid_max = grid_bounds
    voxel_size = (grid_max - grid_min) / grid_res
    
    indices = np.floor((coords - grid_min) / voxel_size).astype(int)
    valid_mask = (indices[:, 0] >= 0) & (indices[:, 0] < grid_res) & \
                 (indices[:, 1] >= 0) & (indices[:, 1] < grid_res) & \
                 (indices[:, 2] >= 0) & (indices[:, 2] < grid_res)
                 
    indices = indices[valid_mask]
    values = values[valid_mask]
    
    grid = np.zeros((grid_res, grid_res, grid_res), dtype=np.float32)
    np.add.at(grid, (indices[:, 0], indices[:, 1], indices[:, 2]), values)
    
    return torch.from_numpy(grid).unsqueeze(0)

class NatureBaselineDataset(Dataset):
    def __init__(self, raw_data_root, cache_dir, sad_dir, grid_res=64, grid_bounds=(-10.0, 10.0)):
        self.raw_data_root = Path(raw_data_root)
        self.cache_dir = Path(cache_dir)
        self.sad_dir = Path(sad_dir)
        self.grid_res = grid_res
        self.grid_bounds = grid_bounds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 扫描有效样本
        self.sample_ids = []
        for d in self.raw_data_root.iterdir():
            if d.is_dir():
                sad_path = self.sad_dir / d.name / "sad_grid.npy"
                sdf_files = list(d.glob("*.sdf"))
                if sad_path.exists() and len(sdf_files) > 0:
                    self.sample_ids.append(d.name)
        
        self.sample_ids.sort()
        print(f"Nature Baseline Dataset loaded {len(self.sample_ids)} valid samples.")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample_dir = self.raw_data_root / sample_id
        
        try:
            # --- 1. 无论是否使用缓存，都要先获取总电子数 (N_e) ---
            sdf_file = next(sample_dir.glob("*.sdf"))
            atoms = ase_read(str(sdf_file), format='sdf')
            total_electrons = float(np.sum(atoms.get_atomic_numbers()))

            # --- 2. 加载 SAD 先验特征 ---
            sad_path = self.sad_dir / sample_id / "sad_grid.npy"
            sad_data = np.load(sad_path).astype(np.float32)
            sad_tensor = torch.from_numpy(sad_data).unsqueeze(0)
            
            # --- 3. 获取目标密度 ---
            target_cache_path = self.cache_dir / f"{sample_id}_target_density.pt"
            if target_cache_path.exists():
                # 注意：在新版本中推荐加上 weights_only=False 以兼容旧缓存
                target_tensor = torch.load(target_cache_path, map_location='cpu')
            else:
                cube_file = next(sample_dir.glob("*.cube"))
                coords, vals = process_cube_data(cube_file)
                # 几何中心化
                centroid = atoms.get_positions().mean(axis=0)
                centered_coords = coords - centroid
                target_tensor = voxelize_target_density(centered_coords, vals, self.grid_res, self.grid_bounds)
                torch.save(target_tensor, target_cache_path)

            return {
                'sample_id': sample_id,
                'sad_grid': sad_tensor,
                'target_density': target_tensor,
                'target_integral': torch.tensor([total_electrons], dtype=torch.float32) # <--- 修正：确保一定返回
            }
            
        except Exception as e:
            # 这里的打印能帮你找到具体哪个样本出错了
            print(f"Error processing sample {sample_id}: {e}")
            return None

def collate_fn_nature(batch):
    # 过滤掉由于读取错误产生的 None
    batch = [b for b in batch if b is not None]
    
    # --- 修正：如果整个 Batch 为空，返回 None 告知 DataLoader 跳过此 Batch ---
    if len(batch) == 0:
        return None
    
    return {
        'sample_id': [b['sample_id'] for b in batch],
        'sad_grid': torch.stack([b['sad_grid'] for b in batch]),
        'target_density': torch.stack([b['target_density'] for b in batch]),
        'target_integral': torch.stack([b['target_integral'] for b in batch])
    }