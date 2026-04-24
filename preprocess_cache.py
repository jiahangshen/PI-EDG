import numpy as np
from pathlib import Path
from tqdm import tqdm
from ase.io import read as ase_read
from cube_tools import cube as CubeToolzLoader

# --- align_molecule 函数现在已不再需要 ---

# ======================================================
# 1. 配置参数
# ======================================================
RAW_DATA_ROOT = Path("./raw_data_cube") # <-- !! 修改为您实际的原始数据路径 !!
CACHE_DIR = Path("./preprocessed_cache") # <-- !! 使用独立的、无对齐的缓存目录 !!

# ======================================================
# 2. 主处理流程 (已移除对齐 & 恢复高效计算)
# ======================================================
def main():
    """
    主函数，执行离线缓存生成流程 (不进行分子姿态对齐，使用高效的向量化计算)。
    """
    print("开始生成轻量级中间缓存文件 (无姿态对齐 & 向量化版)...")
    
    # --- 1. 扫描有效的样本索引 ---
    print(f"正在扫描目录 {RAW_DATA_ROOT}...")
    valid_indices = []
    for item in tqdm(list(RAW_DATA_ROOT.iterdir()), desc="扫描目录"):
        if item.is_dir() and item.name.isdigit():
             if any(item.glob("*.sdf")) and any(item.glob("*.cube")):
                 valid_indices.append(item.name)
    print(f"找到 {len(valid_indices)} 个有效的样本。")
    if not valid_indices:
        print("错误: 未找到任何有效的样本目录，程序终止。")
        return

    # --- 2. 创建输出目录 ---
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"缓存文件将保存在: {CACHE_DIR}")
    
    # --- 3. 遍历并处理每个分子 ---
    print(f"开始处理 {len(valid_indices)} 个分子并生成缓存...")
    BOHR_TO_ANGSTROM = 0.529177210903
    
    for index in tqdm(valid_indices, desc="生成缓存"):
        sample_dir = RAW_DATA_ROOT / index
        output_npz_path = CACHE_DIR / f"{index}.npz"

    # --- 步骤 2: 检查该文件是否已经存在 ---
        if output_npz_path.exists():
            # 如果文件已存在，则跳过这个分子的所有处理步骤
            continue
        try:
            sdf_file_path = next(sample_dir.glob("*.sdf"))
            cube_file_path = next(sample_dir.glob("*.cube"))
            
            atoms_ase = ase_read(str(sdf_file_path), format='sdf')
            atom_coords = atoms_ase.get_positions().astype(np.float32)
            atom_z_numbers = atoms_ase.get_atomic_numbers()
            
            cube_obj = CubeToolzLoader(str(cube_file_path))
            density_data = cube_obj.data
            cube_origin_bohr = cube_obj.origin.astype(np.float32)
            step_vec_x_bohr = cube_obj.X.astype(np.float32)
            step_vec_y_bohr = cube_obj.Y.astype(np.float32)
            step_vec_z_bohr = cube_obj.Z.astype(np.float32)

            # --- !! 姿态对齐步骤已完全移除 !! ---

            # --- c. (恢复高效向量化) 使用矩阵乘法计算所有CUBE点坐标 ---
            nx, ny, nz = density_data.shape
            
            i_indices, j_indices, k_indices = np.indices((nx, ny, nz))
            indices_matrix = np.stack([i_indices.ravel(), j_indices.ravel(), k_indices.ravel()], axis=1)
            step_matrix_bohr = np.stack([step_vec_x_bohr, step_vec_y_bohr, step_vec_z_bohr], axis=0)
            
            all_coords_bohr = cube_origin_bohr + indices_matrix @ step_matrix_bohr
            all_coords_ang = all_coords_bohr * BOHR_TO_ANGSTROM
            # --- 高效计算结束 ---

            density_values = density_data.flatten()

            # --- d. 保存到紧凑的 .npz 文件 ---
            output_npz_path = CACHE_DIR / f"{index}.npz"
            np.savez_compressed(
                output_npz_path,
                # --- 保存的是原始坐标，不是对齐或中心化后的坐标 ---
                atom_coords=atom_coords,
                atom_z_numbers=atom_z_numbers.astype(np.int16),
                density_coords=all_coords_ang,
                density_values=density_values
            )

        except Exception as e:
            print(f"\n处理索引 {index} 时发生错误，已跳过: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    print("\n所有缓存文件生成完成！")

if __name__ == "__main__":
    main()