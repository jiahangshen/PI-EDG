import numpy as np
from pathlib import Path
from tqdm import tqdm
from ase.io import read as ase_read
import os

# ======================================================
# 1. 配置参数 (请修改为您自己的路径)
# ======================================================

# --- !! 核心修改：指定原始数据根目录 !! ---
RAW_DATA_ROOT = Path("./raw_data_cube") # <-- !! 请务必修改为您实际的路径 !!

# --- 抽样参数 ---
# 我们不需要分析所有分子，抽取一部分即可
NUM_SAMPLES_TO_CHECK = 30000 # 抽样1000个分子进行分析

# --- 分析参数 ---
# 我们希望找到一个能包含99.5%数据的边界
PERCENTILE = 100

# ======================================================
# 2. 复制 align_molecule 函数
# (确保这里的对齐逻辑与您训练时完全一致)
# ======================================================
def align_molecule(coords: np.ndarray):
    """使用PCA对分子坐标进行对齐。"""
    centroid = coords.mean(axis=0)
    centered_coords = coords - centroid
    covariance_matrix = np.cov(centered_coords, rowvar=False)
    if np.allclose(covariance_matrix, 0):
        return np.eye(3), centroid
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        return eigenvectors.T, centroid
    except np.linalg.LinAlgError:
         print("警告: 奇异协方差矩阵，返回单位矩阵。")
         return np.eye(3), centroid

# ======================================================
# 3. 主分析流程
# ======================================================
def main():
    print(f"开始分析数据集边界，将从 {RAW_DATA_ROOT} 中抽样 {NUM_SAMPLES_TO_CHECK} 个分子...")

    # --- 扫描目录获取所有样本索引 ---
    if not RAW_DATA_ROOT.is_dir():
        print(f"错误: 路径 {RAW_DATA_ROOT} 不是一个有效的目录。")
        return
        
    all_indices = [item.name for item in RAW_DATA_ROOT.iterdir() if item.is_dir() and item.name.isdigit()]
    if not all_indices:
        print(f"错误: 在 {RAW_DATA_ROOT} 中找不到任何以数字命名的样本子目录。")
        return

    # --- 随机抽样 ---
    if len(all_indices) > NUM_SAMPLES_TO_CHECK:
        sampled_indices = np.random.choice(all_indices, NUM_SAMPLES_TO_CHECK, replace=False)
    else:
        print("警告: 样本总数小于抽样数，将分析所有样本。")
        sampled_indices = all_indices

    print(f"已抽样 {len(sampled_indices)} 个索引，开始处理...")

    # --- 遍历样本，收集对齐后的坐标 ---
    all_aligned_coords = []
    for index in tqdm(sampled_indices, desc="分析分子"):
        sample_dir = RAW_DATA_ROOT / index
        try:
            sdf_file_path = next(sample_dir.glob("*.sdf"))
            atoms = ase_read(str(sdf_file_path), format='sdf')
            atom_coords = atoms.get_positions()

            # 只执行对齐，不进行体素化
            rotation_matrix, centroid = align_molecule(atom_coords)
            aligned_coords = (atom_coords - centroid) @ rotation_matrix
            
            all_aligned_coords.append(aligned_coords)

        except Exception as e:
            print(f"\n处理索引 {index} 时出错: {e}")
            continue

    if not all_aligned_coords:
        print("错误: 未能成功处理任何分子，无法进行分析。")
        return

    # 将所有坐标合并成一个巨大的Numpy数组
    full_coord_array = np.concatenate(all_aligned_coords, axis=0)
    print(f"\n分析完成，共收集到 {full_coord_array.shape[0]} 个原子坐标。")

    # --- 分析坐标分布 ---
    # 计算每个轴的最小值和最大值
    min_coords = full_coord_array.min(axis=0)
    max_coords = full_coord_array.max(axis=0)

    # 计算能包含指定百分比数据的边界
    lower_bound = np.percentile(full_coord_array, 100 - PERCENTILE, axis=0)
    upper_bound = np.percentile(full_coord_array, PERCENTILE, axis=0)

    print("\n--- 坐标分布分析结果 ---")
    print(f"X轴范围: [{min_coords[0]:.2f}, {max_coords[0]:.2f}] Å")
    print(f"Y轴范围: [{min_coords[1]:.2f}, {max_coords[1]:.2f}] Å")
    print(f"Z轴范围: [{min_coords[2]:.2f}, {max_coords[2]:.2f}] Å")
    print("-" * 20)
    print(f"包含 {PERCENTILE}% 数据的边界框 (X, Y, Z):")
    print(f"  下界: [{lower_bound[0]:.2f}, {lower_bound[1]:.2f}, {lower_bound[2]:.2f}]")
    print(f"  上界: [{upper_bound[0]:.2f}, {upper_bound[1]:.2f}, {upper_bound[2]:.2f}]")
    print("-" * 20)
    
    # --- 给出最终建议 ---
    # 为了保证画布是立方体且关于原点对称，我们取所有边界中的最大绝对值
    recommended_bound = np.max(np.abs(np.concatenate([lower_bound, upper_bound])))
    # 稍微留一些余量，比如向上取整
    final_bound = np.ceil(recommended_bound)

    print("\n--- 最终建议 ---")
    print(f"为了保证对称并覆盖 {PERCENTILE}% 的数据，建议的最大坐标值为: {final_bound:.1f} Å")
    print(f"您可以将 train.py 中的 GRID_BOUNDS 参数设置为: (-{final_bound:.1f}, {final_bound:.1f})")
    print("=" * 50)

if __name__ == "__main__":
    main()