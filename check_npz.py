import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ======================================================
# 1. 配置参数 (请仔细修改！)
# ======================================================

# --- 缓存文件路径 ---
# !! 核心修改：指向存储“无对齐”数据的缓存目录 !!
CACHE_DIR = Path("./preprocessed_cache")

# --- 样本索引 ---
# 选择一个您想检查的分子索引
SAMPLE_INDEX = "0"  # <-- 替换成您想检查的任何一个样本的索引

# ======================================================
# 2. 主检查函数
# ======================================================
def main():
    """
    主函数，加载并检查指定的 .npz 缓存文件 (无对齐版本)。
    """
    npz_file_path = CACHE_DIR / f"{SAMPLE_INDEX}.npz"

    if not npz_file_path.exists():
        print(f"错误: 找不到缓存文件 -> {npz_file_path}")
        print("请确认 CACHE_DIR 和 SAMPLE_INDEX 设置是否正确。")
        return

    print(f"正在加载并检查缓存文件: {npz_file_path}")

    try:
        # --- a. 加载 .npz 文件 ---
        data = np.load(npz_file_path)
        
        # --- b. 打印文件内容和统计信息 ---
        print("\n" + "="*50)
        print("文件内容 (数组名称):")
        print(list(data.keys()))
        print("="*50)

        for key in data.keys():
            array = data[key]
            print(f"\n--- 数组: '{key}' ---")
            print(f"  - 数据类型 (dtype): {array.dtype}")
            print(f"  - 形状 (shape): {array.shape}")
            if np.issubdtype(array.dtype, np.floating):
                print(f"  - 最大值 (max): {np.max(array):.4f}")
                print(f"  - 最小值 (min): {np.min(array):.4f}")
            print(f"  - 是否包含 NaN: {np.isnan(array).any()}")
            print(f"  - 是否包含 Inf: {np.isinf(array).any()}")
        
        # --- c. 可视化坐标 ---
        print("\n正在生成三维坐标可视化图像...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # --- !! 核心修改：使用正确的键名 !! ---
        # 绘制原子坐标 (红色，大一点)
        if 'atom_coords' in data: # <-- 键名从 aligned_atom_coords 改为 atom_coords
            atom_coords = data['atom_coords']
            ax.scatter(atom_coords[:, 0], atom_coords[:, 1], atom_coords[:, 2], 
                       c='red', s=50, label='Atoms (Original)', depthshade=True)
        else:
            print("警告: 缓存文件中未找到 'atom_coords'。")

        # 绘制电子密度点云 (蓝色，小一点)
        if 'density_coords' in data: # <-- 键名从 aligned_density_coords 改为 density_coords
            density_coords = data['density_coords']
            subsample_rate = max(1, len(density_coords) // 5000)
            density_coords_sub = density_coords[::subsample_rate]
            
            ax.scatter(density_coords[:, 0], density_coords[:, 1], density_coords[:, 2], 
                       c='blue', s=1, label=f'Density Points (1/{subsample_rate})', alpha=0.3)
        else:
            print("警告: 缓存文件中未找到 'density_coords'。")
        # --- 修改结束 ---
            
        ax.set_title(f"Visualization of Cached Data (No Alignment) for Index {SAMPLE_INDEX}")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend()
        plt.axis('equal')

        # 保存图像
        output_filename = f"debug_cache_no_align_idx_{SAMPLE_INDEX}.png"
        plt.savefig(output_filename, dpi=150)
        print(f"\n可视化快照已成功保存为: {output_filename}")


    except Exception as e:
        print(f"\n检查文件时发生严重错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
