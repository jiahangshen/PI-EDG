# import os
# import numpy as np
# from pathlib import Path
# from ase.io import read as ase_read
# from tqdm import tqdm

# try:
#     import pyscf
#     from pyscf import gto, dft
# except ImportError:
#     raise ImportError("请先安装 PySCF: pip install pyscf")

# # ================= 配置区域 =================
# RAW_DATA_ROOT = Path("./raw_data_cube")
# # === 新增：将生成的 SAD 存放到专门的独立目录 ===
# SAD_OUTPUT_DIR = Path("./preprocessed_sad") 
# GRID_RESOLUTION = 64
# GRID_BOUNDS = (-10.0, 10.0)
# BOHR_TO_ANGSTROM = 0.529177210903
# # ===========================================

# def get_grid_coords_numpy(grid_res, grid_bounds):
#     grid_min, grid_max = grid_bounds
#     x = np.linspace(grid_min, grid_max, grid_res)
#     X, Y, Z = np.meshgrid(x, x, x, indexing='ij') 
#     coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
#     return coords

# def generate_sad_for_molecule(atom_coords_ang, atom_symbols, grid_coords_ang):
#     atom_str = ""
#     for sym, coord in zip(atom_symbols, atom_coords_ang):
#         atom_str += f"{sym} {coord[0]} {coord[1]} {coord[2]}; "
        
#     mol = gto.Mole()
#     mol.atom = atom_str
#     mol.basis = 'def2-svp' 
#     mol.verbose = 0        
    
#     try:
#         mol.build(dump_input=False, parse_arg=False)
#     except Exception as e:
#         print(f"  [警告] PySCF Build 警告 (已忽略): {e}")
        
#     dm_sad = pyscf.scf.hf.init_guess_by_atom(mol)
#     grid_coords_bohr = grid_coords_ang / BOHR_TO_ANGSTROM
#     ao_value = dft.numint.eval_ao(mol, grid_coords_bohr, deriv=0)
#     rho_sad_flat = dft.numint.eval_rho(mol, ao_value, dm_sad, xctype='LDA')
#     rho_sad_3d = rho_sad_flat.reshape(GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION)
#     return np.clip(rho_sad_3d, a_min=0.0, a_max=None)

# def main():
#     grid_coords = get_grid_coords_numpy(GRID_RESOLUTION, GRID_BOUNDS)
    
#     # 确保主输出目录存在
#     SAD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
#     sample_dirs =[d for d in RAW_DATA_ROOT.iterdir() if d.is_dir()]
#     print(f"开始为 {len(sample_dirs)} 个分子生成 SAD 网格先验...")
#     print(f"SAD 数据将保存在: {SAD_OUTPUT_DIR.absolute()}")
    
#     for sample_dir in tqdm(sample_dirs):
#         sample_id = sample_dir.name
        
#         # === 核心修改：将保存路径指向新的输出目录 ===
#         # 在 preprocessed_sad 下为每个样本创建独立文件夹
#         sample_out_dir = SAD_OUTPUT_DIR / sample_id
#         sample_out_dir.mkdir(parents=True, exist_ok=True)
#         sad_save_path = sample_out_dir / "sad_grid.npy"
#         # ===============================================
        
#         if sad_save_path.exists():
#             continue
            
#         try:
#             sdf_file = next(sample_dir.glob("*.sdf"))
#             atoms = ase_read(str(sdf_file), format='sdf')
            
#             coords = atoms.get_positions()
#             centroid = coords.mean(axis=0)
#             centered_coords = coords - centroid
#             symbols = atoms.get_chemical_symbols()
            
#             sad_grid = generate_sad_for_molecule(centered_coords, symbols, grid_coords)
#             np.save(sad_save_path, sad_grid.astype(np.float32))
            
#         except Exception as e:
#             print(f"\n处理样本 {sample_id} 时出错: {e}")

# if __name__ == "__main__":
#     main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

# ================= 配置区域 =================
SAD_OUTPUT_DIR = Path("./preprocessed_sad")
# ===========================================

def inspect_one_sample():
    # 1. 寻找第一个可用的 .npy 文件
    sad_files = list(SAD_OUTPUT_DIR.glob("**/sad_grid.npy"))
    
    if not sad_files:
        print(f"错误: 在 {SAD_OUTPUT_DIR} 目录下没有找到任何 sad_grid.npy 文件。")
        return

    sample_path = sad_files[1]
    sample_id = sample_path.parent.name
    print(f"正在检查样本: {sample_id}")
    print(f"文件路径: {sample_path}")

    # 2. 加载数据
    data = np.load(sample_path)

    # 3. 打印统计数值
    print("\n--- 统计信息 ---")
    print(f"数据形状: {data.shape}")
    print(f"最大值 (Max Density): {data.max():.6e}")
    print(f"最小值 (Min Density): {data.min():.6e}")
    print(f"平均值 (Mean):        {data.mean():.6e}")
    print(f"非零点占比:           {(data > 1e-34).sum() / data.size * 100:.2f}%")

    # 4. 可视化切片
    # 选取中心索引
    mid = data.shape[0] // 2
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- 左图：线性尺度 (反映原子核位置) ---
    im1 = axes[0].imshow(data[mid, :, :], cmap='Spectral_r')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title(f"Linear Scale (XY Slice at z={mid})")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    # --- 右图：对数尺度 (反映电子云衰减) ---
    # 使用 1e-10 作为绘图下限，防止 log(0) 报错
    vmin = max(data.min(), 1e-10)
    vmax = data.max()
    
    im2 = axes[1].imshow(data[mid, :, :], cmap='Spectral_r', 
                         norm=LogNorm(vmin=vmin, vmax=vmax))
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title(f"Log Scale (XY Slice at z={mid})")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

    plt.tight_layout()
    
    # 保存预览图
    save_path = f"sad_preview_{sample_id}.png"
    plt.savefig(save_path)
    print(f"\n可视化预览图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    inspect_one_sample()