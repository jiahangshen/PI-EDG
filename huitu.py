import numpy as np
import pyvista as pv
from ase.io.cube import read_cube_data

pv.start_xvfb()

# ==========================================
# 1. 数据读取
# ==========================================
cube_path = 'raw_data_cube/15/Mol1_Dt.cube'

with open(cube_path, 'r') as f:
    cube_result = read_cube_data(f)
    if isinstance(cube_result, tuple):
        grid_data, cube_atoms = cube_result
    else:
        grid_data = cube_result['data']
        cube_atoms = cube_result['atoms']
grid_data = grid_data * 0.95
# ==========================================
# 2. 核心数学转换
# ==========================================
epsilon = 1e-14  
safe_grid_data = np.clip(grid_data, a_min=epsilon, a_max=None)
log_grid_data = np.log10(safe_grid_data)

# ==========================================
# 3. 构建网格
# ==========================================
grid = pv.ImageData()
grid.dimensions = np.array(grid_data.shape)

cell = cube_atoms.get_cell()
grid_lengths = np.diag(cell)
grid.spacing = grid_lengths / grid.dimensions
grid.origin = (0.0, 0.0, 0.0) 

grid.point_data["Log_Density"] = log_grid_data.flatten(order="F")

# ==========================================
# 4. 渲染可视化 (高颜值版)
# ==========================================
plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])

# 稍微提高底线，滤掉绝对的真空背景噪点
log_vmin = np.log10(1e-34) 
log_vmax = log_grid_data.max()

# 【视觉核心 1】：S形透明度曲线 (剥离雾霾)
# 前几个 0.0 确保低密度区域完全透明，后半段才迅速变得不透明，突显核心结构
s_curve_opacity = [0.01, 0.13, 0.15, 0.15, 0.2, 0.6, 1.0]

plotter.add_volume(grid, 
                   scalars="Log_Density", 
                   cmap="coolwarm",  # 推荐使用 magma 或 plasma，非常有高级科研图表的质感
                   clim=[log_vmin, log_vmax], 
                   opacity=s_curve_opacity, 
                   show_scalar_bar=False,
                   scalar_bar_args={"title": "Log10(Electron Density)", "color": "black"})

# 【视觉核心 2】：柔和、不遮挡的三维参考网格
# plotter.show_bounds(
#     grid='back',             # 关键：网格线只画在背面的墙上，不遮挡云雾！
#     location='outer', 
#     all_edges=False,         # 去掉生硬的纯黑实线外框
#     color='#BBBBBB',         # 使用柔和的浅灰色网格线
#     font_size=0,            # 刻度字体缩小一点，显得更精致
#     xlabel="X (Angstrom)", 
#     ylabel="Y (Angstrom)", 
#     zlabel="Z (Angstrom)"
# )

# 在左下角添加一个小型的 XYZ 方向指示器 (黑色)

# --- 最终设置 ---
plotter.set_background('white')

# 【视觉核心 3】：使用等轴测视角 (Isometric)
# 这是一个自带高级感的三维对角线视角，比自己手动调 elevation 和 azimuth 效果更好
plotter.camera_position = 'iso' 

print("正在服务器上进行离屏体积渲染 (高颜值网格版)...")

output_filename = 'volume_render_beautiful_small.png'
plotter.show(screenshot=output_filename)

print(f"渲染完成！图片已保存为 {output_filename}")
# import numpy as np
# import pyvista as pv
# from ase.io.cube import read_cube_data
# from rdkit import Chem
# pv.start_xvfb()
# # ==========================================
# # 0. 数据读取：获取真实的原子信息和网格空间
# # ==========================================
# sdf_path = 'raw_data_cube/15/Mol1.sdf'
# cube_path = 'raw_data_cube/15/Mol1_Dt.cube'

# # A. 从 SDF 获取真实原子序数 Z_A 和坐标 R_A
# mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
# conf = mol.GetConformer()
# R_A = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
# Z_A = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])

# # B. 从 Cube 获取真实的网格边界信息 (为了完美对齐)
# with open(cube_path, 'r') as f:
#     grid_data, cube_atoms = read_cube_data(f)

# # 获取网格维度形状 (Nx, Ny, Nz)
# grid_shape = grid_data.shape
# # 获取网格的三维物理长度 (Angstrom/Bohr)
# cell = cube_atoms.get_cell()
# grid_lengths = np.diag(cell)
# grid_origin = np.array([0.0, 0.0, 0.0]) # Cube 通常原点有偏移，简单起见设为0或读取真实原点

# # 生成真实物理空间的三维坐标网格
# x_lin = np.linspace(grid_origin[0], grid_origin[0] + grid_lengths[0], grid_shape[0])
# y_lin = np.linspace(grid_origin[1], grid_origin[1] + grid_lengths[1], grid_shape[1])
# z_lin = np.linspace(grid_origin[2], grid_origin[2] + grid_lengths[2], grid_shape[2])
# X, Y, Z = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')

# # ==========================================
# # 核心通用渲染函数
# # ==========================================
# def render_physics_channel(volume_data, output_name, cmap, opacity, title):
#     grid = pv.ImageData()
#     grid.dimensions = np.array(grid_shape)
#     grid.spacing = grid_lengths / grid.dimensions
#     grid.origin = grid_origin
#     grid.point_data["Feature"] = volume_data.flatten(order="F")

#     plotter = pv.Plotter(off_screen=True, window_size=[1000, 1000])
#     plotter.show_bounds(all_edges=True, color='black', font_size=10)
    
#     plotter.add_volume(grid, scalars="Feature", cmap=cmap, opacity=opacity, show_scalar_bar=False)
    
#     # 自动调整到一个看分子比较清楚的经典视角
#     plotter.camera_position = 'iso'
#     plotter.screenshot(output_name, transparent_background=True)
#     print(f"[{title}] 图片已保存为: {output_name}")
#     plotter.close()


# # ==========================================
# # 1. 生成 (a) Gaussian Atom Grid
# # 公式: sum_A Z_A * exp(-beta * |r - R_A|^2)
# # ==========================================
# print("正在计算 Gaussian Atom Grid...")
# gaussian_grid = np.zeros_like(X)
# beta = 2.0 # 控制高斯分布的宽度，可根据论文调节
# for pos, z_val in zip(R_A, Z_A):
#     r_sq = (X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2
#     gaussian_grid += z_val * np.exp(-beta * r_sq)

# # 渲染 (原图风格：蓝色，外围透明)
# opacity_gauss = [0.01, 0.05, 0.2, 0.6, 1.0]
# render_physics_channel(gaussian_grid, 'real_a_gaussian.png', 'Spectral_r', opacity_gauss, "Gaussian Grid")


# # ==========================================
# # 2. 生成 (b) External Potential
# # 公式: sum_A Z_A / |r - R_A| (通常取负，这里取绝对值方便可视化)
# # ==========================================
# print("正在计算 External Potential Grid...")
# potential_grid = np.zeros_like(X)
# epsilon_pot = 0.5 # 软化因子，防止核心处无限大
# for pos, z_val in zip(R_A, Z_A):
#     r_dist = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2)
#     potential_grid += z_val / (r_dist + epsilon_pot)

# # 渲染 (原图风格：绿色，看起来更实体)
# opacity_pot = [0.01, 0.15, 0.3, 0.6, 1.0]
# render_physics_channel(potential_grid, 'real_b_potential.png', 'YlGn', opacity_pot, "Potential Grid")


# # ==========================================
# # 3. 生成 (c) Nuclear Charge Peaks
# # 极度尖锐的电荷峰，体现奇异性
# # ==========================================
# print("正在计算 Nuclear Charge Peaks Grid...")
# peaks_grid = np.zeros_like(X)
# epsilon_peak = 0.05 # 极小的软化因子，让峰非常尖锐
# for pos, z_val in zip(R_A, Z_A):
#     r_dist = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2)
#     # 用平方反比可以更加突出中心峰值的尖锐度
#     peaks_grid += z_val / (r_dist**2 + epsilon_peak)

# # 渲染 (原图风格：红橙色，只在峰值处不透明)
# # 这个数组需要极度偏向右侧，即只有最高值才显示实体
# opacity_peaks = [0.03, 0.5, 1.0] 
# render_physics_channel(peaks_grid, 'real_c_peaks.png', 'YlOrRd', opacity_peaks, "Charge Peaks")

# print("所有真实分子特征图生成完毕！")