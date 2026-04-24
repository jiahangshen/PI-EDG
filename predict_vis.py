import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from ase.io import read as ase_read
from cube_tools import cube as CubeToolzLoader 
from train import DensityModel 
from matplotlib.colors import LogNorm 

def create_coordinate_grid(grid_res: int, device: torch.device = torch.device("cpu")):
    coords = torch.linspace(-1.0, 1.0, grid_res, device=device)
    grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1)
    return grid

def positional_encoding(coords: torch.Tensor, num_freqs: int):
    freqs = 2.0 ** torch.arange(num_freqs, device=coords.device)
    scaled_coords = coords.unsqueeze(-1) * freqs
    sines = torch.sin(scaled_coords)
    cosines = torch.cos(scaled_coords)
    encoded = torch.cat([sines, cosines], dim=-1)
    return encoded.flatten(start_dim=-2)

def align_molecule(coords: np.ndarray):
    centroid = coords.mean(axis=0)
    centered_coords = coords - centroid
    covariance_matrix = np.cov(centered_coords, rowvar=False)
    if np.allclose(covariance_matrix, 0): return np.eye(3), centroid
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        return eigenvectors.T, centroid
    except np.linalg.LinAlgError:
         return np.eye(3), centroid
#sigmoid / log
def voxelize_structure(
    atom_coords: np.ndarray,
    atom_z_numbers: np.ndarray,
    grid_res: int,
    grid_bounds: tuple,
    atom_map: dict,
    sigma: float = 0.8  
) -> torch.Tensor:
    grid_min, grid_max = grid_bounds
    num_channels = len(atom_map)
    voxel_grid = torch.zeros((num_channels, grid_res, grid_res, grid_res), dtype=torch.float32)
    voxel_size = (grid_max - grid_min) / grid_res
    if voxel_size <= 0: return voxel_grid.to(torch.float16)

    coords_1d = torch.linspace(grid_min + voxel_size/2, grid_max - voxel_size/2, grid_res)
    grid_z, grid_y, grid_x = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    voxel_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1) # shape: [res, res, res, 3]
    
    for i in range(atom_coords.shape[0]):
        atom_z = int(atom_z_numbers[i])
        if atom_z in atom_map:
            channel_idx = atom_map[atom_z]
            atom_pos = torch.from_numpy(atom_coords[i]).float()
            distances_sq = torch.sum((voxel_coords - atom_pos) ** 2, dim=-1)
            gaussian_blob = torch.exp(-distances_sq / (2 * (sigma ** 2)))
            voxel_grid[channel_idx] = torch.maximum(voxel_grid[channel_idx], gaussian_blob)
            
    return voxel_grid.to(torch.float16) 

def voxelize_density_integrate(
    cube_data: np.ndarray, cube_origin: np.ndarray,
    step_vec_x_bohr: np.ndarray, step_vec_y_bohr: np.ndarray, step_vec_z_bohr: np.ndarray,
    grid_res: int, grid_bounds: tuple,
    rotation_matrix: np.ndarray, centroid: np.ndarray
) -> torch.Tensor:
    grid_min_ang, grid_max_ang = grid_bounds
    target_voxel_grid = torch.zeros((grid_res, grid_res, grid_res), dtype=torch.float32)
    target_voxel_size_ang = (grid_max_ang - grid_min_ang) / grid_res
    if target_voxel_size_ang <= 0: return target_voxel_grid.unsqueeze(0).to(torch.float16)
    BOHR_TO_ANGSTROM = 0.529177210903
    origin_ang = cube_origin * BOHR_TO_ANGSTROM
    nx, ny, nz = cube_data.shape
    vec_x_ang = step_vec_x_bohr * BOHR_TO_ANGSTROM
    vec_y_ang = step_vec_y_bohr * BOHR_TO_ANGSTROM
    vec_z_ang = step_vec_z_bohr * BOHR_TO_ANGSTROM
    i_indices, j_indices, k_indices = np.indices((nx, ny, nz))
    all_coords_ang = origin_ang + \
                     i_indices.flatten()[:, np.newaxis] * vec_x_ang + \
                     j_indices.flatten()[:, np.newaxis] * vec_y_ang + \
                     k_indices.flatten()[:, np.newaxis] * vec_z_ang
    aligned_all_coords_ang = (all_coords_ang - centroid) @ rotation_matrix
    target_all_indices = ((aligned_all_coords_ang - grid_min_ang) / target_voxel_size_ang).astype(int)
    valid_mask = np.all((target_all_indices >= 0) & (target_all_indices < grid_res), axis=1)
    valid_target_indices = target_all_indices[valid_mask]
    valid_density_values = cube_data.flatten()[valid_mask]
    if len(valid_target_indices) > 0:
        vz, vy, vx = valid_target_indices[:, 2], valid_target_indices[:, 1], valid_target_indices[:, 0]
        target_voxel_grid.index_put_(
            (torch.from_numpy(vz), torch.from_numpy(vy), torch.from_numpy(vx)),
            torch.from_numpy(valid_density_values).float(), accumulate=True )
    return target_voxel_grid.unsqueeze(0)


CHECKPOINT_PATH = Path("./checkpoints_late_fusion/best-model-epoch=01-score=val_overall_score=0.9989.ckpt")
RAW_DATA_ROOT = Path("./raw_data_cube")
SAMPLE_INDEX = "3032"

GRID_RESOLUTION = 64
GRID_BOUNDS = (-10.0, 10.0)
NUM_PE_FREQS = 4


ATOM_Z_TO_CHANNEL = {1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 16: 6, 17: 7, 35: 8}
ATOM_COLOR_MAP = {
    1: 'white',      # H - 氢
    5: 'tan',        # B - 硼
    6: 'gray',       # C - 碳
    7: 'blue',       # N - 氮
    8: 'red',        # O - 氧
    9: 'green',      # F - 氟
    15: 'orange',    # P - 磷
    16: 'yellow',    # S - 硫
    17: 'lightgreen',# Cl - 氯
    35: 'darkred',   # Br - 溴
}
DEFAULT_ATOM_COLOR = 'pink'
ISO_LEVEL = 0
ERROR_THRESHOLD = 0

def main():
    print(f"正在加载模型: {CHECKPOINT_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DensityModel.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH, map_location=device)
    model.to(device)
    model.eval()
    
    print(f"正在为分子索引 '{SAMPLE_INDEX}' 在线预处理数据...")
    
    sample_dir = RAW_DATA_ROOT / SAMPLE_INDEX
    try:
        sdf_file_path = next(sample_dir.glob("*.sdf"))
        cube_file_path = next(sample_dir.glob("*.cube"))
    except StopIteration:
        print(f"错误: 在目录 {sample_dir} 中找不到 .sdf 或 .cube 文件。")
        return

    try:
        atoms_ase = ase_read(str(sdf_file_path), format='sdf')
        atom_coords = atoms_ase.get_positions()
        atom_z_numbers = atoms_ase.get_atomic_numbers()
        
        cube_obj = CubeToolzLoader(str(cube_file_path))
        density_data = cube_obj.data
        density_data =density_data.astype(np.float32)
        cube_origin_bohr = cube_obj.origin
        step_vecs_bohr = [cube_obj.X, cube_obj.Y, cube_obj.Z]
        centroid = atom_coords.mean(axis=0)
        centered_atom_coords = atom_coords - centroid
        rotation_matrix = np.eye(3)
        iso_level = 0.0
        aligned_atom_coords = (atom_coords - centroid) @ rotation_matrix 

        atom_grid = voxelize_structure(
            aligned_atom_coords, atom_z_numbers,
            GRID_RESOLUTION, GRID_BOUNDS, ATOM_Z_TO_CHANNEL
        )
        
        target_tensor = voxelize_density_integrate(
            density_data, cube_origin_bohr,
            step_vecs_bohr[0], step_vecs_bohr[1], step_vecs_bohr[2],
            GRID_RESOLUTION, GRID_BOUNDS,
            rotation_matrix, centroid 
        )
    
        print("在线预处理完成。")
        
    except Exception as e:
        print(f"预处理索引 {SAMPLE_INDEX} 时发生严重错误: {e}")
        return

    print("正在进行预测...")
    with torch.no_grad():
        atom_grid_batch = atom_grid.to(device).float().unsqueeze(0)
        coord_grid = create_coordinate_grid(GRID_RESOLUTION, device=device)
        pe_grid = positional_encoding(coord_grid, NUM_PE_FREQS)
        pe_grid_batch = pe_grid.permute(3, 0, 1, 2).float().unsqueeze(0)
        
        # prediction_log_tensor 是模型在对数空间的原始输出
        prediction_log_tensor = model(atom_grid_batch, pe_grid_batch)

    # --- 核心修改：将预测结果从对数空间转换回真实物理空间 ---
    prediction_tensor = torch.expm1(prediction_log_tensor)
    # 确保转换后的值为非负
    prediction_tensor = torch.relu(prediction_tensor)
    # -------------------------------------------------------------

    # 现在 prediction_grid 和 target_grid 都在真实物理空间中，可以进行有意义的比较和可视化
    prediction_grid = prediction_tensor.squeeze().cpu().numpy()
    target_grid = target_tensor.squeeze().cpu().numpy() # target_tensor 本来就是真实密度
    
    # difference_grid 的计算现在是正确的
    difference_grid = np.abs(prediction_grid - target_grid)
    
    print("正在生成三维可视化图像...")
    fig = plt.figure(figsize=(24, 8))
    # 1. 定义一个统一的显示阈值，过滤掉噪音
    display_threshold = 1e-6 

    # 2. 从 target_grid 和 prediction_grid 中找出所有高于阈值的点
    target_display_values = target_grid[target_grid > display_threshold]
    pred_display_values = prediction_grid[prediction_grid > display_threshold]

    # 3. 确定全局的 vmin 和 vmax
    #    检查是否有任何点高于阈值，以避免在空数据上调用 .min()/.max()
    if len(target_display_values) > 0 or len(pred_display_values) > 0:
        # 将两组值合并，然后找到全局的最小和最大值
        all_display_values = np.concatenate([target_display_values, pred_display_values])
        global_vmin = all_display_values.min()
        global_vmax = all_display_values.max()
    else:
        # 如果两个网格都没有高于阈值的点，设置一个默认范围
        global_vmin = display_threshold
        global_vmax = 1.0

    print(f"为密度图设置统一的颜色范围 (Log Scale): vmin={global_vmin:.2e}, vmax={global_vmax:.2e}")
    # iso_level 的计算现在也是基于真实密度，是正确的
    iso_level_target = np.percentile(target_grid[target_grid > 0.001], 5) if np.any(target_grid > 0.001) else 0.01
    iso_level_pred = np.percentile(prediction_grid[prediction_grid > 0.001], 5) if np.any(prediction_grid > 0.001) else 0.01

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    # 4. 将全局范围和阈值传递给绘图函数
    plot_density_scatter(ax1, target_grid, "Ground Truth Density (Scatter)", fig, 
                         aligned_atom_coords, atom_z_numbers, GRID_BOUNDS,
                         vmin=global_vmin, vmax=global_vmax, threshold=display_threshold) # <-- 传递新参数
    
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    # 5. 对预测图也使用相同的全局范围和阈值
    plot_density_scatter(ax2, prediction_grid, "Predicted Density (Scatter)", fig, 
                         aligned_atom_coords, atom_z_numbers, GRID_BOUNDS,
                         vmin=global_vmin, vmax=global_vmax, threshold=display_threshold) # <-- 传递新参数
    
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    plot_error_scatter(ax3, difference_grid, "Absolute Prediction Error", fig, 
                       aligned_atom_coords, atom_z_numbers, GRID_BOUNDS)


    plt.tight_layout(pad=1.0)
    output_filename = f"online_analysis_scatter_{SAMPLE_INDEX}.png"
    plt.savefig(output_filename, dpi=150)
    print(f"\n可视化图像已成功保存为: {output_filename}")
    analyze_mae_by_magnitude(target_grid, prediction_grid)
def analyze_mae_by_magnitude(target_grid, prediction_grid):
    """
    根据真实电子密度的数量级，分区间计算 MAE 和其他统计数据。
    """
    print("\n--- 按数量级分析平均绝对误差 (MAE) ---")
    
    bins = [0, 1e-14,1e-10,1e-7,1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0,100.0,1000.0,np.inf]
    
    absolute_errors = np.abs(prediction_grid - target_grid)
    

    bin_indices = np.digitize(target_grid, bins)
    
    print(f"{'Density Range':>25} | {'Points Count':>15} | {'% of Total':>12} | {'MAE':>15}")
    print("-" * 75)
    
    total_points = target_grid.size
    
    # 4. 遍历每个区间进行计算和打印
    for i in range(1, len(bins)):
        mask = (bin_indices == i)
        
        points_in_bin = np.sum(mask)
        
        if points_in_bin == 0:
            continue
            
        mae_in_bin = np.mean(absolute_errors[mask])
        
        range_label = f"({bins[i-1]:.1e}, {bins[i]:.1e}]"
        if bins[i-1] == 0:
            range_label = f"[0, {bins[i]:.1e}]" 
        
        percentage = (points_in_bin / total_points) * 100
        
        print(f"{range_label:>25} | {points_in_bin:>15,} | {percentage:11.4f}% | {mae_in_bin:15.4e}")

    print("-" * 75)
    overall_mae = np.mean(absolute_errors)
    print(f"{'Overall MAE':>55} | {overall_mae:15.4e}")
def plot_density_scatter(ax, grid_data, title, fig, atom_coords, atom_z_numbers, grid_bounds, 
                         vmin=None, vmax=None, threshold=1e-6): 

    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)
    
    z, y, x = np.where(grid_data > threshold)
    density_values = grid_data[z, y, x]
    
    if len(x) == 0:
        ax.text(0.5, 0.5, 0.5, f"No density > {threshold:.1e}", transform=ax.transAxes, ha='center', va='center')
        ax.set_xlim(0, grid_data.shape[2]); ax.set_ylim(0, grid_data.shape[1]); ax.set_zlim(0, grid_data.shape[0])
        return

    if vmin is None:
        vmin = density_values.min() if density_values.min() > 0 else threshold
    if vmax is None:
        vmax = density_values.max()

    scatter = ax.scatter(x, y, z, 
                         c=density_values, 
                         norm=LogNorm(vmin=vmin, vmax=vmax), 
                         cmap='Spectral_r',
                         s=(density_values / vmax) * 10 + 0.01, 
                         alpha=0.5, 
                         depthshade=False)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Density Value (Log Scale)")
    
    grid_min, grid_max = grid_bounds
    grid_res = grid_data.shape[0]
    voxel_size = (grid_max - grid_min) / grid_res
    atom_indices = (atom_coords - grid_min) / voxel_size
    atom_colors = [ATOM_COLOR_MAP.get(z, DEFAULT_ATOM_COLOR) for z in atom_z_numbers]
    ax.scatter(atom_indices[:, 0], atom_indices[:, 1], atom_indices[:, 2], 
               s=150, c=atom_colors, alpha=0.9, edgecolors='black', linewidth=0.8, zorder=10)

    ax.set_xlim(0, grid_data.shape[2])
    ax.set_ylim(0, grid_data.shape[1])
    ax.set_zlim(0, grid_data.shape[0])

def plot_error_scatter(ax, diff_grid, title, fig, atom_coords, atom_z_numbers, grid_bounds):
    """
    使用三维散点图可视化误差，突出显示异常值。
    """
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)

    if np.any(diff_grid > 0):
        error_threshold = np.percentile(diff_grid[diff_grid > 0], 50)
    else:
        error_threshold = 1e-6
    
    z, y, x = np.where(diff_grid > error_threshold)
    error_values = diff_grid[z, y, x]
    
    if len(x) == 0:
        ax.text(0.5, 0.5, 0.5, "No significant error", transform=ax.transAxes, ha='center', va='center')
        ax.set_xlim(0, diff_grid.shape[2]); ax.set_ylim(0, diff_grid.shape[1]); ax.set_zlim(0, diff_grid.shape[0])
        return

    min_positive_error = error_values.min() if error_values.min() > 0 else 1e-9

    scatter = ax.scatter(x, y, z, 
                         c=error_values, 
                         norm=LogNorm(vmin=min_positive_error, vmax=error_values.max()),
                         cmap='Reds',
                         s=(error_values / error_values.max()) * 5 + 0.0001, 
                         alpha=0.7,
                         depthshade=False)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Absolute Error (Log Scale)")
    
    grid_min, grid_max = grid_bounds
    grid_res = diff_grid.shape[0]
    voxel_size = (grid_max - grid_min) / grid_res
    atom_indices = (atom_coords - grid_min) / voxel_size
    atom_colors = [ATOM_COLOR_MAP.get(z, DEFAULT_ATOM_COLOR) for z in atom_z_numbers]
    ax.scatter(atom_indices[:, 0], atom_indices[:, 1], atom_indices[:, 2], 
               s=150, c=atom_colors, alpha=0.9, edgecolors='black', linewidth=0.8, zorder=10)

    ax.set_xlim(0, diff_grid.shape[2])
    ax.set_ylim(0, diff_grid.shape[1])
    ax.set_zlim(0, diff_grid.shape[0])

if __name__ == "__main__":
    main()