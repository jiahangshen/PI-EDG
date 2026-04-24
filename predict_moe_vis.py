import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read as ase_read
from cube_tools import cube as CubeToolzLoader 
from matplotlib.colors import LogNorm 
from train_moe import MixtureOfExpertsModel 
from scipy.stats import pearsonr # <--- 请确保在您的脚本顶部添加此导入

def analyze_metrics_by_magnitude(target_grid, prediction_grid):
    """
    根据真实电子密度的数量级，分区间计算 MAE、皮尔逊相关系数和其他统计数据。
    """
    print("\n--- 按数量级分析预测指标 ---")
    bins = [0, 1e-14, 1e-10, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, np.inf]
    
    absolute_errors = np.abs(prediction_grid - target_grid)
    
    # 使用 target_grid 的值来确定每个点属于哪个区间
    bin_indices = np.digitize(target_grid, bins)
    
    # 更新表头
    print(f"{'Density Range':>25} | {'Points Count':>15} | {'% of Total':>12} | {'MAE':>15} | {'Pearson Coeff':>15}")
    print("-" * 93) # 调整分隔线长度
    
    total_points = target_grid.size
    
    for i in range(1, len(bins)):
        mask = (bin_indices == i)
        points_in_bin = np.sum(mask)
        
        # 计算皮尔逊系数至少需要2个点
        if points_in_bin <= 1:
            continue
            
        # 计算区间内的 MAE
        mae_in_bin = np.mean(absolute_errors[mask])
        
        # --- 新增：计算区间内的皮尔逊系数 ---
        pearson_in_bin = np.nan # 默认值为 NaN
        try:
            # 提取该区间内的真实值和预测值
            target_values_in_bin = target_grid[mask]
            prediction_values_in_bin = prediction_grid[mask]
            
            # 计算皮尔逊系数
            p_corr, _ = pearsonr(target_values_in_bin, prediction_values_in_bin)
            pearson_in_bin = p_corr
        except ValueError:
            # 如果一个数组中的所有值都相同（例如，区间内所有值都为0），
            # pearsonr会抛出ValueError，此时相关性无定义。
            pearson_in_bin = np.nan
        # --- 结束新增部分 ---
            
        range_label = f"({bins[i-1]:.1e}, {bins[i]:.1e}]"
        if bins[i-1] == 0:
            range_label = f"[0, {bins[i]:.1e}]" 
        
        percentage = (points_in_bin / total_points) * 100
        
        # 更新打印行，添加皮尔逊系数值
        print(f"{range_label:>25} | {points_in_bin:>15,} | {percentage:11.4f}% | {mae_in_bin:15.4e} | {pearson_in_bin:15.6f}")

    print("-" * 93)
    
    # 计算总体的 MAE 和 皮尔逊系数
    overall_mae = np.mean(absolute_errors)
    overall_pearson, _ = pearsonr(target_grid.flatten(), prediction_grid.flatten())
    
    # 更新总结行
    print(f"{'Overall':>72} | {overall_mae:15.4e} | {overall_pearson:15.6f}")
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
    voxel_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)
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
    centroid: np.ndarray
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
    aligned_all_coords_ang = (all_coords_ang - centroid)
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


CHECKPOINT_PATH = Path("./checkpoints_moe/best-moe-model-epoch=37-iou=val\epoch_iou_macro=0.0000.ckpt") 
RAW_DATA_ROOT = Path("./raw_data_cube")
SAMPLE_INDEX = "2353"

GRID_RESOLUTION = 64
GRID_BOUNDS = (-10.0, 10.0)
NUM_PE_FREQS = 4

ATOM_Z_TO_CHANNEL = {1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 16: 6, 17: 7, 35: 8} 
ATOM_COLOR_MAP = {1: 'white', 6: 'gray', 7: 'blue', 8: 'red', 9: 'green', 16: 'yellow', 17: 'lightgreen', 35: 'darkred'}
DEFAULT_ATOM_COLOR = 'pink'


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
                         vmin=None, vmax=None, threshold=1e-14): 

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
def save_grid_to_cube(filename, grid_data, atoms_z_numbers, atoms_coords_ang, grid_bounds):
    """
    使用你的 cube_tools.py 脚本，将一个 NumPy 网格保存为 .cube 文件。

    Args:
        filename (str): 输出的 .cube 文件名 (例如 "prediction.cube")
        grid_data (np.ndarray): 形状为 [res, res, res] 的 3D 体素数据
        atoms_z_numbers (np.ndarray): 形状为 [N,] 的原子序数 (例如 [6, 1, 1, ...])
        atoms_coords_ang (np.ndarray): 形状为 [N, 3] 的原子坐标 (单位：埃)
        grid_bounds (tuple): 网格的边界, (min, max)，单位：埃
    """
    print(f"正在将网格数据保存到 {filename}...")
    
    # 1. 创建一个新的、空的 cube 对象
    # (这会调用 cube_tools.py 里的 __init__ 和 default_values)
    output_cube = CubeToolzLoader()

    # 2. 填充网格元数据
    grid_min, grid_max = grid_bounds
    grid_res = grid_data.shape[0] # 应该是 64
    voxel_size = (grid_max - grid_min) / grid_res
    
    # 将单位从埃 (Angstrom) 转换回玻尔 (Bohr)
    # 你的 cube_tools 脚本默认使用 Bohr
    BOHR_TO_ANGSTROM = 0.529177210903
    
    output_cube.natoms = len(atoms_z_numbers)
    # .cube 文件的原点是网格的角点，单位是 Bohr
    output_cube.origin = np.array([grid_min / BOHR_TO_ANGSTROM, 
                                   grid_min / BOHR_TO_ANGSTROM, 
                                   grid_min / BOHR_TO_ANGSTROM])
    
    output_cube.NX = grid_res
    output_cube.NY = grid_res
    output_cube.NZ = grid_res
    
    # .cube 文件的体素向量，单位是 Bohr
    voxel_size_bohr = voxel_size / BOHR_TO_ANGSTROM
    output_cube.X = np.array([voxel_size_bohr, 0.0, 0.0])
    output_cube.Y = np.array([0.0, voxel_size_bohr, 0.0])
    output_cube.Z = np.array([0.0, 0.0, voxel_size_bohr])

    # 3. 填充原子数据
    # cube_tools 需要原子的 *符号* (字符串) 和 *坐标* (Bohr)
    output_cube.atoms = [str(int(z)) for z in atoms_z_numbers] 
    output_cube.atomsXYZ = atoms_coords_ang / BOHR_TO_ANGSTROM
    
    # 4. 填充网格数据 (注意：cube_tools 期望数据单位是 Bohr^-3)
    # 你的模型预测的密度单位是什么？这里我们假设它与 target 一致。
    # **注意**: 你的 cube_tools.py 里的 read_cube 和 write_cube
    # 使用 (NX, NY, NZ) 索引，这与你的 (vz, vy, vx) 匹配。
    output_cube.data = grid_data

    # 5. 写入文件
    output_cube.write_cube(filename, comment=f"Grid data saved by script\n{filename}")
    
    print(f"成功保存到 {filename}")
def main():
    print(f"正在加载 MoE 模型: {CHECKPOINT_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_weights=torch.tensor([0.0261, 1.4925, 1.2437, 0.8265, 0.7202, 1.1831, 1.2085, 1.3151, 0.9841])
    model = MixtureOfExpertsModel.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH, map_location=device,class_weights=dummy_weights)
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
        density_data = cube_obj.data.astype(np.float32)
        cube_origin_bohr = cube_obj.origin
        step_vecs_bohr = [cube_obj.X, cube_obj.Y, cube_obj.Z]
        
        centroid = atom_coords.mean(axis=0)
        aligned_atom_coords = atom_coords - centroid

        atom_grid = voxelize_structure(
            aligned_atom_coords, atom_z_numbers,
            GRID_RESOLUTION, GRID_BOUNDS, ATOM_Z_TO_CHANNEL
        )
        
        target_tensor = voxelize_density_integrate(
            density_data, cube_origin_bohr,
            step_vecs_bohr[0], step_vecs_bohr[1], step_vecs_bohr[2],
            GRID_RESOLUTION, GRID_BOUNDS,
            centroid
        )
        print("在线预处理完成。")
        
    except Exception as e:
        print(f"预处理索引 {SAMPLE_INDEX} 时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return

    print("正在进行 MoE 预测...")
    with torch.no_grad():
        atom_grid_batch = atom_grid.to(device).float().unsqueeze(0)
        coord_grid = create_coordinate_grid(GRID_RESOLUTION, device=device)
        pe_grid = positional_encoding(coord_grid, NUM_PE_FREQS)
        pe_grid_batch = pe_grid.permute(3, 0, 1, 2).float().unsqueeze(0)
        logits, expert_outputs = model(atom_grid_batch, pe_grid_batch)
        pred_labels = torch.argmax(logits, dim=1)
        prediction_tensor = torch.gather(expert_outputs, 1, pred_labels.unsqueeze(1)).squeeze(1)

    prediction_grid = prediction_tensor.squeeze().cpu().numpy()
    target_grid = target_tensor.squeeze().cpu().numpy()
    difference_grid = np.abs(prediction_grid - target_grid)
    save_grid_to_cube(
        filename=f"prediction_{SAMPLE_INDEX}.cube",
        grid_data=prediction_grid,
        atoms_z_numbers=atom_z_numbers,
        atoms_coords_ang=aligned_atom_coords, # 使用对齐后的坐标
        grid_bounds=GRID_BOUNDS
    )
    print("正在生成三维可视化图像...")
    fig = plt.figure(figsize=(24, 8))
    
    display_threshold = 0 
    target_display_values = target_grid[target_grid > display_threshold]
    pred_display_values = prediction_grid[prediction_grid > display_threshold]
    if len(target_display_values) > 0 or len(pred_display_values) > 0:
        all_display_values = np.concatenate([target_display_values, pred_display_values])
        global_vmin, global_vmax = all_display_values.min(), all_display_values.max()
    else:
        global_vmin, global_vmax = display_threshold, 1.0

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    plot_density_scatter(ax1, target_grid, "Ground Truth Density", fig, 
                         aligned_atom_coords, atom_z_numbers, GRID_BOUNDS,
                         vmin=global_vmin, vmax=global_vmax, threshold=display_threshold)
    
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    plot_density_scatter(ax2, prediction_grid, "MoE Predicted Density", fig, 
                         aligned_atom_coords, atom_z_numbers, GRID_BOUNDS,
                         vmin=global_vmin, vmax=global_vmax, threshold=display_threshold)
    
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    plot_error_scatter(ax3, difference_grid, "Absolute Prediction Error", fig, 
                       aligned_atom_coords, atom_z_numbers, GRID_BOUNDS)

    plt.tight_layout(pad=1.0)
    output_filename = f"moe_analysis_scatter_{SAMPLE_INDEX}.png"
    plt.savefig(output_filename, dpi=150)
    print(f"\n可视化图像已成功保存为: {output_filename}")
    analyze_metrics_by_magnitude(target_grid, prediction_grid)


if __name__ == "__main__":
    main()