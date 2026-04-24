import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read as ase_read
from cube_tools import cube as CubeToolzLoader
from matplotlib.colors import LogNorm
from train_moe import MixtureOfExpertsModel
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr

# ... (所有之前的辅助函数保持不变) ...

def create_coordinate_grid(grid_res: int, device: torch.device = torch.device("cpu")):
    coords = torch.linspace(-1.0, 1.0, grid_res, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing="ij")
    grid = torch.stack((grid_x, grid_y, grid_z), dim=-1)
    return grid

# ... (positional_encoding, voxelize_structure, voxelize_density_integrate 不变) ...
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


### CORRECTED AND CLARIFIED ROTATION FUNCTION ###
def transform_grid_by_interpolation(source_grid: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Transforms a 3D grid using a given 3x3 transformation matrix and trilinear interpolation.
    This is a generalized version of rotation.

    The core logic is: Value_new(p) = Value_source(T @ p), where T is the transformation.
    To implement this, we do a reverse lookup: for each point 'p' in our new grid,
    we find its corresponding source point 'p_source = T @ p' in the source grid and interpolate.

    Args:
        source_grid (np.ndarray): The (N, N, N) grid to be transformed.
        transform_matrix (np.ndarray): The (3, 3) matrix (e.g., rotation matrix) defining the
                                       transformation of coordinates.

    Returns:
        np.ndarray: The transformed (N, N, N) grid.
    """
    grid_res = source_grid.shape[0]

    # 1. Create an interpolator for the source grid.
    grid_axes = [np.arange(grid_res)] * 3
    interpolator = RegularGridInterpolator(
        grid_axes, source_grid,method="cubic",
        bounds_error=False, fill_value=0
    )

    # 2. Define the physical coordinates of the target (output) grid.
    # These are the points 'p' for which we want to find values.
    target_phys_coords = create_coordinate_grid(grid_res, device=torch.device("cpu")).numpy()

    # 3. Apply the transformation to the target coordinates to find where to sample
    # from the source grid. This is the reverse lookup: p_source = T @ p_target.
    source_phys_coords = target_phys_coords.reshape(-1, 3) @ transform_matrix.T # @T to handle row vectors
    
    # 4. Convert physical coordinates [-1, 1] to index coordinates [0, N-1] for the interpolator.
    source_indices = (source_phys_coords + 1.0) / 2.0 * (grid_res - 1)
    
    # 5. Perform the interpolation.
    transformed_values = interpolator(source_indices)
    
    # 6. Reshape back to grid form.
    transformed_grid = transformed_values.reshape(grid_res, grid_res, grid_res)
    
    return transformed_grid

# ... (Constants and plotting functions remain the same) ...

CHECKPOINT_PATH = Path("./checkpoints_moe/best-moe-model-epoch=43-iou=val\epoch_iou_macro=0.0000.ckpt")
RAW_DATA_ROOT = Path("./raw_data_cube")
SAMPLE_INDEX = "2353"

GRID_RESOLUTION = 64
GRID_BOUNDS = (-10.0, 10.0)
NUM_PE_FREQS = 4

ATOM_Z_TO_CHANNEL = {1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 16: 6, 17: 7, 35: 8}
ATOM_COLOR_MAP = {1: 'white', 6: 'gray', 7: 'blue', 8: 'red', 9: 'green', 16: 'yellow', 17: 'lightgreen', 35: 'darkred'}
DEFAULT_ATOM_COLOR = 'pink'


def analyze_mae_by_magnitude(target_grid, prediction_grid):
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
    scatter = ax.scatter(x, y, z, c=density_values, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='Spectral_r',
                         s=(density_values / vmax) * 10 + 0.01, alpha=0.5, depthshade=False)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Density Value (Log Scale)")
    grid_min, grid_max = grid_bounds
    grid_res = grid_data.shape[0]
    voxel_size = (grid_max - grid_min) / grid_res
    atom_indices = (atom_coords - grid_min) / voxel_size
    atom_colors = [ATOM_COLOR_MAP.get(z, DEFAULT_ATOM_COLOR) for z in atom_z_numbers]
    ax.scatter(atom_indices[:, 0], atom_indices[:, 1], atom_indices[:, 2],
               s=150, c=atom_colors, alpha=0.9, edgecolors='black', linewidth=0.8, zorder=10)
    ax.set_xlim(0, grid_data.shape[2]); ax.set_ylim(0, grid_data.shape[1]); ax.set_zlim(0, grid_data.shape[0])

def plot_error_scatter(ax, diff_grid, title, fig, atom_coords, atom_z_numbers, grid_bounds):
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)
    if np.any(diff_grid > 0):
        error_threshold = np.percentile(np.abs(diff_grid[diff_grid != 0]), 50)
    else:
        error_threshold = 1e-6
    z, y, x = np.where(np.abs(diff_grid) > error_threshold)
    error_values = np.abs(diff_grid[z, y, x])
    if len(x) == 0:
        ax.text(0.5, 0.5, 0.5, "No significant error", transform=ax.transAxes, ha='center', va='center')
        ax.set_xlim(0, diff_grid.shape[2]); ax.set_ylim(0, diff_grid.shape[1]); ax.set_zlim(0, diff_grid.shape[0])
        return
    min_positive_error = error_values.min() if error_values.min() > 0 else 1e-9
    scatter = ax.scatter(x, y, z, c=error_values, norm=LogNorm(vmin=min_positive_error, vmax=error_values.max()),
                         cmap='Reds', s=(error_values / error_values.max()) * 5 + 0.0001, alpha=0.7, depthshade=False)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Absolute Error (Log Scale)")
    grid_min, grid_max = grid_bounds
    grid_res = diff_grid.shape[0]
    voxel_size = (grid_max - grid_min) / grid_res
    atom_indices = (atom_coords - grid_min) / voxel_size
    atom_colors = [ATOM_COLOR_MAP.get(z, DEFAULT_ATOM_COLOR) for z in atom_z_numbers]
    ax.scatter(atom_indices[:, 0], atom_indices[:, 1], atom_indices[:, 2],
               s=150, c=atom_colors, alpha=0.9, edgecolors='black', linewidth=0.8, zorder=10)
    ax.set_xlim(0, diff_grid.shape[2]); ax.set_ylim(0, diff_grid.shape[1]); ax.set_zlim(0, diff_grid.shape[0])


def main():
    # ... (Data loading and standard prediction part is the same) ...
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

    print("\n--- 1. 正在进行标准 MoE 预测 (无旋转)...")
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

    print("\n--- 2. 正在进行旋转不变性测试 (使用修正后的插值逻辑)...")
    random_rotation = Rotation.random()
    rot_matrix = random_rotation.as_matrix()
    inv_rot_matrix = rot_matrix.T
    print("已生成随机旋转矩阵。")

    # Predict on rotated atoms
    rotated_atom_coords = aligned_atom_coords @ rot_matrix.T
    rotated_atom_grid = voxelize_structure(
        rotated_atom_coords, atom_z_numbers,
        GRID_RESOLUTION, GRID_BOUNDS, ATOM_Z_TO_CHANNEL
    )
    with torch.no_grad():
        rotated_atom_grid_batch = rotated_atom_grid.to(device).float().unsqueeze(0)
        logits_rot, expert_outputs_rot = model(rotated_atom_grid_batch, pe_grid_batch)
        pred_labels_rot = torch.argmax(logits_rot, dim=1)
        prediction_tensor_rotated_input = torch.gather(expert_outputs_rot, 1, pred_labels_rot.unsqueeze(1)).squeeze(1)
    
    prediction_grid_in_rotated_space = prediction_tensor_rotated_input.squeeze().cpu().numpy()
    print("已在旋转后的分子上完成预测，得到一个旋转空间中的密度场。")

    # Correctly un-rotate the predicted grid
    print("正在使用修正后的插值逻辑将预测结果旋转回原始方向...")
    # To get the unrotated field, we need Value_unrotated(p) = Value_rotated_space(R @ p)
    # So the transformation matrix passed to the function should be the forward rotation matrix R.
    unrotated_prediction_grid = transform_grid_by_interpolation(prediction_grid_in_rotated_space, rot_matrix)
    print("反向旋转和插值完成。")
    
    mae_rotation = np.mean(np.abs(prediction_grid - unrotated_prediction_grid))
    pearson_corr, _ = pearsonr(prediction_grid.flatten(), unrotated_prediction_grid.flatten())
    
    # ... (The rest of the script, including plotting and the interpolation check, remains valid) ...
    print("\n--- 旋转不变性分析结果 ---")
    print(f"原始预测 vs. 旋转后还原的预测:")
    print(f"  - 平均绝对误差 (MAE): {mae_rotation:.4e}")
    print(f"  - 皮尔逊相关系数:   {pearson_corr:.6f}")

    difference_grid_vs_target = np.abs(prediction_grid - target_grid)

    print("\n正在生成三维可视化图像...")
    fig = plt.figure(figsize=(32, 8))
    display_threshold = 1e-5
    target_display_values = target_grid[target_grid > display_threshold]
    pred_display_values = prediction_grid[prediction_grid > display_threshold]
    if len(target_display_values) > 0 or len(pred_display_values) > 0:
        all_display_values = np.concatenate([target_display_values, pred_display_values])
        global_vmin, global_vmax = all_display_values.min(), all_display_values.max()
    else:
        global_vmin, global_vmax = display_threshold, 1.0

    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    plot_density_scatter(ax1, target_grid, "Ground Truth Density", fig,
                         aligned_atom_coords, atom_z_numbers, GRID_BOUNDS,
                         vmin=global_vmin, vmax=global_vmax, threshold=display_threshold)
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    plot_density_scatter(ax2, prediction_grid, "MoE Predicted Density (Original)", fig,
                         aligned_atom_coords, atom_z_numbers, GRID_BOUNDS,
                         vmin=global_vmin, vmax=global_vmax, threshold=display_threshold)
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    plot_density_scatter(ax3, prediction_grid_in_rotated_space, "MoE Prediction (Rotated & Un-rotated)", fig,
                         rotated_atom_coords, atom_z_numbers, GRID_BOUNDS,
                         vmin=global_vmin, vmax=global_vmax, threshold=display_threshold)
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    plot_error_scatter(ax4, difference_grid_vs_target, "Absolute Prediction Error (vs Truth)", fig,
                       aligned_atom_coords, atom_z_numbers, GRID_BOUNDS)

    plt.tight_layout(pad=1.0)
    output_filename = f"moe_analysis_scatter_{SAMPLE_INDEX}_with_rotation_test_corrected.png"
    plt.savefig(output_filename, dpi=150)
    print(f"\n可视化图像已成功保存为: {output_filename}")
    
    analyze_mae_by_magnitude(target_grid, prediction_grid)

    print("\n--- 3. 正在验证旋转-逆旋转操作的保真度 (使用修正后逻辑) ---")
    
    # Step 1: "Forward" rotate the original prediction grid
    # Value_rotated(p) = Value_original(R_inv @ p)
    rotated_original_prediction = transform_grid_by_interpolation(prediction_grid, inv_rot_matrix)

    # Step 2: "Backward" rotate the result to recover the original
    # Value_recovered(p) = Value_rotated(R @ p)
    recovered_prediction_grid = transform_grid_by_interpolation(rotated_original_prediction, rot_matrix)

    mae_interpolation = np.mean(np.abs(prediction_grid - recovered_prediction_grid))
    pearson_interpolation, _ = pearsonr(prediction_grid.flatten(), recovered_prediction_grid.flatten())

    print("\n--- 插值误差分析结果 ---")
    print("原始预测 vs. 经过(旋转->逆旋转)后的原始预测:")
    print(f"  - 平均绝对误差 (MAE): {mae_interpolation:.4e}")
    print(f"  - 皮尔逊相关系数:   {pearson_interpolation:.6f}")
    print("\n--- 最终结论 ---")
    print(f"模型等变性误差 (MAE): {mae_rotation:.4e}")
    print(f"插值过程误差 (MAE): {mae_interpolation:.4e}")
    if mae_rotation > mae_interpolation:
        model_error = mae_rotation - mae_interpolation
        print(f"可以估计，模型本身缺乏等变性导致的误差约为: {model_error:.4e}")
        print("这表明模型不是完美的旋转等变体，但可能已经非常接近。")
    else:
        print("插值误差大于或等于模型等变性误差。这表明在此测试中，模型表现出了非常好的旋转等变性，")
        print("我们观察到的差异主要来源于插值过程的数值精度限制。")


if __name__ == "__main__":
    main()