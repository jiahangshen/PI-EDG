import copy
import numpy as np
from scipy.stats import pearsonr

from cube_tools import cube,diff_cubes
original_cube = cube(fname="raw_data_cube/1/Mol1_Dt.cube") # 创建一个空的 cube 对象


# 2. 执行旋转 -> 逆旋转
angle_deg = 30.0
rotation_axes = (0, 1) # 绕 Z 轴

# 第一次旋转（正向）
rotated_cube = copy.deepcopy(original_cube)
print(f"\nRotating cube by {angle_deg} degrees...")
rotated_cube.rotate_cube(angle_deg, axes=rotation_axes)
rotated_cube.write_cube("pred_rotated.cube")

# 第二次旋转（逆向）
recovered_cube = copy.deepcopy(rotated_cube)
print(f"Rotating cube back by {-angle_deg} degrees...")
recovered_cube.rotate_cube(-angle_deg, axes=rotation_axes)
recovered_cube.write_cube("pred_recovered.cube")

# 3. 计算差异
print("\nCalculating difference between original and recovered cubes...")
# 注意：这里的 diff_cubes 需要能接受对象列表，而不仅仅是文件名列表
diff_cube_obj = diff_cubes(["raw_data_cube/1/Mol1_Dt.cube", "pred_recovered.cube"])

# 4. 分析差异
print("\nAnalyzing interpolation error...")
diff_data = np.abs(diff_cube_obj.data)
mae_interpolation = np.mean(diff_data)
max_error = np.max(diff_data)

# 计算皮尔逊系数
p_corr, _ = pearsonr(original_cube.data.flatten(), recovered_cube.data.flatten())

print(f"Interpolation Roundtrip Error Analysis:")
print(f"  - Mean Absolute Error (MAE): {mae_interpolation:.4e}")
print(f"  - Max Absolute Error:        {max_error:.4e}")
print(f"  - Pearson Correlation:       {p_corr:.6f}")