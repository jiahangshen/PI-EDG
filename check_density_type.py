import numpy as np
from pathlib import Path
from ase.io import read as ase_read
from cube_tools import cube as CubeToolzLoader 

# 配置您的路径
RAW_DATA_ROOT = Path("./raw_data_cube")
SAMPLE_INDEX = "3032" # 使用您刚才测试的那个样本

def check_density_type():
    sample_dir = RAW_DATA_ROOT / SAMPLE_INDEX
    sdf_file = next(sample_dir.glob("*.sdf"))
    cube_file = next(sample_dir.glob("*.cube"))
    
    # 1. 计算理论总电子数 (Z_total)
    atoms = ase_read(str(sdf_file))
    z_numbers = atoms.get_atomic_numbers()
    total_electrons = sum(z_numbers)
    print(f"分子: {atoms.get_chemical_formula()}")
    print(f"理论总电子数 (Z_total): {total_electrons}")
    
    # 2. 计算 CUBE 积分
    cube = CubeToolzLoader(str(cube_file))
    data = cube.data
    
    # 计算体素体积 dV (单位: Bohr^3)
    # CUBE 文件标准单位是 Bohr，步长向量的模长即为边长
    dx = np.linalg.norm(cube.X)
    dy = np.linalg.norm(cube.Y)
    dz = np.linalg.norm(cube.Z)
    dV = dx * dy * dz
    
    # CUBE 数据值的单位通常是 electrons/Bohr^3
    # 所以积分 = sum(data) * dV
    total_integral = np.sum(data) * dV
    print(f"CUBE 积分结果 (N_cube): {total_integral:.4f}")
    
    # 3. 判断
    ratio = total_integral / total_electrons
    print(f"比率 (N_cube / Z_total): {ratio:.4f}")
    
    if 0.95 <= ratio <= 1.05:
        print(">>> 结论: 这是【总电子密度 (Total Density)】")
    elif 0.45 <= ratio <= 0.55:
        print(">>> 结论: 这是【Alpha 密度 (Alpha Density)】 (约为总数的一半)")
    else:
        print(f">>> 结论: 类型未知，或者是自旋密度差，或者是单位/格式异常。")

if __name__ == "__main__":
    check_density_type()