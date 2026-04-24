from cube_tools import cube
from pathlib import Path
import numpy as np

# --- 请修改为您想测试的 CUBE 文件路径 ---
CUBE_FILE_PATH = Path("raw_data_cube/15172/Mol1_Dt.cube") 

def test_cubetoolz_reader(filepath: Path):
    """
    使用 Cube-Toolz 尝试解析 CUBE 文件并检查密度数据。
    """
    if not filepath.exists():
        print(f"错误: 文件不存在 -> {filepath}")
        return

    print(f"正在使用 Cube-Toolz 尝试读取: {filepath} ...")
    
    try:
        # 使用 Cube-Toolz 加载文件
        cube_obj = cube.read_cube(str(filepath))
        
        # Cube-Toolz 通常将数据存储在 .data 属性中
        # 它的形状可能是 (nx, ny, nz)
        if hasattr(cube_obj, 'data') and isinstance(cube_obj.data, np.ndarray):
            print("成功！在 .data 属性中找到了体积数据！")
            print(f"数据形状 (Shape): {cube_obj.data.shape}")
            
            # 额外检查其他必要信息是否存在
            if hasattr(cube_obj, 'origin') and hasattr(cube_obj, 'cell'):
                 print("网格原点 (origin) 和 单元/步长 (cell) 信息也已找到。")
                 print(f"Origin: {cube_obj.origin}")
                 print(f"Cell/Steps: {cube_obj.cell}") # 需要确认 cell 返回的是什么
                 return cube_obj
            else:
                 print("警告：找到了体积数据，但缺少网格原点或单元/步长信息。")
                 return None
        else:
            print("失败！读取成功，但在返回的对象中未找到 .data 属性或类型不符。")
            print("请检查 Cube-Toolz 是否能正确解析您的 CUBE 文件。")
            print("返回的 cube_obj 对象属性有:", dir(cube_obj))
            return None

    except Exception as e:
        print(f"使用 Cube-Toolz 读取文件时发生严重错误: {e}")
        return None

if __name__ == "__main__":
    parsed_data = test_cubetoolz_reader(CUBE_FILE_PATH)
    
    if parsed_data:
        print("\n测试成功，您可以在您的 dataset.py 中改用 Cube-Toolz 来读取 CUBE 文件。")