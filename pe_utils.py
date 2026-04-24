import torch

def create_coordinate_grid(grid_res: int, device: torch.device = torch.device("cpu")):
    """
    创建一个归一化到[-1, 1]范围的坐标网格。
    这是一个更明确、更健壮的实现。
    """
    # 1. 创建三个独立的一维坐标轴
    x_coords = torch.linspace(-1.0, 1.0, grid_res, device=device)
    y_coords = torch.linspace(-1.0, 1.0, grid_res, device=device)
    z_coords = torch.linspace(-1.0, 1.0, grid_res, device=device)
    
    # 2. 使用 view 和 expand/repeat 手动构建三维网格
    #    这比 meshgrid 更明确
    grid_x = x_coords.view(1, 1, -1).repeat(grid_res, grid_res, 1) # 在 X 维 (dim 2) 变化
    grid_y = y_coords.view(1, -1, 1).repeat(grid_res, 1, grid_res) # 在 Y 维 (dim 1) 变化
    grid_z = z_coords.view(-1, 1, 1).repeat(1, grid_res, grid_res) # 在 Z 维 (dim 0) 变化
    
    # 3. 将三个坐标网格堆叠成最终的 (D, H, W, 3) 张量
    #    注意顺序，我们希望最后一个维度是 (x, y, z)
    #    但PyTorch习惯的图像/体素维度是 (D,H,W)，对应 (z,y,x)
    #    所以我们按 (z, y, x) 的顺序堆叠
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    
    return grid

def positional_encoding(coords: torch.Tensor, num_freqs: int):
    """
    参数:
        coords (torch.Tensor): 坐标张量，形状为 (..., 3)。
        num_freqs (int): 使用的频率数量 L。

    返回:
        torch.Tensor: 编码后的特征张量。
    """
    # 创建频率向量 (2^0, 2^1, ..., 2^(L-1))
    freqs = 2.0 ** torch.arange(num_freqs, device=coords.device)
    
    # 将坐标与频率相乘 (..., 3, L)
    scaled_coords = coords.unsqueeze(-1) * freqs
    
    # 计算 sin 和 cos (..., 3, L)
    sines = torch.sin(scaled_coords)
    cosines = torch.cos(scaled_coords)
    
    # 将 sin 和 cos 拼接在一起 (..., 3, 2*L)
    encoded = torch.cat([sines, cosines], dim=-1)
    
    # 展平成最终的特征向量 (..., 3 * 2 * L)
    return encoded.flatten(start_dim=-2)