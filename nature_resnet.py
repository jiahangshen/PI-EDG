import torch
import torch.nn as nn
import torch.nn.functional as F

class CircularPad3d(nn.Module):
    """论文指定的 Circular Padding，处理周期性/空间连续性"""
    def __init__(self, padding):
        super().__init__()
        self.pad = padding

    def forward(self, x):
        # 在 D, H, W 三个维度上进行循环填充
        return F.pad(x, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), mode='circular')

class ResBlock3D(nn.Module):
    """
    包含 Instance Normalization 的标准超分辨残差块
    """
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            CircularPad3d(1),
            nn.Conv3d(channels, channels, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm3d(channels, affine=True), # 论文指定的归一化
            nn.PReLU(),
            CircularPad3d(1),
            nn.Conv3d(channels, channels, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm3d(channels, affine=True)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class SRResNet3D(nn.Module):
    """
    Nature Comm 2025: Image super-resolution inspired electron density prediction
    完全复刻版：包含 Upscaling Factor=2, Instance Norm 和 电子数硬归一化
    """
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_res_blocks=16, upscale_factor=2):
        super().__init__()
        
        # 1. 初始特征提取
        self.conv_in = nn.Sequential(
            CircularPad3d(1),
            nn.Conv3d(in_channels, num_features, kernel_size=3, padding=0),
            nn.PReLU()
        )
        
        # 2. 16 个残差块 (参数来自原论文标准设定)
        res_blocks =[]
        for _ in range(num_res_blocks):
            res_blocks.append(ResBlock3D(num_features))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # 3. 后残差卷积
        self.conv_post = nn.Sequential(
            CircularPad3d(1),
            nn.Conv3d(num_features, num_features, kernel_size=3, padding=0),
            nn.InstanceNorm3d(num_features, affine=True)
        )
        
        # 4. Upsampling (上采样因子 = 2)
        # 将低分辨率 SAD 转为高分辨率密度
        if upscale_factor == 2:
            self.upsample = nn.Sequential(
                nn.ConvTranspose3d(num_features, num_features, kernel_size=2, stride=2),
                nn.PReLU()
            )
        else:
            self.upsample = nn.Identity() # 如果您的输入输出分辨率本来就一样，可以设为 1
        
        # 5. 最终输出层
        self.conv_out = nn.Sequential(
            CircularPad3d(1),
            nn.Conv3d(num_features, out_channels, kernel_size=3, padding=0)
        )

    def forward(self, x_sad, target_electrons):
        """
        x_sad: 输入 SAD 猜测 [B, 1, 64, 64, 64]
        target_electrons: 真实的分子总电子数 [B, 1] 或 [B]
        """
        # 1. 特征提取与残差学习
        x_init = self.conv_in(x_sad)
        res = self.res_blocks(x_init)
        res = self.conv_post(res)
        
        # 2. 全局跳跃连接 (学习从 SAD 到真值的残差)
        x_fused = x_init + res 
        
        # 3. 空间处理 (如果 upscale_factor=1, 则不改变分辨率)
        out_high_res = self.upsample(x_fused)
        
        # 4. 最终输出层
        raw_density = self.conv_out(out_high_res)
        
        # --- 核心后处理步骤 ---
        
        # A. 确保非负性：改用 Softplus 以避免 ReLU 的梯度死亡问题
        density_activated = F.softplus(raw_density)
        
        # B. 物理硬约束 (积分归一化)
        # 计算当前预测网格的总和 (Integral)
        # dim=(1,2,3,4) 涵盖了 C, D, H, W
        current_integral = torch.sum(density_activated, dim=(1, 2, 3, 4), keepdim=True) + 1e-8
        
        # 准备目标电子数，确保其维度为 [B, 1, 1, 1, 1] 以便广播乘法
        target_integral = target_electrons.view(-1, 1, 1, 1, 1)
        
        # C. 强制缩放：让预测网格的总积分严格等于目标电子数
        final_density = density_activated * (target_integral / current_integral)
        
        # ！！！关键：必须返回归一化后的结果 ！！！
        return final_density