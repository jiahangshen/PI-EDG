import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(卷积 => [批归一化] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样模块：最大池化 + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样模块：转置卷积/插值 + DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            # 正确的逻辑：拼接后的通道数 = 上采样输出(in_channels) + 跳跃连接(in_channels)
            self.conv = DoubleConv(in_channels * 2, out_channels) # <--- 这是修正后的行
        else:
            # 这个分支的逻辑是正确的，因为 ConvTranspose3d 已经将通道数减半了
            # 拼接后通道数 = (in_channels // 2) + (in_channels // 2) = in_channels
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 是来自上一个（更深）层的上采样特征
        # x2 是来自编码器（更浅）层的跳跃连接
        x1 = self.up(x1)
        
        # 处理可能的奇数尺寸问题
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        # F.pad 的参数顺序是 (左, 右, 上, 下, 前, 后)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1) # 在通道维度上拼接
        return self.conv(x)

class OutConv(nn.Module):
    """最终输出层"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    """
    后期融合 (Late Fusion) U-Net 架构。
    - 编码器只处理原子网格。
    - 位置编码在瓶颈处被注入。
    """
    def __init__(self, in_channels_atom, in_channels_pe, out_channels, n_channels=64, bilinear=True):
        """
        参数:
            in_channels_atom (int): 原子网格的输入通道数 (e.g., 14)
            in_channels_pe (int): 位置编码网格的输入通道数 (e.g., 60)
            out_channels (int): 输出通道数 (e.g., 1)
        """
        super(UNet3D, self).__init__()
        self.in_channels_atom = in_channels_atom
        self.in_channels_pe = in_channels_pe
        self.out_channels = out_channels
        self.n_channels = n_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # 编码器部分 (只处理原子)
        self.inc = DoubleConv(in_channels_atom, n_channels)
        self.down1 = Down(n_channels, n_channels * 2)
        self.down2 = Down(n_channels * 2, n_channels * 4)
        self.down3 = Down(n_channels * 4, n_channels * 8)
        self.down4 = Down(n_channels * 8, n_channels * 16 // factor)
        
        # --- 瓶颈层：融合位置编码 ---
        bottleneck_in_channels = (n_channels * 16 // factor) + in_channels_pe
        bottleneck_out_channels = n_channels * 16 // factor
        self.bottleneck_conv = DoubleConv(bottleneck_in_channels, bottleneck_out_channels)
        # -----------------------------
        
        # 解码器部分
        self.up1 = Up(n_channels * 16 // factor, n_channels * 8 // factor, bilinear)
        self.up2 = Up(n_channels * 8 // factor, n_channels * 4 // factor, bilinear)
        self.up3 = Up(n_channels * 4 // factor, n_channels * 2 // factor, bilinear)
        self.up4 = Up(n_channels * 2 // factor, n_channels, bilinear)
        self.outc = OutConv(n_channels, out_channels)
        self.final_activation = nn.ReLU() # 保证输出非负

    def forward(self, atom_grid, pe_grid):
        """
        forward 接收两个独立的输入
        """
        # --- 编码器只处理 atom_grid ---
        x1 = self.inc(atom_grid)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # 瓶颈特征, e.g., [B, 512, 4, 4, 4]
        
        # --- 在瓶颈处融合 PE ---
        # 1. 将 pe_grid (e.g., [B, 60, 64, 64, 64]) 下采样到与瓶颈特征相同的尺寸
        pe_downsampled = F.interpolate(pe_grid, size=x5.shape[2:], mode='trilinear', align_corners=False)
        
        # 2. 在通道维度上拼接
        x5_fused = torch.cat([x5, pe_downsampled], dim=1) # e.g., [B, 512+60, 4, 4, 4]
        
        # 3. 通过融合卷积层
        x5_processed = self.bottleneck_conv(x5_fused) # e.g., [B, 512, 4, 4, 4]
        
        # --- 解码器从融合后的特征开始 ---
        x = self.up1(x5_processed, x4) # x5_processed 替换了 x5
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return self.final_activation(logits)

