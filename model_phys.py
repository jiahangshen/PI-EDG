import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 傅里叶神经算子层 (FNO Layer) - 修复版 ---
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # 动态截断，防止 Grid size 小于 modes
        cur_m1 = min(self.modes1, x_ft.size(-3))
        cur_m2 = min(self.modes2, x_ft.size(-2))
        cur_m3 = min(self.modes3, x_ft.size(-1))

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :cur_m1, :cur_m2, :cur_m3] = \
            self.compl_mul3d(x_ft[:, :, :cur_m1, :cur_m2, :cur_m3], self.weights1[:, :, :cur_m1, :cur_m2, :cur_m3])
        out_ft[:, :, -cur_m1:, :cur_m2, :cur_m3] = \
            self.compl_mul3d(x_ft[:, :, -cur_m1:, :cur_m2, :cur_m3], self.weights2[:, :, :cur_m1, :cur_m2, :cur_m3])
        out_ft[:, :, :cur_m1, -cur_m2:, :cur_m3] = \
            self.compl_mul3d(x_ft[:, :, :cur_m1, -cur_m2:, :cur_m3], self.weights3[:, :, :cur_m1, :cur_m2, :cur_m3])
        out_ft[:, :, -cur_m1:, -cur_m2:, :cur_m3] = \
            self.compl_mul3d(x_ft[:, :, -cur_m1:, -cur_m2:, :cur_m3], self.weights4[:, :, :cur_m1, :cur_m2, :cur_m3])

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

# --- 2. 混合瓶颈层 - 修复版 ---
class HybridBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels), # 使用 GroupNorm 提升稳定性
            nn.SiLU(inplace=True)
        )
        self.fno_branch = SpectralConv3d(in_channels, out_channels, 4, 4, 4)
        self.act = nn.SiLU()
        
    def forward(self, x):
        local_feat = self.conv_branch(x)
        global_feat = self.fno_branch(x)
        return self.act(local_feat + global_feat)

# --- 3. 硬约束归一化层 (最终修复版) ---
class HardConstraintLayer(nn.Module):
    def __init__(self, threshold=1e-2): # 不需要 grid 参数了
        super().__init__()
        self.softplus = nn.Softplus()
        self.threshold = threshold

    def forward(self, x, target_integral): # <--- 接收真实积分
        # 1. 激活
        rho_cubic = self.softplus(x)
        
        # 2. 静噪
        mask = torch.sigmoid((rho_cubic - self.threshold) * 100)
        rho_cubic_masked = rho_cubic * mask
        
        # 3. 变换回线性空间
        rho_real = torch.pow(rho_cubic_masked, 3.0)
        
        # 4. 计算预测积分
        current_sum = torch.sum(rho_real, dim=(1, 2, 3, 4), keepdim=True) + 1e-8
        
        # 5. 归一化目标 (直接使用 Dataset 提供的真实积分)
        # target_integral shape: [B] -> [B, 1, 1, 1, 1]
        target_sum = target_integral.view(-1, 1, 1, 1, 1)
        
        scale = target_sum / current_sum
        
        # 6. 应用缩放
        final_density = rho_real * scale
        
        return final_density

# --- 4. 升级版：残差卷积块 (ResBlock) ---
# 相比 DoubleConv，这能提供更强的学习能力和梯度流动
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU(inplace=True)

        self.project = nn.Identity()
        if in_channels != out_channels:
            self.project = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = self.project(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += residual
        return self.act2(out)

# --- 5. Physics-Informed 3D U-Net ---
class PIUNet3D(nn.Module):
    def __init__(self, in_channels_atom, n_channels=64, grid_res=64, grid_bounds=(-10.0, 10.0)):
        super().__init__()
        self.phys_channels = 2
        
        # 使用 ResBlock 替代 DoubleConv
        self.inc = ResBlock(in_channels_atom + self.phys_channels, n_channels)
        
        self.down1 = nn.MaxPool3d(2)
        self.conv1 = ResBlock(n_channels + self.phys_channels, n_channels * 2)
        
        self.down2 = nn.MaxPool3d(2)
        self.conv2 = ResBlock(n_channels * 2 + self.phys_channels, n_channels * 4)
        
        self.down3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = HybridBottleneck(n_channels * 4 + self.phys_channels, n_channels * 8)
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1 = ResBlock(n_channels * 8 + n_channels * 4 + self.phys_channels, n_channels * 4)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2 = ResBlock(n_channels * 4 + n_channels * 2 + self.phys_channels, n_channels * 2)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec3 = ResBlock(n_channels * 2 + n_channels + self.phys_channels, n_channels)
        
        # Output
        self.outc = nn.Conv3d(n_channels, 1, kernel_size=1)
        
        # Bias Initialization
        nn.init.constant_(self.outc.bias, -5.0)

        grid_span = grid_bounds[1]
        self.constraint = HardConstraintLayer(threshold=1e-2)

    def inject_physics(self, feat, v_ext, z_spike):
        target_size = feat.shape[2:]
        v_scaled = F.interpolate(v_ext, size=target_size, mode='trilinear', align_corners=False)
        z_scaled = F.interpolate(z_spike, size=target_size, mode='trilinear', align_corners=False) 
        return torch.cat([feat, v_scaled, z_scaled], dim=1)

    def forward(self, atom_grid, v_ext, z_spike, total_electrons, use_constraint=True):
        x0 = torch.cat([atom_grid, v_ext, z_spike], dim=1)
        x1 = self.inc(x0) 
        
        x2_pre = self.down1(x1)
        x2_inj = self.inject_physics(x2_pre, v_ext, z_spike)
        x2 = self.conv1(x2_inj)
        
        x3_pre = self.down2(x2)
        x3_inj = self.inject_physics(x3_pre, v_ext, z_spike)
        x3 = self.conv2(x3_inj)
        
        x4_pre = self.down3(x3)
        x4_inj = self.inject_physics(x4_pre, v_ext, z_spike)
        x_bottle = self.bottleneck(x4_inj)
        
        xu1 = self.up1(x_bottle)
        xu1 = torch.cat([xu1, x3], dim=1)
        xu1 = self.inject_physics(xu1, v_ext, z_spike)
        xd1 = self.dec1(xu1)
        
        xu2 = self.up2(xd1)
        xu2 = torch.cat([xu2, x2], dim=1)
        xu2 = self.inject_physics(xu2, v_ext, z_spike)
        xd2 = self.dec2(xu2)
        
        xu3 = self.up3(xd2)
        xu3 = torch.cat([xu3, x1], dim=1)
        xu3 = self.inject_physics(xu3, v_ext, z_spike)
        xd3 = self.dec3(xu3)
        
        pre_density = self.outc(xd3)
        
        # 旁路逻辑
        if not use_constraint:
            rho_cubic = F.softplus(pre_density)
            final_density = torch.pow(rho_cubic, 3.0)
        else:
            final_density = self.constraint(pre_density, total_electrons)
        
        return final_density, pre_density