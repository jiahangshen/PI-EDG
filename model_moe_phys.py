import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. 基础组件 (复用 PIUNet3D 的组件)
# =========================================================================

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels; self.out_channels = out_channels
        self.modes1 = modes1; self.modes2 = modes2; self.modes3 = modes3
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
        cur_m1 = min(self.modes1, x_ft.size(-3))
        cur_m2 = min(self.modes2, x_ft.size(-2))
        cur_m3 = min(self.modes3, x_ft.size(-1))
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :cur_m1, :cur_m2, :cur_m3] = self.compl_mul3d(x_ft[:, :, :cur_m1, :cur_m2, :cur_m3], self.weights1[:, :, :cur_m1, :cur_m2, :cur_m3])
        out_ft[:, :, -cur_m1:, :cur_m2, :cur_m3] = self.compl_mul3d(x_ft[:, :, -cur_m1:, :cur_m2, :cur_m3], self.weights2[:, :, :cur_m1, :cur_m2, :cur_m3])
        out_ft[:, :, :cur_m1, -cur_m2:, :cur_m3] = self.compl_mul3d(x_ft[:, :, :cur_m1, -cur_m2:, :cur_m3], self.weights3[:, :, :cur_m1, :cur_m2, :cur_m3])
        out_ft[:, :, -cur_m1:, -cur_m2:, :cur_m3] = self.compl_mul3d(x_ft[:, :, -cur_m1:, -cur_m2:, :cur_m3], self.weights4[:, :, :cur_m1, :cur_m2, :cur_m3])
        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))

class HybridBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels), 
            nn.SiLU(inplace=True)
        )
        self.fno_branch = SpectralConv3d(in_channels, out_channels, 4, 4, 4)
        self.act = nn.SiLU()
    def forward(self, x): return self.act(self.conv_branch(x) + self.fno_branch(x))

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.project = nn.Conv3d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
    def forward(self, x): return self.act2(self.gn2(self.conv2(self.act1(self.gn1(self.conv1(x))))) + self.project(x))

# =========================================================================
# 2. 物理知情骨干网络 (Physics Backbone)
# =========================================================================

class PhysicsResUNet(nn.Module):
    """
    去掉了 Output Head 的 PIUNet3D。
    专门用于提取包含物理信息的高维特征，供 MoE 使用。
    """
    def __init__(self, in_channels_atom, in_channels_pe=0, n_channels=64):
        super().__init__()
        self.phys_channels = 2 # v_ext + z_spike
        total_in_channels = in_channels_atom + self.phys_channels + in_channels_pe
        
        # Encoder
        self.inc = ResBlock(total_in_channels, n_channels)
        self.down1 = nn.MaxPool3d(2); self.conv1 = ResBlock(n_channels + self.phys_channels, n_channels * 2)
        self.down2 = nn.MaxPool3d(2); self.conv2 = ResBlock(n_channels * 2 + self.phys_channels, n_channels * 4)
        self.down3 = nn.MaxPool3d(2)
        self.bottleneck = HybridBottleneck(n_channels * 4 + self.phys_channels, n_channels * 8)
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1 = ResBlock(n_channels * 8 + n_channels * 4 + self.phys_channels, n_channels * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2 = ResBlock(n_channels * 4 + n_channels * 2 + self.phys_channels, n_channels * 2)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec3 = ResBlock(n_channels * 2 + n_channels + self.phys_channels, n_channels)
        
        # 输出特征而不是密度
        self.out_features = n_channels

    def inject_physics(self, feat, v_ext, z_spike):
        target_size = feat.shape[2:]
        v_scaled = F.interpolate(v_ext, size=target_size, mode='trilinear', align_corners=False)
        z_scaled = F.interpolate(z_spike, size=target_size, mode='trilinear', align_corners=False) 
        return torch.cat([feat, v_scaled, z_scaled], dim=1)

    def forward(self, atom_grid, pe_grid, v_ext, z_spike):
        # 1. Input
        if pe_grid is not None:
            x0 = torch.cat([atom_grid, pe_grid, v_ext, z_spike], dim=1)
        else:
            x0 = torch.cat([atom_grid, v_ext, z_spike], dim=1)
        x1 = self.inc(x0)
        
        # 2. Down
        x2_pre = self.down1(x1)
        x2_inj = self.inject_physics(x2_pre, v_ext, z_spike)
        x2 = self.conv1(x2_inj)
        
        x3_pre = self.down2(x2)
        x3_inj = self.inject_physics(x3_pre, v_ext, z_spike)
        x3 = self.conv2(x3_inj)
        
        x4_pre = self.down3(x3)
        x4_inj = self.inject_physics(x4_pre, v_ext, z_spike)
        x_bottle = self.bottleneck(x4_inj)
        
        # 3. Up
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
        
        return xd3 # [B, 64, D, H, W]

# =========================================================================
# 3. 物理知情 MoE 硬约束层
# =========================================================================

class MoEHardConstraintLayer(nn.Module):
    def __init__(self, threshold=1e-2):
        super().__init__()
        self.threshold = threshold

    def forward(self, rho_cubic, total_electrons):
        """
        rho_cubic: MoE 融合后的结果，代表 n^(1/3)
                   注意：因为 MoE 输出已经经过 Sigmoid/Softplus，保证非负
        """
        # 1. 静噪 (在立方根空间)
        mask = torch.sigmoid((rho_cubic - self.threshold) * 100)
        rho_cubic_masked = rho_cubic * mask
        
        # 2. 变换回线性空间 (Physics Space)
        rho_real = torch.pow(rho_cubic_masked, 3.0)
        
        # 3. 计算积分
        current_sum = torch.sum(rho_real, dim=(1, 2, 3, 4), keepdim=True) + 1e-8
        
        # 4. 归一化目标 (使用真实 Raw Sum 目标)
        target_sum = total_electrons.view(-1, 1, 1, 1, 1)
        scale = target_sum / current_sum
        
        # 5. 应用缩放
        final_density = rho_real * scale
        
        return final_density

# =========================================================================
# 4. PI-MoE 主模型
# =========================================================================

class PhysicsMoE(nn.Module):
    def __init__(self, in_channels_atom, in_channels_pe, num_classes, class_bins, n_channels=64, grid_res=64, grid_bounds=(-10.0, 10.0)):
        super().__init__()
        
        # 1. 物理骨干
        self.backbone = PhysicsResUNet(in_channels_atom, in_channels_pe, n_channels)
        
        # 2. 门控网络
        self.gating_head = nn.Conv3d(n_channels, num_classes, kernel_size=1)
        
        # 3. 专家网络
        # 每个专家输出 1 个通道，代表 n^(1/3)
        self.expert_layer1 = nn.Conv3d(n_channels, 32 * num_classes, kernel_size=3, padding=1)
        
        # 激活
        self.expert_act = nn.ReLU(inplace=True)
        
        # 第二层: 32 * 14 -> 1 * 14
        # 这里必须使用 groups=num_classes，让每个专家只处理属于它自己的 32 个通道
        # Input Channels: 32 * num_classes
        # Output Channels: 1 * num_classes
        # Groups: num_classes
        # 这样第 i 个输出通道只依赖于第 i 组的 32 个输入通道
        self.expert_layer2 = nn.Conv3d(
            32 * num_classes, 
            1 * num_classes, 
            kernel_size=1, 
            groups=num_classes # <--- 关键！实现并行独立
        )
        
        # 4. 约束层
        self.constraint = MoEHardConstraintLayer(threshold=1e-2)
        
        # 5. Bins 处理 (关键：将 Bin 边界转换为立方根空间)
        # 因为专家预测的是 n^(1/3)，所以我们要把 bin 也开三次方
        if not isinstance(class_bins, torch.Tensor):
            class_bins = torch.tensor(class_bins, dtype=torch.float32)
            
        # 处理 inf
        class_bins = torch.nan_to_num(class_bins, posinf=10000.0)
        
        # 转换为立方根空间 [0, 1, 2...] -> [0, 1, 1.26...]
        bins_cubic = torch.pow(class_bins, 1.0/3.0)
        
        min_vals = torch.zeros_like(bins_cubic)
        min_vals[1:] = bins_cubic[:-1]
        max_vals = bins_cubic
        
        self.register_buffer("min_vals", min_vals)
        self.register_buffer("max_vals", max_vals)

    def forward(self, atom_grid, pe_grid, v_ext, z_spike, total_electrons, use_constraint=True):
        # A. 提取特征
        features = self.backbone(atom_grid, pe_grid, v_ext, z_spike)
        
        # B. 门控 logits
        logits = self.gating_head(features)
        # C. 专家
        x = self.expert_layer1(features)
        x = self.expert_act(x)
        
        # 2. Layer 2: 降维 & 分组 [B, 32*K, ...] -> [B, K, ...]
        expert_raw = self.expert_layer2(x)
        
        # 3. Sigmoid 归一化 (0-1)
        normalized = torch.sigmoid(expert_raw)
        
        # 4. 映射到 Bin
        mn = self.min_vals.view(1, -1, 1, 1, 1)
        mx = self.max_vals.view(1, -1, 1, 1, 1)
        expert_outputs_cubic = mn + normalized * (mx - mn)
    
        
        # D. 融合 (Fusion)
        expert_weights = F.softmax(logits, dim=1)
        # 融合结果仍代表 n^(1/3)
        rho_cubic_fused = torch.sum(expert_weights * expert_outputs_cubic, dim=1, keepdim=True)
        
        # E. 约束与输出
        if not use_constraint:
            # 旁路：直接立方返回
            # 加上静噪以保持一致性
            mask = torch.sigmoid((rho_cubic_fused - 1e-2) * 100)
            final_density = torch.pow(rho_cubic_fused * mask, 3.0)
        else:
            # 约束：应用积分校准
            final_density = self.constraint(rho_cubic_fused, total_electrons)
            
        return logits, expert_outputs_cubic, final_density