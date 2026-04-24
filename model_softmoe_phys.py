import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========================================================================
# 1. 基础物理组件 (特征提取 & 傅里叶卷积)
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

class PhysicsResUNet(nn.Module):
    def __init__(self, in_channels_atom, in_channels_pe=0, n_channels=64):
        super().__init__()
        self.phys_channels = 2 # v_ext + z_spike
        total_in_channels = in_channels_atom + self.phys_channels + in_channels_pe
        
        self.inc = ResBlock(total_in_channels, n_channels)
        self.down1 = nn.MaxPool3d(2); self.conv1 = ResBlock(n_channels + self.phys_channels, n_channels * 2)
        self.down2 = nn.MaxPool3d(2); self.conv2 = ResBlock(n_channels * 2 + self.phys_channels, n_channels * 4)
        self.down3 = nn.MaxPool3d(2)
        self.bottleneck = HybridBottleneck(n_channels * 4 + self.phys_channels, n_channels * 8)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1 = ResBlock(n_channels * 8 + n_channels * 4 + self.phys_channels, n_channels * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2 = ResBlock(n_channels * 4 + n_channels * 2 + self.phys_channels, n_channels * 2)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec3 = ResBlock(n_channels * 2 + n_channels + self.phys_channels, n_channels)
        self.out_features = n_channels

    def inject_physics(self, feat, v_ext, z_spike):
        target_size = feat.shape[2:]
        v_scaled = F.interpolate(v_ext, size=target_size, mode='trilinear', align_corners=False)
        z_scaled = F.interpolate(z_spike, size=target_size, mode='trilinear', align_corners=False) 
        return torch.cat([feat, v_scaled, z_scaled], dim=1)

    def forward(self, atom_grid, pe_grid, v_ext, z_spike):
        if pe_grid is not None: x0 = torch.cat([atom_grid, pe_grid, v_ext, z_spike], dim=1)
        else: x0 = torch.cat([atom_grid, v_ext, z_spike], dim=1)
        x1 = self.inc(x0)
        x2_pre = self.down1(x1); x2 = self.conv1(self.inject_physics(x2_pre, v_ext, z_spike))
        x3_pre = self.down2(x2); x3 = self.conv2(self.inject_physics(x3_pre, v_ext, z_spike))
        x4_pre = self.down3(x3); x_bottle = self.bottleneck(self.inject_physics(x4_pre, v_ext, z_spike))
        xu1 = self.inject_physics(torch.cat([self.up1(x_bottle), x3], dim=1), v_ext, z_spike); xd1 = self.dec1(xu1)
        xu2 = self.inject_physics(torch.cat([self.up2(xd1), x2], dim=1), v_ext, z_spike); xd2 = self.dec2(xu2)
        xu3 = self.inject_physics(torch.cat([self.up3(xd2), x1], dim=1), v_ext, z_spike); xd3 = self.dec3(xu3)
        return xd3 

class MoEHardConstraintLayer(nn.Module):
    def __init__(self, threshold=1e-2):
        super().__init__()
        self.threshold = threshold

    def forward(self, rho_cubic, total_electrons):
        mask = torch.sigmoid((rho_cubic - self.threshold) * 100)
        rho_cubic_masked = rho_cubic * mask
        rho_real = torch.pow(rho_cubic_masked, 3.0)
        current_sum = torch.sum(rho_real, dim=(1, 2, 3, 4), keepdim=True) + 1e-8
        target_sum = total_electrons.view(-1, 1, 1, 1, 1)
        scale = target_sum / current_sum
        return rho_real * scale


# =========================================================================
# 2. 对数空间的软分箱路由器 (Log-Space Learnable Router)
# =========================================================================

class LogSpaceLearnableRouter(nn.Module):
    def __init__(self, num_experts, initial_bins, temperature=2.0):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature 
        
        internal_bins = np.array(initial_bins[1:-1]) 
        assert len(internal_bins) == num_experts - 1
        
        log10_bins = np.log10(internal_bins + 1e-35) 
        base_val = -35.0
        self.register_buffer("base_val", torch.tensor(base_val, dtype=torch.float32))
        
        initial_gaps = np.diff(np.insert(log10_bins, 0, base_val))
        self.log_gaps = nn.Parameter(torch.tensor(np.log(initial_gaps + 1e-8), dtype=torch.float32))

    def get_thresholds_log10(self):
        gaps = torch.exp(self.log_gaps)
        thresholds = self.base_val + torch.cumsum(gaps, dim=0)
        return thresholds

    def forward(self, guide_field_log10):
        thresholds = self.get_thresholds_log10() 
        t = thresholds.view(1, -1, 1, 1, 1)
        
        boundaries = torch.sigmoid(self.temperature * (guide_field_log10 - t))
        
        probs = []
        probs.append(1.0 - boundaries[:, 0:1, ...])
        for i in range(self.num_experts - 2):
            probs.append(boundaries[:, i:i+1, ...] - boundaries[:, i+1:i+2, ...])
        if self.num_experts > 1:
            probs.append(boundaries[:, -1:, ...])
            
        routing_weights = torch.cat(probs, dim=1)
        routing_weights = routing_weights / (torch.sum(routing_weights, dim=1, keepdim=True) + 1e-8)
        
        return routing_weights, thresholds


# =========================================================================
# 3. 混合空间的 PI-MoE 主模型
# =========================================================================

class PhysicsMoE(nn.Module):
    def __init__(self, in_channels_atom, in_channels_pe, num_experts, initial_bins, n_channels=64):
        super().__init__()
        self.num_experts = num_experts
        
        self.backbone = PhysicsResUNet(in_channels_atom, in_channels_pe, n_channels)
        
        self.guide_predictor_log10 = nn.Sequential(
            nn.Conv3d(n_channels, n_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_channels // 2, 1, kernel_size=1) 
        )
        
        self.router = LogSpaceLearnableRouter(num_experts, initial_bins, temperature=2.0)
        
        self.expert_layer1 = nn.Conv3d(n_channels, 32 * num_experts, kernel_size=3, padding=1)
        self.expert_act = nn.ReLU(inplace=True)
        self.expert_layer2 = nn.Conv3d(32 * num_experts, 1 * num_experts, kernel_size=1, groups=num_experts)
        
        self.constraint = MoEHardConstraintLayer(threshold=1e-2)

    def forward(self, atom_grid, pe_grid, v_ext, z_spike, total_electrons, use_constraint=True):
        # 1. 骨干特征提取
        features = self.backbone(atom_grid, pe_grid, v_ext, z_spike)
        
        # =================================================================
        # 【核心修正 1】：精确梯度隔离的 Guide Field
        # =================================================================
        # a) attached 版本：参与路由，使得全局损失可以调整主干网络特征
        guide_field_attached = self.guide_predictor_log10(features)
        # b) detached 版本：专门用于返回给外部算 loss_guide，绝对不污染主干
        guide_field_detached = self.guide_predictor_log10(features.detach())
        
        routing_weights, thresholds_log10 = self.router(guide_field_attached)
        
        # 2. 专家预测 (Sigmoid 获取 0~1 的基础归一化值)
        x = self.expert_layer1(features)
        x = self.expert_act(x)
        expert_raw = self.expert_layer2(x)
        normalized_expert = torch.sigmoid(expert_raw) # [B, K, D, H, W]
        
        # =================================================================
        # 【核心修正 2】：动态边界的硬重映射 (Dynamic Hard Remapping)
        # =================================================================
        # 我们根据路由器当前学到的对数阈值，实时计算各个专家的立方根工作区间
        
        # a) 将 log10 阈值转换到立方根空间
        th_linear = torch.pow(10.0, thresholds_log10)
        th_cubic = torch.pow(th_linear, 1.0/3.0) # 形状: [K-1]
        
        # b) 构建每个专家的 min_vals 和 max_vals
        min_vals = torch.zeros(self.num_experts, device=features.device)
        max_vals = torch.zeros(self.num_experts, device=features.device)
        
        if self.num_experts > 1:
            min_vals[1:] = th_cubic
            max_vals[:-1] = th_cubic
            
        # 最高的专家上限，取立方根空间的巨大值 (例如 10000^(1/3) ≈ 21.544)
        max_vals[-1] = 21.5443 
        
        # 变形以支持广播机制 [1, K, 1, 1, 1]
        min_vals = min_vals.view(1, -1, 1, 1, 1)
        max_vals = max_vals.view(1, -1, 1, 1, 1)
        
        # c) 强制重映射！专家只能在这个范围内输出
        # 这保证了无论特征多乱，最低级专家绝不可能输出高密度的值
        expert_outputs_cubic = min_vals + normalized_expert * (max_vals - min_vals)
        
        # 3. 动态加权融合 (Cubic-root Space)
        rho_cubic_fused = torch.sum(routing_weights * expert_outputs_cubic, dim=1, keepdim=True)
        
        # 4. 硬约束与转换为线性物理空间
        if not use_constraint:
            mask = torch.sigmoid((rho_cubic_fused - 1e-2) * 100)
            final_density = torch.pow(rho_cubic_fused * mask, 3.0)
        else:
            final_density = self.constraint(rho_cubic_fused, total_electrons)
            
        # 注意：这里返回的是 detached 的 guide_field，以保障训练逻辑的纯洁性
        return routing_weights, thresholds_log10, final_density, guide_field_detached