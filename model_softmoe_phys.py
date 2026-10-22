import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np


def normalize_density_to_electrons(
    density, target_electrons, voxel_volume_bohr3=0.064, eps=1e-8
):
    """Differentiably normalize a nonnegative grid to a physical electron count."""
    positive = torch.clamp(density, min=0.0)
    current_electrons = positive.sum(
        dim=(1, 2, 3, 4), keepdim=True
    ) * float(voxel_volume_bohr3)
    target = target_electrons.reshape(-1, 1, 1, 1, 1)
    return positive * target / (current_electrons + float(eps))


def _resolve_group_count(num_channels, preferred_groups):
    groups = min(preferred_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return max(1, groups)


def _build_res_stage(in_channels, out_channels, num_blocks, groups):
    blocks = []
    for block_idx in range(num_blocks):
        block_in = in_channels if block_idx == 0 else out_channels
        blocks.append(ResBlock(block_in, out_channels, groups=groups))
    return nn.Sequential(*blocks)

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
    def __init__(
        self,
        in_channels,
        out_channels,
        modes=(4, 4, 4),
        groups=8,
    ):
        super().__init__()
        norm_groups = _resolve_group_count(out_channels, groups)
        self.conv_branch = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(inplace=True)
        )
        self.fno_branch = SpectralConv3d(in_channels, out_channels, *modes)
        self.act = nn.SiLU()
    def forward(self, x):
        result = self.conv_branch(x)
        result = result + self.fno_branch(x)
        return self.act(result)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        norm_groups = _resolve_group_count(out_channels, groups)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(norm_groups, out_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(norm_groups, out_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.project = nn.Conv3d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
    def forward(self, x): return self.act2(self.gn2(self.conv2(self.act1(self.gn1(self.conv1(x))))) + self.project(x))

class PhysicsResUNet(nn.Module):
    def __init__(
        self,
        in_channels_atom,
        in_channels_pe=0,
        n_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        bottleneck_modes=(4, 4, 4),
        norm_groups=8,
        encoder_block_counts=(1, 1, 1, 1),
        decoder_block_counts=(1, 1, 1),
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.phys_channels = 2
        total_in_channels = in_channels_atom + self.phys_channels + in_channels_pe
        c1 = n_channels * channel_multipliers[0]
        c2 = n_channels * channel_multipliers[1]
        c3 = n_channels * channel_multipliers[2]
        c4 = n_channels * channel_multipliers[3]

        self.inc = _build_res_stage(total_in_channels, c1, encoder_block_counts[0], groups=norm_groups)
        self.down1 = nn.MaxPool3d(2); self.conv1 = _build_res_stage(c1 + self.phys_channels, c2, encoder_block_counts[1], groups=norm_groups)
        self.down2 = nn.MaxPool3d(2); self.conv2 = _build_res_stage(c2 + self.phys_channels, c3, encoder_block_counts[2], groups=norm_groups)
        self.down3 = nn.MaxPool3d(2)
        bottleneck_blocks = [
            HybridBottleneck(
                c3 + self.phys_channels,
                c4,
                modes=bottleneck_modes,
                groups=norm_groups,
            )
        ]
        for _ in range(max(0, encoder_block_counts[3] - 1)):
            bottleneck_blocks.append(
                HybridBottleneck(
                    c4,
                    c4,
                    modes=bottleneck_modes,
                    groups=norm_groups,
                )
            )
        self.bottleneck = nn.Sequential(*bottleneck_blocks)

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1 = _build_res_stage(c4 + c3 + self.phys_channels, c3, decoder_block_counts[0], groups=norm_groups)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2 = _build_res_stage(c3 + c2 + self.phys_channels, c2, decoder_block_counts[1], groups=norm_groups)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec3 = _build_res_stage(c2 + c1 + self.phys_channels, c1, decoder_block_counts[2], groups=norm_groups)
        self.out_features = c1
        self.use_activation_checkpointing = use_activation_checkpointing

    def _run_block(self, module, *args):
        if self.use_activation_checkpointing and self.training:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def inject_physics(self, feat, v_ext, z_spike):
        target_size = feat.shape[2:]
        v_scaled = F.interpolate(v_ext, size=target_size, mode='trilinear', align_corners=False)
        z_scaled = F.interpolate(z_spike, size=target_size, mode='trilinear', align_corners=False)
        return torch.cat([feat, v_scaled, z_scaled], dim=1)

    def forward(self, atom_grid, pe_grid, v_ext, z_spike):
        if v_ext.dim() == 4:
            v_ext = v_ext.unsqueeze(1)
        if z_spike.dim() == 4:
            z_spike = z_spike.unsqueeze(1)

        if pe_grid is not None:
            if pe_grid.dim() == 4:
                B = atom_grid.shape[0]
                pe_grid = pe_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)

            x0 = torch.cat([atom_grid, pe_grid, v_ext, z_spike], dim=1)
        else:
            x0 = torch.cat([atom_grid, v_ext, z_spike], dim=1)
        x1 = self._run_block(self.inc, x0)
        x2_pre = self.down1(x1); x2 = self._run_block(self.conv1, self.inject_physics(x2_pre, v_ext, z_spike))
        x3_pre = self.down2(x2); x3 = self._run_block(self.conv2, self.inject_physics(x3_pre, v_ext, z_spike))
        x4_pre = self.down3(x3); x_bottle = self._run_block(self.bottleneck, self.inject_physics(x4_pre, v_ext, z_spike))
        xu1 = self.inject_physics(torch.cat([self.up1(x_bottle), x3], dim=1), v_ext, z_spike); xd1 = self._run_block(self.dec1, xu1)
        xu2 = self.inject_physics(torch.cat([self.up2(xd1), x2], dim=1), v_ext, z_spike); xd2 = self._run_block(self.dec2, xu2)
        xu3 = self.inject_physics(torch.cat([self.up3(xd2), x1], dim=1), v_ext, z_spike); xd3 = self._run_block(self.dec3, xu3)
        return xd3

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
        boundary_logits = self.temperature * (guide_field_log10 - t)

        boundaries = torch.sigmoid(boundary_logits)

        probs = []
        probs.append(1.0 - boundaries[:, 0:1, ...])
        for i in range(self.num_experts - 2):
            probs.append(boundaries[:, i:i+1, ...] - boundaries[:, i+1:i+2, ...])
        probs.append(boundaries[:, -1:, ...])

        routing_weights = torch.cat(probs, dim=1)
        routing_weights = routing_weights / (torch.sum(routing_weights, dim=1, keepdim=True) + 1e-8)

        return routing_weights, thresholds

class PhysicsMoE(nn.Module):
    def __init__(
        self,
        in_channels_atom,
        in_channels_pe,
        num_experts,
        initial_bins,
        n_channels=64,
        backbone_channel_multipliers=(1, 2, 4, 8),
        bottleneck_modes=(4, 4, 4),
        norm_groups=8,
        encoder_block_counts=(1, 1, 1, 1),
        decoder_block_counts=(1, 1, 1),
        guide_hidden_channels=None,
        expert_hidden_channels=32,
        shared_expert_depth=2,
        routing_backbone_grad_scale=1.0,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.routing_backbone_grad_scale = routing_backbone_grad_scale
        self.use_activation_checkpointing = use_activation_checkpointing

        self.backbone = PhysicsResUNet(
            in_channels_atom,
            in_channels_pe,
            n_channels=n_channels,
            channel_multipliers=backbone_channel_multipliers,
            bottleneck_modes=bottleneck_modes,
            norm_groups=norm_groups,
            encoder_block_counts=encoder_block_counts,
            decoder_block_counts=decoder_block_counts,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        guide_hidden_channels = guide_hidden_channels or max(self.backbone.out_features // 2, 16)

        self.guide_predictor_log10 = nn.Sequential(
            nn.Conv3d(self.backbone.out_features, guide_hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(guide_hidden_channels, 1, kernel_size=1)
        )

        self.router = LogSpaceLearnableRouter(num_experts, initial_bins, temperature=2.0)

        expert_norm_groups = _resolve_group_count(expert_hidden_channels, norm_groups)
        shared_expert_layers = []
        shared_in_channels = self.backbone.out_features
        for _ in range(shared_expert_depth):
            shared_expert_layers.extend([
                nn.Conv3d(shared_in_channels, expert_hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(expert_norm_groups, expert_hidden_channels),
                nn.SiLU(inplace=True),
            ])
            shared_in_channels = expert_hidden_channels
        self.shared_expert_trunk = nn.Sequential(*shared_expert_layers)
        self.expert_out = nn.Conv3d(expert_hidden_channels, num_experts, kernel_size=1)

    def forward(self, atom_grid, pe_grid, v_ext, z_spike):
        features = self.backbone(atom_grid, pe_grid, v_ext, z_spike)
        guide_field_attached = self.guide_predictor_log10(features)
        guide_field_detached = self.guide_predictor_log10(features.detach())
        guide_field_routing = (
            self.routing_backbone_grad_scale * guide_field_attached
            + (1.0 - self.routing_backbone_grad_scale) * guide_field_attached.detach()
        )

        routing_weights, thresholds_log10 = self.router(guide_field_routing)

        if self.use_activation_checkpointing and self.training:
            expert_features = checkpoint(self.shared_expert_trunk, features, use_reentrant=False)
        else:
            expert_features = self.shared_expert_trunk(features)
        expert_raw = self.expert_out(expert_features)

        th_linear = torch.pow(10.0, thresholds_log10)
        th_cubic = torch.pow(th_linear, 1.0/3.0)

        min_vals = torch.zeros(self.num_experts, device=features.device)
        max_vals = torch.zeros(self.num_experts, device=features.device)

        min_vals[1:] = th_cubic
        max_vals[:-1] = th_cubic

        max_vals[-1] = 21.5443
        min_vals = min_vals.view(1, -1, 1, 1, 1)
        max_vals = max_vals.view(1, -1, 1, 1, 1)

        normalized_expert = torch.sigmoid(expert_raw)
        expert_outputs_cubic = min_vals + normalized_expert * (max_vals - min_vals)
        rho_cubic_fused = torch.sum(
            routing_weights * expert_outputs_cubic, dim=1, keepdim=True
        )
        mask = torch.sigmoid((rho_cubic_fused - 1e-2) * 100)
        final_density = torch.pow(rho_cubic_fused * mask, 3.0)

        return routing_weights, thresholds_log10, final_density, guide_field_detached
