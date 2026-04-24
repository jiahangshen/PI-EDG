import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PhysicsUtils:
    @staticmethod
    def get_grid_coords(grid_res, grid_bounds, device):
        """生成物理坐标网格"""
        grid_min, grid_max = grid_bounds
        coords = torch.linspace(grid_min, grid_max, grid_res, device=device)
        grid_z, grid_y, grid_x = torch.meshgrid(coords, coords, coords, indexing='ij')
        # shape: [1, res, res, res, 3]
        return torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(0)

    @staticmethod
    def build_vext(atom_coords, atom_z, grid_coords, epsilon=1e-6):
        """
        在 GPU 上动态构建外部势 v_ext。
        V_ext(r) = sum_A -Z_A / |r - R_A|
        """
        # grid_coords: [1, res, res, res, 3]
        # atom_coords: [B, N_atoms, 3]
        # atom_z: [B, N_atoms]
        
        B, N_atoms, _ = atom_coords.shape
        # 调整维度以进行广播计算
        # pos: [1, res, res, res, 3] -> [1, 1, res, res, res, 3]
        grid_tensor = grid_coords.unsqueeze(1) 
        
        # atoms: [B, N_atoms, 3] -> [B, N_atoms, 1, 1, 1, 3]
        atom_pos = atom_coords.view(B, N_atoms, 1, 1, 1, 3)
        atom_charges = atom_z.view(B, N_atoms, 1, 1, 1)
        
        # 计算距离 |r - R_A|
        # dist: [B, N_atoms, res, res, res]
        dist = torch.norm(grid_tensor - atom_pos, dim=-1)
        
        # 计算电势 (添加 epsilon 防止除零)
        vext = -atom_charges / (dist + epsilon)
        
        # 对所有原子求和 -> [B, 1, res, res, res]
        total_vext = torch.sum(vext, dim=1, keepdim=True)
        return total_vext

class DifferentiablePoissonSolver(nn.Module):
    """基于 FFT 的泊松方程求解器，用于计算 Hartree 势"""
    def __init__(self, grid_res, grid_bounds):
        super().__init__()
        self.grid_res = grid_res
        grid_min, grid_max = grid_bounds
        L = grid_max - grid_min
        
        # 构建倒空间 k 向量
        freqs = torch.fft.fftfreq(grid_res, d=L/grid_res) * 2 * np.pi
        k_z, k_y, k_x = torch.meshgrid(freqs, freqs, freqs, indexing='ij')
        k_squared = k_x**2 + k_y**2 + k_z**2
        
        # 处理 k=0 奇点
        k_squared[0, 0, 0] = 1.0 
        self.register_buffer('inverse_k_squared', 1.0 / k_squared)
        self.inverse_k_squared[0, 0, 0] = 0.0 # V_H(k=0) 设为 0 (对应背景电荷中和)

    def forward(self, density):
        """
        Input: density [B, 1, D, H, W]
        Output: Hartree Potential V_H [B, 1, D, H, W]
        Formula: V_H(k) = 4 * pi * n(k) / |k|^2
        """
        # 1. 实空间 -> 倒空间
        density_k = torch.fft.fftn(density, dim=(-3, -2, -1))
        
        # 2. 求解泊松方程 (代数除法)
        vh_k = 4 * np.pi * density_k * self.inverse_k_squared
        
        # 3. 倒空间 -> 实空间
        vh_r = torch.fft.ifftn(vh_k, dim=(-3, -2, -1)).real
        
        return vh_r

class EnergyLoss(nn.Module):
    def __init__(self, grid_res, grid_bounds, lambda_1=1.0, lambda_2=0.1):
        super().__init__()
        self.lambda_1 = lambda_1 # 回归损失权重 (动态调整时会被覆盖)
        self.lambda_2 = lambda_2 # 物理能量损失权重 (动态调整时会被覆盖)
        
        self.poisson = DifferentiablePoissonSolver(grid_res, grid_bounds)
        self.mse = nn.MSELoss()
        
        # 空间微分步长
        self.h = (grid_bounds[1] - grid_bounds[0]) / grid_res


    def forward(self, n_pred, n_target, v_ext, curr_lambda_reg=None, curr_lambda_phys=None):
        """
        计算总损失
        新增参数: curr_lambda_reg, curr_lambda_phys 用于接收动态权重
        """
        # 使用传入的动态权重，如果没有传入则使用初始权重
        w_reg = curr_lambda_reg if curr_lambda_reg is not None else self.lambda_1
        w_phys = curr_lambda_phys if curr_lambda_phys is not None else self.lambda_2

        # --- 修改点: 使用立方根 (Cubic Root) 代替 Log ---
        # 虽然 n^(1/3) 在 n=0 时有定义，但导数是无穷大
        # 所以加一个极小的 epsilon (如 1e-8) 仅为了防止梯度爆炸（NaN）
        epsilon = 1e-8   
        
        # Power Transform
        # 注意：PyTorch 的 pow 对负数处理可能会产生 NaN，但我们的密度经过 Softplus 保证是非负的
        # 加上 epsilon 双重保险
        root_pred = torch.pow(n_pred + epsilon, 1.0/3.0)
        root_target = torch.pow(n_target + epsilon, 1.0/3.0)
        
        # 计算变换后的 MSE
        loss_density = self.mse(root_pred, root_target)
        
        # 2. 物理能量项计算 (积分 = 逐点求和 * 体积元 dV)
        dV = self.h ** 3
        
        # A. E_ne (核吸引) - 保留
        # 这是一个很强的负值，拉拢电子
        E_ne = torch.sum(n_pred * v_ext) * dV
        
        # B. E_H (电子排斥) - 保留
        # 这是一个正值，推开电子
        V_H = self.poisson(n_pred)
        E_H = 0.5 * torch.sum(n_pred * V_H) * dV
        
        # C. 动能正则化 - 只要 Weizsacker
        # 计算梯度
        grad_x = (torch.roll(n_pred, -1, dims=-1) - torch.roll(n_pred, 1, dims=-1)) / (2 * self.h)
        grad_y = (torch.roll(n_pred, -1, dims=-2) - torch.roll(n_pred, 1, dims=-2)) / (2 * self.h)
        grad_z = (torch.roll(n_pred, -1, dims=-3) - torch.roll(n_pred, 1, dims=-3)) / (2 * self.h)
        grad_sq = grad_x**2 + grad_y**2 + grad_z**2
        
        # T_W = (1/8) * |grad n|^2 / n
        # 加上 epsilon 防止除零
        t_w_density = (1.0/8.0) * grad_sq / (n_pred + 1e-8)
        T_W = torch.sum(t_w_density) * dV
        
        # D. 新的物理损失
        # 我们不再追求 E_total 的绝对值最小化（因为它缺项，肯定不准）
        # 我们追求的是：在满足 MSE 的前提下，让分布尽可能符合库仑力和动能平滑
        # E_ne 是负的，E_H 和 T_W 是正的，它们之间会形成一种制衡
        L_physics = E_ne + E_H + T_W 
        
        # 归一化到 Batch
        L_physics_mean = L_physics / n_pred.shape[0]
        
        # 总损失
        loss_total = w_reg * loss_density + w_phys * L_physics_mean
        
        return loss_total, loss_density, L_physics_mean, E_ne, E_H

    def lda_exchange_correlation(self, n):
        """
        LDA 交换相关能近似 (Slater Exchange + VWN Correlation 近似)
        这里使用简化版的 Dirac Exchange 和 Thomas-Fermi 风格
        E_x[n] = -Cx * integral(n^(4/3))
        """
        # 加上 epsilon 防止梯度爆炸
        rho = n + 1e-8
        Cx = (3/4) * (3/np.pi)**(1/3)
        e_x = -Cx * torch.pow(rho, 4/3)
        
        # 简化的相关能 (Wigner parametrization 示例)
        # 实际 DFT 软件通常用更复杂的 VWN，这里用 Wigner 足以提供负相关能
        rs = (3 / (4 * np.pi * rho))**(1/3)
        e_c = -0.88 / (rs + 7.8) * rho
        
        return e_x + e_c # 返回能量密度

    def thomas_fermi_weizsacker_kinetic(self, n):
        """
        Orbital-Free Kinetic Energy (T_s 近似)
        T[n] = T_TF[n] + \lambda * T_W[n]
        T_TF = C_F * integral(n^(5/3))
        T_W  = (1/8) * integral(|grad n|^2 / n)
        """
        rho = n + 1e-8
        
        # 1. Thomas-Fermi Term
        C_F = (3/10) * (3 * np.pi**2)**(2/3)
        t_tf = C_F * torch.pow(rho, 5/3)
        
        # 2. Weizsacker Term (Gradient correction)
        # 使用中心差分计算梯度
        grad_x = (torch.roll(rho, -1, dims=-1) - torch.roll(rho, 1, dims=-1)) / (2 * self.h)
        grad_y = (torch.roll(rho, -1, dims=-2) - torch.roll(rho, 1, dims=-2)) / (2 * self.h)
        grad_z = (torch.roll(rho, -1, dims=-3) - torch.roll(rho, 1, dims=-3)) / (2 * self.h)
        
        grad_sq = grad_x**2 + grad_y**2 + grad_z**2
        t_w = (1/8) * grad_sq / rho
        
        return t_tf + t_w

    