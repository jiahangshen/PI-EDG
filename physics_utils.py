import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PhysicsUtils:
    @staticmethod
    def get_grid_coords(grid_res, grid_bounds, device):
        grid_min, grid_max = grid_bounds
        coords = torch.linspace(grid_min, grid_max, grid_res, device=device)
        grid_z, grid_y, grid_x = torch.meshgrid(coords, coords, coords, indexing='ij')
        return torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(0)

    @staticmethod
    def build_vext(atom_coords, atom_z, grid_coords, epsilon=1e-6):
        B, N_atoms, _ = atom_coords.shape
        grid_tensor = grid_coords.unsqueeze(1)
        atom_pos = atom_coords.view(B, N_atoms, 1, 1, 1, 3)
        atom_charges = atom_z.view(B, N_atoms, 1, 1, 1)

        dist = torch.norm(grid_tensor - atom_pos, dim=-1)
        vext = -atom_charges / (dist + epsilon)
        total_vext = torch.sum(vext, dim=1, keepdim=True)
        return total_vext

class DifferentiablePoissonSolver(nn.Module):
    def __init__(self, grid_res, grid_bounds):
        super().__init__()
        self.grid_res = grid_res
        grid_min, grid_max = grid_bounds
        L = grid_max - grid_min

        freqs = torch.fft.fftfreq(grid_res, d=L/grid_res) * 2 * np.pi
        k_z, k_y, k_x = torch.meshgrid(freqs, freqs, freqs, indexing='ij')
        k_squared = k_x**2 + k_y**2 + k_z**2

        k_squared[0, 0, 0] = 1.0
        self.register_buffer('inverse_k_squared', 1.0 / k_squared)
        self.inverse_k_squared[0, 0, 0] = 0.0

    def forward(self, density):
        density_k = torch.fft.fftn(density, dim=(-3, -2, -1))
        vh_k = 4 * np.pi * density_k * self.inverse_k_squared
        vh_r = torch.fft.ifftn(vh_k, dim=(-3, -2, -1)).real
        return vh_r

class EnergyLoss(nn.Module):
    def __init__(self, grid_res, grid_bounds, lambda_1=1.0, lambda_2=0.1):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.poisson = DifferentiablePoissonSolver(grid_res, grid_bounds)
        self.mse = nn.MSELoss()
        self.h = (grid_bounds[1] - grid_bounds[0]) / grid_res

    def forward(self, n_pred, n_target, v_ext, curr_lambda_reg=None, curr_lambda_phys=None):
        w_reg = curr_lambda_reg if curr_lambda_reg is not None else self.lambda_1
        w_phys = curr_lambda_phys if curr_lambda_phys is not None else self.lambda_2

        epsilon = 1e-8
        root_pred = torch.pow(n_pred + epsilon, 1.0/3.0)
        root_target = torch.pow(n_target + epsilon, 1.0/3.0)

        loss_density = self.mse(root_pred, root_target)
        dV = self.h ** 3

        E_ne = torch.sum(n_pred * v_ext) * dV
        V_H = self.poisson(n_pred)
        E_H = 0.5 * torch.sum(n_pred * V_H) * dV

        grad_x = (torch.roll(n_pred, -1, dims=-1) - torch.roll(n_pred, 1, dims=-1)) / (2 * self.h)
        grad_y = (torch.roll(n_pred, -1, dims=-2) - torch.roll(n_pred, 1, dims=-2)) / (2 * self.h)
        grad_z = (torch.roll(n_pred, -1, dims=-3) - torch.roll(n_pred, 1, dims=-3)) / (2 * self.h)
        grad_sq = grad_x**2 + grad_y**2 + grad_z**2

        t_w_density = (1.0/8.0) * grad_sq / (n_pred + 1e-8)
        T_W = torch.sum(t_w_density) * dV

        L_physics = E_ne + E_H + T_W
        L_physics_mean = L_physics / n_pred.shape[0]
        loss_total = w_reg * loss_density + w_phys * L_physics_mean
        return loss_total, loss_density, L_physics_mean, E_ne, E_H

    def lda_exchange_correlation(self, n):
        rho = n + 1e-8
        Cx = (3/4) * (3/np.pi)**(1/3)
        e_x = -Cx * torch.pow(rho, 4/3)

        rs = (3 / (4 * np.pi * rho))**(1/3)
        e_c = -0.88 / (rs + 7.8) * rho

        return e_x + e_c

    def thomas_fermi_weizsacker_kinetic(self, n):
        rho = n + 1e-8

        C_F = (3/10) * (3 * np.pi**2)**(2/3)
        t_tf = C_F * torch.pow(rho, 5/3)

        grad_x = (torch.roll(rho, -1, dims=-1) - torch.roll(rho, 1, dims=-1)) / (2 * self.h)
        grad_y = (torch.roll(rho, -1, dims=-2) - torch.roll(rho, 1, dims=-2)) / (2 * self.h)
        grad_z = (torch.roll(rho, -1, dims=-3) - torch.roll(rho, 1, dims=-3)) / (2 * self.h)

        grad_sq = grad_x**2 + grad_y**2 + grad_z**2
        t_w = (1/8) * grad_sq / rho

        return t_tf + t_w
