"""Geometry-derived nuclear external potential for PI-EDG."""

import torch


class PhysicsUtils:
    @staticmethod
    def get_grid_coords(grid_res, grid_bounds, device):
        grid_min, grid_max = grid_bounds
        coords = torch.linspace(grid_min, grid_max, grid_res, device=device)
        grid_z, grid_y, grid_x = torch.meshgrid(coords, coords, coords, indexing="ij")
        return torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(0)

    @staticmethod
    def build_vext(atom_coords, atom_z, grid_coords, epsilon=1e-6):
        batch_size, n_atoms, _ = atom_coords.shape
        grid_tensor = grid_coords.unsqueeze(1)
        atom_pos = atom_coords.view(batch_size, n_atoms, 1, 1, 1, 3)
        atom_charges = atom_z.view(batch_size, n_atoms, 1, 1, 1)
        dist = torch.norm(grid_tensor - atom_pos, dim=-1)
        return torch.sum(-atom_charges / (dist + epsilon), dim=1, keepdim=True)
