"""Geometry voxelization used by the PI-EDG input pipeline."""

import numpy as np
import torch


ATOM_MAP = {
    1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 14: 6,
    15: 7, 16: 8, 17: 9, 32: 10, 34: 11, 35: 12,
}


def voxelize_structure(
    atom_coords: np.ndarray,
    atom_z_numbers: np.ndarray,
    grid_res: int,
    grid_bounds: tuple,
    atom_map: dict = ATOM_MAP,
    sigma: float = 0.1,
) -> torch.Tensor:
    grid_min, grid_max = grid_bounds
    num_channels = len(atom_map)
    voxel_grid = torch.zeros((num_channels, grid_res, grid_res, grid_res), dtype=torch.float32)

    voxel_size = (grid_max - grid_min) / grid_res
    if voxel_size <= 0:
        raise ValueError("grid_bounds must define a positive box")

    coords_1d = torch.linspace(grid_min + voxel_size / 2, grid_max - voxel_size / 2, grid_res)
    gaussian_radius = max(1, int(np.ceil((4.0 * sigma) / voxel_size)))
    denom = 2.0 * sigma**2

    for atom_pos_np, atom_z in zip(atom_coords, atom_z_numbers):
        atom_z = int(atom_z)
        if atom_z not in atom_map:
            raise ValueError(f"Unsupported atomic number: {atom_z}")

        channel_idx = atom_map[atom_z]
        atom_pos = torch.as_tensor(atom_pos_np, dtype=torch.float32)
        center_idx = torch.round((atom_pos - grid_min) / voxel_size - 0.5).to(torch.long)
        x0 = max(0, int(center_idx[0]) - gaussian_radius)
        x1 = min(grid_res, int(center_idx[0]) + gaussian_radius + 1)
        y0 = max(0, int(center_idx[1]) - gaussian_radius)
        y1 = min(grid_res, int(center_idx[1]) + gaussian_radius + 1)
        z0 = max(0, int(center_idx[2]) - gaussian_radius)
        z1 = min(grid_res, int(center_idx[2]) + gaussian_radius + 1)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            continue

        local_z, local_y, local_x = coords_1d[z0:z1], coords_1d[y0:y1], coords_1d[x0:x1]
        grid_z, grid_y, grid_x = torch.meshgrid(local_z, local_y, local_x, indexing="ij")
        dx = grid_x - atom_pos[0]
        dy = grid_y - atom_pos[1]
        dz = grid_z - atom_pos[2]
        gaussian_blob = torch.exp(-(dx * dx + dy * dy + dz * dz) / denom)

        current = voxel_grid[channel_idx, z0:z1, y0:y1, x0:x1]
        voxel_grid[channel_idx, z0:z1, y0:y1, x0:x1] = torch.maximum(current, gaussian_blob)

    return voxel_grid
