import argparse
from pathlib import Path

import numpy as np
from ase.io import read as ase_read
from cube_reader import read_cube
from tqdm import tqdm

RAW_DATA_ROOT = Path("data/raw_data_cube")
CACHE_DIR = Path("data/preprocessed_cache")
BOHR_TO_ANGSTROM = 0.529177210903


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate PI-EDG preprocessed cache files.")
    parser.add_argument("--raw_data_root", type=Path, default=RAW_DATA_ROOT)
    parser.add_argument("--cache_dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--sample_id", type=str, default=None, help="Only rebuild one sample id, e.g. 1533.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache files.")
    parser.add_argument(
        "--recursive_sample_scan",
        action="store_true",
        help="Recursively scan nested shard/sample directories instead of assuming sample dirs sit directly under raw_data_root.",
    )
    parser.add_argument("--grid_res", type=int, default=64)
    parser.add_argument("--grid_bounds", type=float, nargs=2, default=(-10.0, 10.0))
    parser.add_argument(
        "--write_batch_size",
        type=int,
        default=64,
        help="Maximum number of prepared samples held in memory before writing their individual cache files.",
    )
    return parser.parse_args()


def load_atoms_with_cube_fallback(sdf_file_path, cube_file_path):
    try:
        atoms_ase = ase_read(str(sdf_file_path), format="sdf")
        atom_coords = atoms_ase.get_positions().astype(np.float32)
        atom_z_numbers = atoms_ase.get_atomic_numbers().astype(np.int16)
        return atom_coords, atom_z_numbers
    except Exception as exc:
        print(f"Failed to read SDF; falling back to the CUBE header atom list: {sdf_file_path} ({exc})")

    with cube_file_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 7:
        raise ValueError(f"Incomplete CUBE header: {cube_file_path}")

    num_atoms = abs(int(lines[2].split()[0]))
    atom_lines = lines[6 : 6 + num_atoms]
    if len(atom_lines) != num_atoms:
        raise ValueError(f"Incomplete CUBE atom list: {cube_file_path}")

    atom_z_numbers = []
    atom_coords_bohr = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed CUBE atom line: {line.rstrip()}")
        atom_z_numbers.append(int(float(parts[0])))
        atom_coords_bohr.append([float(parts[2]), float(parts[3]), float(parts[4])])

    atom_coords = np.asarray(atom_coords_bohr, dtype=np.float32) * BOHR_TO_ANGSTROM
    atom_z_numbers = np.asarray(atom_z_numbers, dtype=np.int16)
    return atom_coords, atom_z_numbers


def build_target_density_fast(
    density_data,
    cube_origin_bohr,
    step_vec_x_bohr,
    step_vec_y_bohr,
    step_vec_z_bohr,
    centroid_ang,
    grid_res,
    grid_bounds,
):
    target = np.zeros((grid_res, grid_res, grid_res), dtype=np.float32)
    grid_min, grid_max = grid_bounds
    voxel_size = (grid_max - grid_min) / grid_res
    if voxel_size <= 0:
        return target

    step_matrix = np.stack([step_vec_x_bohr, step_vec_y_bohr, step_vec_z_bohr], axis=0)
    offdiag = step_matrix.copy()
    np.fill_diagonal(offdiag, 0.0)

    if np.max(np.abs(offdiag)) < 1e-8:
        nx, ny, nz = density_data.shape
        x_coords = (cube_origin_bohr[0] + np.arange(nx, dtype=np.float32) * step_vec_x_bohr[0]) * BOHR_TO_ANGSTROM - centroid_ang[0]
        y_coords = (cube_origin_bohr[1] + np.arange(ny, dtype=np.float32) * step_vec_y_bohr[1]) * BOHR_TO_ANGSTROM - centroid_ang[1]
        z_coords = (cube_origin_bohr[2] + np.arange(nz, dtype=np.float32) * step_vec_z_bohr[2]) * BOHR_TO_ANGSTROM - centroid_ang[2]

        x_bins = ((x_coords - grid_min) / voxel_size).astype(np.int32)
        y_bins = ((y_coords - grid_min) / voxel_size).astype(np.int32)
        z_bins = ((z_coords - grid_min) / voxel_size).astype(np.int32)

        valid_x = (x_bins >= 0) & (x_bins < grid_res)
        valid_y = (y_bins >= 0) & (y_bins < grid_res)
        valid_z = (z_bins >= 0) & (z_bins < grid_res)
        if not (valid_x.any() and valid_y.any() and valid_z.any()):
            return target

        x_bins = x_bins[valid_x]
        y_bins = y_bins[valid_y]
        z_bins = z_bins[valid_z]
        sub_density = density_data[np.ix_(valid_x, valid_y, valid_z)].astype(np.float32, copy=False)

        gx, gy, gz = np.meshgrid(x_bins, y_bins, z_bins, indexing="ij")
        np.add.at(target, (gz.ravel(), gy.ravel(), gx.ravel()), sub_density.ravel())
        return target

    nx, ny, nz = density_data.shape
    i_indices, j_indices, k_indices = np.indices((nx, ny, nz), dtype=np.int32)
    indices_matrix = np.stack([i_indices.ravel(), j_indices.ravel(), k_indices.ravel()], axis=1)
    all_coords_bohr = cube_origin_bohr + indices_matrix @ step_matrix
    all_coords_ang = all_coords_bohr * BOHR_TO_ANGSTROM
    centered_coords_ang = all_coords_ang - centroid_ang[None, :]
    target_all_indices = ((centered_coords_ang - grid_min) / voxel_size).astype(np.int32)
    valid_mask = np.all((target_all_indices >= 0) & (target_all_indices < grid_res), axis=1)
    valid_target_indices = target_all_indices[valid_mask]
    valid_density_values = density_data.ravel()[valid_mask].astype(np.float32, copy=False)
    if len(valid_target_indices) == 0:
        return target
    vx = valid_target_indices[:, 0]
    vy = valid_target_indices[:, 1]
    vz = valid_target_indices[:, 2]
    np.add.at(target, (vz, vy, vx), valid_density_values)
    return target


def build_cache_payload(atom_coords, atom_z_numbers, density_data, cube_origin_bohr, step_vec_x_bohr, step_vec_y_bohr, step_vec_z_bohr, grid_res, grid_bounds):
    centroid = atom_coords.mean(axis=0).astype(np.float32)
    centered_atom_coords = (atom_coords - centroid).astype(np.float32, copy=False)
    target_density = build_target_density_fast(
        density_data=density_data,
        cube_origin_bohr=cube_origin_bohr,
        step_vec_x_bohr=step_vec_x_bohr,
        step_vec_y_bohr=step_vec_y_bohr,
        step_vec_z_bohr=step_vec_z_bohr,
        centroid_ang=centroid,
        grid_res=grid_res,
        grid_bounds=grid_bounds,
    )
    return {
        "atom_coords": centered_atom_coords,
        "atom_z_numbers": atom_z_numbers.astype(np.int16, copy=False),
        "target_density": target_density,
    }


def flush_pending_writes(pending_writes):
    for output_npz_path, payload in pending_writes:
        np.savez(output_npz_path, **payload)
    pending_writes.clear()


def main():
    args = parse_args()
    raw_data_root = args.raw_data_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()

    print(f"Building cache files (write_batch_size={args.write_batch_size})...")
    print(f"Scanning {raw_data_root}...")
    valid_samples = []
    if args.sample_id is not None:
        if raw_data_root.name == args.sample_id:
            sample_dirs = [raw_data_root]
        elif (raw_data_root / args.sample_id).is_dir():
            sample_dirs = [raw_data_root / args.sample_id]
        else:
            sample_dirs = [path for path in raw_data_root.rglob(args.sample_id) if path.is_dir()]
    else:
        if args.recursive_sample_scan:
            discovered = {}
            for sdf_path in raw_data_root.rglob("*.sdf"):
                sample_dir = sdf_path.parent
                if sample_dir.name.isdigit():
                    discovered[str(sample_dir)] = sample_dir
            sample_dirs = list(discovered.values())
        else:
            sample_dirs = list(raw_data_root.iterdir())

    for sample_dir in tqdm(sample_dirs, desc="Scanning directories"):
        if not sample_dir.name.isdigit():
            continue
        if any(sample_dir.glob("*.sdf")) and any(sample_dir.glob("*.cube")):
            sample_id = sample_dir.name
            valid_samples.append((sample_id, sample_dir))
    sample_ids = [sample_id for sample_id, _ in valid_samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate numeric sample IDs across raw-data shards")
    print(f"Found {len(valid_samples)} valid samples.")
    if not valid_samples:
        raise FileNotFoundError(f"No usable numeric sample directories under {raw_data_root}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache files will be saved to: {cache_dir}")
    print(f"Processing {len(valid_samples)} molecules and building cache files...")

    grid_bounds = tuple(args.grid_bounds)
    pending_writes = []

    for sample_id, sample_dir in tqdm(valid_samples, desc="Building cache"):
        output_npz_path = cache_dir / f"{sample_id}.npz"
        if output_npz_path.exists() and not args.force:
            continue
        try:
            sdf_file_path = next(iter(sorted(sample_dir.glob("*.sdf"))))
            cube_files = sorted(sample_dir.glob("*.cube"))
            if len(cube_files) != 1:
                raise ValueError(f"Expected one density CUBE in {sample_dir}, found {len(cube_files)}")
            cube_file_path = cube_files[0]
            atom_coords, atom_z_numbers = load_atoms_with_cube_fallback(sdf_file_path, cube_file_path)
            cube_obj = read_cube(cube_file_path)
            density_data = np.asarray(cube_obj.data, dtype=np.float32)
            payload = build_cache_payload(
                atom_coords=atom_coords,
                atom_z_numbers=atom_z_numbers,
                density_data=density_data,
                cube_origin_bohr=cube_obj.origin.astype(np.float32),
                step_vec_x_bohr=cube_obj.X.astype(np.float32),
                step_vec_y_bohr=cube_obj.Y.astype(np.float32),
                step_vec_z_bohr=cube_obj.Z.astype(np.float32),
                grid_res=args.grid_res,
                grid_bounds=grid_bounds,
            )
            pending_writes.append((output_npz_path, payload))
            if len(pending_writes) >= args.write_batch_size:
                flush_pending_writes(pending_writes)
        except Exception as e:
            raise RuntimeError(f"Cannot preprocess sample {sample_id}") from e

    if pending_writes:
        flush_pending_writes(pending_writes)

    print("\nAll cache files have been built.")


if __name__ == "__main__":
    main()
