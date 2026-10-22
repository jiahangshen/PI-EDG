# PI-EDG: paper-method code

This directory is a **release candidate** for the voxel PI-EDG method in
*PI-EDG: Physics-Informed Full-Space Electron Density Generation from Molecular
Geometry* (latest PDF supplied by the author, 22 October 2026). It contains
the physics-aware 3D backbone, hybrid FNO–CNN bottleneck, 37-way density-space
MoE, geometry-to-cache preprocessing, training, single-molecule prediction, and
ID-paired grid evaluation. It deliberately excludes later F3/S3 continuous-core
experiments, model weights, raw EDBench data, caches, logs, and third-party
baseline source.

**Read [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) before publishing or claiming
exact numerical reproduction.** The manuscript and recorded reviewer-run
configuration disagree on several material details. The code here preserves
the implementation; it does not invent a checkpoint or claim that running the
commands below automatically reproduces a reported MAE.

## Code map

| File | Purpose |
| --- | --- |
| `model_softmoe_phys.py` | PhysicsResUNet, Hybrid FNO–CNN, log-space router, bounded experts, optional inference normalization |
| `physics_utils.py` | External nuclear potential on the voxel grid |
| `dataset_phys.py`, `dataset_moe_phys.py` | Geometry and DFT-density voxelization, 37-bin cached dataset |
| `cube_reader.py`, `preprocess_cache.py` | Read single-field CUBE and SDF files; build portable NPZ cache |
| `train_softmoe_phys_duoka.py`, `training_protocol.py` | Training loop and scheduling controls |
| `predict.py` | Raw and neutral-electron-normalized density from SDF/XYZ geometry |
| `evaluate.py` | Per-molecule raw/normalized MAE, RMSE, Pearson, electron-number error |
| `splits/` | Portable sample IDs only; no machine-specific paths |

## Setup

The manuscript reports Python 3.8.20, PyTorch 2.4.1+cu121 and CUDA 12.1.
Install a PyTorch build compatible with your GPU, then install
`requirements.txt`. Other CUDA/PyTorch versions may work but have not been
verified for the paper results.
The optional `--optimizer_type adafactor` switch additionally requires a
compatible `transformers` installation; the paper-method command uses AdamW.

```bash
python -m pip install -r requirements.txt
```

Obtain the EDBench SDF/CUBE data from its original distributor. This repository
does not redistribute EDBench or any DFT grids. The preprocessing command
expects numeric sample directories containing one `.sdf` and one single-field
`.cube` each, directly or under nested shards:

```bash
python preprocess_cache.py \
  --raw_data_root /path/to/edbench/raw \
  --cache_dir /path/to/edbench/cache \
  --recursive_sample_scan \
  --cache_format train_ready_grid64
```

The `train_ready_grid64` cache stores the historical hard-binned `64³`
density target and centered atomic coordinates, as in the current research
cache. `legacy_points` is available for an older point-list workflow but is
larger and requires re-binning at load time. Preprocessing now
fails loudly on a malformed sample instead of silently reducing the data set.
The reader supports ordinary single-field CUBE files with Bohr axis vectors.

## Train the paper-method configuration

The portable ID lists in `splits/` came from the research split manifests:
18,689 training IDs and 2,076 validation IDs. The reported 2,000-molecule
subset is contained **within** those 2,076 validation IDs; it is not an
independent held-out test set. See `splits/provenance.json` for counts and
checksums. Keep this distinction in mind when interpreting reported results.

```bash
python train_softmoe_phys_duoka.py \
  --devices 0 \
  --raw_data_root /path/to/edbench/raw \
  --cache_dir /path/to/edbench/cache \
  --train_id_file splits/train_ids.npy \
  --val_id_file splits/validation_ids.npy \
  --strict_data --bins_mode decade_exact \
  --model_scale base --batch_size 8 --max_epochs 30 \
  --learning_rate 1e-5 --threshold_lr_scale 0.1 \
  --lambda_global 1 --lambda_guide 1e-5 \
  --constraint_start_epoch -1 --hard_routing_epoch -1 \
  --skip_lr_find --logger_type csv
```

This uses the paper's tabulated learning rate. A recorded reviewer run used
`1e-4`, stopped after 12 epochs, and set
`--routing_backbone_grad_scale 0.3`; see the audit. Training uses the raw
network output, **not** training-time electron normalization. The CLI contains
additional research switches, but the paper method keeps core-precision and
packed-data extensions disabled.

## Predict and evaluate

After training, point to a Lightning `.ckpt` saved under `checkpoints_hybrid`:

```bash
python predict.py --checkpoint /path/to/best.ckpt \
  --structure /path/to/molecule.sdf --output outputs/molecule.npz \
  --voxel-volume-bohr3 0.064

python evaluate.py --checkpoint /path/to/best.ckpt \
  --cache-dir /path/to/edbench/cache \
  --sample-ids splits/reported_2000_ids.txt \
  --output-dir outputs/evaluation --voxel-volume-bohr3 0.064
```

`predict.py` writes arrays in internal `(z,y,x)` order and records the
centroid used to center the molecule. The normalized grid uses
`N_e max(ρ,0)/(Σ max(ρ,0) ΔV + 1e-8)` with `N_e=Σ Z_A` for a neutral
molecule. **The voxel volume must match the density cache's integration
convention**; `0.064 Bohr³` corresponds to a `0.4 Bohr` source CUBE step, not
to the geometric volume of the displayed `64³` box. `evaluate.py` compares
both raw and normalized predictions with the cached, unnormalized DFT target.

## Scope and release checks

Only the original voxel model is packaged. The later trilinear F3 labels,
continuous S3 atom-centered branch, large-scale 2-million-pool pipeline, and
GPU-specific cache paths are intentionally absent. No model weights are
included. The package is source code, not a validated reproduction of all
tables or baseline implementations in the paper. Run `python -m unittest
discover -s tests` for the included smoke checks.

The source code in this release is available under the MIT license. The named
GitHub release should also be checked against any still-active anonymous-review
requirements.
