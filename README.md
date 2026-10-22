# PI-EDG

**Physics-Informed Full-Space Electron Density Generation from Molecular Geometry**

PI-EDG predicts a molecule's electron density on a `64³` grid from its atomic elements and coordinates. This repository provides the paper-version voxel model, training code, and inference scripts. It does not include EDBench data or pretrained weights.

## Download EDBench

Get the data from the [official EDBench repository](https://github.com/HongxinXiang/EDBench):

- [Full EDBench release (Baidu Drive)](https://pan.baidu.com/s/1VNoN8Uwo5XCyzk7z19X7Rg?pwd=yxy4), for the complete molecular collection. The extraction code is `yxy4`.
- [ED5-EDP electron-density prediction subset (Harvard Dataverse)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SLYBD1), listed by EDBench as `ed_prediction_5w`.

Use each molecule's geometry `.sdf` and **electron-density** `Mol1_Dt.cube`. Do not use `Mol1_ESP.cube`, which contains electrostatic potential rather than electron density. The provided [`splits/`](splits/) contain molecule IDs only; check that the download contains every ID you intend to train or evaluate on.

The cache builder expects one numeric directory per molecule with one `.sdf` and one `.cube` file. If the downloaded directories also contain an ESP CUBE, stage only the density CUBE in the input tree:

```text
/path/to/edbench/input/
└── shard_000/
    └── 19412/
        ├── Mol1.sdf
        └── Mol1_Dt.cube
```

Nested shard directories are supported. EDBench remains governed by its own distribution terms; it is not redistributed here.

## Install and prepare

Use a CUDA-compatible PyTorch installation, then install the remaining requirements:

```bash
python -m pip install -r requirements.txt
```

Build the reusable `64³` cache once:

```bash
python preprocess_cache.py \
  --raw_data_root /path/to/edbench/input \
  --cache_dir /path/to/edbench/cache \
  --recursive_sample_scan
```

## Train

```bash
python train.py \
  --devices 0 \
  --cache-dir /path/to/edbench/cache \
  --train-ids splits/train_ids.npy \
  --val-ids splits/validation_ids.npy
```

The default model is the PI-EDG Hybrid FNO–CNN backbone with density-space experts. Adjust training options with `--help` for your hardware and use case.

## Predict and evaluate

Predict from a neutral SDF or XYZ structure using a trained Lightning checkpoint:

```bash
python predict.py \
  --checkpoint /path/to/best.ckpt \
  --structure /path/to/molecule.sdf \
  --output outputs/molecule.npz \
  --voxel-volume-bohr3 0.064
```

Evaluate an ID list against the prepared cache:

```bash
python evaluate.py \
  --checkpoint /path/to/best.ckpt \
  --cache-dir /path/to/edbench/cache \
  --sample-ids /path/to/evaluation_ids.txt \
  --output-dir outputs/evaluation \
  --voxel-volume-bohr3 0.064
```

Set `--voxel-volume-bohr3` to match your source-density integration convention. The supplied 2,000 evaluation IDs are a subset of the validation IDs, not an independent test set; see [`splits/provenance.json`](splits/provenance.json). Numerical details that differ from simplified manuscript equations are listed in [Implementation notes](IMPLEMENTATION_NOTES.md).

## License

Code is released under the [MIT License](LICENSE).
