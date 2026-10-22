import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cube_reader import read_cube
from dataset_moe_phys import create_pe_grid
from model_softmoe_phys import normalize_density_to_electrons
from train_softmoe_phys_duoka import MoEPhysModule, build_initial_bins


class ReleaseSmokeTests(unittest.TestCase):
    def test_cube_reader_axis_and_data_order(self):
        content = (
            "comment one\ncomment two\n"
            "1 0.0 0.0 0.0\n"
            "2 0.4 0.0 0.0\n"
            "2 0.0 0.4 0.0\n"
            "2 0.0 0.0 0.4\n"
            "1 0.0 0.0 0.0 0.0\n"
            "0 1 2 3 4 5 6 7\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.cube"
            path.write_text(content, encoding="utf-8")
            parsed = read_cube(path)
        self.assertEqual(parsed.data.shape, (2, 2, 2))
        self.assertEqual(float(parsed.data[1, 0, 1]), 5.0)
        np.testing.assert_allclose(parsed.X, [0.4, 0, 0])

    def test_neutral_normalization_conserves_electrons(self):
        density = torch.tensor([[[[[1.0, -1.0], [2.0, 0.0]]]]])
        result = normalize_density_to_electrons(density, torch.tensor([6.0]), 0.5)
        self.assertTrue(bool((result >= 0).all()))
        self.assertAlmostEqual(float(result.sum() * 0.5), 6.0, places=5)

    def test_nested_preprocessing_keeps_numeric_split_id(self):
        cube = (
            "comment one\ncomment two\n"
            "1 0.0 0.0 0.0\n"
            "2 0.4 0.0 0.0\n"
            "2 0.0 0.4 0.0\n"
            "2 0.0 0.0 0.4\n"
            "1 0.0 0.0 0.0 0.0\n"
            "1 1 1 1 1 1 1 1\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "raw" / "shard_000" / "19412"
            sample.mkdir(parents=True)
            (sample / "density.cube").write_text(cube, encoding="utf-8")
            (sample / "geometry.sdf").write_text("invalid SDF\n", encoding="utf-8")
            command = [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "preprocess_cache.py"),
                "--raw_data_root", str(root / "raw"),
                "--cache_dir", str(root / "cache"),
                "--recursive_sample_scan", "--grid_res", "8",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertTrue((root / "cache" / "19412.npz").exists())
            self.assertFalse((root / "cache" / "shard_000__19412.npz").exists())

    def test_paper_base_has_37_experts_and_expected_parameters(self):
        args = SimpleNamespace(expert_count=0, bins_mode="decade_exact",
                               decade_min_exp=-34, decade_max_exp=1)
        bins = build_initial_bins(args)
        self.assertEqual(len(bins) - 1, 37)
        model = MoEPhysModule(13, 24, 37, bins)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 50_661_546)
        self.assertEqual(create_pe_grid(8).shape, (1, 24, 8, 8, 8))

    def test_portable_split_sets(self):
        root = Path(__file__).resolve().parents[1] / "splits"
        train = set((root / "train_ids.txt").read_text().splitlines())
        validation = set((root / "validation_ids.txt").read_text().splitlines())
        reported = set((root / "reported_2000_ids.txt").read_text().splitlines())
        self.assertEqual((len(train), len(validation), len(reported)), (18689, 2076, 2000))
        self.assertFalse(train & validation)
        self.assertTrue(reported <= validation)


if __name__ == "__main__":
    unittest.main()
