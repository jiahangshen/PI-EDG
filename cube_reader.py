"""Read single-field Gaussian CUBE files with coordinates in Bohr."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Cube:
    origin: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    data: np.ndarray


def read_cube(path: str | Path) -> Cube:
    with Path(path).open("r", encoding="utf-8") as handle:
        handle.readline()
        handle.readline()
        header = handle.readline().split()
        if len(header) < 4:
            raise ValueError(f"Incomplete CUBE origin header: {path}")
        n_atoms = int(header[0])
        if n_atoms < 0:
            raise ValueError("Multi-orbital CUBE files are not supported")
        origin = np.asarray(header[1:4], dtype=np.float32)
        counts = []
        steps = []
        for _ in range(3):
            axis = handle.readline().split()
            if len(axis) < 4:
                raise ValueError(f"Incomplete CUBE axis header: {path}")
            count = int(axis[0])
            if count <= 0:
                raise ValueError("Expected positive CUBE axis counts in Bohr")
            counts.append(count)
            steps.append(np.asarray(axis[1:4], dtype=np.float32))
        for _ in range(n_atoms):
            if len(handle.readline().split()) < 5:
                raise ValueError(f"Incomplete CUBE atom header: {path}")
        values = np.fromstring(handle.read().replace("D", "E"), sep=" ", dtype=np.float32)
    expected = int(np.prod(counts))
    if values.size != expected or not np.isfinite(values).all():
        raise ValueError(f"CUBE data count/values invalid: got {values.size}, expected {expected}")
    return Cube(origin, steps[0], steps[1], steps[2], values.reshape(counts))
