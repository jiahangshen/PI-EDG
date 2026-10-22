"""Extract portable ID-only split files from research JSON manifests.

No dataset paths, predictions, or machine-specific metadata are copied.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def ids(rows):
    values = [str(row["sample_id"]) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError("Duplicate sample ID in manifest")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    subset = json.loads(args.subset_manifest.read_text(encoding="utf-8"))
    values = {
        "train": ids(source["train_samples"]),
        "validation": ids(source["eval_samples"]),
        "reported_2000": ids(subset["samples"]),
    }
    if set(values["train"]) & set(values["validation"]):
        raise ValueError("Train/validation IDs overlap")
    if not set(values["reported_2000"]) <= set(values["validation"]):
        raise ValueError("Reported 2000 IDs must be a validation subset")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"reported_2000_is_validation_subset": True, "splits": {}}
    for name, items in values.items():
        text = "\n".join(items) + "\n"
        (args.output_dir / f"{name}_ids.txt").write_text(text, encoding="utf-8")
        width = max(map(len, items))
        np.save(args.output_dir / f"{name}_ids.npy", np.asarray(items, dtype=f"S{width}"))
        provenance["splits"][name] = {
            "count": len(items),
            "sha256_of_text": hashlib.sha256(text.encode()).hexdigest(),
        }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
