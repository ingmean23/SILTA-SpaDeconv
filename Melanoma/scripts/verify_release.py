#!/usr/bin/env python3
"""Verify hashes and locked ds6 T evidence without requiring the raw dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "checkpoints/ds6_T_seed42_epoch2/model.model":
        "fcd6040058a61ba6dd9011379a6c2acbb841294d3964e14b744e587e92e07e41",
    "expected/ds6_T/fractions.csv":
        "cf7cde5149f00116c3d17ded16af2f3b00801688311b8807aa9e7ff12ecdb0c4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    failures = []
    hashes = {}
    for relative, expected in EXPECTED.items():
        path = root / relative
        observed = sha256(path) if path.is_file() else "MISSING"
        hashes[relative] = observed
        if observed != expected:
            failures.append(f"{relative}: expected {expected}, observed {observed}")

    table = pd.read_csv(root / "expected" / "ds6_T" / "per_cell_type.csv")
    row = table.loc[table["cell_type"].astype(str) == "T"]
    if len(row) != 1:
        failures.append(f"Expected one T row, found {len(row)}")
        metrics = {}
    else:
        metrics = {
            "pcc": float(row.iloc[0]["mean_marker_pcc_strict"]),
            "spearman": float(row.iloc[0]["mean_marker_spearman_strict"]),
        }
        if abs(metrics["pcc"] - 0.0699059108699994) > 1e-12:
            failures.append(f"T PCC drift: {metrics['pcc']}")
        if abs(metrics["spearman"] - 0.0658180680674039) > 1e-12:
            failures.append(f"T Spearman drift: {metrics['spearman']}")

    report = {
        "status": "FAIL" if failures else "PASS",
        "hashes": hashes,
        "ds6_T": metrics,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
