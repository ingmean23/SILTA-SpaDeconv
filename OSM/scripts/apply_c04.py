from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT  # noqa: F401
from silta_osm.calibration import apply_bundle
from silta_osm.io import write_fraction_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the restored SILTA C01+C04 state to exported logits"
    )
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--calibration-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    separator = "\t" if ".tsv" in args.logits.name else ","
    logits = pd.read_csv(args.logits, sep=separator, index_col=0)
    logits.index = logits.index.astype(str)
    logits.columns = logits.columns.astype(str)
    fractions = apply_bundle(logits, args.calibration_state)
    write_fraction_table(args.output, fractions)
    print(args.output)


if __name__ == "__main__":
    main()

