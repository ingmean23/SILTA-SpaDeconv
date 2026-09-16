from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT  # noqa: F401
from silta_osm.prepared import PreparedInputs, save_prepared


def _table(path: Path) -> pd.DataFrame:
    separator = "\t" if ".tsv" in path.name or path.suffix == ".txt" else ","
    frame = pd.read_csv(path, sep=separator, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a strict SILTA prepared-input NPZ from authorized tables")
    parser.add_argument("--expression", required=True, type=Path)
    parser.add_argument("--expression-adjacency", required=True, type=Path)
    parser.add_argument("--spatial-adjacency", required=True, type=Path)
    parser.add_argument("--coordinates", required=True, type=Path)
    parser.add_argument("--reference-signature", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    expression = _table(args.expression)
    expression_adjacency = _table(args.expression_adjacency)
    spatial_adjacency = _table(args.spatial_adjacency)
    coordinates = _table(args.coordinates)
    signature = _table(args.reference_signature)
    spot_ids = list(expression.index)
    genes = list(expression.columns)
    if list(expression_adjacency.index) != spot_ids or list(
        expression_adjacency.columns) != spot_ids:
        raise ValueError("expression adjacency must use the expression spot order")
    if list(spatial_adjacency.index) != spot_ids or list(
        spatial_adjacency.columns) != spot_ids:
        raise ValueError("spatial adjacency must use the expression spot order")
    if list(coordinates.index) != spot_ids or coordinates.shape[1] != 2:
        raise ValueError("coordinates must contain exactly x/y columns in spot order")
    if list(signature.columns) != genes:
        raise ValueError("reference signature must use the expression gene order")
    save_prepared(args.output, PreparedInputs(
        expression=expression.to_numpy(),
        expression_adjacency=expression_adjacency.to_numpy(),
        spatial_adjacency=spatial_adjacency.to_numpy(),
        coordinates=coordinates.to_numpy(),
        reference_signature=signature.to_numpy(),
        spot_ids=tuple(spot_ids),
        gene_names=tuple(genes),
        cell_types=tuple(signature.index),
    ))
    print(args.output)


if __name__ == "__main__":
    main()
