#!/usr/bin/env python3
"""Render SILTA HBC dominant-type and per-cell-type fraction maps."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_inputs(prediction: Path, metadata: Path):
    pred = pd.read_csv(prediction)
    meta = pd.read_csv(metadata, sep="\t")
    if "spot_id" not in pred:
        raise ValueError("prediction CSV must contain spot_id")
    id_column = "ID" if "ID" in meta else next(
        (column for column in meta if meta[column].dtype == object), None
    )
    if id_column is None:
        raise ValueError("metadata must contain an ID column")
    coordinate_pairs = (
        ("scaled_x", "scaled_y"), ("x", "y"), ("coor_X", "coor_Y"),
        ("array_row", "array_col"), ("imagecol", "imagerow"),
    )
    coords = next(
        ((left, right) for left, right in coordinate_pairs
         if left in meta and right in meta), None
    )
    if coords is None:
        raise ValueError("metadata has no recognized coordinate pair")
    work = meta.merge(pred, left_on=id_column, right_on="spot_id", how="inner")
    if len(work) != len(meta):
        raise ValueError(f"prediction covers {len(work)}/{len(meta)} spots")
    cell_types = [
        column for column in pred.columns
        if column != "spot_id" and pd.api.types.is_numeric_dtype(pred[column])
    ]
    return work, cell_types, coords


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    work, cell_types, (x_name, y_name) = load_inputs(args.prediction, args.metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    values = work[cell_types].to_numpy(float)
    dominant = np.argmax(values, axis=1)
    palette = plt.get_cmap("tab20", len(cell_types))
    fig, axis = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    for index, name in enumerate(cell_types):
        mask = dominant == index
        axis.scatter(
            work.loc[mask, x_name], work.loc[mask, y_name], s=9,
            color=palette(index), linewidths=0, label=name,
        )
    axis.set_title("SILTA dominant cell type")
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.savefig(args.output_dir / "dominant_cell_type.png", dpi=args.dpi)
    plt.close(fig)

    columns = 4
    rows = math.ceil(len(cell_types) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(3.0 * columns, 2.7 * rows),
        constrained_layout=True, squeeze=False,
    )
    for axis, name in zip(axes.flat, cell_types):
        artist = axis.scatter(
            work[x_name], work[y_name], c=work[name], cmap="viridis",
            s=7, linewidths=0, vmin=0, vmax=float(work[name].quantile(0.99)),
        )
        axis.set_title(name, fontsize=9)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(artist, ax=axis, fraction=0.045, pad=0.02)
    for axis in axes.flat[len(cell_types):]:
        axis.set_visible(False)
    fig.savefig(args.output_dir / "cell_type_fraction_maps.png", dpi=args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
