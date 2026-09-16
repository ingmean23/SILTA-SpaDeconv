"""Render a publication-ready marker-gene spatial comparison for one cell type."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from silta.preprocessing import read_coordinates


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name.strip(), Path(path)


def read_prediction(path: Path, spot_ids: list[str], cell_type: str) -> np.ndarray:
    frame = pd.read_csv(path)
    id_column = next(
        (name for name in ("spot_id", "barcode", "cell_id") if name in frame),
        None,
    )
    if id_column is None:
        raise ValueError(f"prediction has no explicit spot identifier: {path}")
    frame[id_column] = frame[id_column].astype(str)
    if frame[id_column].duplicated().any():
        raise ValueError(f"prediction contains duplicate spots: {path}")
    indexed = frame.set_index(id_column)
    missing = [spot for spot in spot_ids if spot not in indexed.index]
    if missing or cell_type not in indexed.columns:
        raise ValueError(
            f"prediction contract mismatch for {path}: "
            f"missing_spots={len(missing)}, cell_type={cell_type in indexed.columns}"
        )
    return indexed.loc[spot_ids, cell_type].to_numpy(dtype=float)


def read_metric(path: Path, cell_type: str) -> tuple[float, float]:
    frame = pd.read_csv(path)
    row = frame[frame["cell_type"].astype(str) == cell_type]
    if len(row) != 1:
        raise ValueError(f"expected one metric row for {cell_type}: {path}")
    return (
        float(row.iloc[0]["mean_marker_pcc_strict"]),
        float(row.iloc[0]["mean_marker_spearman_strict"]),
    )


def normalized_marker_expression(
    st_path: Path, spot_ids: list[str], genes: list[str]
) -> np.ndarray:
    adata = sc.read_h5ad(st_path)
    adata.var_names_make_unique()
    if list(map(str, adata.obs_names)) != spot_ids:
        index = {str(value): position for position, value in enumerate(adata.obs_names)}
        missing = [spot for spot in spot_ids if spot not in index]
        if missing:
            raise ValueError(f"ST input misses {len(missing)} prediction spots")
        adata = adata[[index[spot] for spot in spot_ids]].copy()
    missing_genes = [gene for gene in genes if gene not in adata.var_names]
    if missing_genes:
        raise ValueError(f"ST input misses marker genes: {missing_genes}")
    full = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    selected = adata[:, genes].X
    selected = selected.toarray() if hasattr(selected, "toarray") else np.asarray(selected)
    if np.any(full < 0):
        raise ValueError("plotting requires non-negative raw ST counts")
    totals = np.asarray(full.sum(axis=1), dtype=float).reshape(-1)
    return np.log1p(
        np.asarray(selected, dtype=float)
        / np.maximum(totals[:, None], 1e-12)
        * 10000.0
    )


def clipped(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    low, high = np.quantile(values, [0.01, 0.99])
    if high <= low:
        high = low + 1.0
    return np.clip(values, low, high), float(low), float(high)


def render(
    st: Path,
    coordinates: Path,
    panel: Path,
    cell_type: str,
    predictions: list[tuple[str, Path]],
    metrics: list[tuple[str, Path]],
    output_dir: Path,
) -> list[Path]:
    if not predictions or predictions[0][0].upper() != "SILTA":
        raise ValueError("the first --prediction must be SILTA=PATH")
    metric_paths = dict(metrics)
    if set(metric_paths) != {name for name, _ in predictions}:
        raise ValueError("prediction and metric method sets must match exactly")
    panel_data = json.loads(panel.read_text(encoding="utf-8"))
    if cell_type not in panel_data["markers_by_cell_type"]:
        raise ValueError(f"unknown cell type: {cell_type}")
    genes = [
        str(item["gene"] if isinstance(item, dict) else item)
        for item in panel_data["markers_by_cell_type"][cell_type]
    ]
    first = pd.read_csv(predictions[0][1])
    id_column = next(
        (name for name in ("spot_id", "barcode", "cell_id") if name in first),
        None,
    )
    if id_column is None:
        raise ValueError("SILTA prediction has no explicit spot identifier")
    spot_ids = first[id_column].astype(str).tolist()
    if len(set(spot_ids)) != len(spot_ids):
        raise ValueError("SILTA prediction contains duplicate spot identifiers")
    xy = read_coordinates(coordinates, spot_ids)
    marker_values = normalized_marker_expression(st, spot_ids, genes)
    method_values = {
        name: read_prediction(path, spot_ids, cell_type)
        for name, path in predictions
    }
    method_metrics = {
        name: read_metric(metric_paths[name], cell_type)
        for name, _ in predictions
    }
    pooled = np.concatenate(list(method_values.values()))
    prediction_limit = max(float(np.quantile(pooled, 0.99)), 1e-6)
    columns = max(len(genes), len(predictions) + 2)
    figure = plt.figure(
        figsize=(2.55 * columns, 6.2),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(2, columns)
    source_rows = []
    for index, gene in enumerate(genes):
        axis = figure.add_subplot(grid[0, index])
        values, low, high = clipped(marker_values[:, index])
        image = axis.scatter(
            xy[:, 0], xy[:, 1], c=values, cmap="viridis", s=9,
            marker="s", linewidths=0, vmin=low, vmax=high, rasterized=True,
        )
        axis.set_title(gene, fontsize=9)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
        source_rows.extend(
            {"spot_id": spot, "panel": gene, "value": float(value)}
            for spot, value in zip(spot_ids, marker_values[:, index])
        )
    for index in range(len(genes), columns):
        figure.add_subplot(grid[0, index]).set_axis_off()
    for index, (name, _) in enumerate(predictions):
        axis = figure.add_subplot(grid[1, index])
        values = method_values[name]
        image = axis.scatter(
            xy[:, 0], xy[:, 1], c=values, cmap="viridis", s=9,
            marker="s", linewidths=0, vmin=0, vmax=prediction_limit,
            rasterized=True,
        )
        pcc, spearman = method_metrics[name]
        axis.set_title(
            f"{name}: {cell_type}\nPCC={pcc:.3f}, Spearman={spearman:.3f}",
            fontsize=9,
        )
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
        source_rows.extend(
            {"spot_id": spot, "panel": name, "value": float(value)}
            for spot, value in zip(spot_ids, values)
        )
    names = [name for name, _ in predictions]
    for offset, (metric_name, metric_index) in enumerate(
        (("PCC", 0), ("Spearman", 1))
    ):
        axis = figure.add_subplot(grid[1, len(predictions) + offset])
        values = np.asarray([method_metrics[name][metric_index] for name in names])
        order = np.argsort(values)
        colors = ["#D1495B" if names[i] == "SILTA" else "#7A8A99" for i in order]
        axis.barh(np.asarray(names)[order], values[order], color=colors)
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_title(metric_name)
        axis.set_xlim(min(-0.05, float(values.min()) - 0.03), max(0.05, float(values.max()) + 0.03))
        axis.grid(axis="x", alpha=0.2)
    for index in range(len(predictions) + 2, columns):
        figure.add_subplot(grid[1, index]).set_axis_off()
    figure.suptitle(
        f"SILTA DLPFC direct marker colocalization: {cell_type}",
        fontsize=13,
        fontweight="bold",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = cell_type.replace("/", "_").replace(" ", "_")
    outputs = []
    for suffix in ("png", "pdf", "svg", "tiff"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    pd.DataFrame(source_rows).to_csv(
        output_dir / f"{stem}_source_data.csv", index=False
    )
    pd.DataFrame([
        {"method": name, "cell_type": cell_type, "pcc": pcc, "spearman": spearman}
        for name, (pcc, spearman) in method_metrics.items()
    ]).to_csv(output_dir / f"{stem}_metrics.csv", index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--st", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--cell-type", required=True)
    parser.add_argument(
        "--prediction", action="append", type=parse_named_path, required=True
    )
    parser.add_argument(
        "--metrics", action="append", type=parse_named_path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in render(
        args.st,
        args.coordinates,
        args.panel,
        args.cell_type,
        args.prediction,
        args.metrics,
        args.output_dir,
    ):
        print(path)


if __name__ == "__main__":
    main()
