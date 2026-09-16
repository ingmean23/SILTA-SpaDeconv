from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def spatial_maps(prediction_path: Path | str, coordinates_path: Path | str,
                 output_path: Path | str, max_types: int = 12) -> None:
    prediction = pd.read_csv(prediction_path, sep="\t", index_col=0)
    prediction.index = prediction.index.astype(str)
    coordinates = pd.read_csv(coordinates_path, sep="\t")
    if not {"spot_id", "x", "y"}.issubset(coordinates.columns):
        raise ValueError("coordinates must contain spot_id, x, and y")
    coordinates["spot_id"] = coordinates["spot_id"].astype(str)
    coordinates = coordinates.set_index("spot_id").loc[prediction.index]
    types = list(prediction.mean(axis=0).sort_values(ascending=False).head(max_types).index)
    columns = 4
    rows = int(np.ceil(len(types) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(11.5, 2.7 * rows), squeeze=False)
    for axis, cell_type in zip(axes.ravel(), types):
        points = axis.scatter(
            coordinates["x"], coordinates["y"], c=prediction[cell_type],
            s=10, cmap="viridis", vmin=0, linewidths=0,
        )
        axis.set_title(cell_type, fontsize=9)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(points, ax=axis, fraction=0.046, pad=0.02)
    for axis in axes.ravel()[len(types):]:
        axis.set_visible(False)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def metric_bars(metrics_path: Path | str, output_path: Path | str) -> None:
    frame = pd.read_csv(metrics_path)
    required = {"method", "ST_RMSE", "ST_JSD", "ST_SSIM", "ST_Pearson"}
    if not required.issubset(frame.columns):
        raise ValueError(f"metrics table missing columns: {sorted(required - set(frame.columns))}")
    metrics = ["ST_RMSE", "ST_JSD", "ST_SSIM", "ST_Pearson"]
    labels = ["RMSE", "JSD", "SSIM", "PCC"]
    figure, axes = plt.subplots(1, 4, figsize=(12.5, 3.3), constrained_layout=True)
    colors = ["#2F7F8F" if method == "SILTA" else "#A9B0B7" for method in frame["method"]]
    for axis, metric, label in zip(axes, metrics, labels):
        ascending = metric in {"ST_RMSE", "ST_JSD"}
        ordered = frame.sort_values(metric, ascending=ascending)
        ordered_colors = [colors[index] for index in ordered.index]
        axis.barh(ordered["method"], ordered[metric], color=ordered_colors)
        axis.invert_yaxis()
        axis.set_title(label, fontsize=11, weight="bold")
        axis.grid(axis="x", color="#D8DDE2", linewidth=0.7, linestyle="--")
        axis.spines[["top", "right"]].set_visible(False)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
