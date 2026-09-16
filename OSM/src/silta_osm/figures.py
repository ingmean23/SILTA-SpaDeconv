from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Patch, Wedge

from .io import read_fraction_table, resolve_relative


METRICS = ("RMSE", "JSD", "SSIM", "PCC")
LOWER_IS_BETTER = {"RMSE", "JSD"}
METHOD_COLORS = (
    "#126782", "#E07A5F", "#6A994E", "#7B6D8D", "#D4A72C",
    "#3D5A80", "#B56576", "#4D908E", "#9C6644", "#577590", "#8A817C",
)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Figure manifest must be a YAML mapping")
    for key in ("benchmark", "ground_truth", "coordinates", "metrics", "output_dir", "methods"):
        if key not in payload:
            raise ValueError(f"Figure manifest missing {key}")
    return payload


def _coordinates(path: Path, spot_ids: list[str]) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t")
    if "spot_id" in frame.columns:
        frame["spot_id"] = frame["spot_id"].astype(str)
        frame = frame.set_index("spot_id").reindex(spot_ids)
    if not {"x", "y"}.issubset(frame.columns) or frame[["x", "y"]].isna().any().any():
        raise ValueError("Coordinates must align to all spots and contain x/y")
    return frame[["x", "y"]].to_numpy(dtype=float)


def _cell_colors(count: int) -> list[Any]:
    return [plt.get_cmap("turbo")(value) for value in np.linspace(0.02, 0.98, count)]


def _method_tables(manifest_path: Path, payload: Mapping[str, Any], truth: pd.DataFrame):
    methods = []
    for item in payload["methods"]:
        prediction = read_fraction_table(resolve_relative(manifest_path, item["prediction"]))
        if set(prediction.index) != set(truth.index) or set(prediction.columns) != set(truth.columns):
            raise ValueError(f"Method {item['name']} does not match GT sets")
        methods.append((dict(item), prediction.reindex(index=truth.index, columns=truth.columns)))
    return methods


def _spatial_axes(method_count: int, width: float = 16.0):
    columns = 4
    rows = math.ceil(method_count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(width, 3.8 * rows), squeeze=False)
    return figure, axes.ravel()


def render_dominant(
    output: Path, coordinates: np.ndarray, tables, cell_types: list[str], colors
) -> None:
    figure, axes = _spatial_axes(len(tables))
    for axis, (name, values) in zip(axes, tables):
        dominant = np.argmax(values, axis=1)
        axis.scatter(
            coordinates[:, 0], coordinates[:, 1], c=[colors[index] for index in dominant],
            marker="s", s=42, linewidths=0,
        )
        axis.set_title(name, fontsize=14, fontweight="bold")
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.axis("off")
    for axis in axes[len(tables):]:
        axis.axis("off")
    legend = [Patch(facecolor=color, label=name) for name, color in zip(cell_types, colors)]
    figure.legend(handles=legend, loc="lower center", ncol=6, fontsize=8, frameon=True)
    figure.subplots_adjust(bottom=0.15, wspace=0.08, hspace=0.18)
    _save(figure, output / "dominant_maps")


def _draw_pies(axis, coordinates: np.ndarray, values: np.ndarray, colors) -> None:
    distances = np.sqrt(np.sum((coordinates[:, None] - coordinates[None, :]) ** 2, axis=2))
    distances[distances == 0] = np.nan
    radius = 0.42 * float(np.nanmedian(np.nanmin(distances, axis=1)))
    for (x, y), row in zip(coordinates, values):
        start = 0.0
        for fraction, color in zip(row, colors):
            if fraction <= 0:
                continue
            end = start + 360.0 * float(fraction)
            axis.add_patch(Wedge((x, y), radius, start, end, facecolor=color, edgecolor="none"))
            start = end
    axis.set_xlim(coordinates[:, 0].min() - radius, coordinates[:, 0].max() + radius)
    axis.set_ylim(coordinates[:, 1].max() + radius, coordinates[:, 1].min() - radius)


def render_pies(output: Path, coordinates: np.ndarray, tables, cell_types, colors) -> None:
    figure, axes = _spatial_axes(len(tables))
    for axis, (name, values) in zip(axes, tables):
        _draw_pies(axis, coordinates, values, colors)
        axis.set_title(name, fontsize=14, fontweight="bold")
        axis.set_aspect("equal")
        axis.axis("off")
    for axis in axes[len(tables):]:
        axis.axis("off")
    legend = [Patch(facecolor=color, label=name) for name, color in zip(cell_types, colors)]
    figure.legend(handles=legend, loc="lower center", ncol=6, fontsize=8, frameon=True)
    figure.subplots_adjust(bottom=0.15, wspace=0.08, hspace=0.18)
    _save(figure, output / "spot_pie_maps")


def render_spatial_composite(
    output: Path, coordinates: np.ndarray, tables, cell_types, colors
) -> None:
    method_columns = 4
    rows = math.ceil(len(tables) / method_columns)
    figure = plt.figure(figsize=(18, 4.2 * rows + 1.3))
    grid = figure.add_gridspec(rows, method_columns * 2, wspace=0.03, hspace=0.16)
    for index, (name, values) in enumerate(tables):
        row, column = divmod(index, method_columns)
        dominant_axis = figure.add_subplot(grid[row, column * 2])
        pie_axis = figure.add_subplot(grid[row, column * 2 + 1])
        dominant = np.argmax(values, axis=1)
        dominant_axis.scatter(
            coordinates[:, 0], coordinates[:, 1],
            c=[colors[value] for value in dominant], marker="s", s=46, linewidths=0,
        )
        dominant_axis.set_title(name, fontsize=16, fontweight="bold", pad=8)
        dominant_axis.set_aspect("equal")
        dominant_axis.invert_yaxis()
        dominant_axis.axis("off")
        _draw_pies(pie_axis, coordinates, values, colors)
        pie_axis.set_aspect("equal")
        pie_axis.axis("off")
    for index in range(len(tables), rows * method_columns):
        row, column = divmod(index, method_columns)
        figure.add_subplot(grid[row, column * 2]).axis("off")
        figure.add_subplot(grid[row, column * 2 + 1]).axis("off")
    legend = [Patch(facecolor=color, label=name) for name, color in zip(cell_types, colors)]
    figure.legend(
        handles=legend, loc="lower center", ncol=min(8, len(cell_types)),
        fontsize=10, frameon=True, handlelength=1.2, columnspacing=1.4,
    )
    figure.subplots_adjust(bottom=0.12)
    _save(figure, output / "spatial_composite")


def render_metrics(output: Path, metrics: pd.DataFrame, primary: str) -> None:
    colors = {name: METHOD_COLORS[index % len(METHOD_COLORS)] for index, name in enumerate(metrics["method"])}
    for metric in METRICS:
        ranked = metrics.sort_values(metric, ascending=metric in LOWER_IS_BETTER)
        figure, axis = plt.subplots(figsize=(7.2, max(3.2, 0.42 * len(ranked))))
        bars = axis.barh(ranked["method"], ranked[metric], color=[colors[name] for name in ranked["method"]])
        axis.invert_yaxis()
        axis.set_title(metric, fontsize=15, fontweight="bold")
        axis.grid(axis="x", alpha=0.2)
        for bar, value in zip(bars, ranked[metric]):
            axis.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.4f}", va="center", fontsize=9)
        for label in axis.get_yticklabels():
            label.set_fontweight("bold" if label.get_text() == primary else "normal")
        figure.tight_layout()
        _save(figure, output / metric)


def _save(figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def render_manifest(manifest_path: Path) -> list[Path]:
    payload = _load_manifest(manifest_path)
    truth = read_fraction_table(resolve_relative(manifest_path, payload["ground_truth"]))
    coordinates = _coordinates(
        resolve_relative(manifest_path, payload["coordinates"]), truth.index.tolist()
    )
    methods = _method_tables(manifest_path, payload, truth)
    tables = [("Ground truth", truth.to_numpy())] + [
        (item["name"], frame.to_numpy()) for item, frame in methods
    ]
    colors = _cell_colors(len(truth.columns))
    output = resolve_relative(manifest_path, payload["output_dir"])
    render_dominant(output, coordinates, tables, truth.columns.tolist(), colors)
    render_pies(output, coordinates, tables, truth.columns.tolist(), colors)
    render_spatial_composite(output, coordinates, tables, truth.columns.tolist(), colors)
    metrics_path = resolve_relative(manifest_path, payload["metrics"])
    metrics = pd.read_csv(metrics_path)
    required = {"method", *METRICS}
    if not required.issubset(metrics.columns):
        raise ValueError(f"Metric table missing {sorted(required - set(metrics.columns))}")
    render_metrics(output, metrics, str(payload.get("primary_method", "SILTA")))
    return sorted(output.glob("*"))
