#!/usr/bin/env python3
"""Render observed marker maps, model fraction maps, and supplied rankings."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


REQUIRED = {
    "coordinates": ("spot_id", "x", "y"),
    "markers": ("spot_id", "cell_type", "gene", "value"),
    "predictions": ("spot_id", "method", "cell_type", "fraction"),
    "metrics": ("cell_type", "method", "pcc", "spearman"),
}
DEFAULT_STYLE = {
    "invert_y": True,
    "map_columns": 5,
    "cmap": "viridis",
    "marker_quantiles": [0.01, 0.99],
    "prediction_quantile": 0.99,
    "formats": ["png", "svg", "pdf", "tiff"],
    "dpi": 600,
    "dacg_color": "#7B3294",
    "baseline_color": "#A9B0B7",
    "map_geometry": "points",
    "grid_wspace": 0.48,
    "grid_hspace": 0.42,
    "ranking_width": 1.55,
    "ranking_gap": 0.0,
    "shared_prediction_colorbar": False,
    "aligned_section_layout": False,
}


def read_manifest(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML requires PyYAML; use JSON or install PyYAML") from exc
        value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    else:
        raise ValueError("Manifest must be JSON or YAML")
    if not isinstance(value, dict):
        raise ValueError("Manifest root must be an object")
    return value


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def require_columns(frame: pd.DataFrame, names: Sequence[str], table: str) -> None:
    missing = set(names).difference(frame.columns)
    if missing:
        raise ValueError(f"{table} missing columns: {sorted(missing)}")


def finite(frame: pd.DataFrame, columns: Sequence[str], table: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{table}.{column} contains non-finite values")


def aligned(frame: pd.DataFrame, expected: pd.Index, label: str) -> pd.DataFrame:
    ids = frame["spot_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"{label} has duplicate spot IDs")
    observed, wanted = set(ids), set(expected.astype(str))
    if observed != wanted:
        raise ValueError(
            f"{label} spot mismatch; missing={sorted(wanted-observed)[:5]}, "
            f"extra={sorted(observed-wanted)[:5]}")
    return frame.assign(spot_id=ids).set_index("spot_id").loc[expected.astype(str)]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "cell_type"


def marker_display(values: np.ndarray, quantiles: Sequence[float]):
    low, high = np.quantile(values, list(map(float, quantiles)))
    if high <= low:
        return np.zeros_like(values), float(low), float(high)
    return (np.clip(values, low, high) - low) / (high - low), float(low), float(high)


def letter(index: int) -> str:
    return chr(ord("a") + index) if index < 26 else f"p{index + 1}"


def mpl_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    })


def grid_step(values: np.ndarray) -> float:
    unique = np.unique(np.asarray(values, dtype=float))
    differences = np.diff(unique)
    positive = differences[differences > 1e-12]
    return float(positive.min()) if len(positive) else 1.0


def spatial_panel(axis, x, y, values, title, style, vmax, size, panel):
    geometry = str(style.get("map_geometry", "points"))
    if geometry == "square_tiles":
        width, height = grid_step(x), grid_step(y)
        patches = [
            Rectangle((x_value - width / 2, y_value - height / 2), width, height)
            for x_value, y_value in zip(x, y)
        ]
        artist = PatchCollection(
            patches, cmap=style["cmap"], edgecolor="none", linewidth=0,
            rasterized=True)
        artist.set_array(np.asarray(values, dtype=float))
        artist.set_clim(0.0, vmax)
        axis.add_collection(artist)
        axis.set_xlim(float(np.min(x) - width / 2), float(np.max(x) + width / 2))
        axis.set_ylim(float(np.min(y) - height / 2), float(np.max(y) + height / 2))
    elif geometry == "points":
        artist = axis.scatter(
            x, y, c=values, cmap=style["cmap"], vmin=0.0, vmax=vmax,
            s=size, linewidths=0, rasterized=True)
    else:
        raise ValueError(f"Unsupported map_geometry: {geometry}")
    axis.set_title(title, fontsize=7.2, pad=3)
    axis.set_aspect("equal", adjustable="box")
    if style["invert_y"]:
        axis.invert_yaxis()
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.text(-0.06, 1.04, panel, transform=axis.transAxes,
              fontsize=8, fontweight="bold", ha="right", va="bottom")
    return artist


def ranking_panel(axis, metrics, metric, style, panel):
    ordered = metrics.sort_values(metric, ascending=True).reset_index(drop=True)
    values = ordered[metric].to_numpy(dtype=float)
    positions = np.arange(len(ordered))
    colors = [style["dacg_color"] if name == "DACG" else style["baseline_color"]
              for name in ordered["method"]]
    axis.axvline(0, color="#C7CBD0", lw=0.8)
    axis.hlines(positions, np.minimum(values, 0), values, color=colors, lw=1.2)
    axis.scatter(values, positions, color=colors, s=24, zorder=3)
    axis.set_yticks(positions, ordered["method"].astype(str))
    if bool(style.get("aligned_section_layout", False)):
        axis.set_xlabel("")
        axis.set_title(metric.upper(), fontsize=8, fontweight="bold")
    else:
        axis.set_xlabel(f"Supplied {metric.capitalize()}")
        axis.set_title(f"{metric.capitalize()} ranking", fontsize=8)
    axis.grid(axis="x", color="#ECEFF1", lw=0.6)
    span = max(float(values.max() - values.min()), 0.1)
    axis.set_xlim(min(float(values.min()), 0) - 0.08 * span,
                  max(float(values.max()), 0) + 0.32 * span)
    for row, value in enumerate(values):
        rank = len(values) - row
        is_dacg = ordered.loc[row, "method"] == "DACG"
        axis.text(value + 0.025 * span, row, f"#{rank}  {value:.3f}",
                  va="center", fontsize=6.4,
                  color=style["dacg_color"] if is_dacg else "#59616A",
                  fontweight="bold" if is_dacg else "normal")
    axis.text(-0.08, 1.03, panel, transform=axis.transAxes,
              fontsize=8, fontweight="bold", ha="right", va="bottom")


def validate(manifest_path: Path) -> dict[str, Any]:
    manifest = read_manifest(manifest_path)
    for key in ("dataset", "inputs", "methods", "cell_types", "output_dir"):
        if key not in manifest:
            raise ValueError(f"Manifest missing: {key}")
    methods = [str(item) for item in manifest["methods"]]
    if not methods or methods[0] != "DACG" or len(methods) != len(set(methods)):
        raise ValueError("methods must be unique and start with DACG")
    frames, paths = {}, {}
    for table, columns in REQUIRED.items():
        paths[table] = resolve(manifest_path.parent, manifest["inputs"][table])
        frames[table] = read_table(paths[table])
        require_columns(frames[table], columns, table)
    coords = frames["coordinates"].copy()
    coords["spot_id"] = coords["spot_id"].astype(str)
    if coords["spot_id"].duplicated().any():
        raise ValueError("coordinates has duplicate spot IDs")
    finite(coords, ["x", "y"], "coordinates")
    finite(frames["markers"], ["value"], "markers")
    finite(frames["predictions"], ["fraction"], "predictions")
    finite(frames["metrics"], ["pcc", "spearman"], "metrics")
    if (pd.to_numeric(frames["markers"]["value"]) < 0).any():
        raise ValueError("marker values must be nonnegative")
    fractions = pd.to_numeric(frames["predictions"]["fraction"])
    if ((fractions < 0) | (fractions > 1)).any():
        raise ValueError("fractions must lie in [0, 1]")
    for metric in ("pcc", "spearman"):
        values = pd.to_numeric(frames["metrics"][metric])
        if ((values < -1) | (values > 1)).any():
            raise ValueError(f"{metric} must lie in [-1, 1]")
    markers = frames["markers"].assign(
        spot_id=frames["markers"]["spot_id"].astype(str),
        cell_type=frames["markers"]["cell_type"].astype(str),
        gene=frames["markers"]["gene"].astype(str))
    predictions = frames["predictions"].assign(
        spot_id=frames["predictions"]["spot_id"].astype(str),
        method=frames["predictions"]["method"].astype(str),
        cell_type=frames["predictions"]["cell_type"].astype(str))
    metrics = frames["metrics"].assign(
        method=frames["metrics"]["method"].astype(str),
        cell_type=frames["metrics"]["cell_type"].astype(str))
    spots = pd.Index(coords["spot_id"], name="spot_id")
    seen = set()
    for spec in manifest["cell_types"]:
        cell_type = str(spec["id"])
        genes = [str(gene) for gene in spec["markers"]]
        if cell_type in seen or not genes or len(genes) != len(set(genes)):
            raise ValueError(f"Invalid cell type or marker list: {cell_type}")
        seen.add(cell_type)
        for gene in genes:
            aligned(markers[(markers.cell_type == cell_type) & (markers.gene == gene)],
                    spots, f"marker {cell_type}/{gene}")
        for method in methods:
            aligned(predictions[(predictions.cell_type == cell_type)
                                & (predictions.method == method)],
                    spots, f"prediction {cell_type}/{method}")
            count = len(metrics[(metrics.cell_type == cell_type) & (metrics.method == method)])
            if count != 1:
                raise ValueError(f"Expected one metric row for {cell_type}/{method}; got {count}")
    return {"manifest": manifest, "coordinates": coords, "markers": markers,
            "predictions": predictions, "metrics": metrics, "spots": spots,
            "paths": paths}


def render(context: Mapping[str, Any], spec: Mapping[str, Any], manifest_path: Path):
    manifest = context["manifest"]
    style = {**DEFAULT_STYLE, **dict(manifest.get("style", {}))}
    methods = [str(item) for item in manifest["methods"]]
    cell_type = str(spec["id"])
    label = str(spec.get("label", cell_type))
    title_label = f"{label[:-6]}-cell" if label.lower().endswith(" cells") else label
    genes = [str(gene) for gene in spec["markers"]]
    coords = context["coordinates"].set_index("spot_id").loc[context["spots"]]
    x, y = coords.x.to_numpy(float), coords.y.to_numpy(float)
    size = float(style.get("spot_size", max(1.5, min(11.0, 14000 / len(coords)))))
    marker_maps, marker_audit = [], {}
    for gene in genes:
        frame = context["markers"][(context["markers"].cell_type == cell_type)
                                   & (context["markers"].gene == gene)]
        raw = aligned(frame, context["spots"], gene).value.to_numpy(float)
        display, low, high = marker_display(raw, style["marker_quantiles"])
        marker_maps.append((gene, raw, display))
        marker_audit[gene] = {"q_low": low, "q_high": high, "degenerate": high <= low}
    prediction_maps, pooled = [], []
    for method in methods:
        frame = context["predictions"][(context["predictions"].cell_type == cell_type)
                                       & (context["predictions"].method == method)]
        values = aligned(frame, context["spots"], method).fraction.to_numpy(float)
        prediction_maps.append((method, values))
        pooled.append(values)
    prediction_vmax = max(float(np.quantile(np.concatenate(pooled),
                                             style["prediction_quantile"])), 1e-8)
    metrics = context["metrics"][context["metrics"].cell_type == cell_type]
    metrics = metrics.set_index("method").loc[methods].reset_index()
    columns = max(1, min(int(style["map_columns"]), max(len(genes), len(methods))))
    marker_rows = math.ceil(len(genes) / columns)
    method_rows = math.ceil(len(methods) / columns)
    total_rows = marker_rows + method_rows
    aligned_layout = bool(style.get("aligned_section_layout", False))
    if aligned_layout:
        figure = plt.figure(figsize=(2.52 * columns + 3.9, 2.34 * total_rows + 1.85))
        outer = figure.add_gridspec(
            1, 2, width_ratios=[columns, float(style["ranking_width"])],
            left=0.035, right=0.985, bottom=0.055, top=0.925, wspace=0.16)
        marker_start = 1
        marker_cbar_row = marker_start + marker_rows
        prediction_header_row = marker_cbar_row + 1
        prediction_start = prediction_header_row + 1
        prediction_cbar_row = prediction_start + method_rows
        row_heights = (
            [0.10] + [1.0] * marker_rows + [0.065, 0.10]
            + [1.0] * method_rows + [0.065]
        )
        map_grid = outer[0].subgridspec(
            len(row_heights), columns, height_ratios=row_heights,
            wspace=float(style["grid_wspace"]), hspace=0.12)
        marker_header = figure.add_subplot(map_grid[0, :])
        marker_header.axis("off")
        marker_header.text(
            0.0, 0.45, "Observed marker localization (q01-q99 per gene)",
            fontsize=8.5, fontweight="bold", ha="left", va="center")
        prediction_header = figure.add_subplot(map_grid[prediction_header_row, :])
        prediction_header.axis("off")
        prediction_header.text(
            0.0, 0.45, f"Model-predicted {title_label} fraction (shared scale)",
            fontsize=8.5, fontweight="bold", ha="left", va="center")
        marker_slots = [map_grid[marker_start + row, column]
                        for row in range(marker_rows) for column in range(columns)]
        prediction_slots = [map_grid[prediction_start + row, column]
                            for row in range(method_rows) for column in range(columns)]
        marker_cbar_axis = figure.add_subplot(map_grid[marker_cbar_row, 1:-1])
        prediction_cbar_axis = figure.add_subplot(map_grid[prediction_cbar_row, 1:-1])
        rank_grid = outer[1].subgridspec(2, 1, hspace=0.30)
    else:
        figure = plt.figure(figsize=(2.42 * columns + 4.1, 2.22 * total_rows + 1.25))
        ranking_gap = float(style["ranking_gap"])
        grid = figure.add_gridspec(
            total_rows, columns + 2,
            width_ratios=[1] * columns + [max(ranking_gap, 1e-6), float(style["ranking_width"])],
            left=0.025, right=0.99, bottom=0.065, top=0.90,
            wspace=float(style["grid_wspace"]), hspace=float(style["grid_hspace"]))
        marker_slots = [grid[row, column]
                        for row in range(marker_rows) for column in range(columns)]
        prediction_slots = [grid[marker_rows + row, column]
                            for row in range(method_rows) for column in range(columns)]
        marker_cbar_axis = None
        prediction_cbar_axis = None
        rank_grid = grid[:, -1].subgridspec(2, 1, hspace=0.38)
    panel = 0
    for index, (gene, _, values) in enumerate(marker_maps):
        axis = figure.add_subplot(marker_slots[index])
        artist = spatial_panel(axis, x, y, values, gene, style, 1.0, size, letter(panel))
        if not aligned_layout:
            figure.colorbar(artist, ax=axis, fraction=0.04, pad=0.025)
        panel += 1
    for index in range(len(genes), marker_rows * columns):
        figure.add_subplot(marker_slots[index]).axis("off")
    if aligned_layout:
        marker_colorbar = figure.colorbar(artist, cax=marker_cbar_axis, orientation="horizontal")
        marker_colorbar.ax.tick_params(labelsize=6, length=2)
    prediction_axes, prediction_artist = [], None
    for index, (method, values) in enumerate(prediction_maps):
        axis = figure.add_subplot(prediction_slots[index])
        title = method if aligned_layout else f"{method}: {label}"
        artist = spatial_panel(axis, x, y, np.clip(values, 0, prediction_vmax),
            title, style, prediction_vmax, size, letter(panel))
        prediction_axes.append(axis)
        prediction_artist = artist
        if not aligned_layout and not bool(style["shared_prediction_colorbar"]):
            figure.colorbar(artist, ax=axis, fraction=0.04, pad=0.025)
        panel += 1
    for index in range(len(methods), method_rows * columns):
        figure.add_subplot(prediction_slots[index]).axis("off")
    if aligned_layout:
        colorbar = figure.colorbar(
            prediction_artist, cax=prediction_cbar_axis, orientation="horizontal")
        colorbar.ax.tick_params(labelsize=6, length=2)
    elif bool(style["shared_prediction_colorbar"]):
        colorbar = figure.colorbar(
            prediction_artist, ax=prediction_axes, orientation="horizontal",
            fraction=0.018, pad=0.025, aspect=65)
        colorbar.set_label("Predicted T-cell fraction (shared scale)", fontsize=6.5)
        colorbar.ax.tick_params(labelsize=6, length=2)
    ranking_panel(figure.add_subplot(rank_grid[0]), metrics, "pcc", style, letter(panel))
    panel += 1
    ranking_panel(figure.add_subplot(rank_grid[1]), metrics, "spearman", style, letter(panel))
    if aligned_layout:
        figure.suptitle(
            f"{manifest['dataset']} | {title_label} spatial concordance",
            fontsize=11, fontweight="bold", y=0.975)
    else:
        figure.suptitle(f"{manifest['dataset']} | {label}: observed markers and predicted fractions",
                        fontsize=11, fontweight="bold", y=0.975)
        figure.text(0.035, 0.925, "Observed marker localization (each gene q01-q99 scaled)",
                    fontsize=8.5, fontweight="bold")
        figure.text(0.035, 0.90 - marker_rows / total_rows * 0.845,
                    "Predicted target-cell fraction (shared scale across methods)",
                    fontsize=8.5, fontweight="bold")
        figure.text(0.5, 0.015,
            "PCC and Spearman are externally supplied; marker localization is not fraction ground truth.",
            ha="center", fontsize=6.5, color="#555B61")
    output = resolve(manifest_path.parent, manifest["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    stem = safe_name(cell_type)
    formats = [str(item).lower() for item in style["formats"]]
    for suffix in formats:
        if suffix not in {"png", "svg", "pdf", "tiff", "tif"}:
            raise ValueError(f"Unsupported output format: {suffix}")
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix in {"png", "tiff", "tif"}:
            kwargs["dpi"] = int(style["dpi"])
        figure.savefig(output / f"{stem}.{suffix}", **kwargs)
    plt.close(figure)
    marker_source = pd.concat([pd.DataFrame({
        "spot_id": context["spots"], "cell_type": cell_type, "gene": gene,
        "raw_value": raw, "display_value": display})
        for gene, raw, display in marker_maps], ignore_index=True)
    prediction_source = pd.concat([pd.DataFrame({
        "spot_id": context["spots"], "cell_type": cell_type, "method": method,
        "fraction": values}) for method, values in prediction_maps], ignore_index=True)
    marker_source.to_csv(output / f"{stem}_marker_source_data.csv", index=False)
    prediction_source.to_csv(output / f"{stem}_prediction_source_data.csv", index=False)
    metrics.to_csv(output / f"{stem}_metrics_source_data.csv", index=False)
    audit = {"dataset": manifest["dataset"], "cell_type": cell_type,
        "n_spots": len(coords), "marker_order": genes, "method_map_order": methods,
        "ranking_values_recomputed": False, "marker_scaling": marker_audit,
        "prediction_vmin": 0.0, "prediction_vmax": prediction_vmax,
        "prediction_quantile": style["prediction_quantile"],
        "invert_y": style["invert_y"], "spot_size": size,
        "map_geometry": style["map_geometry"],
        "grid_wspace": style["grid_wspace"],
        "grid_hspace": style["grid_hspace"],
        "ranking_width": style["ranking_width"],
        "ranking_gap": style["ranking_gap"],
        "shared_prediction_colorbar": style["shared_prediction_colorbar"],
        "aligned_section_layout": aligned_layout,
        "map_columns": columns, "formats": formats}
    (output / f"{stem}_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    path = args.manifest.resolve()
    mpl_style()
    context = validate(path)
    if args.validate_only:
        print(json.dumps({"status": "valid", "dataset": context["manifest"]["dataset"],
            "n_spots": len(context["spots"]),
            "n_methods": len(context["manifest"]["methods"]),
            "n_cell_types": len(context["manifest"]["cell_types"])}, indent=2))
        return
    audits = [render(context, spec, path) for spec in context["manifest"]["cell_types"]]
    print(json.dumps({"status": "complete", "dataset": context["manifest"]["dataset"],
        "cell_types": [item["cell_type"] for item in audits],
        "output_dir": str(resolve(path.parent, context["manifest"]["output_dir"]))},
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
