#!/usr/bin/env python
"""Render standardized SILTA benchmark spatial and metric figures."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle, Wedge


METRICS = ("RMSE", "JSD", "SSIM", "PCC")
LOWER_IS_BETTER = {"RMSE", "JSD"}
METRIC_TITLES = {metric: metric for metric in METRICS}
FORMATS = ("png", "tiff", "pdf", "svg")
SPATIAL_COLUMNS = 4

METHOD_COLORS = {
    "silta": "#146C75",
    "dacg": "#146C75",
    "stereoscope": "#8C72B2",
    "rctd": "#D6AD50",
    "spatialdecon": "#CC896F",
    "card": "#949575",
    "stdgcn": "#6F9AC0",
    "cell2location": "#A580A2",
    "tangram": "#5FA395",
    "stride": "#97B77F",
    "spotlight": "#D494AC",
    "gmgat": "#B8BEC2",
    "harmodecon": "#7B8DA4",
    "redeconv": "#C88692",
}


class FigureContractError(ValueError):
    """Raised when benchmark inputs violate the plotting contract."""


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _separator(path: Path) -> str:
    suffixes = {suffix.lower() for suffix in path.suffixes}
    return "\t" if suffixes & {".tsv", ".txt"} else ","


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=_separator(path), compression="infer")


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0"}:
        return False
    raise FigureContractError(f"{field} must be boolean, got {value!r}")


def load_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FigureContractError("Manifest root must be a mapping")
    required = {
        "benchmark", "ground_truth", "coordinates", "metrics",
        "output_dir", "dacg_method", "methods",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise FigureContractError(f"Manifest missing keys: {missing}")
    if not isinstance(payload["methods"], list) or not payload["methods"]:
        raise FigureContractError("Manifest methods must be a non-empty list")
    base = path.parent
    for key in ("ground_truth", "coordinates", "metrics", "output_dir"):
        payload[key] = _resolve(base, payload[key])
    for method in payload["methods"]:
        if not isinstance(method, dict) or "name" not in method:
            raise FigureContractError("Each method must define name")
        if "prediction" not in method:
            raise FigureContractError(
                f"Method {method['name']!r} must define prediction")
        method["prediction"] = _resolve(base, method["prediction"])
    payload["manifest_path"] = path
    return payload


def _validate_identifiers(values: Iterable[Any], *, label: str) -> pd.Index:
    index = pd.Index([str(value) for value in values], name=label)
    if index.hasnans:
        raise FigureContractError(f"{label} contains missing values")
    if index.duplicated().any():
        duplicates = index[index.duplicated()].unique().tolist()[:5]
        raise FigureContractError(f"{label} contains duplicates: {duplicates}")
    return index


def load_fraction_table(
    path: Path,
    *,
    spot_id_column: str,
    row_sum_atol: float,
    renormalize_atol: float,
    negative_atol: float,
    expected_spots: pd.Index | None = None,
    expected_types: pd.Index | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    table = _read_table(path)
    if spot_id_column not in table.columns:
        raise FigureContractError(
            f"{path}: missing spot ID column {spot_id_column!r}")
    spot_ids = _validate_identifiers(
        table.pop(spot_id_column), label=spot_id_column)
    if table.columns.duplicated().any():
        duplicates = table.columns[table.columns.duplicated()].tolist()[:5]
        raise FigureContractError(
            f"{path}: duplicate cell-type columns {duplicates}")
    if table.shape[1] == 0:
        raise FigureContractError(f"{path}: no cell-type columns")
    try:
        values = table.apply(pd.to_numeric, errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise FigureContractError(f"{path}: nonnumeric fraction value") from exc
    if not np.isfinite(values).all():
        raise FigureContractError(f"{path}: NaN or Inf fraction value")
    minimum = float(values.min())
    if minimum < -negative_atol:
        raise FigureContractError(
            f"{path}: negative fraction {minimum:.6g} below tolerance")
    clipped_count = int(np.count_nonzero(values < 0))
    values = np.clip(values, 0.0, None)
    row_sums = values.sum(axis=1)
    if np.any(row_sums <= 0):
        raise FigureContractError(f"{path}: zero-sum fraction row")
    deviations = np.abs(row_sums - 1.0)
    max_deviation = float(deviations.max(initial=0.0))
    if max_deviation > renormalize_atol:
        raise FigureContractError(
            f"{path}: row sum deviation {max_deviation:.6g} exceeds "
            f"renormalize_atol={renormalize_atol}")
    renormalized = deviations > row_sum_atol
    if np.any(renormalized) or clipped_count:
        values = values / values.sum(axis=1, keepdims=True)
    frame = pd.DataFrame(
        values,
        index=spot_ids,
        columns=pd.Index([str(value) for value in table.columns]),
    )
    spot_reordered = False
    type_reordered = False
    if expected_spots is not None:
        missing = expected_spots.difference(frame.index)
        extra = frame.index.difference(expected_spots)
        if len(missing) or len(extra):
            raise FigureContractError(
                f"{path}: spot mismatch; missing={missing[:5].tolist()}, "
                f"extra={extra[:5].tolist()}")
        spot_reordered = not frame.index.equals(expected_spots)
        frame = frame.reindex(expected_spots)
    if expected_types is not None:
        missing = expected_types.difference(frame.columns)
        extra = frame.columns.difference(expected_types)
        if len(missing) or len(extra):
            raise FigureContractError(
                f"{path}: cell-type mismatch; missing={missing[:5].tolist()}, "
                f"extra={extra[:5].tolist()}")
        type_reordered = not frame.columns.equals(expected_types)
        frame = frame.reindex(columns=expected_types)
    audit = {
        "path": str(path),
        "spot_count": int(frame.shape[0]),
        "cell_type_count": int(frame.shape[1]),
        "max_row_sum_deviation_before": max_deviation,
        "renormalized_row_count": int(np.count_nonzero(renormalized)),
        "clipped_negative_value_count": clipped_count,
        "spot_reordered": spot_reordered,
        "cell_type_reordered": type_reordered,
    }
    return frame, audit


def load_coordinates(
    path: Path,
    *,
    spot_id_column: str | None,
    coordinate_columns: Sequence[str],
    expected_spots: pd.Index,
) -> tuple[pd.DataFrame, bool]:
    if len(coordinate_columns) != 2:
        raise FigureContractError("coordinate_columns must contain x and y")
    table = _read_table(path)
    required = list(coordinate_columns)
    if spot_id_column is not None:
        required.insert(0, spot_id_column)
    missing_columns = [column for column in required if column not in table]
    if missing_columns:
        raise FigureContractError(
            f"{path}: missing coordinate columns {missing_columns}")
    if spot_id_column is None:
        if len(table) != len(expected_spots):
            raise FigureContractError(
                f"{path}: positional coordinates contain {len(table)} rows; "
                f"expected {len(expected_spots)}")
        spot_ids = expected_spots.copy()
    else:
        spot_ids = _validate_identifiers(
            table[spot_id_column], label=spot_id_column)
    coordinates = table.loc[:, coordinate_columns].apply(
        pd.to_numeric, errors="raise")
    coordinates.index = spot_ids
    values = coordinates.to_numpy(float)
    if not np.isfinite(values).all():
        raise FigureContractError(f"{path}: NaN or Inf coordinate")
    missing = expected_spots.difference(coordinates.index)
    extra = coordinates.index.difference(expected_spots)
    if len(missing) or len(extra):
        raise FigureContractError(
            f"{path}: coordinate spot mismatch; "
            f"missing={missing[:5].tolist()}, extra={extra[:5].tolist()}")
    reordered = not coordinates.index.equals(expected_spots)
    return coordinates.reindex(expected_spots), reordered


def _method_key(name: str) -> str:
    key = "".join(character for character in name.lower()
                  if character.isalnum())
    if key.startswith("silta"):
        return "silta"
    return "dacg" if key.startswith("dacg") else key


def fallback_method_color(name: str) -> str:
    digest = hashlib.sha256(_method_key(name).encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    saturation = 0.34 + digest[2] / 255.0 * 0.16
    lightness = 0.54 + digest[3] / 255.0 * 0.10
    return mpl.colors.to_hex(colorsys.hls_to_rgb(hue, lightness, saturation))


def method_color(name: str, overrides: Mapping[str, str] | None = None) -> str:
    if overrides and name in overrides:
        value = str(overrides[name])
    else:
        value = METHOD_COLORS.get(_method_key(name), fallback_method_color(name))
    if not mpl.colors.is_color_like(value):
        raise FigureContractError(f"Invalid method color for {name}: {value}")
    return mpl.colors.to_hex(value)


def cell_type_colors(
    cell_types: Sequence[str], colormap: str = "turbo",
) -> dict[str, str]:
    count = len(cell_types)
    if count == 0:
        raise FigureContractError("Cannot color an empty cell-type set")
    if colormap not in mpl.colormaps:
        raise FigureContractError(
            f"Unknown cell-type colormap: {colormap}")
    cmap = mpl.colormaps[colormap].resampled(max(count, 2))
    return {
        str(cell_type): mpl.colors.to_hex(cmap(index))
        for index, cell_type in enumerate(cell_types)
    }


def normalize_methods(
    manifest: Mapping[str, Any], metric_table: pd.DataFrame,
) -> pd.DataFrame:
    names = [str(item["name"]) for item in manifest["methods"]]
    if len(set(names)) != len(names):
        raise FigureContractError("Manifest contains duplicate method names")
    dacg = str(manifest["dacg_method"])
    if dacg not in names:
        raise FigureContractError(f"DACG method {dacg!r} is absent")
    if "method" not in metric_table:
        raise FigureContractError("Metric table missing method column")
    metric_table = metric_table.copy()
    metric_table["method"] = metric_table["method"].astype(str)
    if metric_table["method"].duplicated().any():
        duplicates = metric_table.loc[
            metric_table["method"].duplicated(), "method"].tolist()
        raise FigureContractError(
            f"Metric table contains duplicate methods: {duplicates[:5]}")
    missing_metrics = [metric for metric in METRICS
                       if metric not in metric_table]
    if missing_metrics:
        raise FigureContractError(
            f"Metric table missing columns: {missing_metrics}")
    metric_table = metric_table.set_index("method", drop=False)
    missing_methods = [name for name in names if name not in metric_table.index]
    if missing_methods:
        raise FigureContractError(
            f"Metric table missing methods: {missing_methods}")
    rows: list[dict[str, Any]] = []
    overrides = manifest.get("method_color_overrides", {})
    ordered_specs = sorted(
        manifest["methods"], key=lambda item: 0 if item["name"] == dacg else 1)
    for order, spec in enumerate(ordered_specs):
        name = str(spec["name"])
        metric_row = metric_table.loc[name]
        valid_source = spec.get("valid", metric_row.get("valid", True))
        valid = _as_bool(valid_source, field=f"{name}.valid")
        values = {
            metric: float(pd.to_numeric(metric_row[metric], errors="coerce"))
            for metric in METRICS
        }
        if valid and not np.isfinite(list(values.values())).all():
            raise FigureContractError(f"Valid method {name} has missing metric")
        n_value = spec.get("n", metric_row.get("n", None))
        n = None if pd.isna(n_value) else int(n_value)
        if n is not None and n <= 0:
            raise FigureContractError(f"{name}.n must be positive")
        rows.append({
            "method": name,
            "prediction": str(spec["prediction"]),
            "implementation": str(spec.get("implementation", name)),
            "spot_id_column": spec.get("spot_id_column", None),
            "valid": valid,
            "invalid_reason": str(spec.get("invalid_reason", "")),
            "n": n,
            "spatial_order": order,
            "color": method_color(name, overrides),
            **values,
        })
    result = pd.DataFrame(rows)
    if not bool(result.loc[result["method"] == dacg, "valid"].iloc[0]):
        raise FigureContractError("DACG must be marked valid")
    valid_names = result.loc[
        result["valid"] & (result["method"] != dacg), "method"
    ].tolist()
    invalid_names = result.loc[~result["valid"], "method"].tolist()
    spatial_order = {
        name: order
        for order, name in enumerate([dacg, *valid_names, *invalid_names])
    }
    result["spatial_order"] = result["method"].map(spatial_order).astype(int)
    return result


def metric_improvement(
    dacg_value: float, baseline_value: float, metric: str,
) -> float:
    if metric not in METRICS:
        raise FigureContractError(f"Unsupported metric {metric}")
    if baseline_value == 0:
        return math.nan
    numerator = (
        baseline_value - dacg_value
        if metric in LOWER_IS_BETTER
        else dacg_value - baseline_value
    )
    return numerator / abs(baseline_value) * 100.0


def sort_metric_frame(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in METRICS:
        raise FigureContractError(f"Unsupported metric {metric}")
    ascending = metric in LOWER_IS_BETTER
    valid = frame.loc[frame["valid"]].sort_values(
        [metric, "method"], ascending=[ascending, True], na_position="last")
    invalid = frame.loc[~frame["valid"]].sort_values(
        [metric, "method"], ascending=[ascending, True], na_position="last")
    return pd.concat([valid, invalid], ignore_index=True)


def spatial_grid_shape(panel_count: int) -> tuple[int, int, int]:
    if panel_count <= 0:
        raise FigureContractError("panel_count must be positive")
    rows = math.ceil((panel_count + 1) / SPATIAL_COLUMNS)
    legend_row, legend_column = divmod(panel_count, SPATIAL_COLUMNS)
    return rows, legend_row, legend_column


def _style_spatial_axis(
    ax: plt.Axes,
    coordinates: pd.DataFrame,
    *,
    coordinate_columns: Sequence[str],
    invert_y: bool,
    frame: bool,
    extra_margin: float = 0.0,
) -> None:
    x = coordinates[coordinate_columns[0]].to_numpy(float)
    y = coordinates[coordinate_columns[1]].to_numpy(float)
    span = max(float(np.ptp(x)), float(np.ptp(y)), 1.0)
    margin = max(span * 0.025, float(extra_margin))
    ax.set_xlim(float(x.min() - margin), float(x.max() + margin))
    if invert_y:
        ax.set_ylim(float(y.max() + margin), float(y.min() - margin))
    else:
        ax.set_ylim(float(y.min() - margin), float(y.max() + margin))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(frame)
        if frame:
            spine.set_color("#D2D4D6")
            spine.set_linewidth(0.75)


def _nearest_distance(coordinates: np.ndarray) -> float:
    count = len(coordinates)
    if count < 2:
        return 1.0
    if count > 2500:
        rng = np.random.default_rng(17)
        query = coordinates[rng.choice(count, 2500, replace=False)]
    else:
        query = coordinates
    minima: list[np.ndarray] = []
    for start in range(0, len(query), 256):
        block = query[start:start + 256]
        delta = block[:, None, :] - coordinates[None, :, :]
        squared = np.einsum("ijk,ijk->ij", delta, delta)
        squared[squared <= 1e-20] = np.inf
        minima.append(np.sqrt(squared.min(axis=1)))
    distances = np.concatenate(minima)
    finite = distances[np.isfinite(distances) & (distances > 0)]
    return float(np.median(finite)) if len(finite) else 1.0


def _minimum_grid_step(values: np.ndarray) -> float:
    unique = np.unique(np.asarray(values, dtype=float))
    differences = np.diff(unique)
    positive = differences[np.isfinite(differences) & (differences > 1e-12)]
    return float(positive.min()) if len(positive) else 1.0


def _draw_dominant_map(
    ax: plt.Axes,
    coordinates: np.ndarray,
    dominant: np.ndarray,
    palette: np.ndarray,
    *,
    marker_size: float,
    render_mode: str,
) -> tuple[float, float]:
    if render_mode == "scatter":
        ax.scatter(
            coordinates[:, 0], coordinates[:, 1], c=palette[dominant],
            marker="s", s=marker_size, linewidths=0, rasterized=True,
        )
        return 0.0, 0.0
    if render_mode != "tiles":
        raise FigureContractError(
            "dominant_render_mode must be 'scatter' or 'tiles'")

    width = _minimum_grid_step(coordinates[:, 0])
    height = _minimum_grid_step(coordinates[:, 1])
    rectangles = [
        Rectangle(
            (float(x) - width / 2.0, float(y) - height / 2.0),
            width,
            height,
        )
        for x, y in coordinates
    ]
    collection = PatchCollection(
        rectangles,
        facecolors=palette[dominant],
        edgecolors="none",
        linewidths=0,
        antialiaseds=False,
        rasterized=False,
    )
    collection.set_snap(True)
    ax.add_collection(collection)
    return width, height


def _draw_pies(
    ax: plt.Axes,
    fractions: pd.DataFrame,
    coordinates: pd.DataFrame,
    colors: Mapping[str, str],
    *,
    coordinate_columns: Sequence[str],
    radius: float,
) -> None:
    wedges: list[Wedge] = []
    facecolors: list[str] = []
    positions = coordinates.loc[
        fractions.index, coordinate_columns].to_numpy(float)
    for (x, y), row in zip(positions, fractions.to_numpy(float), strict=True):
        start = 90.0
        for cell_type, fraction in zip(
                fractions.columns, row, strict=True):
            if fraction <= 1e-8:
                continue
            stop = start + 360.0 * float(fraction)
            wedges.append(Wedge((float(x), float(y)), radius, start, stop))
            facecolors.append(colors[str(cell_type)])
            start = stop
    collection = PatchCollection(
        wedges,
        facecolors=facecolors,
        edgecolors="white",
        linewidths=0.12,
        rasterized=True,
    )
    ax.add_collection(collection)


def _add_cell_type_legend(
    ax: plt.Axes,
    colors: Mapping[str, str],
    *,
    full_row: bool,
) -> None:
    ax.axis("off")
    handles = [
        Patch(facecolor=color, edgecolor="none", label=cell_type)
        for cell_type, color in colors.items()
    ]
    columns = 4 if full_row else 2
    legend = ax.legend(
        handles=handles,
        loc="center",
        ncol=columns,
        frameon=True,
        fontsize=7.2,
        handlelength=1.0,
        handleheight=0.9,
        columnspacing=1.1,
        labelspacing=0.35,
        borderpad=0.7,
    )
    legend.get_frame().set_edgecolor("#C9CDD1")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_facecolor("white")


def spatial_panel_title(method: str, *, invalid: bool = False) -> str:
    """Return the publication title for a spatial panel."""
    del invalid
    return str(method)


def _spatial_figure(
    *,
    benchmark: str,
    truth: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    coordinates: pd.DataFrame,
    methods: pd.DataFrame,
    colors: Mapping[str, str],
    coordinate_columns: Sequence[str],
    invert_y: bool,
    context: str,
    marker_size: float,
    dominant_render_mode: str,
    pie_radius_scale: float,
    figure_width: float,
    panel_height: float,
    legend_height: float,
    horizontal_space: float,
    vertical_space: float,
    kind: str,
) -> plt.Figure:
    method_order = methods.sort_values("spatial_order")["method"].tolist()
    frames = [("Ground truth", truth)] + [
        (method, predictions[method]) for method in method_order]
    rows, legend_row, legend_column = spatial_grid_shape(len(frames))
    legend_only_row = legend_column == 0
    ratios = [1.0] * rows
    if legend_only_row:
        ratios[-1] = 0.48
    height = panel_height * (rows - int(legend_only_row))
    height += legend_height if legend_only_row else 0.0
    fig = plt.figure(
        figsize=(figure_width, max(height, panel_height + 1.2)),
        facecolor="white",
    )
    grid = fig.add_gridspec(
        rows, SPATIAL_COLUMNS, height_ratios=ratios,
        left=0.025, right=0.985, bottom=0.035, top=0.885,
        wspace=horizontal_space, hspace=vertical_space,
    )
    metric_lookup = methods.set_index("method")
    palette = np.asarray([colors[str(name)] for name in truth.columns])
    xy = coordinates.loc[truth.index, coordinate_columns].to_numpy(float)
    radius = _nearest_distance(xy) * pie_radius_scale
    for index, (method, fractions) in enumerate(frames):
        row, column = divmod(index, SPATIAL_COLUMNS)
        ax = fig.add_subplot(grid[row, column])
        invalid = (
            False if method == "Ground truth"
            else not bool(metric_lookup.loc[method, "valid"])
        )
        title_name = spatial_panel_title(method, invalid=invalid)
        if kind == "dominant":
            dominant = fractions.to_numpy(float).argmax(axis=1)
            tile_width, tile_height = _draw_dominant_map(
                ax, xy, dominant, palette, marker_size=marker_size,
                render_mode=dominant_render_mode)
            dominant_margin = max(tile_width, tile_height) * 0.52
            frame_axis = False
        else:
            _draw_pies(
                ax, fractions, coordinates, colors,
                coordinate_columns=coordinate_columns, radius=radius)
            frame_axis = True
            dominant_margin = 0.0
        ax.set_title(
            title_name, fontsize=13.0, pad=7,
            color="#6B747B" if invalid else "#252A2E")
        _style_spatial_axis(
            ax, coordinates, coordinate_columns=coordinate_columns,
            invert_y=invert_y, frame=frame_axis,
            extra_margin=(
                radius * 1.08 if kind == "pie" else dominant_margin))
    if legend_column == 0:
        legend_ax = fig.add_subplot(grid[legend_row, :])
    else:
        legend_ax = fig.add_subplot(grid[legend_row, legend_column:])
    _add_cell_type_legend(
        legend_ax, colors, full_row=legend_column == 0)
    if kind == "dominant":
        title = f"{benchmark} dominant cell-type maps"
        subtitle = (
            "Square color denotes the highest-fraction cell type on each spot."
        )
    else:
        title = f"{benchmark} cell-fraction pie maps"
        subtitle = context or (
            f"Ground truth and predictions on the same {len(truth)} spots."
        )
    fig.suptitle(title, y=0.972, fontsize=16, color="#252A2E")
    fig.text(
        0.5, 0.942, subtitle, ha="center", va="center",
        fontsize=9.2, color="#606870")
    return fig


def _format_percent(value: float, *, focus_method: str) -> str:
    if math.isnan(value):
        return f"{focus_method} n/a"
    sign = "+" if value >= 0 else ""
    return f"{focus_method} {sign}{value:.1f}%"


def _metric_figure(
    frame: pd.DataFrame,
    *,
    metric: str,
    dacg_method: str,
) -> plt.Figure:
    count = len(frame)
    fig, ax = plt.subplots(
        figsize=(8.8, max(3.8, 0.53 * count + 1.25)), facecolor="white")
    _draw_metric_axis(
        ax, frame, metric=metric, dacg_method=dacg_method, compact=False)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.89, bottom=0.12)
    return fig


def _draw_metric_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    metric: str,
    dacg_method: str,
    compact: bool,
    annotation_gap: float = 0.19,
    annotation_margin: float | None = None,
    font_scale: float = 1.0,
) -> None:
    ordered = sort_metric_frame(frame, metric)
    count = len(ordered)
    y = np.arange(count)
    values = ordered[metric].to_numpy(float)
    plot_values = np.where(np.isfinite(values), values, 0.0)
    colors = ordered["color"].tolist()
    bars = ax.barh(
        y, plot_values, height=0.64, color=colors, edgecolor="none")
    for bar, (_, row) in zip(bars, ordered.iterrows(), strict=True):
        method = str(row["method"])
        if method == dacg_method:
            bar.set_edgecolor("#0B4F59")
            bar.set_linewidth(2.0)
        if not bool(row["valid"]):
            bar.set_hatch("///")
            bar.set_edgecolor("#7B848B")
            bar.set_linewidth(0.8)
            bar.set_alpha(0.72)
    labels: list[str] = []
    for _, row in ordered.iterrows():
        method = str(row["method"])
        label = f"{method} *" if method == dacg_method else method
        if not compact and pd.notna(row["n"]):
            label += f"  (n={int(row['n'])})"
        labels.append(label)
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    for tick, (_, row) in zip(
            ax.get_yticklabels(), ordered.iterrows(), strict=True):
        if str(row["method"]) == dacg_method:
            tick.set_color("#0B4F59")
            tick.set_fontweight("bold")
        else:
            tick.set_color("#66747D")
    finite_values = values[np.isfinite(values)]
    data_min = min(0.0, float(finite_values.min(initial=0.0)))
    data_max = max(0.0, float(finite_values.max(initial=0.0)))
    span = max(data_max - data_min, 1e-6)
    value_x = data_max + span * 0.025
    gap_x = data_max + span * annotation_gap
    if annotation_margin is None:
        annotation_margin = 0.55 if compact else 0.75
    ax.set_xlim(data_min, data_max + span * annotation_margin)
    dacg_value = float(
        ordered.loc[ordered["method"] == dacg_method, metric].iloc[0])
    for index, (_, row) in enumerate(ordered.iterrows()):
        method = str(row["method"])
        value = float(row[metric])
        value_label = f"{value:.5f}" if np.isfinite(value) else "n/a"
        ax.text(
            value_x, index, value_label, va="center", ha="left",
            fontsize=(7.4 if compact else 8.8) * font_scale,
            color="#61717B")
        if method == dacg_method:
            gap_label = dacg_method
        elif not np.isfinite(value):
            gap_label = "n/a"
        else:
            gap_label = _format_percent(
                metric_improvement(dacg_value, value, metric),
                focus_method=dacg_method,
            )
        ax.text(
            gap_x, index, gap_label, va="center", ha="left",
            fontsize=(7.4 if compact else 8.8) * font_scale,
            color="#0B4F59" if method == dacg_method else "#66747D")
    ax.set_title(
        METRIC_TITLES[metric],
        fontsize=(12.5 if compact else 15) * font_scale,
        pad=8 if compact else 14, color="#252A2E", fontweight="bold")
    ax.xaxis.grid(True, color="#D9E0E4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="x", colors="#63717A", length=3,
        labelsize=(7.6 if compact else 9) * font_scale)
    ax.tick_params(
        axis="y", length=0,
        labelsize=(7.6 if compact else 9) * font_scale)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#65737C")


def _paired_composite_figure(
    *,
    benchmark: str,
    truth: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    coordinates: pd.DataFrame,
    methods: pd.DataFrame,
    colors: Mapping[str, str],
    coordinate_columns: Sequence[str],
    invert_y: bool,
    marker_size: float,
    dominant_render_mode: str,
    pie_radius_scale: float,
    figure_width: float,
    spatial_panel_height: float,
    legend_height: float,
    metric_panel_height: float,
    metric_layout: str,
    metric_wspace: float,
    include_metrics: bool,
    outer_wspace: float,
    outer_hspace: float,
    tile_wspace: float,
    show_title: bool,
    title_fontsize: float,
    method_fontsize: float,
    text_scale: float,
    bold_method_titles: bool,
    legend_fontsize: float,
    legend_ncol: int,
    legend_expand: bool,
    dacg_method: str,
) -> plt.Figure:
    method_order = methods.sort_values("spatial_order")["method"].tolist()
    frames = [("Ground truth", truth)] + [
        (method, predictions[method]) for method in method_order]
    spatial_rows = math.ceil(len(frames) / SPATIAL_COLUMNS)
    metric_rows = 0 if not include_metrics else (
        1 if metric_layout == "single_row" else 2)
    height = (
        spatial_rows * spatial_panel_height
        + legend_height + metric_rows * metric_panel_height + 1.25
    )
    fig = plt.figure(figsize=(figure_width, height), facecolor="white")
    height_ratios = (
        [spatial_panel_height] * spatial_rows
        + [legend_height] + [metric_panel_height] * metric_rows
    )
    outer = fig.add_gridspec(
        spatial_rows + 1 + metric_rows, SPATIAL_COLUMNS,
        height_ratios=height_ratios,
        left=0.025, right=0.985, bottom=0.025, top=0.945,
        wspace=outer_wspace, hspace=outer_hspace,
    )
    metric_lookup = methods.set_index("method")
    palette = np.asarray([colors[str(name)] for name in truth.columns])
    xy = coordinates.loc[truth.index, coordinate_columns].to_numpy(float)
    radius = _nearest_distance(xy) * pie_radius_scale

    for index, (method, fractions) in enumerate(frames):
        row, column = divmod(index, SPATIAL_COLUMNS)
        tile = outer[row, column].subgridspec(
            2, 2, height_ratios=(0.10, 0.90),
            wspace=tile_wspace, hspace=0.0)
        title_ax = fig.add_subplot(tile[0, :])
        title_ax.axis("off")
        invalid = (
            False if method == "Ground truth"
            else not bool(metric_lookup.loc[method, "valid"])
        )
        values = fractions.to_numpy(float)
        dominant = values.argmax(axis=1)
        title = spatial_panel_title(method, invalid=invalid)
        title_ax.text(
            0.5, 0.48, title, ha="center", va="center",
            fontsize=method_fontsize * text_scale,
            color=(
                "#252A2E"
                if bold_method_titles or not invalid else "#6B747B"),
            fontweight=(
                "bold" if bold_method_titles or method == dacg_method
                else "normal"),
        )

        dominant_ax = fig.add_subplot(tile[1, 0])
        tile_width, tile_height = _draw_dominant_map(
            dominant_ax, xy, dominant, palette, marker_size=marker_size,
            render_mode=dominant_render_mode)
        _style_spatial_axis(
            dominant_ax, coordinates, coordinate_columns=coordinate_columns,
            invert_y=invert_y, frame=False,
            extra_margin=max(tile_width, tile_height) * 0.52)

        pie_ax = fig.add_subplot(tile[1, 1])
        _draw_pies(
            pie_ax, fractions, coordinates, colors,
            coordinate_columns=coordinate_columns, radius=radius)
        _style_spatial_axis(
            pie_ax, coordinates, coordinate_columns=coordinate_columns,
            invert_y=invert_y, frame=False, extra_margin=radius * 1.03)

    for index in range(len(frames), spatial_rows * SPATIAL_COLUMNS):
        row, column = divmod(index, SPATIAL_COLUMNS)
        blank = fig.add_subplot(outer[row, column])
        blank.axis("off")

    legend_ax = fig.add_subplot(outer[spatial_rows, :])
    legend_ax.axis("off")
    handles = [
        Patch(facecolor=color, edgecolor="none", label=cell_type)
        for cell_type, color in colors.items()
    ]
    legend_kwargs: dict[str, Any] = {}
    if legend_expand:
        legend_kwargs.update({
            "bbox_to_anchor": (0.0, 0.02, 1.0, 0.96),
            "mode": "expand",
            "borderaxespad": 0.0,
        })
    legend = legend_ax.legend(
        handles=handles, loc="center", ncol=legend_ncol, frameon=True,
        fontsize=legend_fontsize * text_scale,
        handlelength=1.0 * text_scale, handleheight=0.85 * text_scale,
        columnspacing=1.0, labelspacing=0.30, borderpad=0.55,
        **legend_kwargs)
    legend.get_frame().set_edgecolor("#C9CDD1")
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_facecolor("white")

    metric_grid = None
    if include_metrics and metric_layout == "single_row":
        metric_grid = outer[spatial_rows + 1, :].subgridspec(
            1, len(METRICS), wspace=metric_wspace)
    if include_metrics:
        for index, metric in enumerate(METRICS):
            if metric_layout == "single_row":
                assert metric_grid is not None
                ax = fig.add_subplot(metric_grid[0, index])
            else:
                row = spatial_rows + 1 + index // 2
                column = (index % 2) * 2
                ax = fig.add_subplot(outer[row, column:column + 2])
            _draw_metric_axis(
                ax, methods, metric=metric, dacg_method=dacg_method,
                compact=True,
                annotation_gap=(
                    0.29 if metric_layout == "single_row" else 0.19),
                annotation_margin=(
                    0.72 if metric_layout == "single_row" else None),
                font_scale=text_scale,
            )

    if show_title:
        figure_title = (
            f"{benchmark}: spatial fractions and aggregate accuracy"
            if include_metrics
            else f"{benchmark}: dominant and spot-fraction maps"
        )
        fig.suptitle(
            figure_title,
            y=0.987, fontsize=title_fontsize, color="#252A2E")
    if include_metrics:
        fig.text(
            0.025, 0.958, "(a)", ha="left", va="center",
            fontsize=11 * text_scale, fontweight="bold", color="#252A2E")
        metric_top = 0.025 + (
            metric_rows * metric_panel_height + legend_height) / height
        fig.text(
            0.025, metric_top - 0.005, "(b)", ha="left", va="center",
            fontsize=11 * text_scale, fontweight="bold", color="#252A2E")
    return fig


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for figure_format in formats:
        if figure_format not in FORMATS:
            raise FigureContractError(
                f"Unsupported output format: {figure_format}")
        path = output_dir / f"{stem}.{figure_format}"
        if figure_format == "png":
            fig.savefig(path, dpi=360, bbox_inches="tight")
        elif figure_format == "tiff":
            fig.savefig(
                path, dpi=600, format="tiff", bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(path, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def render_suite(
    manifest_path: Path,
    *,
    formats: Sequence[str] = FORMATS,
    validate_only: bool = False,
) -> dict[str, Any]:
    configure_style()
    manifest = load_manifest(manifest_path)
    spot_id_column = str(manifest.get("spot_id_column", "spot_id"))
    coordinate_spot_id_raw = manifest.get(
        "coordinate_spot_id_column", spot_id_column)
    coordinate_spot_id_column = (
        None if coordinate_spot_id_raw is None
        else str(coordinate_spot_id_raw)
    )
    coordinate_columns = tuple(
        str(value) for value in manifest.get(
            "coordinate_columns", ("x", "y")))
    row_sum_atol = float(manifest.get("row_sum_atol", 1e-6))
    renormalize_atol = float(manifest.get("renormalize_atol", 1e-3))
    negative_atol = float(manifest.get("negative_atol", 1e-10))
    if not (0 <= row_sum_atol <= renormalize_atol):
        raise FigureContractError(
            "Require 0 <= row_sum_atol <= renormalize_atol")
    truth, truth_audit = load_fraction_table(
        manifest["ground_truth"], spot_id_column=spot_id_column,
        row_sum_atol=row_sum_atol,
        renormalize_atol=renormalize_atol,
        negative_atol=negative_atol)
    coordinates, coordinate_reordered = load_coordinates(
        manifest["coordinates"], spot_id_column=coordinate_spot_id_column,
        coordinate_columns=coordinate_columns,
        expected_spots=truth.index)
    raw_metrics = _read_table(manifest["metrics"])
    methods = normalize_methods(manifest, raw_metrics)
    predictions: dict[str, pd.DataFrame] = {}
    input_audits = {"ground_truth": truth_audit, "predictions": {}}
    for _, row in methods.sort_values("spatial_order").iterrows():
        method = str(row["method"])
        method_spot_id = row.get("spot_id_column", None)
        prediction_spot_id_column = (
            spot_id_column if method_spot_id is None or pd.isna(method_spot_id)
            else str(method_spot_id)
        )
        prediction, audit = load_fraction_table(
            Path(str(row["prediction"])),
            spot_id_column=prediction_spot_id_column,
            row_sum_atol=row_sum_atol,
            renormalize_atol=renormalize_atol,
            negative_atol=negative_atol,
            expected_spots=truth.index,
            expected_types=truth.columns)
        predictions[method] = prediction
        audit["spot_id_column"] = prediction_spot_id_column
        input_audits["predictions"][method] = audit
    dacg_method = str(manifest["dacg_method"])
    cell_colormap = str(manifest.get("cell_type_colormap", "turbo"))
    cell_colors = cell_type_colors(
        truth.columns.tolist(), colormap=cell_colormap)
    spatial_rows = []
    gt_values = truth.to_numpy(float)
    gt_dominant = gt_values.argmax(axis=1)
    for method, prediction in predictions.items():
        values = prediction.to_numpy(float)
        spatial_rows.append({
            "method": method,
            "RMSE": float(np.sqrt(np.mean((values - gt_values) ** 2))),
            "dominant_agreement": float(np.mean(
                values.argmax(axis=1) == gt_dominant)),
            "active_dominant_types": int(np.unique(
                values.argmax(axis=1)).size),
        })
    spatial_summary = pd.DataFrame(spatial_rows)
    metric_methods = set(methods["method"])
    extra_metric_methods = sorted(
        set(raw_metrics["method"].astype(str)).difference(metric_methods))
    output_dir = Path(manifest["output_dir"])
    provenance: dict[str, Any] = {
        "benchmark": str(manifest["benchmark"]),
        "manifest": str(manifest["manifest_path"]),
        "renderer": str(Path(__file__).resolve()),
        "dacg_method": dacg_method,
        "spot_count": int(len(truth)),
        "cell_type_count": int(len(truth.columns)),
        "method_count": int(len(methods)),
        "method_implementations": dict(zip(
            methods["method"], methods["implementation"], strict=True)),
        "coordinate_alignment": (
            "positional" if coordinate_spot_id_column is None else "spot_id"
        ),
        "coordinate_reordered": coordinate_reordered,
        "extra_metric_methods_ignored": extra_metric_methods,
        "input_audits": input_audits,
        "inputs": {},
        "formats": list(formats),
        "visual_contract": {
            "spatial_columns": SPATIAL_COLUMNS,
            "metric_error_bars": False,
            "invalid_methods_last": True,
            "cell_type_palette_scope": "benchmark_local",
            "cell_type_colormap": cell_colormap,
            "method_palette_scope": "project_global",
        },
    }
    input_paths = [
        Path(manifest["ground_truth"]), Path(manifest["coordinates"]),
        Path(manifest["metrics"]),
        *[Path(value) for value in methods["prediction"]],
    ]
    for path in input_paths:
        provenance["inputs"][str(path)] = {
            "sha256": _sha256(path), "bytes": int(path.stat().st_size)}
    if validate_only:
        return provenance
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_summary.to_csv(
        output_dir / "spatial_panel_metrics.csv", index=False)
    methods.to_csv(
        output_dir / "method_manifest_normalized.csv", index=False)
    methods.loc[:, ["method", *METRICS, "valid", "n"]].to_csv(
        output_dir / "metric_values_normalized.csv", index=False)
    pd.DataFrame({
        "cell_type": list(cell_colors),
        "color_hex": list(cell_colors.values()),
    }).to_csv(output_dir / "cell_type_colors.csv", index=False)
    methods.loc[:, ["method", "color"]].rename(
        columns={"color": "color_hex"}).to_csv(
            output_dir / "method_colors.csv", index=False)
    generated: list[Path] = []
    context = str(manifest.get("spatial_context", ""))
    invert_y = _as_bool(
        manifest.get("invert_y", True), field="invert_y")
    marker_size = float(manifest.get("dominant_marker_size", 20.0))
    dominant_render_mode = str(
        manifest.get("dominant_render_mode", "scatter")).strip().lower()
    if dominant_render_mode not in {"scatter", "tiles"}:
        raise FigureContractError(
            "dominant_render_mode must be 'scatter' or 'tiles'")
    radius_scale = float(manifest.get("pie_radius_scale", 0.42))
    figure_width = float(manifest.get("spatial_figure_width", 16.0))
    panel_height = float(manifest.get("spatial_panel_height", 4.6))
    legend_height = float(manifest.get("spatial_legend_height", 2.4))
    horizontal_space = float(manifest.get("spatial_wspace", 0.20))
    vertical_space = float(manifest.get("spatial_hspace", 0.34))
    paired_composite = _as_bool(
        manifest.get("paired_composite", True), field="paired_composite")
    composite_tile_wspace = float(
        manifest.get("composite_tile_wspace", 0.12))
    composite_show_title = _as_bool(
        manifest.get("composite_show_title", True),
        field="composite_show_title")
    composite_title_fontsize = float(
        manifest.get("composite_title_fontsize", 16.0))
    composite_method_fontsize = float(
        manifest.get("composite_method_fontsize", 12.6))
    composite_metric_layout = str(
        manifest.get("composite_metric_layout", "2x2")).strip().lower()
    composite_metric_wspace = float(
        manifest.get("composite_metric_wspace", 0.32))
    composite_text_scale = float(
        manifest.get("composite_text_scale", 1.0))
    composite_bold_method_titles = _as_bool(
        manifest.get("composite_bold_method_titles", False),
        field="composite_bold_method_titles")
    composite_legend_fontsize = float(
        manifest.get("composite_legend_fontsize", 6.8))
    composite_legend_ncol = int(
        manifest.get("composite_legend_ncol", 8))
    composite_legend_expand = _as_bool(
        manifest.get("composite_legend_expand", False),
        field="composite_legend_expand")
    composite_include_metrics = _as_bool(
        manifest.get("composite_include_metrics", True),
        field="composite_include_metrics")
    if composite_metric_layout not in {"2x2", "single_row"}:
        raise FigureContractError(
            "composite_metric_layout must be '2x2' or 'single_row'")
    if composite_tile_wspace <= -0.5:
        raise FigureContractError(
            "composite_tile_wspace must be greater than -0.5")
    if composite_metric_wspace < 0:
        raise FigureContractError(
            "composite_metric_wspace must be non-negative")
    if composite_text_scale <= 0:
        raise FigureContractError(
            "composite_text_scale must be positive")
    if composite_legend_fontsize <= 0:
        raise FigureContractError(
            "composite_legend_fontsize must be positive")
    if composite_legend_ncol < 1:
        raise FigureContractError(
            "composite_legend_ncol must be at least 1")
    provenance["visual_contract"].update({
        "dominant_marker_size": marker_size,
        "dominant_render_mode": dominant_render_mode,
        "pie_radius_scale": radius_scale,
        "spatial_figure_width": figure_width,
        "spatial_panel_height": panel_height,
        "spatial_legend_height": legend_height,
        "spatial_wspace": horizontal_space,
        "spatial_hspace": vertical_space,
        "paired_composite": paired_composite,
        "composite_tile_wspace": composite_tile_wspace,
        "composite_show_title": composite_show_title,
        "composite_metric_layout": composite_metric_layout,
        "composite_metric_wspace": composite_metric_wspace,
        "composite_text_scale": composite_text_scale,
        "composite_bold_method_titles": composite_bold_method_titles,
        "composite_legend_fontsize": composite_legend_fontsize,
        "composite_legend_ncol": composite_legend_ncol,
        "composite_legend_expand": composite_legend_expand,
        "composite_include_metrics": composite_include_metrics,
        "composite_title_fontsize": composite_title_fontsize,
        "composite_method_fontsize": composite_method_fontsize,
        "spatial_title_mode": "method_only",
    })
    dominant = _spatial_figure(
        benchmark=str(manifest["benchmark"]), truth=truth,
        predictions=predictions, coordinates=coordinates, methods=methods,
        colors=cell_colors, coordinate_columns=coordinate_columns,
        invert_y=invert_y, context=context, marker_size=marker_size,
        dominant_render_mode=dominant_render_mode,
        pie_radius_scale=radius_scale, figure_width=figure_width,
        panel_height=panel_height, legend_height=legend_height,
        horizontal_space=horizontal_space, vertical_space=vertical_space,
        kind="dominant")
    generated.extend(save_figure(
        dominant, output_dir, "dominant_maps", formats))
    pies = _spatial_figure(
        benchmark=str(manifest["benchmark"]), truth=truth,
        predictions=predictions, coordinates=coordinates, methods=methods,
        colors=cell_colors, coordinate_columns=coordinate_columns,
        invert_y=invert_y, context=context, marker_size=marker_size,
        dominant_render_mode=dominant_render_mode,
        pie_radius_scale=radius_scale, figure_width=figure_width,
        panel_height=panel_height, legend_height=legend_height,
        horizontal_space=horizontal_space, vertical_space=vertical_space,
        kind="pie")
    generated.extend(save_figure(
        pies, output_dir, "spot_pie_maps", formats))
    for metric in METRICS:
        figure = _metric_figure(
            methods, metric=metric, dacg_method=dacg_method)
        generated.extend(save_figure(
            figure, output_dir, metric, formats))
    if paired_composite:
        composite = _paired_composite_figure(
            benchmark=str(manifest["benchmark"]), truth=truth,
            predictions=predictions, coordinates=coordinates,
            methods=methods, colors=cell_colors,
            coordinate_columns=coordinate_columns, invert_y=invert_y,
            marker_size=float(manifest.get(
                "composite_dominant_marker_size", marker_size)),
            dominant_render_mode=dominant_render_mode,
            pie_radius_scale=float(manifest.get(
                "composite_pie_radius_scale", radius_scale)),
            figure_width=float(manifest.get(
                "composite_figure_width", 19.2)),
            spatial_panel_height=float(manifest.get(
                "composite_spatial_panel_height", 4.45)),
            legend_height=float(manifest.get(
                "composite_legend_height", 2.05)),
            metric_panel_height=float(manifest.get(
                "composite_metric_panel_height", 4.7)),
            metric_layout=composite_metric_layout,
            metric_wspace=composite_metric_wspace,
            include_metrics=composite_include_metrics,
            outer_wspace=float(manifest.get(
                "composite_outer_wspace", 0.06)),
            outer_hspace=float(manifest.get(
                "composite_outer_hspace", 0.10)),
            tile_wspace=composite_tile_wspace,
            show_title=composite_show_title,
            title_fontsize=composite_title_fontsize,
            method_fontsize=composite_method_fontsize,
            text_scale=composite_text_scale,
            bold_method_titles=composite_bold_method_titles,
            legend_fontsize=composite_legend_fontsize,
            legend_ncol=composite_legend_ncol,
            legend_expand=composite_legend_expand,
            dacg_method=dacg_method,
        )
        composite_stem = (
            "paired_spatial_metric_composite"
            if composite_include_metrics else "paired_spatial_composite")
        generated.extend(save_figure(
            composite, output_dir, composite_stem, formats))
    provenance["outputs"] = [str(path) for path in generated]
    (output_dir / "figure_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=True),
        encoding="utf-8")
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render standardized SILTA benchmark figures")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = render_suite(
        args.manifest, formats=tuple(args.formats),
        validate_only=bool(args.validate_only))
    if args.validate_only:
        print(json.dumps(result, indent=2, ensure_ascii=True))
    else:
        output_dir = load_manifest(args.manifest)["output_dir"]
        print(f"complete={output_dir}")


if __name__ == "__main__":
    main()
