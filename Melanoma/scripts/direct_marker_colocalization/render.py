"""Deterministic visual outputs for the direct marker benchmark."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:100]


def read_locations(path: str | Path, spot_ids: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=None, engine="python")
    id_col = next(
        (name for name in ("barcode", "spot_id", "cell_id") if name in frame),
        frame.columns[0],
    )
    x_col = next((name for name in ("x", "imagecol", "pxl_col_in_fullres") if name in frame), None)
    y_col = next((name for name in ("y", "imagerow", "pxl_row_in_fullres") if name in frame), None)
    if x_col is None or y_col is None:
        raise ValueError(f"Location file has no supported x/y columns: {path}")
    indexed = frame.assign(_spot=frame[id_col].astype(str)).set_index("_spot")
    missing = [spot for spot in spot_ids if spot not in indexed.index]
    if missing:
        raise ValueError(f"Locations miss {len(missing)} spots")
    return indexed.loc[spot_ids, [x_col, y_col]].rename(columns={x_col: "x", y_col: "y"})


def render_proxy_maps(
    output_dir: str | Path,
    locations: pd.DataFrame,
    fractions: np.ndarray,
    proxies: dict[str, np.ndarray],
    cell_types: list[str],
) -> list[str]:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, cell_type in enumerate(cell_types):
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
        fraction_values = fractions[:, index]
        fraction_limit = max(float(np.quantile(fraction_values, 0.99)), 0.05)
        fraction = axes[0].scatter(
            locations.x, locations.y, c=fraction_values, s=12, marker="s",
            linewidths=0, cmap="viridis", vmin=0, vmax=fraction_limit,
        )
        proxy_values = proxies[cell_type]
        finite = proxy_values[np.isfinite(proxy_values)]
        limit = max(float(np.quantile(np.abs(finite), 0.98)), 1.0) if len(finite) else 1.0
        proxy = axes[1].scatter(
            locations.x, locations.y, c=proxy_values, s=12, marker="s",
            linewidths=0, cmap="coolwarm", vmin=-limit, vmax=limit,
        )
        axes[0].set_title(f"{cell_type}\nPredicted fraction")
        axes[1].set_title(f"{cell_type}\nMarker-derived proxy")
        for axis in axes:
            axis.set_aspect("equal")
            axis.invert_yaxis()
            axis.set_axis_off()
        figure.colorbar(fraction, ax=axes[0], fraction=0.046, pad=0.03)
        figure.colorbar(proxy, ax=axes[1], fraction=0.046, pad=0.03)
        path = output_dir / f"{_safe_name(cell_type)}.png"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))
    return paths


def render_method_heatmap(frame: pd.DataFrame, output: str | Path) -> None:
    import matplotlib.pyplot as plt

    pivot = frame.pivot(index="method", columns="cell_type", values="mean_marker_pcc_strict")
    width = max(7.0, 0.36 * len(pivot.columns) + 2.5)
    height = max(3.2, 0.45 * len(pivot.index) + 1.8)
    figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = axis.imshow(pivot.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=90)
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    axis.set_title("Direct marker-fraction PCC by cell type")
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="Mean marker PCC")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
