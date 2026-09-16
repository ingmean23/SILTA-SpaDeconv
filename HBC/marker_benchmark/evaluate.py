"""Evaluate one prediction by direct marker-derived cell colocalization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marker_benchmark.controls import score_negative_controls
from marker_benchmark.bootstrap import spatial_block_bootstrap
from marker_benchmark.h5ad_io import (
    inspect_h5ad, read_row_sums, read_selected_columns,
)
from marker_benchmark.metrics import safe_corr, score_direct_colocalization
from marker_benchmark.render import read_locations, render_proxy_maps
from marker_benchmark.schema import (
    DatasetConfig, file_sha256, json_safe, load_config,
)


def _dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def _read_h5ad_expression(path: Path, genes: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    info = inspect_h5ad(path)
    index = {gene: position for position, gene in enumerate(info.var_names)}
    missing = [gene for gene in genes if gene not in index]
    if missing:
        raise ValueError(f"ST H5AD misses marker genes: {missing}")
    raw = _dense(read_selected_columns(path, [index[gene] for gene in genes])).astype(float)
    totals = read_row_sums(path)
    if np.any(raw < 0) or np.any(totals < 0):
        raise ValueError("Signed/transformed ST X is not accepted as raw counts")
    return info.obs_index, raw, totals


def _read_count_table(path: Path, spot_ids: list[str], genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, sep=None, engine="python")
    gene_column = frame.columns[0]
    names = frame[gene_column].astype(str).str.split().str[0]
    numeric = frame.drop(columns=[gene_column]).apply(pd.to_numeric, errors="raise")
    numeric.index = names
    numeric = numeric.groupby(level=0, sort=False).sum()
    missing_genes = [gene for gene in genes if gene not in numeric.index]
    missing_spots = [spot for spot in spot_ids if spot not in numeric.columns]
    if missing_genes or missing_spots:
        raise ValueError(
            f"Raw count table mismatch: genes={missing_genes}, spots={len(missing_spots)}"
        )
    raw = numeric.loc[genes, spot_ids].to_numpy(dtype=float).T
    totals = numeric.loc[:, spot_ids].sum(axis=0).to_numpy(dtype=float)
    if np.any(raw < 0) or np.any(totals < 0):
        raise ValueError("Raw count table contains negative values")
    return raw, totals


def read_observed(config: DatasetConfig, slice_name: str) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    spec = config.slices[slice_name]
    info = inspect_h5ad(spec.st)
    genes = config.panel.genes
    if spec.st_counts:
        raw, totals = _read_count_table(spec.st_counts, info.obs_index, genes)
    else:
        spot_ids, raw, totals = _read_h5ad_expression(spec.st, genes)
        if spot_ids != info.obs_index:
            raise ValueError("Unexpected H5AD spot-order mismatch")
    cpm = raw / np.maximum(totals[:, None], 1e-12) * 10000.0
    return info.obs_index, genes, raw, np.log1p(cpm)


def read_prediction(
    path: str | Path,
    spot_ids: list[str],
    cell_types: list[str],
    aliases: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path, sep=None, engine="python")
    id_column = next((name for name in ("spot_id", "barcode", "cell_id") if name in frame), None)
    if id_column is None and len(frame.columns) and str(frame.columns[0]).startswith("Unnamed:"):
        id_column = frame.columns[0]
    if id_column is None:
        raise ValueError(f"Prediction has no explicit spot ID: {path}")
    frame[id_column] = frame[id_column].astype(str)
    if frame[id_column].duplicated().any():
        raise ValueError("Prediction contains duplicate spot IDs")
    indexed = frame.set_index(id_column)
    missing_spots = [spot for spot in spot_ids if spot not in indexed.index]
    if missing_spots:
        raise ValueError(f"Prediction misses {len(missing_spots)} spots")
    aliases = aliases or {}
    columns: dict[str, str] = {}
    missing_types = []
    for cell_type in cell_types:
        candidates = [cell_type, *aliases.get(cell_type, [])]
        matched = next((name for name in candidates if name in indexed.columns), None)
        if matched is None:
            missing_types.append(cell_type)
        else:
            columns[cell_type] = matched
    values = np.zeros((len(spot_ids), len(cell_types)), dtype=float)
    for index, cell_type in enumerate(cell_types):
        if cell_type in columns:
            values[:, index] = indexed.loc[spot_ids, columns[cell_type]].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < -1e-8):
        raise ValueError("Prediction contains negative or non-finite fractions")
    values = np.maximum(values, 0)
    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 1e-12):
        raise ValueError("Prediction contains zero-sum rows")
    return values / row_sums, missing_types


def evaluate_prediction(
    config_path: str | Path,
    slice_name: str,
    method: str,
    prediction: str | Path,
    output_dir: str | Path,
    render: bool = False,
    controls: bool = True,
    observed_data: tuple[list[str], list[str], np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    if slice_name not in config.slices:
        raise ValueError(f"Unknown slice {slice_name}; choices={sorted(config.slices)}")
    spot_ids, genes, raw, observed = (
        observed_data if observed_data is not None
        else read_observed(config, slice_name)
    )
    fractions, missing_types = read_prediction(
        prediction, spot_ids, config.panel.cell_types,
        config.payload.get("prediction_column_aliases"),
    )
    per_gene, per_type, metrics, proxies = score_direct_colocalization(
        fractions, observed, config.panel.cell_types, genes,
        config.panel.markers_by_type, config.strict_invalid_score,
        config.primary_cell_types,
    )
    gene_index = {gene: index for index, gene in enumerate(genes)}
    type_index = {cell_type: index for index, cell_type in enumerate(config.panel.cell_types)}
    per_gene["pcc_raw_expression"] = [
        safe_corr(
            pearsonr,
            fractions[:, type_index[row.cell_type]],
            raw[:, gene_index[row.gene]],
        )
        for row in per_gene.itertuples(index=False)
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_gene.insert(0, "slice", slice_name)
    per_gene.insert(0, "method", method)
    per_type.insert(0, "slice", slice_name)
    per_type.insert(0, "method", method)
    per_gene.to_csv(output_dir / "per_gene.csv", index=False)
    per_type.to_csv(output_dir / "per_cell_type.csv", index=False)
    proxy_output = {"spot_id": spot_ids}
    proxy_output.update({f"fraction::{name}": fractions[:, index] for index, name in enumerate(config.panel.cell_types)})
    proxy_output.update({f"marker_proxy::{name}": proxies[name] for name in config.panel.cell_types})
    pd.DataFrame(proxy_output).to_csv(output_dir / "spot_marker_proxies.csv", index=False)

    control_rows = []
    base_seed = int(config.payload.get("control_seed", 42))
    slice_seed = base_seed + 1009 * list(config.slices).index(slice_name)
    if controls:
        control_rows = score_negative_controls(
            fractions, observed, config.panel.cell_types, genes,
            config.panel.markers_by_type, config.strict_invalid_score,
            slice_seed,
            config.primary_cell_types,
        )
        for row in control_rows:
            row["pcc_delta_from_observed"] = (
                metrics["macro_marker_pcc_strict"] - row["macro_marker_pcc_strict"]
            )
        pd.DataFrame(control_rows).to_csv(output_dir / "negative_controls.csv", index=False)
    figure_paths: list[str] = []
    coordinates = config.slices[slice_name].coordinates
    locations_frame = read_locations(coordinates, spot_ids) if coordinates else None
    bootstrap = (
        spatial_block_bootstrap(
            fractions, observed, locations_frame, config.panel.cell_types, genes,
            config.panel.markers_by_type, config.strict_invalid_score,
            config.primary_cell_types, config.bootstrap_repeats,
            config.bootstrap_grid_size, slice_seed,
        )
        if locations_frame is not None and config.bootstrap_repeats > 0
        else {"spatial_block_bootstrap_repeats": 0}
    )
    if render and coordinates:
        figure_paths = render_proxy_maps(
            output_dir / "figures" / "cell_types",
            locations_frame, fractions, proxies,
            config.panel.cell_types,
        )
    control_pass = bool(control_rows) and all(
        row["pcc_delta_from_observed"] > config.control_min_delta
        for row in control_rows
    )
    control_min_observed_pcc_delta = (
        min(row["pcc_delta_from_observed"] for row in control_rows)
        if control_rows else None
    )
    summary = {
        "protocol": "direct marker-fraction colocalization v1",
        "definition": "predicted cell fraction versus observed ST marker expression; no P@S_sc reconstruction",
        "marker_proxy_definition": "equal-weight mean of per-gene standardized observed log1p(CPM)",
        "dataset_id": config.dataset_id,
        "slice": slice_name,
        "method": method,
        "prediction": str(Path(prediction).resolve()),
        "prediction_sha256": file_sha256(prediction),
        "config": str(config.path),
        "config_sha256": file_sha256(config.path),
        "panel": str(config.panel.path),
        "panel_sha256": config.panel.payload.get("direct_panel_sha256", config.panel.payload.get("panel_sha256")),
        "st": str(config.slices[slice_name].st),
        "st_sha256": file_sha256(config.slices[slice_name].st),
        "st_counts": str(config.slices[slice_name].st_counts) if config.slices[slice_name].st_counts else None,
        "st_counts_sha256": file_sha256(config.slices[slice_name].st_counts) if config.slices[slice_name].st_counts else None,
        "normalization": "log1p(CPM from full raw ST library size); raw-expression PCC is sensitivity only",
        "missing_prediction_types": missing_types,
        "negative_controls": control_rows,
        "negative_control_min_delta": config.control_min_delta,
        "negative_control_pass": control_pass,
        "negative_control_min_observed_pcc_delta": control_min_observed_pcc_delta,
        "rendered_figures": figure_paths,
        **metrics,
        **bootstrap,
    }
    summary = json_safe(summary)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--slice", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-controls", action="store_true")
    args = parser.parse_args()
    summary = evaluate_prediction(
        args.config, args.slice, args.method, args.prediction, args.output_dir,
        args.render, not args.no_controls,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
