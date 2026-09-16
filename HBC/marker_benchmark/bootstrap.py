"""Deterministic spatial block bootstrap for marker-colocalization metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .metrics import EPS, safe_corr


def _axis_bins(values: np.ndarray, grid_size: int) -> np.ndarray:
    ranks = pd.Series(values).rank(method="first").to_numpy(dtype=float)
    scaled = (ranks - 1.0) / max(len(ranks), 1)
    return np.minimum((scaled * grid_size).astype(int), grid_size - 1)


def _primary_metrics(
    fractions: np.ndarray,
    observed: np.ndarray,
    cell_types: list[str],
    genes: list[str],
    markers_by_type: dict[str, list[str]],
    primary_cell_types: list[str],
    strict_invalid_score: float,
) -> tuple[float, float]:
    type_index = {cell_type: index for index, cell_type in enumerate(cell_types)}
    gene_index = {gene: index for index, gene in enumerate(genes)}
    type_pcc, type_spearman = [], []
    for cell_type in primary_cell_types:
        target = fractions[:, type_index[cell_type]]
        target_flat = np.std(target) <= EPS
        marker_pcc, marker_spearman = [], []
        for gene in markers_by_type[cell_type]:
            signal = observed[:, gene_index[gene]]
            if np.std(signal) <= EPS:
                continue
            if target_flat:
                marker_pcc.append(strict_invalid_score)
                marker_spearman.append(strict_invalid_score)
            else:
                marker_pcc.append(safe_corr(pearsonr, target, signal))
                marker_spearman.append(safe_corr(spearmanr, target, signal))
        type_pcc.append(float(np.nanmean(marker_pcc)))
        type_spearman.append(float(np.nanmean(marker_spearman)))
    return float(np.nanmean(type_pcc)), float(np.nanmean(type_spearman))


def spatial_block_bootstrap(
    fractions: np.ndarray,
    observed: np.ndarray,
    locations: pd.DataFrame,
    cell_types: list[str],
    genes: list[str],
    markers_by_type: dict[str, list[str]],
    strict_invalid_score: float,
    primary_cell_types: list[str],
    repeats: int,
    grid_size: int,
    seed: int,
) -> dict[str, Any]:
    if repeats <= 0:
        return {"spatial_block_bootstrap_repeats": 0}
    x_bins = _axis_bins(locations.x.to_numpy(dtype=float), grid_size)
    y_bins = _axis_bins(locations.y.to_numpy(dtype=float), grid_size)
    block_ids = x_bins * grid_size + y_bins
    blocks = [np.flatnonzero(block_ids == value) for value in np.unique(block_ids)]
    if len(blocks) < 2:
        raise ValueError("Spatial block bootstrap requires at least two populated blocks")
    rng = np.random.default_rng(seed)
    pcc, spearman = [], []
    for _ in range(repeats):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in selected])
        pcc_value, spearman_value = _primary_metrics(
            fractions[indices], observed[indices], cell_types, genes,
            markers_by_type, primary_cell_types, strict_invalid_score,
        )
        pcc.append(pcc_value)
        spearman.append(spearman_value)
    pcc_bounds = np.nanpercentile(pcc, [2.5, 97.5])
    spearman_bounds = np.nanpercentile(spearman, [2.5, 97.5])
    return {
        "spatial_block_bootstrap_repeats": repeats,
        "spatial_block_count": len(blocks),
        "spatial_block_grid_size": grid_size,
        "macro_marker_pcc_strict_ci95_low": float(pcc_bounds[0]),
        "macro_marker_pcc_strict_ci95_high": float(pcc_bounds[1]),
        "macro_marker_spearman_strict_ci95_low": float(spearman_bounds[0]),
        "macro_marker_spearman_strict_ci95_high": float(spearman_bounds[1]),
        "macro_marker_pcc_strict_bootstrap_samples": [float(value) for value in pcc],
        "macro_marker_spearman_strict_bootstrap_samples": [
            float(value) for value in spearman
        ],
    }
