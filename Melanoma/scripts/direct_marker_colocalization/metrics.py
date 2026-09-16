"""Metric definitions for direct marker-derived cell proxies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


EPS = 1e-12


def safe_corr(function, left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if np.std(left) <= EPS or np.std(right) <= EPS:
        return float("nan")
    result = function(left, right)
    result = result.statistic if hasattr(result, "statistic") else result[0]
    return float(result) if np.isfinite(result) else float("nan")


def marker_consensus(values: np.ndarray) -> np.ndarray:
    """Infer a cell-type proxy as the mean standardized marker expression."""
    values = np.asarray(values, dtype=float)
    means = values.mean(axis=0, keepdims=True)
    scales = values.std(axis=0, keepdims=True)
    valid = scales.ravel() > EPS
    if not np.any(valid):
        return np.full(values.shape[0], np.nan)
    standardized = (values[:, valid] - means[:, valid]) / scales[:, valid]
    return standardized.mean(axis=1)


def score_direct_colocalization(
    fractions: np.ndarray,
    observed: np.ndarray,
    cell_types: list[str],
    genes: list[str],
    markers_by_type: dict[str, list[str]],
    strict_invalid_score: float = -1.0,
    primary_cell_types: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    """Compare predicted fractions directly with real ST marker expression."""
    fractions = np.asarray(fractions, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if fractions.shape != (observed.shape[0], len(cell_types)):
        raise ValueError("Fraction shape does not match spots/cell types")
    if observed.shape[1] != len(genes):
        raise ValueError("Observed expression shape does not match genes")
    gene_index = {gene: index for index, gene in enumerate(genes)}
    type_index = {cell_type: index for index, cell_type in enumerate(cell_types)}
    per_gene_rows: list[dict[str, Any]] = []
    per_type_rows: list[dict[str, Any]] = []
    proxies: dict[str, np.ndarray] = {}

    for cell_type in cell_types:
        target = fractions[:, type_index[cell_type]]
        target_flat = np.std(target) <= EPS
        marker_names = markers_by_type[cell_type]
        marker_columns = [gene_index[gene] for gene in marker_names]
        proxy = marker_consensus(observed[:, marker_columns])
        proxies[cell_type] = proxy
        marker_rows: list[dict[str, Any]] = []
        for gene, column in zip(marker_names, marker_columns):
            signal = observed[:, column]
            observed_flat = np.std(signal) <= EPS
            status = "observed_flat" if observed_flat else "prediction_flat" if target_flat else "valid"
            pcc = safe_corr(pearsonr, target, signal)
            spearman = safe_corr(spearmanr, target, signal)
            strict_pcc = pcc if np.isfinite(pcc) else strict_invalid_score if target_flat else np.nan
            strict_spearman = spearman if np.isfinite(spearman) else strict_invalid_score if target_flat else np.nan
            competitors = []
            target_rank = np.nan
            specificity_gap = np.nan
            target_is_top = False
            if target_flat and not observed_flat:
                specificity_gap = float(strict_invalid_score)
                target_rank = len(cell_types)
            elif not observed_flat:
                correlations = np.asarray([
                    safe_corr(pearsonr, fractions[:, index], signal)
                    for index in range(len(cell_types))
                ])
                valid_corr = np.where(np.isfinite(correlations), correlations, strict_invalid_score)
                intended = valid_corr[type_index[cell_type]]
                competitors = np.delete(valid_corr, type_index[cell_type]).tolist()
                competitor_max = max(competitors) if competitors else np.nan
                specificity_gap = float(intended - competitor_max) if competitors else np.nan
                target_rank = int(1 + np.sum(valid_corr > intended))
                unique_top = int(np.sum(np.isclose(valid_corr, intended, atol=1e-12))) == 1
                target_is_top = bool(target_rank == 1 and unique_top)
            row = {
                "cell_type": cell_type,
                "gene": gene,
                "status": status,
                "pcc": pcc,
                "spearman": spearman,
                "pcc_strict": strict_pcc,
                "spearman_strict": strict_spearman,
                "specificity_gap": specificity_gap,
                "target_rank": target_rank,
                "target_is_top": target_is_top,
                "observed_std": float(np.std(signal)),
                "prediction_std": float(np.std(target)),
            }
            marker_rows.append(row)
            per_gene_rows.append(row)

        marker_frame = pd.DataFrame(marker_rows)
        evaluable = marker_frame.status != "observed_flat"
        valid = marker_frame.status == "valid"
        strict_pcc_values = marker_frame.loc[evaluable, "pcc_strict"].to_numpy(dtype=float)
        strict_spearman_values = marker_frame.loc[evaluable, "spearman_strict"].to_numpy(dtype=float)
        proxy_pcc = safe_corr(pearsonr, target, proxy)
        proxy_spearman = safe_corr(spearmanr, target, proxy)
        has_evaluable_markers = bool(evaluable.any())
        per_type_rows.append({
            "cell_type": cell_type,
            "marker_count": len(marker_frame),
            "evaluable_marker_count": int(evaluable.sum()),
            "valid_marker_count": int(valid.sum()),
            "mean_marker_pcc": float(marker_frame.loc[valid, "pcc"].mean()) if valid.any() else np.nan,
            "mean_marker_spearman": float(marker_frame.loc[valid, "spearman"].mean()) if valid.any() else np.nan,
            "mean_marker_pcc_strict": float(np.nanmean(strict_pcc_values)) if len(strict_pcc_values) else np.nan,
            "mean_marker_spearman_strict": float(np.nanmean(strict_spearman_values)) if len(strict_spearman_values) else np.nan,
            "marker_proxy_pcc": proxy_pcc,
            "marker_proxy_spearman": proxy_spearman,
            "marker_proxy_pcc_strict": (
                proxy_pcc if np.isfinite(proxy_pcc)
                else strict_invalid_score if has_evaluable_markers else np.nan
            ),
            "marker_proxy_spearman_strict": (
                proxy_spearman if np.isfinite(proxy_spearman)
                else strict_invalid_score if has_evaluable_markers else np.nan
            ),
            "mean_specificity_gap": float(marker_frame.loc[evaluable, "specificity_gap"].mean()),
            "target_is_top_fraction": float(marker_frame.loc[evaluable, "target_is_top"].mean()),
            "positive_marker_fraction": float((marker_frame.loc[evaluable, "pcc_strict"] > 0).mean()),
            "prediction_std": float(np.std(target)),
            "prediction_flat": bool(target_flat),
        })

    per_gene = pd.DataFrame(per_gene_rows)
    per_type = pd.DataFrame(per_type_rows)
    primary_cell_types = list(primary_cell_types or cell_types)
    primary_type = per_type[per_type.cell_type.isin(primary_cell_types)]
    primary_gene = per_gene[per_gene.cell_type.isin(primary_cell_types)]
    if len(primary_type) != len(primary_cell_types):
        raise ValueError("Primary cell types are absent from metric inputs")
    evaluable_gene = per_gene.status != "observed_flat"
    evaluable_primary_gene = primary_gene.status != "observed_flat"
    summary = {
        "primary_cell_types": primary_cell_types,
        "macro_marker_pcc_strict": float(primary_type.mean_marker_pcc_strict.mean()),
        "macro_marker_spearman_strict": float(primary_type.mean_marker_spearman_strict.mean()),
        "macro_marker_proxy_pcc_strict": float(primary_type.marker_proxy_pcc_strict.mean()),
        "macro_marker_proxy_spearman_strict": float(primary_type.marker_proxy_spearman_strict.mean()),
        "macro_specificity_gap": float(primary_type.mean_specificity_gap.mean()),
        "target_is_top_fraction": float(primary_gene.loc[evaluable_primary_gene, "target_is_top"].mean()),
        "positive_marker_fraction": float((primary_gene.loc[evaluable_primary_gene, "pcc_strict"] > 0).mean()),
        "valid_type_fraction": float((~primary_type.prediction_flat).mean()),
        "secondary_all_source_macro_marker_pcc_strict": float(per_type.mean_marker_pcc_strict.mean()),
        "secondary_all_source_macro_marker_spearman_strict": float(per_type.mean_marker_spearman_strict.mean()),
        "secondary_all_source_valid_type_fraction": float((~per_type.prediction_flat).mean()),
        "n_primary_cell_types": len(primary_cell_types),
        "n_cell_types": len(cell_types),
        "n_marker_genes": len(per_gene),
    }
    return per_gene, per_type, summary, proxies
