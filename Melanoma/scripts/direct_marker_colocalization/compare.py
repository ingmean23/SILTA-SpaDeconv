"""Evaluate and compare locked predictions across methods and slices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.direct_marker_colocalization.evaluate import evaluate_prediction, read_observed
from scripts.direct_marker_colocalization.render import render_method_heatmap
from scripts.direct_marker_colocalization.schema import json_safe, load_config


SUMMARY_METRICS = (
    "macro_marker_pcc_strict",
    "macro_marker_spearman_strict",
    "macro_marker_proxy_pcc_strict",
    "macro_marker_proxy_spearman_strict",
    "macro_specificity_gap",
    "target_is_top_fraction",
    "positive_marker_fraction",
    "valid_type_fraction",
    "secondary_all_source_macro_marker_pcc_strict",
    "secondary_all_source_macro_marker_spearman_strict",
    "negative_control_pass",
    "negative_control_min_observed_pcc_delta",
)


def _predictions(item: dict[str, Any], base: Path) -> dict[str, str]:
    value = item.get("predictions", item)
    result = {}
    for key, value_path in value.items():
        if key in {"metadata", "notes"}:
            continue
        path = Path(value_path)
        result[str(key)] = str(path if path.is_absolute() else base / path)
    return result


def _holm(p_values: list[float]) -> list[float]:
    result = np.full(len(p_values), np.nan)
    valid = [(index, value) for index, value in enumerate(p_values) if np.isfinite(value)]
    ordered = sorted(valid, key=lambda item: item[1])
    running = 0.0
    total = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        result[index] = running
    return result.tolist()


def _paired_statistics(per_type: pd.DataFrame, primary_method: str) -> pd.DataFrame:
    primary = per_type[per_type.method == primary_method].set_index(["slice", "cell_type"])
    rows = []
    for method in sorted(set(per_type.method) - {primary_method}):
        other = per_type[per_type.method == method].set_index(["slice", "cell_type"])
        common = primary.index.intersection(other.index)
        left = primary.loc[common, "mean_marker_pcc_strict"].to_numpy(dtype=float)
        right = other.loc[common, "mean_marker_pcc_strict"].to_numpy(dtype=float)
        difference = left - right
        if len(difference) and np.any(np.abs(difference) > 1e-12):
            test = wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
            p_value = float(test.pvalue)
        else:
            p_value = 1.0
        rows.append({
            "primary_method": primary_method,
            "baseline": method,
            "paired_units": len(common),
            "mean_pcc_difference": float(np.mean(difference)) if len(difference) else np.nan,
            "median_pcc_difference": float(np.median(difference)) if len(difference) else np.nan,
            "wins": int(np.sum(difference > 1e-12)),
            "losses": int(np.sum(difference < -1e-12)),
            "ties": int(np.sum(np.abs(difference) <= 1e-12)),
            "wilcoxon_p": p_value,
        })
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["wilcoxon_p_holm"] = _holm(frame.wilcoxon_p.tolist())
    return frame


def compare_methods(
    config_path: str | Path,
    predictions_manifest: str | Path,
    output_dir: str | Path,
    primary_method: str | None = None,
    render: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    manifest_path = Path(predictions_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    methods = manifest.get("methods", manifest)
    if not isinstance(methods, dict) or not methods:
        raise ValueError("Prediction manifest contains no methods")
    primary_method = primary_method or manifest.get("primary_method", "DACG")
    if primary_method not in methods:
        raise ValueError(f"Primary method is absent: {primary_method}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries, gene_frames, type_frames = [], [], []
    prediction_maps = {
        method: _predictions(item, manifest_path.parent)
        for method, item in methods.items()
    }
    for method, predictions in prediction_maps.items():
        missing_slices = sorted(set(config.slices) - set(predictions))
        if missing_slices:
            raise ValueError(f"{method} misses slices: {missing_slices}")
    for slice_name in config.slices:
        observed_data = read_observed(config, slice_name)
        for method, predictions in prediction_maps.items():
            run_dir = output_dir / "methods" / method / slice_name
            summary = evaluate_prediction(
                config.path, slice_name, method, predictions[slice_name], run_dir,
                render=render, controls=True,
                observed_data=observed_data,
            )
            summaries.append(summary)
            gene_frames.append(pd.read_csv(run_dir / "per_gene.csv"))
            type_frames.append(pd.read_csv(run_dir / "per_cell_type.csv"))

    per_gene = pd.concat(gene_frames, ignore_index=True)
    per_type = pd.concat(type_frames, ignore_index=True)
    per_slice = pd.DataFrame([
        {"method": item["method"], "slice": item["slice"], **{
            metric: item[metric] for metric in SUMMARY_METRICS
        }} for item in summaries
    ])
    ranking = per_slice.groupby("method", as_index=False)[list(SUMMARY_METRICS)].mean()
    per_method_sd = per_slice.groupby("method").agg(
        macro_marker_pcc_strict_sd=("macro_marker_pcc_strict", "std"),
        macro_marker_spearman_strict_sd=("macro_marker_spearman_strict", "std"),
    )
    ranking = ranking.merge(per_method_sd, on="method", how="left")
    bootstrap_rows = []
    for method in methods:
        method_summaries = [item for item in summaries if item["method"] == method]
        pcc_samples = [
            item.get("macro_marker_pcc_strict_bootstrap_samples", [])
            for item in method_summaries
        ]
        spearman_samples = [
            item.get("macro_marker_spearman_strict_bootstrap_samples", [])
            for item in method_summaries
        ]
        if pcc_samples and all(pcc_samples) and all(spearman_samples):
            pcc_macro = np.mean(np.asarray(pcc_samples, dtype=float), axis=0)
            spearman_macro = np.mean(np.asarray(spearman_samples, dtype=float), axis=0)
            pcc_bounds = np.nanpercentile(pcc_macro, [2.5, 97.5])
            spearman_bounds = np.nanpercentile(spearman_macro, [2.5, 97.5])
            bootstrap_rows.append({
                "method": method,
                "macro_marker_pcc_strict_ci95_low": pcc_bounds[0],
                "macro_marker_pcc_strict_ci95_high": pcc_bounds[1],
                "macro_marker_spearman_strict_ci95_low": spearman_bounds[0],
                "macro_marker_spearman_strict_ci95_high": spearman_bounds[1],
            })
    if bootstrap_rows:
        ranking = ranking.merge(pd.DataFrame(bootstrap_rows), on="method", how="left")
    control_pass = per_slice.groupby("method")["negative_control_pass"].all()
    ranking["negative_control_pass_all_slices"] = ranking.method.map(control_pass)
    ranking = ranking.sort_values(
        ["negative_control_pass_all_slices", "macro_marker_pcc_strict"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    type_macro = per_type.groupby(["method", "cell_type"], as_index=False).agg({
        "mean_marker_pcc_strict": "mean",
        "mean_marker_spearman_strict": "mean",
        "marker_proxy_pcc_strict": "mean",
        "mean_specificity_gap": "mean",
        "target_is_top_fraction": "mean",
    })
    best_by_type = type_macro.loc[
        type_macro.groupby("cell_type")["mean_marker_pcc_strict"].idxmax(),
        ["cell_type", "method", "mean_marker_pcc_strict"],
    ].rename(columns={"method": "winner", "mean_marker_pcc_strict": "winning_pcc"})
    statistics = _paired_statistics(per_type, primary_method)

    primary_types = type_macro[
        (type_macro.method == primary_method)
        & type_macro.cell_type.isin(config.primary_cell_types)
    ].copy()
    best_primary = primary_types.sort_values("mean_marker_pcc_strict", ascending=False).head(1)
    baseline_types = type_macro[
        (type_macro.method != primary_method)
        & type_macro.cell_type.isin(config.primary_cell_types)
    ]
    baseline_best = baseline_types.groupby("cell_type")["mean_marker_pcc_strict"].max()
    primary_types["gain_vs_best_baseline"] = primary_types.apply(
        lambda row: row.mean_marker_pcc_strict - baseline_best.get(row.cell_type, np.nan), axis=1,
    )
    best_gain = (
        primary_types.sort_values("gain_vs_best_baseline", ascending=False).head(1)
        if len(baseline_types) else primary_types.iloc[0:0]
    )

    per_gene.to_csv(output_dir / "all_methods_per_gene.csv", index=False)
    per_type.to_csv(output_dir / "all_methods_per_cell_type.csv", index=False)
    per_slice.to_csv(output_dir / "all_methods_per_slice.csv", index=False)
    type_macro.to_csv(output_dir / "all_methods_type_macro.csv", index=False)
    best_by_type.to_csv(output_dir / "cell_type_winners.csv", index=False)
    ranking.to_csv(output_dir / "method_ranking.csv", index=False)
    statistics.to_csv(output_dir / "paired_statistics.csv", index=False)
    render_method_heatmap(type_macro, output_dir / "figures" / "all_types_heatmap.png")

    summary = {
        "protocol": "direct marker-fraction colocalization v1",
        "definition": "marker genes infer ST cell proxies; predictions are correlated directly with those proxies",
        "dataset_id": config.dataset_id,
        "primary_method": primary_method,
        "primary_ranking_metric": "equal-slice mean macro_marker_pcc_strict",
        "primary_cell_types": config.primary_cell_types,
        "ranking_requires_negative_control_pass": True,
        "method_ranking": ranking.to_dict(orient="records"),
        "exploratory_best_primary_type": best_primary.to_dict(orient="records"),
        "exploratory_largest_primary_gain": best_gain.to_dict(orient="records"),
        "exploratory_results_are_not_primary": True,
        "n_methods": len(methods),
        "n_slices": len(config.slices),
    }
    summary = json_safe(summary)
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primary-method")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    summary = compare_methods(
        args.config, args.predictions_manifest, args.output_dir,
        args.primary_method, args.render,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
