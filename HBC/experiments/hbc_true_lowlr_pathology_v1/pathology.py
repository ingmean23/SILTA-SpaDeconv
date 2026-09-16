#!/usr/bin/env python3
"""Robust, leakage-safe HBC pathology selector for DACG and baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.hbc_pathology_trackb_v3 import converter as v3
from experiments.hbc_pseudo_pathology_ab_v1 import pathology_converter_v2 as v2
from tools import evaluate_hbc_annotation as legacy


Candidate = v3.Candidate


def make_features(
    fine: np.ndarray, fine_names: Sequence[str],
    coarse: np.ndarray, coarse_names: Sequence[str],
    coords: np.ndarray, scales: Sequence[int],
) -> pd.DataFrame:
    """Reuse Track-B v3 features and add signed inter-scale ring contrasts."""
    frame = v3.make_multiscale_features(
        fine, fine_names, coarse, coarse_names, coords, scales)
    fine = v2.normalize(fine)
    coarse = v2.normalize(coarse)
    neighborhoods: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for scale in sorted({int(value) for value in scales}):
        graph = v2.knn_indices(coords, scale)
        neighborhoods[scale] = (
            v2.neighbor_mean(fine, graph), v2.neighbor_mean(coarse, graph))
    ordered = sorted(neighborhoods)
    for left, right in zip(ordered, ordered[1:]):
        left_fine, left_coarse = neighborhoods[left]
        right_fine, right_coarse = neighborhoods[right]
        for index, name in enumerate(fine_names):
            frame[f"ring_k{left}_k{right}_fine::{name}"] = (
                left_fine[:, index] - right_fine[:, index])
        for index, name in enumerate(coarse_names):
            frame[f"ring_k{left}_k{right}_coarse::{name}"] = (
                left_coarse[:, index] - right_coarse[:, index])
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Non-finite robust pathology features")
    return frame


def candidates(preset: Mapping[str, Any]) -> list[Candidate]:
    model_lookup = {item["id"]: item for item in preset["models"]}
    rows: list[Candidate] = []
    for feature in preset["feature_sets"]:
        for model in preset["models"]:
            for refinement in preset["refinement_strengths"]:
                rows.append(Candidate(
                    feature["id"], model["id"], model["kind"], model,
                    refinement_strength=float(refinement)))
        for ensemble in preset.get("ensembles", []):
            left = model_lookup[ensemble["left"]]
            for refinement in preset["refinement_strengths"]:
                rows.append(Candidate(
                    feature["id"], left["id"], left["kind"], left,
                    ensemble_model_id=str(ensemble["right"]),
                    ensemble_weight=float(ensemble["weight"]),
                    refinement_strength=float(refinement)))
    identifiers = [row.candidate_id for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate robust pathology candidate IDs")
    return rows


def _minimum_recall(confusion: pd.DataFrame) -> float:
    matrix = confusion.to_numpy(dtype=float)
    denominator = matrix.sum(axis=1)
    recall = np.divide(
        np.diag(matrix), denominator,
        out=np.zeros_like(denominator), where=denominator > 0)
    supported = recall[denominator > 0]
    return float(supported.min()) if len(supported) else 0.0


def robust_score(
    folds: pd.DataFrame, objective: Mapping[str, float],
) -> tuple[float, bool, dict[str, float]]:
    means = folds[["balanced_accuracy", "macro_f1", "macro_auroc"]].mean()
    fold_core = 0.5 * (
        folds["balanced_accuracy"].to_numpy()
        + folds["macro_f1"].to_numpy())
    weighted = (
        float(objective["balanced_accuracy"]) * float(means["balanced_accuracy"])
        + float(objective["macro_f1"]) * float(means["macro_f1"])
        + float(objective["macro_auroc"]) * float(means["macro_auroc"]))
    worst = float(fold_core.min())
    std = float(fold_core.std(ddof=0))
    minimum_recall = float(folds["minimum_class_recall"].min())
    score = (
        weighted
        + float(objective["worst_fold_weight"]) * worst
        - float(objective["fold_std_penalty"]) * std)
    valid = bool(
        worst >= float(objective["minimum_worst_fold"])
        and minimum_recall >= float(objective["minimum_class_recall"]))
    return score, valid, {
        "inner_mean_balanced_accuracy": float(means["balanced_accuracy"]),
        "inner_mean_macro_f1": float(means["macro_f1"]),
        "inner_mean_macro_auroc": float(means["macro_auroc"]),
        "inner_worst_fold_core": worst,
        "inner_fold_core_std": std,
        "inner_minimum_class_recall": minimum_recall,
    }


def _inner_search(
    method: str, split: str, outer_train: np.ndarray,
    work: pd.DataFrame, labels: np.ndarray,
    fine: np.ndarray, coords: np.ndarray,
    feature_tables: Mapping[str, pd.DataFrame],
    feature_specs: Mapping[str, Mapping[str, Any]],
    preset: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, Candidate]:
    inner_work = work.iloc[outer_train].reset_index(drop=True)
    inner_labels = labels[outer_train]
    folds = v3.spatial_block_folds(
        inner_work, inner_labels, split, int(preset["inner_folds"]),
        float(preset["inner_validation_fraction"]), int(preset["seed"]) + 17)
    model_lookup = {item["id"]: item for item in preset["models"]}
    all_candidates = candidates(preset)
    fold_rows: list[dict[str, Any]] = []
    for feature_id, spec in feature_specs.items():
        matrix = feature_tables[feature_id].to_numpy(dtype=float)[outer_train]
        relevant = [item for item in all_candidates if item.feature_set == feature_id]
        classes = np.array(sorted(pd.unique(inner_labels)))
        for fold_index, (train_index, valid_index) in enumerate(folds):
            probability_cache: dict[str, np.ndarray] = {}
            for model_id, settings in model_lookup.items():
                fitted_classes, probability = v3.fit_model(
                    settings, matrix[train_index], inner_labels[train_index],
                    matrix, int(preset["seed"]) + fold_index)
                probability_cache[model_id] = v3.align_probabilities(
                    probability, fitted_classes, classes)
            for candidate in relevant:
                probability = v3.combine_candidate_probabilities(
                    candidate, probability_cache)
                probability = v2.refine_probabilities(
                    probability, fine[outer_train], coords[outer_train],
                    min(spec["scales"]), candidate.refinement_strength,
                    iterations=2)
                values = v2.metrics(
                    inner_labels[valid_index], probability[valid_index], classes)
                fold_rows.append({
                    "method": method,
                    "outer_split": split,
                    "candidate_id": candidate.candidate_id,
                    "fold": fold_index,
                    "balanced_accuracy": values["balanced_accuracy"],
                    "macro_f1": values["macro_f1"],
                    "macro_auroc": values["macro_auroc"],
                    "minimum_class_recall": _minimum_recall(values["confusion"]),
                })
    fold_table = pd.DataFrame(fold_rows)
    summaries: list[dict[str, Any]] = []
    for candidate in all_candidates:
        frame = fold_table[fold_table["candidate_id"] == candidate.candidate_id]
        score, valid, diagnostics = robust_score(frame, preset["objective"])
        summaries.append({
            "method": method,
            "outer_split": split,
            "candidate_id": candidate.candidate_id,
            "feature_set": candidate.feature_set,
            "model_id": candidate.model_id,
            "ensemble_model_id": candidate.ensemble_model_id,
            "ensemble_weight": candidate.ensemble_weight,
            "refinement_strength": candidate.refinement_strength,
            "inner_selection_score": score,
            "inner_hard_gate_valid": valid,
            **diagnostics,
        })
    summary = pd.DataFrame(summaries).sort_values(
        ["inner_hard_gate_valid", "inner_selection_score"],
        ascending=[False, False])
    selected_id = str(summary.iloc[0]["candidate_id"])
    selected = next(item for item in all_candidates if item.candidate_id == selected_id)
    return fold_table, summary, selected


def development_score(
    prediction: Path, metadata: Path, preset: Mapping[str, Any],
) -> dict[str, Any]:
    """Score a DACG prediction using outer-train labels only."""
    work, fine, fine_names, coarse, coarse_names, coords = v2.load_data(
        prediction, metadata)
    labels = work["region"].astype(str).to_numpy()
    feature_specs = {item["id"]: item for item in preset["feature_sets"]}
    feature_tables = {
        key: make_features(
            fine, fine_names, coarse, coarse_names, coords, spec["scales"])
        for key, spec in feature_specs.items()}
    rows = []
    for split in preset["outer_splits"]:
        outer_train, _ = legacy.split_indices(
            work, split, float(preset["outer_test_size"]), int(preset["seed"]))
        _, summary, selected = _inner_search(
            "dacg", split, outer_train, work, labels, fine, coords,
            feature_tables, feature_specs, preset)
        winner = summary.iloc[0]
        rows.append({
            "outer_split": split,
            "candidate_id": selected.candidate_id,
            "score": float(winner["inner_selection_score"]),
            "valid": bool(winner["inner_hard_gate_valid"]),
            "worst_fold": float(winner["inner_worst_fold_core"]),
            "minimum_class_recall": float(
                winner["inner_minimum_class_recall"]),
        })
    return {
        "pathology_development_score": float(np.mean([row["score"] for row in rows])),
        "pathology_development_valid": bool(all(row["valid"] for row in rows)),
        "pathology_inner_splits": rows,
        "pathology_outer_labels_used": False,
    }


def run_method(
    method: str, prediction: Path, metadata: Path,
    preset: Mapping[str, Any], output_dir: Path,
) -> pd.DataFrame:
    """Lock on inner folds, then open each sealed outer split once."""
    work, fine, fine_names, coarse, coarse_names, coords = v2.load_data(
        prediction, metadata)
    labels = work["region"].astype(str).to_numpy()
    feature_specs = {item["id"]: item for item in preset["feature_sets"]}
    features = {
        key: make_features(
            fine, fine_names, coarse, coarse_names, coords, spec["scales"])
        for key, spec in feature_specs.items()}
    model_lookup = {item["id"]: item for item in preset["models"]}
    rows = []
    for split in preset["outer_splits"]:
        outer_train, outer_test = legacy.split_indices(
            work, split, float(preset["outer_test_size"]), int(preset["seed"]))
        folds, summary, selected = _inner_search(
            method, split, outer_train, work, labels, fine, coords,
            features, feature_specs, preset)
        matrix = features[selected.feature_set].to_numpy(dtype=float)
        classes, raw_probability = v3._fit_candidate(
            selected, model_lookup, matrix[outer_train], labels[outer_train],
            matrix, int(preset["seed"]))
        probability = v2.refine_probabilities(
            raw_probability, fine, coords,
            min(feature_specs[selected.feature_set]["scales"]),
            selected.refinement_strength, iterations=2)
        values = v2.metrics(labels[outer_test], probability[outer_test], classes)
        raw_values = v2.metrics(
            labels[outer_test], raw_probability[outer_test], classes)
        split_dir = output_dir / method / split
        split_dir.mkdir(parents=True, exist_ok=True)
        folds.to_csv(split_dir / "inner_fold_metrics.csv", index=False)
        summary.to_csv(split_dir / "inner_candidate_summary.csv", index=False)
        all_raw = np.asarray(classes)[np.argmax(raw_probability, axis=1)]
        all_selected = np.asarray(classes)[np.argmax(probability, axis=1)]
        prediction_frame = work[
            ["ID", "spot_id", "region", "scaled_x", "scaled_y"]].copy()
        prediction_frame["outer_split"] = "train"
        prediction_frame.loc[
            prediction_frame.index[outer_test], "outer_split"] = "test"
        prediction_frame["raw_region"] = all_raw
        prediction_frame["selected_region"] = all_selected
        prediction_frame.to_csv(
            split_dir / "outer_test_predictions.csv", index=False)
        v2.save_maps(
            work, classes, all_raw, all_selected, probability, split_dir,
            f"{method} robust Track-B {split}")
        payload = {
            "method": method,
            "outer_split": split,
            "selected_candidate_id": selected.candidate_id,
            "selected_inner_score": float(summary.iloc[0]["inner_selection_score"]),
            "selected_inner_hard_gate_valid": bool(
                summary.iloc[0]["inner_hard_gate_valid"]),
            "outer_test_balanced_accuracy": values["balanced_accuracy"],
            "outer_test_macro_f1": values["macro_f1"],
            "outer_test_macro_auroc": values["macro_auroc"],
            "outer_test_raw_balanced_accuracy": raw_values["balanced_accuracy"],
            "outer_test_raw_macro_f1": raw_values["macro_f1"],
            "outer_test_raw_macro_auroc": raw_values["macro_auroc"],
            "outer_test_labels_used_for_selection": False,
            "absolute_coordinates_used_as_features": False,
        }
        (split_dir / "outer_test_metrics.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        rows.append(payload)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / method / "outer_test_metrics.csv", index=False)
    return result
