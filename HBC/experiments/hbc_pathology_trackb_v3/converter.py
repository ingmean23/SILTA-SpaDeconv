#!/usr/bin/env python3
"""Track-B v3 HBC pathology converter with sealed outer spatial holdouts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.hbc_pseudo_pathology_ab_v1 import pathology_converter_v2 as v2
from tools import evaluate_hbc_annotation as legacy


@dataclass(frozen=True)
class Candidate:
    feature_set: str
    model_id: str
    model_kind: str
    model_settings: Mapping[str, Any]
    ensemble_model_id: str | None = None
    ensemble_weight: float | None = None
    refinement_strength: float = 0.0

    @property
    def candidate_id(self) -> str:
        parts = [self.feature_set, self.model_id]
        if self.ensemble_model_id is not None:
            parts.extend([self.ensemble_model_id, f"blend{self.ensemble_weight:g}"])
        parts.append(f"smooth{self.refinement_strength:g}")
        return "__".join(parts)


def _block(data: dict[str, np.ndarray], prefix: str, values: np.ndarray,
           names: Sequence[str]) -> None:
    for index, name in enumerate(names):
        data[f"{prefix}::{name}"] = values[:, index]


def make_multiscale_features(
    fine: np.ndarray,
    fine_names: Sequence[str],
    coarse: np.ndarray,
    coarse_names: Sequence[str],
    coords: np.ndarray,
    scales: Sequence[int],
) -> pd.DataFrame:
    """Build coordinate-free composition features at several graph scales."""
    fine = v2.normalize(fine)
    coarse = v2.normalize(coarse)
    eps = 1e-5
    fine_log = np.log(fine + eps)
    coarse_log = np.log(coarse + eps)
    data: dict[str, np.ndarray] = {}
    _block(data, "fine", fine, fine_names)
    _block(data, "coarse", coarse, coarse_names)
    _block(data, "fine_clr", fine_log - fine_log.mean(1, keepdims=True), fine_names)
    _block(data, "coarse_clr", coarse_log - coarse_log.mean(1, keepdims=True), coarse_names)

    for scale in sorted({int(value) for value in scales}):
        graph = v2.knn_indices(coords, scale)
        n1_fine = v2.neighbor_mean(fine, graph)
        n1_coarse = v2.neighbor_mean(coarse, graph)
        n2_coarse = v2.neighbor_mean(n1_coarse, graph)
        std_fine = np.sqrt(np.maximum(
            v2.neighbor_mean(np.square(fine), graph) - np.square(n1_fine), 0.0))
        std_coarse = np.sqrt(np.maximum(
            v2.neighbor_mean(np.square(coarse), graph) - np.square(n1_coarse), 0.0))
        prefix = f"k{scale}"
        _block(data, f"{prefix}_n1_fine", n1_fine, fine_names)
        _block(data, f"{prefix}_n1_coarse", n1_coarse, coarse_names)
        _block(data, f"{prefix}_n2_coarse", n2_coarse, coarse_names)
        _block(data, f"{prefix}_delta_fine", fine - n1_fine, fine_names)
        _block(data, f"{prefix}_delta_coarse", coarse - n1_coarse, coarse_names)
        _block(data, f"{prefix}_std_fine", std_fine, fine_names)
        _block(data, f"{prefix}_std_coarse", std_coarse, coarse_names)

    group_index = {name: index for index, name in enumerate(coarse_names)}
    for left, right in (
        ("Epithelial/Tumor", "Lymphoid"),
        ("Epithelial/Tumor", "Myeloid/DC"),
        ("Epithelial/Tumor", "Stromal/CAF"),
        ("Epithelial/Tumor", "Endothelial"),
        ("Stromal/CAF", "Endothelial"),
        ("Lymphoid", "Myeloid/DC"),
    ):
        if left in group_index and right in group_index:
            data[f"logratio::{left}/{right}"] = (
                coarse_log[:, group_index[left]] - coarse_log[:, group_index[right]])
    sorted_values = np.sort(fine, axis=1)
    data["summary::entropy"] = (
        -(fine * np.log(fine + 1e-12)).sum(1) / math.log(fine.shape[1]))
    data["summary::max"] = sorted_values[:, -1]
    data["summary::margin"] = sorted_values[:, -1] - sorted_values[:, -2]
    frame = pd.DataFrame(data)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Non-finite multiscale pathology features")
    if any(name in frame.columns for name in ("scaled_x", "scaled_y")):
        raise AssertionError("Absolute coordinates must not enter pathology features")
    return frame


def spatial_block_folds(
    frame: pd.DataFrame,
    labels: np.ndarray,
    split: str,
    n_folds: int,
    validation_fraction: float,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create repeated contiguous validation blocks inside an outer train set."""
    coord_name = "scaled_x" if split == "spatial_x" else "scaled_y"
    order = np.argsort(frame[coord_name].to_numpy(dtype=float))
    n_valid = max(1, int(round(len(order) * float(validation_fraction))))
    max_start = max(0, len(order) - n_valid)
    starts = np.linspace(0, max_start, max(2, int(n_folds)) + 2)[1:-1]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(len(frame))
    for start in np.unique(np.rint(starts).astype(int)):
        valid = order[start:start + n_valid]
        train = np.setdiff1d(all_indices, valid)
        if len(np.unique(labels[train])) != len(np.unique(labels)):
            continue
        if len(np.unique(labels[valid])) < 2:
            continue
        folds.append((train, valid))
    if len(folds) < 2:
        raise ValueError(f"Could not create at least two valid inner spatial folds for {split}")
    return folds


def _standardize(x_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x_train, axis=0, keepdims=True)
    std = np.nanstd(x_train, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return np.nan_to_num((x_train - mean) / std), np.nan_to_num((x_eval - mean) / std)


def fit_model(
    settings: Mapping[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    kind = settings["kind"]
    train, evaluate = _standardize(x_train, x_eval)
    if kind == "elasticnet":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            C=float(settings["c"]), penalty="elasticnet", solver="saga",
            l1_ratio=float(settings["l1_ratio"]), class_weight="balanced",
            max_iter=1200, tol=5e-4, n_jobs=1, random_state=int(seed))
    elif kind == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.utils.class_weight import compute_sample_weight

        model = HistGradientBoostingClassifier(
            learning_rate=float(settings["learning_rate"]),
            max_iter=int(settings["max_iter"]),
            max_depth=int(settings["max_depth"]),
            l2_regularization=float(settings["l2_regularization"]),
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
            random_state=int(seed))
        model.fit(train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
        return model.classes_.astype(str), model.predict_proba(evaluate)
    else:
        raise ValueError(f"Unknown pathology model kind: {kind}")
    model.fit(train, y_train)
    return model.classes_.astype(str), model.predict_proba(evaluate)


def align_probabilities(
    probability: np.ndarray,
    classes: Sequence[str],
    target_classes: Sequence[str],
) -> np.ndarray:
    lookup = {str(name): index for index, name in enumerate(classes)}
    aligned = np.zeros((len(probability), len(target_classes)), dtype=float)
    for index, name in enumerate(target_classes):
        if str(name) in lookup:
            aligned[:, index] = probability[:, lookup[str(name)]]
    return aligned / np.maximum(aligned.sum(1, keepdims=True), 1e-12)


def score_metrics(values: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    return float(
        weights["balanced_accuracy"] * values["balanced_accuracy"]
        + weights["macro_f1"] * values["macro_f1"]
        + weights["macro_auroc"] * values["macro_auroc"])


def selection_score(folds: pd.DataFrame, weights: Mapping[str, float]) -> float:
    means = folds[["balanced_accuracy", "macro_f1", "macro_auroc"]].mean()
    per_fold = folds.apply(lambda row: score_metrics(row, weights), axis=1)
    return score_metrics(means, weights) - float(weights["fold_std_penalty"]) * float(per_fold.std(ddof=0))


def candidates(preset: Mapping[str, Any]) -> list[Candidate]:
    rows: list[Candidate] = []
    for feature_set in preset["feature_sets"]:
        for model in preset["models"]:
            for smooth in preset["refinement_strengths"]:
                rows.append(Candidate(
                    feature_set["id"], model["id"], model["kind"], model,
                    refinement_strength=float(smooth)))
        linear = [model for model in preset["models"] if model["kind"] == "elasticnet"]
        boosting = [model for model in preset["models"] if model["kind"] == "hist_gradient_boosting"]
        for left in linear:
            for right in boosting:
                for blend in preset["ensemble_weights"]:
                    for smooth in preset["refinement_strengths"]:
                        rows.append(Candidate(
                            feature_set["id"], left["id"], left["kind"], left,
                            ensemble_model_id=right["id"],
                            ensemble_weight=float(blend),
                            refinement_strength=float(smooth)))
    return rows


def _fit_candidate(
    candidate: Candidate,
    model_lookup: Mapping[str, Mapping[str, Any]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_classes = np.array(sorted(pd.unique(y_train.astype(str))))
    classes, probability = fit_model(candidate.model_settings, x_train, y_train, x_eval, seed)
    probability = align_probabilities(probability, classes, target_classes)
    if candidate.ensemble_model_id is not None:
        settings = model_lookup[candidate.ensemble_model_id]
        other_classes, other = fit_model(settings, x_train, y_train, x_eval, seed + 101)
        other = align_probabilities(other, other_classes, target_classes)
        weight = float(candidate.ensemble_weight)
        probability = weight * probability + (1.0 - weight) * other
    return target_classes, probability


def combine_candidate_probabilities(
    candidate: Candidate,
    cache: Mapping[str, np.ndarray],
) -> np.ndarray:
    probability = cache[candidate.model_id]
    if candidate.ensemble_model_id is None:
        return probability.copy()
    weight = float(candidate.ensemble_weight)
    return (
        weight * probability
        + (1.0 - weight) * cache[candidate.ensemble_model_id])


def run_method(
    method: str,
    prediction: Path,
    metadata: Path,
    preset: Mapping[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    work, fine, fine_names, coarse, coarse_names, coords = v2.load_data(prediction, metadata)
    labels = work["region"].astype(str).to_numpy()
    feature_specs = {item["id"]: item for item in preset["feature_sets"]}
    features = {
        name: make_multiscale_features(
            fine, fine_names, coarse, coarse_names, coords, spec["scales"])
        for name, spec in feature_specs.items()
    }
    model_lookup = {item["id"]: item for item in preset["models"]}
    all_candidates = candidates(preset)
    outer_rows: list[dict[str, Any]] = []
    for split in preset["outer_splits"]:
        outer_train, outer_test = legacy.split_indices(
            work, split, preset["outer_test_size"], preset["seed"])
        inner_work = work.iloc[outer_train].reset_index(drop=True)
        inner_labels = labels[outer_train]
        folds = spatial_block_folds(
            inner_work, inner_labels, split, preset["inner_folds"],
            preset["inner_validation_fraction"], preset["seed"] + 17)
        candidate_rows: list[dict[str, Any]] = []
        fold_rows: list[dict[str, Any]] = []
        inner_fine = fine[outer_train]
        inner_coords = coords[outer_train]
        inner_classes = np.array(sorted(pd.unique(inner_labels)))
        for feature_set, spec in feature_specs.items():
            matrix = features[feature_set].to_numpy(dtype=float)[outer_train]
            feature_candidates = [
                candidate for candidate in all_candidates
                if candidate.feature_set == feature_set]
            for fold_index, (train_index, valid_index) in enumerate(folds):
                probability_cache: dict[str, np.ndarray] = {}
                for model_id, settings in model_lookup.items():
                    classes, probability = fit_model(
                        settings, matrix[train_index], inner_labels[train_index],
                        matrix, preset["seed"] + fold_index)
                    probability_cache[model_id] = align_probabilities(
                        probability, classes, inner_classes)
                for candidate in feature_candidates:
                    probability = combine_candidate_probabilities(
                        candidate, probability_cache)
                    probability = v2.refine_probabilities(
                        probability, inner_fine, inner_coords,
                        min(spec["scales"]), candidate.refinement_strength,
                        iterations=2)
                    values = v2.metrics(
                        inner_labels[valid_index], probability[valid_index],
                        inner_classes)
                    fold_rows.append({
                        "method": method, "outer_split": split,
                        "candidate_id": candidate.candidate_id, "fold": fold_index,
                        "balanced_accuracy": values["balanced_accuracy"],
                        "macro_f1": values["macro_f1"],
                        "macro_auroc": values["macro_auroc"],
                    })
        fold_table = pd.DataFrame(fold_rows)
        for candidate in all_candidates:
            fold_frame = fold_table[
                fold_table["candidate_id"] == candidate.candidate_id]
            candidate_rows.append({
                "method": method, "outer_split": split,
                "candidate_id": candidate.candidate_id,
                "feature_set": candidate.feature_set,
                "model_id": candidate.model_id,
                "ensemble_model_id": candidate.ensemble_model_id,
                "ensemble_weight": candidate.ensemble_weight,
                "refinement_strength": candidate.refinement_strength,
                "inner_mean_balanced_accuracy": fold_frame["balanced_accuracy"].mean(),
                "inner_mean_macro_f1": fold_frame["macro_f1"].mean(),
                "inner_mean_macro_auroc": fold_frame["macro_auroc"].mean(),
                "inner_selection_score": selection_score(fold_frame, preset["objective"]),
            })
        search = pd.DataFrame(candidate_rows).sort_values("inner_selection_score", ascending=False)
        selected_id = str(search.iloc[0]["candidate_id"])
        selected = next(item for item in all_candidates if item.candidate_id == selected_id)
        matrix = features[selected.feature_set].to_numpy(dtype=float)
        classes, raw_probability = _fit_candidate(
            selected, model_lookup, matrix[outer_train], labels[outer_train],
            matrix, preset["seed"])
        probability = v2.refine_probabilities(
            raw_probability, fine, coords,
            min(feature_specs[selected.feature_set]["scales"]),
            selected.refinement_strength, iterations=2)
        raw_values = v2.metrics(
            labels[outer_test], raw_probability[outer_test], classes)
        values = v2.metrics(labels[outer_test], probability[outer_test], classes)
        split_dir = output_dir / method / split
        split_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(fold_rows).to_csv(split_dir / "inner_fold_metrics.csv", index=False)
        search.to_csv(split_dir / "inner_candidate_summary.csv", index=False)
        outer_payload = {
            "method": method, "outer_split": split,
            "selected_candidate_id": selected_id,
            "selected_inner_score": float(search.iloc[0]["inner_selection_score"]),
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
            json.dumps(outer_payload, indent=2) + "\n", encoding="utf-8")
        raw_all = np.asarray(classes)[np.argmax(raw_probability, axis=1)]
        selected_all = np.asarray(classes)[np.argmax(probability, axis=1)]
        prediction_frame = work[
            ["ID", "spot_id", "region", "scaled_x", "scaled_y"]].copy()
        prediction_frame["outer_split"] = "train"
        prediction_frame.loc[
            prediction_frame.index[outer_test], "outer_split"] = "test"
        prediction_frame["raw_region"] = raw_all
        prediction_frame["selected_region"] = selected_all
        for class_index, name in enumerate(classes):
            prediction_frame[f"raw_prob::{name}"] = raw_probability[:, class_index]
            prediction_frame[f"selected_prob::{name}"] = probability[:, class_index]
        prediction_frame.to_csv(
            split_dir / "outer_test_predictions.csv", index=False)
        raw_values["confusion"].to_csv(
            split_dir / "outer_test_raw_confusion.csv")
        values["confusion"].to_csv(
            split_dir / "outer_test_selected_confusion.csv")
        v2.save_maps(
            work, classes, raw_all, selected_all, probability,
            split_dir, f"{method} Track-B v3 {split}")
        outer_rows.append(outer_payload)
    frame = pd.DataFrame(outer_rows)
    frame.to_csv(output_dir / method / "outer_test_metrics.csv", index=False)
    return frame
