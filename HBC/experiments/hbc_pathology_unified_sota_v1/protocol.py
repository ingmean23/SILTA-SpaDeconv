"""Leakage-controlled HBC pathology tracks shared by DACG and baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from experiments.hbc_pathology_corrected_v1 import protocol as corrected
from experiments.hbc_pathology_trackb_v3 import converter
from experiments.hbc_pseudo_pathology_ab_v1 import pathology_converter_v2 as v2
from experiments.hbc_true_lowlr_pathology_v1 import pathology as robust


METRICS = ("balanced_accuracy", "macro_f1", "macro_auroc")


@dataclass
class Loaded:
    candidate_id: str
    prediction: Path
    work: pd.DataFrame
    labels: np.ndarray
    fine: np.ndarray
    fine_names: list[str]
    coarse: np.ndarray
    coarse_names: list[str]
    coords: np.ndarray


def load(candidate_id: str, prediction: Path, metadata: Path) -> Loaded:
    work, fine, fine_names, coarse, coarse_names, coords = corrected.load_data(
        prediction, metadata)
    return Loaded(
        candidate_id, prediction, work, work["region"].astype(str).to_numpy(),
        fine, list(fine_names), coarse, list(coarse_names), coords)


def _score(values: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return float(sum(float(weights[key]) * float(values[key]) for key in METRICS))


def _mean(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in METRICS}


def _inner_random(
    labels: np.ndarray, outer_train: np.ndarray, n_splits: int, seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(
        n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    local = np.arange(len(outer_train))
    return [
        (outer_train[train], outer_train[valid])
        for train, valid in splitter.split(local, labels[outer_train])]


def _random_cv_score(
    item: Loaded, outer_train: np.ndarray, settings: Mapping[str, Any], seed: int,
) -> float:
    features = corrected.direct_features(item.fine, item.fine_names).to_numpy(float)
    rows = []
    for train, valid in _inner_random(
            item.labels, outer_train, int(settings["inner_splits"]), seed):
        classes, probability = corrected.fit_logistic(
            features, item.labels, train, seed)
        rows.append(v2.metrics(item.labels[valid], probability[valid], classes))
    return _score(_mean(rows), settings["selection_weights"])


def _evaluate_random_outer(
    item: Loaded, split_id: str, train: np.ndarray, test: np.ndarray,
    output: Path, seed: int,
) -> dict[str, Any]:
    features = corrected.direct_features(item.fine, item.fine_names).to_numpy(float)
    classes, probability = corrected.fit_logistic(features, item.labels, train, seed)
    row, values = corrected.metric_row(
        "random_direct", item.candidate_id, split_id, item.labels, test,
        classes, probability)
    corrected.save_predictions_and_maps(
        output, f"{item.candidate_id}: Random-Direct {split_id}", item.work,
        item.labels, train, test, classes, probability, values)
    return row


def _feature_tables(
    item: Loaded, settings: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    return {
        str(spec["id"]): robust.make_features(
            item.fine, item.fine_names, item.coarse, item.coarse_names,
            item.coords, spec["scales"]).to_numpy(float)
        for spec in settings["feature_sets"]
    }


def _spatial_inner_splits(
    item: Loaded, outer_train: np.ndarray, settings: Mapping[str, Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    local_frame = item.work.iloc[outer_train].reset_index(drop=True)
    local_labels = item.labels[outer_train]
    local = corrected.spatial_block_splits(
        local_frame, local_labels, int(settings["inner_splits"]),
        int(settings["bins_per_axis"]), 1)
    return [(outer_train[train], outer_train[valid]) for _, train, valid in local]


def _converter_cv_score(
    item: Loaded, matrix: np.ndarray, model: Mapping[str, Any],
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    weights: Mapping[str, float], seed: int,
) -> float:
    rows = []
    for fold, (train, valid) in enumerate(inner_splits):
        classes, probability = converter.fit_model(
            model, matrix[train], item.labels[train], matrix, seed + fold)
        rows.append(v2.metrics(item.labels[valid], probability[valid], classes))
    fold_scores = np.array([_score(row, weights) for row in rows], dtype=float)
    return float(_score(_mean(rows), weights) - 0.1 * fold_scores.std(ddof=0))


def _evaluate_spatial_outer(
    item: Loaded, split_id: str, train: np.ndarray, test: np.ndarray,
    feature_id: str, model: Mapping[str, Any], matrix: np.ndarray,
    inner_score: float, output: Path, seed: int,
) -> dict[str, Any]:
    classes, probability = converter.fit_model(
        model, matrix[train], item.labels[train], matrix, seed)
    row, values = corrected.metric_row(
        "spatial_trackb", item.candidate_id, split_id, item.labels, test,
        classes, probability)
    row.update({
        "selected_feature_set": feature_id,
        "selected_model": str(model["id"]),
        "inner_selection_score": float(inner_score),
        "absolute_coordinates_used_as_features": False,
        "probability_refinement_used": False,
    })
    corrected.save_predictions_and_maps(
        output, f"{item.candidate_id}: Spatial-TrackB {split_id}", item.work,
        item.labels, train, test, classes, probability, values)
    return row


def evaluate_single(
    method: str, prediction: Path, metadata: Path,
    pathology_preset: Mapping[str, Any], output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one locked method; only converter settings use inner folds."""
    item = load(method, prediction, metadata)
    random_cfg = dict(pathology_preset["random_direct"])
    random_cfg["selection_weights"] = pathology_preset["selection_weights"]
    random_rows = []
    for split_id, train, test in corrected.stratified_random_splits(
            item.labels, random_cfg["seeds"], random_cfg["test_size"]):
        seed = int(split_id.rsplit("_", 1)[-1])
        random_rows.append(_evaluate_random_outer(
            item, split_id, train, test,
            output / "random_direct" / split_id, seed))

    spatial_cfg = pathology_preset["spatial_trackb"]
    tables = _feature_tables(item, spatial_cfg)
    feature_specs = {str(row["id"]): row for row in spatial_cfg["feature_sets"]}
    spatial_rows = []
    splits = corrected.spatial_block_splits(
        item.work, item.labels, int(spatial_cfg["outer_splits"]),
        int(spatial_cfg["bins_per_axis"]),
        int(spatial_cfg["minimum_test_count"]))
    for fold, (split_id, train, test) in enumerate(splits):
        inner = _spatial_inner_splits(item, train, spatial_cfg)
        choices = []
        for feature_id, matrix in tables.items():
            for model in spatial_cfg["models"]:
                choices.append((
                    _converter_cv_score(
                        item, matrix, model, inner,
                        pathology_preset["selection_weights"], 89 + fold),
                    feature_id, model, matrix))
        score, feature_id, model, matrix = max(choices, key=lambda row: row[0])
        spatial_rows.append(_evaluate_spatial_outer(
            item, split_id, train, test, feature_id, model, matrix, score,
            output / "spatial_trackb" / split_id, 89 + fold))
        (output / "spatial_trackb" / split_id / "selected.json").write_text(
            json.dumps({
                "feature_set": feature_id,
                "feature_scales": feature_specs[feature_id]["scales"],
                "model": model["id"], "inner_score": score,
            }, indent=2) + "\n", encoding="utf-8")
    return pd.DataFrame(random_rows), pd.DataFrame(spatial_rows)


def evaluate_pool(
    predictions: Mapping[str, Path], health: Mapping[str, bool], metadata: Path,
    pathology_preset: Mapping[str, Any], output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nested selection over a fixed DACG pool; sealed outer labels never select."""
    items = {
        key: load(key, path, metadata) for key, path in predictions.items()
        if bool(health.get(key, False))
    }
    if not items:
        raise ValueError("no structurally healthy DACG candidate predictions")
    first = next(iter(items.values()))
    for item in items.values():
        if not np.array_equal(item.labels, first.labels):
            raise ValueError("candidate pathology label order mismatch")

    selection_rows = []
    random_rows = []
    random_cfg = dict(pathology_preset["random_direct"])
    random_cfg["selection_weights"] = pathology_preset["selection_weights"]
    for split_id, train, test in corrected.stratified_random_splits(
            first.labels, random_cfg["seeds"], random_cfg["test_size"]):
        seed = int(split_id.rsplit("_", 1)[-1])
        scored = [
            (_random_cv_score(item, train, random_cfg, seed + 1000), key)
            for key, item in items.items()]
        inner_score, selected = max(scored)
        selection_rows.extend({
            "track": "random_direct", "outer_split": split_id,
            "candidate_id": key, "inner_score": score,
            "selected": key == selected,
        } for score, key in scored)
        row = _evaluate_random_outer(
            items[selected], split_id, train, test,
            output / "random_direct" / split_id, seed)
        row["selected_candidate"] = selected
        row["inner_selection_score"] = inner_score
        random_rows.append(row)

    spatial_cfg = pathology_preset["spatial_trackb"]
    table_cache = {
        key: _feature_tables(item, spatial_cfg) for key, item in items.items()}
    screen_feature = str(spatial_cfg["screen_feature_set"]["id"])
    screen_model = next(
        model for model in spatial_cfg["models"] if model["id"] == "enet_c1")
    spatial_rows = []
    splits = corrected.spatial_block_splits(
        first.work, first.labels, int(spatial_cfg["outer_splits"]),
        int(spatial_cfg["bins_per_axis"]),
        int(spatial_cfg["minimum_test_count"]))
    for fold, (split_id, train, test) in enumerate(splits):
        inner = _spatial_inner_splits(first, train, spatial_cfg)
        screen = []
        for key, item in items.items():
            score = _converter_cv_score(
                item, table_cache[key][screen_feature], screen_model, inner,
                pathology_preset["selection_weights"], 1089 + fold)
            screen.append((score, key))
        finalists = [
            key for _, key in sorted(screen, reverse=True)[
                :int(spatial_cfg["screen_top_k"])]]
        choices = []
        for key in finalists:
            item = items[key]
            for feature_id, matrix in table_cache[key].items():
                for model in spatial_cfg["models"]:
                    choices.append((
                        _converter_cv_score(
                            item, matrix, model, inner,
                            pathology_preset["selection_weights"], 2089 + fold),
                        key, feature_id, model, matrix))
        inner_score, selected, feature_id, model, matrix = max(
            choices, key=lambda row: row[0])
        selection_rows.extend({
            "track": "spatial_trackb", "outer_split": split_id,
            "candidate_id": key, "inner_score": score,
            "selected": key == selected and fid == feature_id
                        and candidate_model["id"] == model["id"],
            "feature_set": fid, "model": candidate_model["id"],
        } for score, key, fid, candidate_model, _ in choices)
        row = _evaluate_spatial_outer(
            items[selected], split_id, train, test, feature_id, model, matrix,
            inner_score, output / "spatial_trackb" / split_id, 89 + fold)
        row["selected_candidate"] = selected
        spatial_rows.append(row)
    return (
        pd.DataFrame(random_rows), pd.DataFrame(spatial_rows),
        pd.DataFrame(selection_rows))


def summary(frame: pd.DataFrame) -> dict[str, float]:
    return {key: float(frame[key].mean()) for key in METRICS}

