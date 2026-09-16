"""Shared data, split, feature, metric, and plotting helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.hbc_pseudo_pathology_ab_v1 import pathology_converter_v2 as v2
from experiments.hbc_pathology_trackb_v3.converter import make_multiscale_features
from tools import evaluate_hbc_annotation as legacy


LABEL_COLUMN = "old_annot_type"
FINE_LABEL_COLUMN = "old_fine_annot_type"
REGIONS = tuple(legacy.REGION_ORDER)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(prediction: Path, metadata: Path):
    """Load fractions and use the untouched public SEDR annotation columns."""
    pred = pd.read_csv(prediction)
    meta = pd.read_csv(metadata, sep="\t")
    required_meta = {"ID", LABEL_COLUMN, FINE_LABEL_COLUMN, "scaled_x", "scaled_y"}
    required_pred = {"spot_id"}
    if missing := required_meta.difference(meta.columns):
        raise ValueError(f"metadata is missing corrected-label columns: {sorted(missing)}")
    if missing := required_pred.difference(pred.columns):
        raise ValueError(f"prediction is missing columns: {sorted(missing)}")
    pred_columns = [
        name for name in pred.columns
        if name != "spot_id" and pd.api.types.is_numeric_dtype(pred[name])]
    if len(pred_columns) != 13:
        raise ValueError(f"expected 13 HBC fraction columns, found {len(pred_columns)}")

    fractions = v2.normalize(pred[pred_columns].to_numpy(dtype=float))
    fraction_frame = pd.DataFrame(fractions, columns=pred_columns)
    fraction_frame.insert(0, "spot_id", pred["spot_id"].astype(str).to_numpy())
    work = meta.merge(fraction_frame, left_on="ID", right_on="spot_id", how="inner")
    if len(work) != len(meta):
        raise ValueError(
            f"prediction covers {len(work)}/{len(meta)} metadata spots; full coverage required")
    work["region"] = work[LABEL_COLUMN].astype(str)
    unknown = set(work["region"]).difference(REGIONS)
    if unknown:
        raise ValueError(f"unexpected corrected pathology labels: {sorted(unknown)}")

    coarse, _, _, _ = legacy.build_coarse_predictions(
        work[["spot_id", *pred_columns]], legacy.DEFAULT_GROUPS, "spot_id")
    coarse_names = list(coarse.columns)
    fine = v2.normalize(work[pred_columns].to_numpy(dtype=float))
    coarse_values = v2.normalize(coarse.to_numpy(dtype=float))
    coords = work[["scaled_x", "scaled_y"]].to_numpy(dtype=float)
    return work, fine, pred_columns, coarse_values, coarse_names, coords


def direct_features(fine: np.ndarray, fine_names: Sequence[str]) -> pd.DataFrame:
    """Raw-fraction features only: no coarse groups, neighbors, or coordinates."""
    fine = v2.normalize(fine)
    ordered = np.sort(fine, axis=1)
    data = {f"fraction::{name}": fine[:, index] for index, name in enumerate(fine_names)}
    data["summary::entropy"] = (
        -(fine * np.log(fine + 1e-12)).sum(1) / math.log(fine.shape[1]))
    data["summary::max"] = ordered[:, -1]
    data["summary::margin"] = ordered[:, -1] - ordered[:, -2]
    return pd.DataFrame(data)


def stratified_random_splits(
    labels: Sequence[str], seeds: Iterable[int], test_size: float
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    from sklearn.model_selection import train_test_split

    labels = np.asarray(labels).astype(str)
    indices = np.arange(len(labels))
    result = []
    for seed in seeds:
        train, test = train_test_split(
            indices, test_size=float(test_size), random_state=int(seed),
            stratify=labels)
        _require_label_coverage(labels, train, test, minimum_test_count=1)
        result.append((f"seed_{int(seed)}", np.sort(train), np.sort(test)))
    return result


def spatial_block_splits(
    frame: pd.DataFrame,
    labels: Sequence[str],
    n_splits: int,
    bins_per_axis: int,
    minimum_test_count: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Keep local coordinate bins intact while balancing pathology labels across folds."""
    from sklearn.model_selection import StratifiedGroupKFold

    labels = np.asarray(labels).astype(str)
    x_bin = pd.qcut(
        frame["scaled_x"], q=int(bins_per_axis), labels=False, duplicates="drop")
    y_bin = pd.qcut(
        frame["scaled_y"], q=int(bins_per_axis), labels=False, duplicates="drop")
    groups = (x_bin.astype(str) + "::" + y_bin.astype(str)).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=int(n_splits), shuffle=False)
    result = []
    for fold, (train, test) in enumerate(splitter.split(frame, labels, groups)):
        _require_label_coverage(labels, train, test, minimum_test_count)
        result.append((f"fold_{fold}", np.sort(train), np.sort(test)))
    return result


def _require_label_coverage(
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    minimum_test_count: int,
) -> None:
    expected = set(REGIONS)
    if set(labels[train]) != expected or set(labels[test]) != expected:
        raise ValueError("every pathology label must occur in train and test")
    counts = pd.Series(labels[test]).value_counts()
    if int(counts.reindex(REGIONS, fill_value=0).min()) < int(minimum_test_count):
        raise ValueError(
            f"test fold violates minimum class count {minimum_test_count}: "
            f"{counts.to_dict()}")


def class_counts(labels: Sequence[str], indices: np.ndarray) -> dict[str, int]:
    counts = pd.Series(np.asarray(labels)[indices]).value_counts()
    return {name: int(counts.get(name, 0)) for name in REGIONS}


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression

    x_train = features[train]
    mean = np.nanmean(x_train, axis=0, keepdims=True)
    std = np.nanstd(x_train, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    train_values = np.nan_to_num((x_train - mean) / std)
    all_values = np.nan_to_num((features - mean) / std)
    model = LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", class_weight="balanced",
        max_iter=2000, n_jobs=1, random_state=int(seed))
    model.fit(train_values, labels[train])
    return model.classes_.astype(str), model.predict_proba(all_values)


def metric_row(
    track: str,
    method: str,
    split_id: str,
    labels: np.ndarray,
    test: np.ndarray,
    classes: np.ndarray,
    probability: np.ndarray,
) -> tuple[dict, dict]:
    values = v2.metrics(labels[test], probability[test], classes)
    row = {
        "track": track,
        "method": method,
        "split_id": split_id,
        "balanced_accuracy": values["balanced_accuracy"],
        "macro_f1": values["macro_f1"],
        "macro_auroc": values["macro_auroc"],
        "n_test": int(len(test)),
    }
    for name, count in class_counts(labels, test).items():
        row[f"test_count::{name}"] = count
    return row, values


def save_predictions_and_maps(
    output: Path,
    title: str,
    work: pd.DataFrame,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    classes: np.ndarray,
    probability: np.ndarray,
    values: dict,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    predicted = classes[np.argmax(probability, axis=1)]
    frame = work[["ID", "spot_id", LABEL_COLUMN, FINE_LABEL_COLUMN,
                  "scaled_x", "scaled_y"]].copy()
    frame["region"] = labels
    frame["split"] = "train"
    frame.loc[frame.index[test], "split"] = "test"
    frame["predicted_region"] = predicted
    for index, name in enumerate(classes):
        frame[f"prob::{name}"] = probability[:, index]
    frame.to_csv(output / "predictions.csv", index=False)
    values["confusion"].to_csv(output / "confusion.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for axis, column, panel_title in (
        (axes[0], "region", "Public pathology annotation"),
        (axes[1], "predicted_region", "Predicted pathology region"),
    ):
        for name in REGIONS:
            mask = frame[column].eq(name).to_numpy()
            axis.scatter(
                frame.loc[mask, "scaled_x"], frame.loc[mask, "scaled_y"],
                s=8, linewidths=0, c=legacy.REGION_COLORS[name], label=name)
        axis.set_title(panel_title)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_xticks([])
        axis.set_yticks([])
    axes[1].legend(frameon=False, fontsize=8, markerscale=2)
    fig.suptitle(title)
    fig.savefig(output / "pathology_map.png", dpi=300)
    plt.close(fig)

    audit = {
        "label_column": LABEL_COLUMN,
        "fine_label_column": FINE_LABEL_COLUMN,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_counts": class_counts(labels, train),
        "test_counts": class_counts(labels, test),
    }
    (output / "split_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def trackb_features(
    fine: np.ndarray,
    fine_names: Sequence[str],
    coarse: np.ndarray,
    coarse_names: Sequence[str],
    coords: np.ndarray,
    scales: Sequence[int],
) -> pd.DataFrame:
    return make_multiscale_features(
        fine, fine_names, coarse, coarse_names, coords, scales)
