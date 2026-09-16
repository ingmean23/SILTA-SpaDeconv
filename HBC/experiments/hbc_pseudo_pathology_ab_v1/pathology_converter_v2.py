#!/usr/bin/env python3
"""Leakage-safe HBC pathology converter with compositional spatial features."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import evaluate_hbc_annotation as legacy


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    values = np.clip(values, 0.0, None)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def knn_indices(coords: np.ndarray, neighbors: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    count = len(coords)
    k = min(max(1, int(neighbors)), max(1, count - 1))
    model = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    return model.kneighbors(return_distance=False)[:, 1:]


def neighbor_mean(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return values[indices].mean(axis=1)


def make_features(
    fine: np.ndarray,
    fine_names: Sequence[str],
    coarse: np.ndarray,
    coarse_names: Sequence[str],
    coords: np.ndarray,
    neighbors: int,
) -> pd.DataFrame:
    fine = normalize(fine)
    coarse = normalize(coarse)
    eps = 1e-5
    fine_log = np.log(fine + eps)
    coarse_log = np.log(coarse + eps)
    fine_clr = fine_log - fine_log.mean(axis=1, keepdims=True)
    coarse_clr = coarse_log - coarse_log.mean(axis=1, keepdims=True)
    graph = knn_indices(coords, neighbors)
    n1_fine = neighbor_mean(fine, graph)
    n1_coarse = neighbor_mean(coarse, graph)
    graph2_values = neighbor_mean(n1_coarse, graph)

    blocks: list[tuple[str, np.ndarray, Sequence[str]]] = [
        ("fine", fine, fine_names),
        ("coarse", coarse, coarse_names),
        ("fine_clr", fine_clr, fine_names),
        ("coarse_clr", coarse_clr, coarse_names),
        ("n1_fine", n1_fine, fine_names),
        ("n1_coarse", n1_coarse, coarse_names),
        ("n2_coarse", graph2_values, coarse_names),
        ("delta_fine", fine - n1_fine, fine_names),
        ("delta_coarse", coarse - n1_coarse, coarse_names),
    ]
    data: dict[str, np.ndarray] = {}
    for prefix, block, names in blocks:
        for index, name in enumerate(names):
            data[f"{prefix}::{name}"] = block[:, index]

    group_index = {name: index for index, name in enumerate(coarse_names)}
    pairs = [
        ("Epithelial/Tumor", "Lymphoid"),
        ("Epithelial/Tumor", "Myeloid/DC"),
        ("Epithelial/Tumor", "Stromal/CAF"),
        ("Epithelial/Tumor", "Endothelial"),
        ("Stromal/CAF", "Endothelial"),
        ("Lymphoid", "Myeloid/DC"),
    ]
    for left, right in pairs:
        if left in group_index and right in group_index:
            data[f"logratio::{left}/{right}"] = (
                coarse_log[:, group_index[left]]
                - coarse_log[:, group_index[right]])

    sorted_values = np.sort(fine, axis=1)
    data["summary::entropy"] = (
        -(fine * np.log(fine + 1e-12)).sum(axis=1) / math.log(fine.shape[1]))
    data["summary::max"] = sorted_values[:, -1]
    data["summary::margin"] = sorted_values[:, -1] - sorted_values[:, -2]
    return pd.DataFrame(data)


def spatial_inner_split(
    frame: pd.DataFrame,
    labels: np.ndarray,
    split: str,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train, valid = legacy.split_indices(frame, split, fraction, seed)
    if len(np.unique(labels[train])) == len(np.unique(labels)):
        return train, valid
    fallback = frame.copy()
    fallback["region"] = labels
    return legacy.split_indices(fallback, "stratified", fraction, seed)


def fit_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    c_value: float,
    l1_ratio: float,
    seed: int,
):
    from sklearn.linear_model import LogisticRegression

    mean = np.nanmean(x_train, axis=0, keepdims=True)
    std = np.nanstd(x_train, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    train = np.nan_to_num((x_train - mean) / std)
    evaluate = np.nan_to_num((x_eval - mean) / std)
    classifier = LogisticRegression(
        C=float(c_value),
        penalty="elasticnet",
        solver="saga",
        l1_ratio=float(l1_ratio),
        class_weight="balanced",
        max_iter=800,
        tol=1e-3,
        n_jobs=-1,
        random_state=int(seed),
    )
    classifier.fit(train, y_train)
    return classifier.classes_.astype(str), classifier.predict_proba(evaluate)


def refine_probabilities(
    probability: np.ndarray,
    composition: np.ndarray,
    coords: np.ndarray,
    neighbors: int,
    strength: float,
    iterations: int = 3,
) -> np.ndarray:
    if strength <= 0.0:
        return probability.copy()
    graph = knn_indices(coords, neighbors)
    difference = composition[:, None, :] - composition[graph]
    distance2 = np.square(difference).sum(axis=2)
    positive = distance2[distance2 > 0]
    sigma2 = float(np.median(positive)) if len(positive) else 1.0
    weights = np.exp(-distance2 / max(sigma2, 1e-8))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    base = probability.copy()
    current = probability.copy()
    for _ in range(int(iterations)):
        neighbor = (current[graph] * weights[:, :, None]).sum(axis=1)
        current = (base + float(strength) * neighbor) / (1.0 + float(strength))
    return current / np.maximum(current.sum(axis=1, keepdims=True), 1e-12)


def metrics(
    truth: np.ndarray,
    probability: np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    prediction = np.asarray(classes)[np.argmax(probability, axis=1)]
    order = [name for name in legacy.REGION_ORDER if name in set(truth)]
    order += [name for name in sorted(set(truth)) if name not in order]
    values = legacy.classification_metrics(truth, prediction, order)
    _, auroc = legacy.region_auroc(truth, probability, classes)
    return {
        "accuracy": float(values["accuracy"]),
        "balanced_accuracy": float(values["balanced_accuracy"]),
        "macro_f1": float(values["macro_f1"]),
        "macro_auroc": None if auroc is None else float(auroc),
        "prediction": prediction,
        "confusion": values["confusion"],
    }


def objective(values: Mapping[str, Any]) -> float:
    terms = [values["balanced_accuracy"], values["macro_f1"]]
    if values.get("macro_auroc") is not None:
        terms.append(values["macro_auroc"])
    return float(np.mean(terms))


def save_maps(
    work: pd.DataFrame,
    classes: Sequence[str],
    raw_prediction: np.ndarray,
    refined_prediction: np.ndarray,
    refined_probability: np.ndarray,
    output_dir: Path,
    title: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = work["scaled_x"].to_numpy(dtype=float)
    y = work["scaled_y"].to_numpy(dtype=float)
    truth = work["region"].astype(str).to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    panels = ((truth, "GT"), (raw_prediction, "Raw converter"),
              (refined_prediction, "Graph-refined converter"))
    for axis, (labels, name) in zip(axes, panels):
        for label in legacy.REGION_ORDER:
            mask = labels == label
            if mask.any():
                axis.scatter(
                    x[mask], y[mask], s=9,
                    c=legacy.REGION_COLORS.get(label, "#888888"),
                    label=label, linewidths=0)
        axis.set_title(name)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_xticks([])
        axis.set_yticks([])
    axes[-1].legend(frameon=False, fontsize=8, markerscale=2)
    fig.suptitle(title)
    fig.savefig(output_dir / "hbc_gt_raw_refined_pathology_map.png", dpi=300)
    plt.close(fig)

    rows = int(math.ceil(len(classes) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4.8 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for index, name in enumerate(classes):
        scatter = axes[index].scatter(
            x, y, c=refined_probability[:, index], s=9,
            cmap="viridis", vmin=0.0, vmax=1.0, linewidths=0)
        axes[index].set_title(str(name))
        axes[index].set_aspect("equal")
        axes[index].invert_yaxis()
        axes[index].set_xticks([])
        axes[index].set_yticks([])
        fig.colorbar(scatter, ax=axes[index], fraction=0.035, pad=0.01)
    for axis in axes[len(classes):]:
        axis.axis("off")
    fig.savefig(output_dir / "hbc_refined_pathology_probabilities.png", dpi=300)
    plt.close(fig)


def load_data(prediction: Path, metadata: Path):
    pred = pd.read_csv(prediction)
    meta = pd.read_csv(metadata, sep="\t")
    if "spot_id" not in pred:
        raise ValueError("Prediction must contain spot_id")
    coarse, fine_names, _, _ = legacy.build_coarse_predictions(
        pred, legacy.DEFAULT_GROUPS, "spot_id")
    coarse.insert(0, "spot_id", pred["spot_id"].astype(str).values)
    work = meta.merge(coarse, left_on="ID", right_on="spot_id", how="inner")
    fine_by_id = pred.set_index("spot_id")
    for name in fine_names:
        work[name] = work["spot_id"].map(fine_by_id[name])
    work = work.rename(columns={"annot_type": "region"})
    coarse_names = [name for name in coarse.columns if name != "spot_id"]
    fine = normalize(work[fine_names].to_numpy(dtype=float))
    coarse_values = normalize(work[coarse_names].to_numpy(dtype=float))
    coords = work[["scaled_x", "scaled_y"]].to_numpy(dtype=float)
    return work, fine, fine_names, coarse_values, coarse_names, coords


def run_split(
    prediction: Path,
    metadata: Path,
    output_dir: Path,
    split: str,
    settings: Mapping[str, Any],
    title: str,
) -> dict[str, Any]:
    work, fine, fine_names, coarse, coarse_names, coords = load_data(
        prediction, metadata)
    labels = work["region"].astype(str).to_numpy()
    feature_frame = make_features(
        fine, fine_names, coarse, coarse_names, coords,
        int(settings["neighbors"]))
    outer_train, outer_test = legacy.split_indices(
        work, split, float(settings["test_size"]), int(settings["seed"]))
    inner_frame = work.iloc[outer_train].reset_index(drop=True)
    inner_labels = labels[outer_train]
    inner_train, inner_valid = spatial_inner_split(
        inner_frame, inner_labels, split, float(settings["inner_fraction"]),
        int(settings["seed"]) + 17)

    search_rows = []
    best = None
    for c_value in settings["c_values"]:
        for l1_ratio in settings["l1_ratios"]:
            classes, probability = fit_classifier(
                feature_frame.to_numpy()[outer_train][inner_train],
                inner_labels[inner_train],
                feature_frame.to_numpy()[outer_train],
                float(c_value), float(l1_ratio), int(settings["seed"]))
            for smooth in settings["smooth_lambdas"]:
                refined = refine_probabilities(
                    probability, fine[outer_train], coords[outer_train],
                    int(settings["neighbors"]), float(smooth))
                result = metrics(
                    inner_labels[inner_valid], refined[inner_valid], classes)
                row = {
                    "c_value": float(c_value),
                    "l1_ratio": float(l1_ratio),
                    "smooth_lambda": float(smooth),
                    "balanced_accuracy": result["balanced_accuracy"],
                    "macro_f1": result["macro_f1"],
                    "macro_auroc": result["macro_auroc"],
                    "objective": objective(result),
                }
                search_rows.append(row)
                if best is None or row["objective"] > best["objective"]:
                    best = row
    if best is None:
        raise RuntimeError("No valid pathology converter candidate")

    classes, raw_probability = fit_classifier(
        feature_frame.to_numpy()[outer_train], labels[outer_train],
        feature_frame.to_numpy(), best["c_value"], best["l1_ratio"],
        int(settings["seed"]))
    refined_probability = refine_probabilities(
        raw_probability, fine, coords, int(settings["neighbors"]),
        best["smooth_lambda"])
    raw = metrics(labels[outer_test], raw_probability[outer_test], classes)
    refined = metrics(
        labels[outer_test], refined_probability[outer_test], classes)
    raw_all = np.asarray(classes)[np.argmax(raw_probability, axis=1)]
    refined_all = np.asarray(classes)[np.argmax(refined_probability, axis=1)]

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search_rows).sort_values(
        "objective", ascending=False).to_csv(
            output_dir / "inner_spatial_search.csv", index=False)
    feature_frame.insert(0, "spot_id", work["spot_id"].values)
    feature_frame.to_csv(output_dir / "features.csv", index=False)
    prediction_frame = work[["ID", "spot_id", "region", "scaled_x", "scaled_y"]].copy()
    prediction_frame["outer_split"] = "train"
    prediction_frame.loc[prediction_frame.index[outer_test], "outer_split"] = "test"
    prediction_frame["raw_region"] = raw_all
    prediction_frame["refined_region"] = refined_all
    for index, name in enumerate(classes):
        prediction_frame[f"raw_prob::{name}"] = raw_probability[:, index]
        prediction_frame[f"refined_prob::{name}"] = refined_probability[:, index]
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)
    raw["confusion"].to_csv(output_dir / "raw_confusion.csv")
    refined["confusion"].to_csv(output_dir / "refined_confusion.csv")
    save_maps(
        work, classes, raw_all, refined_all, refined_probability,
        output_dir, title)

    payload = {
        "prediction": str(prediction),
        "split": split,
        "n_train": int(len(outer_train)),
        "n_test": int(len(outer_test)),
        "chosen": best,
        "raw": {key: value for key, value in raw.items()
                if key not in {"prediction", "confusion"}},
        "refined": {key: value for key, value in refined.items()
                    if key not in {"prediction", "confusion"}},
        "outer_test_labels": {
            str(key): int(value) for key, value in
            pd.Series(labels[outer_test]).value_counts().items()},
        "outer_test_labels_used_for_selection": False,
        "pathology_labels_used_to_train_dacg": False,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run(args: argparse.Namespace) -> None:
    preset = json.loads(Path(args.preset).read_text(encoding="utf-8-sig"))
    settings = dict(preset["pathology"])
    prediction = Path(args.prediction)
    metadata = Path(args.metadata or settings["metadata"])
    output = Path(args.output_dir)
    splits = list(settings["splits"]) if args.split == "all" else [args.split]
    rows = []
    for split in splits:
        result = run_split(
            prediction, metadata, output / split, split, settings,
            args.title or prediction.stem)
        for variant in ("raw", "refined"):
            rows.append({
                "split": split, "variant": variant,
                **result[variant], **result["chosen"],
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_split_metrics.csv", index=False)
    means = frame.groupby("variant", as_index=False)[
        ["balanced_accuracy", "macro_f1", "macro_auroc"]].mean()
    means.to_csv(output / "mean_metrics.csv", index=False)
    print(means.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--split", choices=["spatial_x", "spatial_y", "all"],
                        default="all")
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--preset", default=str(Path(__file__).with_name("preset.json")))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
