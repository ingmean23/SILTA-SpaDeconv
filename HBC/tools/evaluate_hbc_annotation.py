import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_GROUPS = {
    "Epithelial/Tumor": [
        "luminal cell",
        "luminal progenitor",
        "myoepithelial cell",
    ],
    "Lymphoid": [
        "T cell",
        "B cell",
        "NK cell",
        "plasma cell",
    ],
    "Myeloid/DC": [
        "macrophage/DC/monocyte",
        "macrophage_DC_monocyte",
        "pDC",
    ],
    "Stromal/CAF": [
        "fibroblast",
        "muscle cell",
    ],
    "Endothelial": [
        "vascular endothelial cell",
        "lymphatic endothelial cell",
    ],
}

REGION_ORDER = ["Healthy", "Surrounding tumor", "Tumor", "Invasive"]
REGION_COLORS = {
    "Healthy": "#6F3FA0",
    "Surrounding tumor": "#B08B2D",
    "Tumor": "#E23B3B",
    "Invasive": "#2E7FB8",
}
GROUP_COLORS = {
    "Epithelial/Tumor": "#C44E52",
    "Lymphoid": "#4C72B0",
    "Myeloid/DC": "#DD8452",
    "Stromal/CAF": "#55A868",
    "Endothelial": "#8172B2",
    "Other": "#8C8C8C",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HBC prediction consistency with pathology annotation."
    )
    parser.add_argument(
        "--mode",
        choices=["composition", "region_converter", "both"],
        default="composition",
        help=(
            "composition: region-vs-cell-compartment diagnostics; "
            "region_converter: train/evaluate a pathology-region classifier "
            "from predicted fractions; both: run both modes."
        ),
    )
    parser.add_argument(
        "--metadata",
        default="Human_Breast_Cancer/Human_Breast_Cancer/metadata.tsv",
        help="HBC metadata.tsv with spot-level pathology annotation.",
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="Prediction CSV containing spot_id and cell type fractions.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <pred_dir>/hbc_region_consistency.",
    )
    parser.add_argument(
        "--region-col",
        default="annot_type",
        help="Coarse region column in metadata.tsv.",
    )
    parser.add_argument(
        "--fine-region-col",
        default="fine_annot_type",
        help="Fine region column in metadata.tsv.",
    )
    parser.add_argument(
        "--spot-id-col",
        default="spot_id",
        help="Spot barcode column in prediction CSV.",
    )
    parser.add_argument(
        "--metadata-id-col",
        default="ID",
        help="Spot barcode column in metadata.tsv.",
    )
    parser.add_argument(
        "--groups-json",
        default=None,
        help="Optional JSON mapping coarse group names to prediction columns.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title prefix used in figures.",
    )
    parser.add_argument(
        "--split",
        choices=["stratified", "spatial_x", "spatial_y", "all"],
        default="stratified",
        help=(
            "Validation split for region_converter. Use 'all' only for "
            "visualization, not for unbiased reporting."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Held-out fraction for region_converter when split is not all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for region_converter split/model.",
    )
    return parser.parse_args()


def load_groups(path):
    if not path:
        return DEFAULT_GROUPS
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dir(pred_path, output_dir):
    if output_dir:
        out = Path(output_dir)
    else:
        out = Path(pred_path).resolve().parent / "hbc_region_consistency"
    out.mkdir(parents=True, exist_ok=True)
    return out


def normalize_rows(values):
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    sums = values.sum(axis=1, keepdims=True)
    sums[sums <= 0] = 1.0
    return values / sums


def build_coarse_predictions(pred_df, groups, spot_col):
    pred_cols = [
        c for c in pred_df.columns
        if c != spot_col and pd.api.types.is_numeric_dtype(pred_df[c])
    ]
    used = set()
    coarse = pd.DataFrame(index=pred_df.index)
    missing = {}
    for group, columns in groups.items():
        present = [c for c in columns if c in pred_cols]
        missing[group] = []
        for col in columns:
            if col in pred_cols:
                continue
            alias = col.replace("_DC_", "/DC/")
            reverse_alias = col.replace("/DC/", "_DC_")
            if alias in pred_cols or reverse_alias in pred_cols:
                continue
            missing[group].append(col)
        used.update(present)
        if present:
            coarse[group] = pred_df[present].sum(axis=1)
        else:
            coarse[group] = 0.0
    leftovers = [c for c in pred_cols if c not in used]
    if leftovers:
        coarse["Other"] = pred_df[leftovers].sum(axis=1)
    coarse_values = normalize_rows(coarse.values)
    coarse = pd.DataFrame(coarse_values, columns=coarse.columns, index=pred_df.index)
    return coarse, pred_cols, missing, leftovers


def effect_size_eta2(values, labels):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    overall = np.nanmean(values)
    ss_total = np.nansum((values - overall) ** 2)
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for label in pd.unique(labels):
        mask = labels == label
        if mask.sum() == 0:
            continue
        group_mean = np.nanmean(values[mask])
        ss_between += mask.sum() * (group_mean - overall) ** 2
    return float(ss_between / ss_total)


def normalized_mutual_info(a, b):
    a = pd.Series(a).astype(str)
    b = pd.Series(b).astype(str)
    table = pd.crosstab(a, b).to_numpy(dtype=float)
    total = table.sum()
    if total <= 0:
        return 0.0
    pxy = table / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = pxy > 0
    mi = float(np.sum(pxy[mask] * np.log(pxy[mask] / expected[mask])))
    hx = -float(np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = -float(np.sum(py[py > 0] * np.log(py[py > 0])))
    denom = math.sqrt(hx * hy)
    if denom <= 0:
        return 0.0
    return mi / denom


def centroid_balanced_accuracy(features, labels):
    labels = np.asarray(labels).astype(str)
    features = np.asarray(features, dtype=float)
    unique = np.array(sorted(pd.unique(labels)))
    pred = []
    for i in range(len(labels)):
        distances = []
        for label in unique:
            mask = labels == label
            mask[i] = False
            if mask.sum() == 0:
                centroid = features[labels == label].mean(axis=0)
            else:
                centroid = features[mask].mean(axis=0)
            distances.append(np.linalg.norm(features[i] - centroid))
        pred.append(unique[int(np.argmin(distances))])
    pred = np.asarray(pred)
    recalls = []
    for label in unique:
        mask = labels == label
        if mask.sum() > 0:
            recalls.append(float((pred[mask] == label).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def save_region_maps(df, coarse_cols, out_dir, title):
    x = df["scaled_x"].to_numpy()
    y = df["scaled_y"].to_numpy()
    region = df["region"].astype(str)
    dom = df["dominant_coarse"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, labels, colors, name in [
        (axes[0], region, REGION_COLORS, "Pathology region"),
        (axes[1], dom, GROUP_COLORS, "Predicted dominant biological compartment"),
    ]:
        for label in sorted(pd.unique(labels)):
            mask = labels == label
            ax.scatter(
                x[mask], y[mask],
                s=9, c=colors.get(label, "#8C8C8C"),
                label=label, linewidths=0, alpha=0.9,
            )
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=2, fontsize=8, loc="best", frameon=False)
    if title:
        fig.suptitle(title)
    fig.savefig(out_dir / "hbc_region_vs_predicted_coarse_map.png", dpi=300)
    plt.close(fig)

    n = len(coarse_cols)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(12, 5.2 * nrows), constrained_layout=True
    )
    axes = np.asarray(axes).reshape(-1)
    for ax, col in zip(axes, coarse_cols):
        vals = df[col].to_numpy(dtype=float)
        sc = ax.scatter(x, y, s=9, c=vals, cmap="viridis", linewidths=0)
        ax.set_title(col)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(f"{title}: biological compartment fractions")
    fig.savefig(out_dir / "hbc_biological_compartment_heatmaps.png", dpi=300)
    fig.savefig(out_dir / "hbc_coarse_group_heatmaps.png", dpi=300)
    plt.close(fig)


def save_heatmap(matrix, out_path, title):
    fig, ax = plt.subplots(figsize=(1.5 + 1.3 * matrix.shape[1], 3.2))
    im = ax.imshow(matrix.values, aspect="auto", cmap="magma", vmin=0)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center",
                    va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_boxplots(df, coarse_cols, out_path, region_order, title):
    fig, axes = plt.subplots(
        len(coarse_cols), 1, figsize=(9, 2.7 * len(coarse_cols)),
        sharex=True, constrained_layout=True
    )
    if len(coarse_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, coarse_cols):
        data = [
            df.loc[df["region"] == r, col].to_numpy(dtype=float)
            for r in region_order if (df["region"] == r).any()
        ]
        labels = [r for r in region_order if (df["region"] == r).any()]
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].tick_params(axis="x", labelrotation=25)
    if title:
        fig.suptitle(title)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def row_entropy(values):
    values = normalize_rows(values)
    ent = -(values * np.log(values + 1e-12)).sum(axis=1)
    denom = math.log(values.shape[1]) if values.shape[1] > 1 else 1.0
    return ent / denom


def make_region_converter_features(df, pred_cols, coarse_cols):
    feature_cols = list(coarse_cols)
    if pred_cols:
        feature_cols += [c for c in pred_cols if c not in feature_cols]
    features = df[feature_cols].to_numpy(dtype=float)
    raw_values = df[pred_cols].to_numpy(dtype=float) if pred_cols else features
    extra = pd.DataFrame(index=df.index)
    extra["fraction_entropy"] = row_entropy(raw_values)
    extra["max_fraction"] = np.nanmax(raw_values, axis=1)
    extra["dominant_margin"] = 0.0
    if raw_values.shape[1] >= 2:
        sorted_values = np.sort(raw_values, axis=1)
        extra["dominant_margin"] = sorted_values[:, -1] - sorted_values[:, -2]
    feature_frame = pd.concat(
        [df[feature_cols].reset_index(drop=True), extra.reset_index(drop=True)],
        axis=1,
    )
    return feature_frame


def split_indices(df, split, test_size, seed):
    n = len(df)
    all_idx = np.arange(n)
    if split == "all":
        return all_idx, all_idx
    test_size = min(max(float(test_size), 0.05), 0.80)
    if split in {"spatial_x", "spatial_y"}:
        coord = "scaled_x" if split == "spatial_x" else "scaled_y"
        order = np.argsort(df[coord].to_numpy(dtype=float))
        n_test = max(1, int(round(n * test_size)))
        test_idx = order[-n_test:]
        train_idx = np.setdiff1d(all_idx, test_idx)
        return train_idx, test_idx

    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    labels = df["region"].astype(str).to_numpy()
    for label in sorted(pd.unique(labels)):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        if len(idx) < 2:
            train_parts.append(idx)
            continue
        n_test = min(len(idx) - 1, max(1, int(round(len(idx) * test_size))))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def standardize_train_test(x_train, x_eval):
    mean = np.nanmean(x_train, axis=0, keepdims=True)
    std = np.nanstd(x_train, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (x_train - mean) / std, (x_eval - mean) / std


def fit_predict_region_classifier(x_train, y_train, x_eval, seed):
    try:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
        clf.fit(x_train, y_train)
        return clf.classes_.astype(str), clf.predict(x_eval).astype(str), clf.predict_proba(x_eval), "logistic_regression"
    except Exception:
        classes = np.array(sorted(pd.unique(y_train.astype(str))))
        centroids = np.vstack([x_train[y_train == c].mean(axis=0) for c in classes])
        dists = ((x_eval[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        logits = -dists
        logits = logits - logits.max(axis=1, keepdims=True)
        proba = np.exp(logits)
        proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)
        pred = classes[np.argmax(proba, axis=1)]
        return classes.astype(str), pred.astype(str), proba, "nearest_centroid_fallback"


def classification_metrics(y_true, y_pred, classes):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    table = pd.crosstab(
        pd.Series(y_true, name="true"),
        pd.Series(y_pred, name="pred"),
    ).reindex(index=classes, columns=classes).fillna(0).astype(int)
    recalls = []
    precisions = []
    f1s = []
    for cls in classes:
        tp = float(table.loc[cls, cls])
        row = float(table.loc[cls].sum())
        col = float(table[cls].sum())
        recall = tp / row if row > 0 else 0.0
        precision = tp / col if col > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "confusion": table,
    }


def region_auroc(y_true, proba, classes):
    try:
        from sklearn.metrics import roc_auc_score

        scores = {}
        for j, cls in enumerate(classes):
            binary = (np.asarray(y_true).astype(str) == cls).astype(int)
            if binary.min() == binary.max():
                scores[str(cls)] = None
            else:
                scores[str(cls)] = float(roc_auc_score(binary, proba[:, j]))
        valid = [v for v in scores.values() if v is not None]
        return scores, float(np.mean(valid)) if valid else None
    except Exception:
        return {}, None


def save_confusion_heatmap(table, out_path, title):
    row_norm = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(table.columns), 4.2))
    im = ax.imshow(row_norm.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(row_norm.shape[1]))
    ax.set_xticklabels(row_norm.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(row_norm.shape[0]))
    ax.set_yticklabels(row_norm.index)
    ax.set_xlabel("Predicted pathology region")
    ax.set_ylabel("GT pathology region")
    ax.set_title(title)
    for i in range(row_norm.shape[0]):
        for j in range(row_norm.shape[1]):
            ax.text(j, i, f"{row_norm.iloc[i, j]:.2f}", ha="center",
                    va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_pathology_region_maps(df, out_dir, title):
    x = df["scaled_x"].to_numpy(dtype=float)
    y = df["scaled_y"].to_numpy(dtype=float)
    gt = df["region"].astype(str)
    pred = df["predicted_region"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, labels, name in [
        (axes[0], gt, "GT pathology region"),
        (axes[1], pred, "Predicted pathology region from decon fractions"),
    ]:
        for label in REGION_ORDER + sorted(set(labels) - set(REGION_ORDER)):
            if label not in set(labels):
                continue
            mask = labels == label
            ax.scatter(
                x[mask], y[mask], s=9, c=REGION_COLORS.get(label, "#8C8C8C"),
                label=label, linewidths=0, alpha=0.9,
            )
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=2, fontsize=8, loc="best", frameon=False)
    if title:
        fig.suptitle(title)
    fig.savefig(out_dir / "hbc_gt_vs_predicted_pathology_region_map.png", dpi=300)
    plt.close(fig)


def save_region_probability_maps(df, classes, out_dir, title):
    n = len(classes)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    x = df["scaled_x"].to_numpy(dtype=float)
    y = df["scaled_y"].to_numpy(dtype=float)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(12, 5.2 * nrows), constrained_layout=True
    )
    axes = np.asarray(axes).reshape(-1)
    for ax, cls in zip(axes, classes):
        col = f"prob_{cls}"
        vals = df[col].to_numpy(dtype=float)
        sc = ax.scatter(x, y, s=9, c=vals, cmap="viridis", vmin=0, vmax=1, linewidths=0)
        ax.set_title(cls)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(f"{title}: predicted pathology-region probabilities")
    fig.savefig(out_dir / "hbc_predicted_pathology_region_probability_maps.png", dpi=300)
    plt.close(fig)


def run_region_converter(merged, pred, pred_cols, coarse_cols, args, out_dir, title):
    work = merged.copy()
    pred_by_id = pred.set_index(args.spot_id_col)
    for col in pred_cols:
        work[col] = work[args.spot_id_col].map(pred_by_id[col])

    feature_frame = make_region_converter_features(work, pred_cols, coarse_cols)
    feature_cols = list(feature_frame.columns)
    x_all = feature_frame.to_numpy(dtype=float)
    y_all = work["region"].astype(str).to_numpy()

    train_idx, test_idx = split_indices(work, args.split, args.test_size, args.seed)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Region converter split produced empty train/test indices.")

    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    x_train_std, x_all_std = standardize_train_test(x_train, x_all)
    classes, pred_all, proba_all, classifier_name = fit_predict_region_classifier(
        x_train_std, y_train, x_all_std, args.seed
    )

    work["predicted_region"] = pred_all
    work["split"] = "train"
    work.loc[work.index[test_idx], "split"] = "test"
    if args.split == "all":
        work["split"] = "all"
    for j, cls in enumerate(classes):
        work[f"prob_{cls}"] = proba_all[:, j]

    eval_classes = [r for r in REGION_ORDER if r in set(y_all)]
    eval_classes += [r for r in sorted(set(y_all)) if r not in eval_classes]
    heldout = classification_metrics(y_all[test_idx], pred_all[test_idx], eval_classes)
    apparent = classification_metrics(y_all, pred_all, eval_classes)
    auc_by_region, macro_auc = region_auroc(y_all[test_idx], proba_all[test_idx], classes)

    heldout["confusion"].to_csv(out_dir / "hbc_region_converter_confusion_matrix.csv")
    apparent["confusion"].to_csv(out_dir / "hbc_region_converter_apparent_confusion_matrix.csv")
    work_cols = [
        args.metadata_id_col,
        args.spot_id_col,
        "region",
        args.fine_region_col,
        "scaled_x",
        "scaled_y",
        "split",
        "predicted_region",
    ] + [f"prob_{cls}" for cls in classes] + coarse_cols + pred_cols
    work[work_cols].to_csv(out_dir / "hbc_region_converter_predictions.csv", index=False)
    feature_frame.insert(0, args.spot_id_col, work[args.spot_id_col].values)
    feature_frame.insert(1, "region", work["region"].values)
    feature_frame.to_csv(out_dir / "hbc_region_converter_features.csv", index=False)

    metrics = {
        "classifier": classifier_name,
        "split": args.split,
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "feature_columns": feature_cols,
        "classes": [str(c) for c in classes],
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "heldout": {k: v for k, v in heldout.items() if k != "confusion"},
        "apparent_all_spots": {k: v for k, v in apparent.items() if k != "confusion"},
        "heldout_auroc_by_region": auc_by_region,
        "heldout_macro_auroc": macro_auc,
        "region_vs_predicted_region_nmi_all_spots": normalized_mutual_info(
            y_all, pred_all
        ),
    }
    with (out_dir / "hbc_region_converter_metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    summary_rows = []
    for scope, values in [
        ("heldout", metrics["heldout"]),
        ("apparent_all_spots", metrics["apparent_all_spots"]),
    ]:
        for key, value in values.items():
            summary_rows.append({"scope": scope, "metric": key, "value": value})
    summary_rows.append({
        "scope": "heldout",
        "metric": "macro_auroc",
        "value": macro_auc,
    })
    summary_rows.append({
        "scope": "all_spots",
        "metric": "region_vs_predicted_region_nmi",
        "value": metrics["region_vs_predicted_region_nmi_all_spots"],
    })
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "hbc_region_converter_summary.csv", index=False
    )

    save_confusion_heatmap(
        heldout["confusion"],
        out_dir / "hbc_region_converter_confusion_matrix.png",
        "Held-out pathology-region confusion matrix",
    )
    save_pathology_region_maps(work, out_dir, title)
    save_region_probability_maps(work, classes, out_dir, title)

    print(
        "region_converter "
        f"heldout_bacc={metrics['heldout']['balanced_accuracy']:.4f} "
        f"heldout_macro_f1={metrics['heldout']['macro_f1']:.4f} "
        f"heldout_acc={metrics['heldout']['accuracy']:.4f} "
        f"classifier={classifier_name}"
    )


def main():
    args = parse_args()
    metadata_path = Path(args.metadata)
    pred_path = Path(args.pred)
    out_dir = ensure_output_dir(pred_path, args.output_dir)
    groups = load_groups(args.groups_json)

    meta = pd.read_csv(metadata_path, sep="\t")
    pred = pd.read_csv(pred_path)
    if args.spot_id_col not in pred.columns:
        raise ValueError(
            f"Prediction CSV must contain '{args.spot_id_col}'. "
            "Use a raw inference pred.csv with spot_id."
        )
    for col in [args.metadata_id_col, args.region_col, "scaled_x", "scaled_y"]:
        if col not in meta.columns:
            raise ValueError(f"metadata missing required column: {col}")

    coarse, pred_cols, missing, leftovers = build_coarse_predictions(
        pred, groups, args.spot_id_col
    )
    coarse.insert(0, args.spot_id_col, pred[args.spot_id_col].astype(str).values)
    merged = meta.merge(
        coarse,
        left_on=args.metadata_id_col,
        right_on=args.spot_id_col,
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping spot IDs between metadata and prediction.")

    coarse_cols = [c for c in coarse.columns if c != args.spot_id_col]
    merged = merged.rename(columns={args.region_col: "region"})
    merged["dominant_coarse"] = merged[coarse_cols].idxmax(axis=1)
    region_order = [r for r in REGION_ORDER if r in set(merged["region"])]
    region_order += [r for r in sorted(set(merged["region"])) if r not in region_order]

    region_group_means = (
        merged.groupby("region")[coarse_cols]
        .mean()
        .reindex(region_order)
    )
    region_group_stds = (
        merged.groupby("region")[coarse_cols]
        .std()
        .reindex(region_order)
    )
    region_counts = merged["region"].value_counts().reindex(region_order)
    region_group_means.to_csv(out_dir / "hbc_region_coarse_group_means.csv")
    region_group_stds.to_csv(out_dir / "hbc_region_coarse_group_stds.csv")
    region_counts.rename("n_spots").to_csv(out_dir / "hbc_region_counts.csv")

    raw_cell_means = (
        meta.merge(
            pred[[args.spot_id_col] + pred_cols],
            left_on=args.metadata_id_col,
            right_on=args.spot_id_col,
            how="inner",
        )
        .rename(columns={args.region_col: "region"})
        .groupby("region")[pred_cols]
        .mean()
        .reindex(region_order)
    )
    raw_cell_means.to_csv(out_dir / "hbc_region_celltype_means.csv")

    eta2 = {
        col: effect_size_eta2(merged[col].to_numpy(), merged["region"].to_numpy())
        for col in coarse_cols
    }
    centroid_bacc = centroid_balanced_accuracy(
        merged[coarse_cols].to_numpy(), merged["region"].to_numpy()
    )
    nmi = normalized_mutual_info(
        merged["region"].to_numpy(), merged["dominant_coarse"].to_numpy()
    )
    dom_table = pd.crosstab(
        merged["region"], merged["dominant_coarse"], normalize="index"
    ).reindex(region_order).fillna(0.0)
    dom_table.to_csv(out_dir / "hbc_region_by_dominant_coarse.csv")

    metrics = {
        "metadata": str(metadata_path),
        "prediction": str(pred_path),
        "n_metadata_spots": int(len(meta)),
        "n_prediction_spots": int(len(pred)),
        "n_joined_spots": int(len(merged)),
        "n_prediction_cell_types": int(len(pred_cols)),
        "coarse_groups": coarse_cols,
        "leftover_cell_types_assigned_to_other": leftovers,
        "missing_group_columns": missing,
        "region_column": args.region_col,
        "fine_region_column": args.fine_region_col,
        "region_counts": {
            str(k): int(v) for k, v in region_counts.dropna().items()
        },
        "coarse_eta2_by_region": eta2,
        "mean_coarse_eta2": float(np.mean(list(eta2.values()))) if eta2 else 0.0,
        "region_centroid_balanced_accuracy": centroid_bacc,
        "region_vs_dominant_coarse_nmi": nmi,
    }
    with (out_dir / "hbc_region_consistency_metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    pd.DataFrame([{
        "metric": "mean_coarse_eta2",
        "value": metrics["mean_coarse_eta2"],
    }, {
        "metric": "region_centroid_balanced_accuracy",
        "value": metrics["region_centroid_balanced_accuracy"],
    }, {
        "metric": "region_vs_dominant_coarse_nmi",
        "value": metrics["region_vs_dominant_coarse_nmi"],
    }]).to_csv(out_dir / "hbc_region_consistency_summary.csv", index=False)

    merged_out_cols = [
        args.metadata_id_col,
        args.spot_id_col,
        "region",
        args.fine_region_col,
        "scaled_x",
        "scaled_y",
        "dominant_coarse",
    ] + coarse_cols
    merged[merged_out_cols].to_csv(
        out_dir / "hbc_annotation_prediction_joined.csv", index=False
    )

    title = args.title or pred_path.stem
    save_region_maps(merged, coarse_cols, out_dir, title)
    save_heatmap(
        region_group_means,
        out_dir / "hbc_region_coarse_group_mean_heatmap.png",
        "Mean predicted biological compartment by pathology region",
    )
    save_heatmap(
        dom_table,
        out_dir / "hbc_region_dominant_coarse_heatmap.png",
        "Dominant predicted biological compartment distribution by pathology region",
    )
    save_boxplots(
        merged,
        coarse_cols,
        out_dir / "hbc_region_coarse_group_boxplots.png",
        region_order,
        title,
    )

    print(f"Joined spots: {len(merged)}/{len(meta)} metadata, {len(pred)} predictions")
    print(f"Output: {out_dir}")
    print(f"mean_coarse_eta2={metrics['mean_coarse_eta2']:.4f}")
    print(f"region_centroid_balanced_accuracy={centroid_bacc:.4f}")
    print(f"region_vs_dominant_coarse_nmi={nmi:.4f}")
    if args.mode in {"region_converter", "both"}:
        run_region_converter(
            merged, pred, pred_cols, coarse_cols, args, out_dir, title
        )


if __name__ == "__main__":
    main()
