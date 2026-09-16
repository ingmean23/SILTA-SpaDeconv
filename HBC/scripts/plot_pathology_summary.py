#!/usr/bin/env python3
"""Plot Track-B fold metrics and the best sealed-fold pathology map."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = (
    ("balanced_accuracy", "Balanced accuracy"),
    ("macro_f1", "Macro-F1"),
    ("macro_auroc", "Macro-AUROC"),
)
REGION_COLORS = {
    "Healthy": "#6F3FA0",
    "Surrounding tumor": "#B08B2D",
    "Tumor": "#E23B3B",
    "Invasive": "#2E7FB8",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    metrics_path = args.evaluation_dir / "spatial_trackb_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    core = metrics[[key for key, _ in METRICS]].mean(axis=1)
    best_row = metrics.iloc[int(np.argmax(core.to_numpy()))]
    fold = str(best_row.get("split_id", best_row.get("outer_split", "fold_0")))
    prediction_path = (
        args.evaluation_dir / "spatial_trackb" / fold / "predictions.csv"
    )
    prediction = pd.read_csv(prediction_path)

    fig = plt.figure(figsize=(12.4, 5.1), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.15))
    for panel, column, title in (
        (fig.add_subplot(grid[0, 0]), "region", "Pathology annotation"),
        (fig.add_subplot(grid[0, 1]), "predicted_region", "SILTA + Track-B"),
    ):
        for name, color in REGION_COLORS.items():
            mask = prediction[column].astype(str).eq(name)
            panel.scatter(
                prediction.loc[mask, "scaled_x"],
                prediction.loc[mask, "scaled_y"],
                s=9, linewidths=0, color=color, label=name,
            )
        panel.set_title(title)
        panel.set_aspect("equal")
        panel.invert_yaxis()
        panel.set_xticks([])
        panel.set_yticks([])
    axis = fig.add_subplot(grid[0, 2])
    x = np.arange(len(METRICS))
    means = [metrics[key].mean() for key, _ in METRICS]
    stds = [metrics[key].std(ddof=1) for key, _ in METRICS]
    axis.bar(x, means, yerr=stds, color=("#4C78A8", "#F2A541", "#8064A2"), capsize=3)
    axis.set_xticks(x, [label for _, label in METRICS], rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Five-fold score")
    axis.set_title("Spatial generalization")
    axis.spines[["top", "right"]].set_visible(False)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Representative sealed fold: {fold}")


if __name__ == "__main__":
    main()
