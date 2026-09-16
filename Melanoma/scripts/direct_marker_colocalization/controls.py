"""Negative controls for direct marker colocalization."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import score_direct_colocalization


def score_negative_controls(
    fractions: np.ndarray,
    observed: np.ndarray,
    cell_types: list[str],
    genes: list[str],
    markers_by_type: dict[str, list[str]],
    strict_invalid_score: float,
    seed: int = 42,
    primary_cell_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    variants = {
        "spot_permuted": fractions[rng.permutation(len(fractions))],
        "type_mismatched": np.roll(fractions, shift=1, axis=1),
        "uniform": np.full_like(fractions, 1.0 / fractions.shape[1]),
    }
    rows = []
    for name, values in variants.items():
        _, _, summary, _ = score_direct_colocalization(
            values, observed, cell_types, genes, markers_by_type,
            strict_invalid_score, primary_cell_types,
        )
        rows.append({"control": name, **summary})
    return rows
