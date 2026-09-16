"""Build a comparison manifest from locked fixed-native baseline predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DISPLAY_NAMES = {
    "card": "CARD",
    "cell2location": "Cell2location",
    "rctd": "RCTD",
    "spatialdecon": "SpatialDecon",
    "spotlight": "SPOTlight",
    "stdgcn": "STdGCN",
    "stereoscope": "Stereoscope",
    "stride": "STRIDE",
    "tangram": "Tangram",
}


def discover_predictions(
    results_root: Path,
    dataset: str,
    track: str,
) -> dict[str, dict[str, str]]:
    """Return complete method-to-slice mappings under one locked result track."""
    run_root = results_root / "runs" / dataset
    discovered: dict[str, dict[str, str]] = {}
    for prediction in sorted(run_root.rglob("predictions_locked.csv")):
        parts = prediction.relative_to(run_root).parts
        if dataset == "melanoma":
            if len(parts) < 6 or parts[1] != track:
                continue
            slice_name, method = parts[0], parts[2]
        else:
            if len(parts) < 5 or parts[0] != track:
                continue
            slice_name, method = "ds10" if dataset == "hbc" else "ds11", parts[1]
        display_name = DISPLAY_NAMES.get(method, method)
        existing = discovered.setdefault(display_name, {})
        if slice_name in existing:
            raise ValueError(f"Duplicate prediction for {display_name}/{slice_name}")
        existing[slice_name] = str(prediction.resolve())
    return discovered


def build_manifest(
    results_root: Path,
    dataset: str,
    track: str,
    expected_slices: list[str],
    primary_method: str,
) -> dict[str, object]:
    discovered = discover_predictions(results_root, dataset, track)
    expected = set(expected_slices)
    complete = {
        method: {"predictions": predictions}
        for method, predictions in sorted(discovered.items())
        if set(predictions) == expected
    }
    if not complete:
        raise ValueError("No method has a complete locked prediction set")
    if primary_method not in complete:
        available = ", ".join(complete)
        raise ValueError(f"Primary method {primary_method!r} is unavailable; found: {available}")
    return {
        "primary_method": primary_method,
        "metadata": {
            "selection": "all complete predictions_locked.csv sets",
            "parameter_search": False,
            "algorithm_seed": 42,
            "track": track,
        },
        "methods": complete,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("hbc", "dlpfc", "melanoma"), required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--slices", required=True, help="Comma-separated expected slice IDs")
    parser.add_argument("--primary-method", default="Stereoscope")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(
        args.results_root.resolve(),
        args.dataset,
        args.track,
        [item.strip() for item in args.slices.split(",") if item.strip()],
        args.primary_method,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(manifest['methods'])} complete methods: {args.output}")


if __name__ == "__main__":
    main()
