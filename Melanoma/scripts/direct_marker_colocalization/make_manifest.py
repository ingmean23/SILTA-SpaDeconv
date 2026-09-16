"""Build a locked prediction manifest for direct marker comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.direct_marker_colocalization.schema import load_config


def build_manifest(
    config_path: str | Path,
    baseline_root: str | Path,
    output: str | Path,
    extra_methods: list[str] | None = None,
) -> Path:
    config = load_config(config_path)
    baseline_root = Path(baseline_root).resolve()
    methods: dict[str, dict[str, str]] = {}
    for slice_name in config.slices:
        for prediction in sorted((baseline_root / slice_name).glob("**/predictions_locked.csv")):
            method = prediction.relative_to(baseline_root / slice_name).parts[0]
            if slice_name in methods.setdefault(method, {}):
                raise ValueError(f"Duplicate prediction for {method}/{slice_name}")
            methods[method][slice_name] = str(prediction.resolve())
    for method, predictions in methods.items():
        missing = sorted(set(config.slices) - set(predictions))
        if missing:
            raise ValueError(f"Baseline {method} misses slices: {missing}")
    for item in extra_methods or []:
        if "=" not in item:
            raise ValueError("Extra method must use NAME=PATH_WITH_{slice}")
        method, pattern = item.split("=", 1)
        if method in methods:
            raise ValueError(f"Duplicate method name: {method}")
        predictions = {
            slice_name: str(Path(pattern.format(slice=slice_name)).resolve())
            for slice_name in config.slices
        }
        missing = [name for name, path in predictions.items() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Extra method {method} misses slices: {missing}")
        methods[method] = predictions
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "direct marker-fraction colocalization v1",
        "config": str(Path(config_path).resolve()),
        "methods": {
            method: {"predictions": predictions}
            for method, predictions in sorted(methods.items())
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", action="append", default=[])
    args = parser.parse_args()
    print(build_manifest(args.config, args.baseline_root, args.output, args.method))


if __name__ == "__main__":
    main()
