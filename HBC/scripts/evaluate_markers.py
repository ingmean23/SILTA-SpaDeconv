#!/usr/bin/env python3
"""Run the locked HBC direct marker-fraction colocalization evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from marker_benchmark.evaluate import evaluate_prediction  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs" / "direct_marker_config.json",
    )
    parser.add_argument("--slice", default="ds10")
    parser.add_argument("--method", default="SILTA")
    parser.add_argument("--no-controls", action="store_true")
    args = parser.parse_args()
    summary = evaluate_prediction(
        args.config, args.slice, args.method, args.prediction,
        args.output_dir, True, not args.no_controls,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
