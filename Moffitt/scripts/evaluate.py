from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from silta.metrics import align, fraction_metrics, read_fractions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Moffitt fraction prediction")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prediction, truth = align(read_fractions(args.prediction), read_fractions(args.truth))
    metrics = fraction_metrics(prediction.to_numpy(), truth.to_numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
