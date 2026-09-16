from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from silta.plotting import metric_bars, spatial_maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SILTA Moffitt result figures")
    subparsers = parser.add_subparsers(dest="kind", required=True)
    spatial = subparsers.add_parser("spatial")
    spatial.add_argument("--prediction", type=Path, required=True)
    spatial.add_argument("--coordinates", type=Path, required=True)
    spatial.add_argument("--output", type=Path, required=True)
    spatial.add_argument("--max-types", type=int, default=12)
    bars = subparsers.add_parser("metrics")
    bars.add_argument("--metrics", type=Path, required=True)
    bars.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "spatial":
        spatial_maps(args.prediction, args.coordinates, args.output, args.max_types)
    else:
        metric_bars(args.metrics, args.output)


if __name__ == "__main__":
    main()
