from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from silta_osm.evaluation import evaluate_files
from silta_osm.io import write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Evaluate SILTA OSM fractions")
    result.add_argument("--prediction", type=Path, required=True)
    result.add_argument("--truth", type=Path, required=True)
    result.add_argument("--coordinates", type=Path, required=True)
    result.add_argument("--type-counts", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    metrics = evaluate_files(
        args.prediction, args.truth, args.coordinates, args.type_counts
    )
    write_json(args.output, metrics)
    print(args.output)


if __name__ == "__main__":
    main()

