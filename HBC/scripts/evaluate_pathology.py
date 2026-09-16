#!/usr/bin/env python3
"""Evaluate one locked HBC prediction with Random-Direct and Track-B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.hbc_pathology_unified_sota_v1 import protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument(
        "--protocol", type=Path,
        default=ROOT / "configs" / "pathology_protocol.json",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--method", default="SILTA")
    args = parser.parse_args()

    settings = json.loads(args.protocol.read_text(encoding="utf-8"))["pathology"]
    random_frame, spatial_frame = protocol.evaluate_single(
        args.method, args.prediction, args.metadata, settings, args.output_dir
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random_frame.to_csv(args.output_dir / "random_direct_metrics.csv", index=False)
    spatial_frame.to_csv(args.output_dir / "spatial_trackb_metrics.csv", index=False)
    summary = {
        "method": args.method,
        "random_direct": protocol.summary(random_frame),
        "spatial_trackb": protocol.summary(spatial_frame),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
