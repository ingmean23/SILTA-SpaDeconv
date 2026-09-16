from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from silta_osm.inference import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SILTA OSM inference")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "silta_osm_b_fold42_seed42.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-c04", action="store_true",
        help="Write checkpoint and C01-only pre-C04 outputs without final C04.",
    )
    args = parser.parse_args()
    outputs = run(args.config, device=args.device, apply_c04=not args.no_c04)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
