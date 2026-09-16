from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from silta_osm.figures import render_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SILTA OSM benchmark figures")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    for path in render_manifest(args.manifest):
        print(path)


if __name__ == "__main__":
    main()

