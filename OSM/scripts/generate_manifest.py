from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _bootstrap import ROOT


MANIFEST = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".git"}
EXCLUDED_NAMES = {MANIFEST.name, ".release_build_in_progress"}


def included_files() -> list[Path]:
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED_NAMES
        and not any(part in EXCLUDED_PARTS for part in path.parts)
    )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def content() -> str:
    return "".join(
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n"
        for path in included_files()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write or verify the release manifest")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = content()
    if args.check:
        if not MANIFEST.is_file():
            raise SystemExit("MANIFEST.sha256 is missing")
        if MANIFEST.read_text(encoding="utf-8") != expected:
            raise SystemExit("MANIFEST.sha256 is stale or invalid")
        print(f"PASS: {len(included_files())} files match MANIFEST.sha256")
        return
    MANIFEST.write_text(expected, encoding="utf-8", newline="\n")
    print(f"WROTE: {MANIFEST.name} ({len(included_files())} files)")


if __name__ == "__main__":
    main()

