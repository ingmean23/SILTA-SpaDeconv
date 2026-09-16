from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path

import torch

from _bootstrap import ROOT


TEXT_SUFFIXES = {
    ".csv", ".ini", ".json", ".md", ".py", ".toml", ".tsv", ".txt",
    ".yaml", ".yml",
}
FIGURE_SUFFIXES = {".pdf", ".png", ".tif", ".tiff"}
FORBIDDEN_DIRS = {"__pycache__", ".pytest_cache", ".git"}
FORBIDDEN_FILES = {".release_build_in_progress"}
ALLOWED_BINARY_ASSETS = {
    "checkpoint/silta_osm_b_fold42_seed42.model",
    "calibration/C01_C04_state.pt",
}
SUSPICIOUS_FILENAMES = re.compile(
    r"(?i)(id_rsa|id_ed25519|credentials|\.pem$|\.p12$|\.pfx$|\.key$|\.env$)"
)
PATTERNS = {
    "private_key": re.compile("BEGIN " + ".{0,30}" + "PRIVATE " + "KEY"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "gitlab_token": re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    "git_credential_url": re.compile(
        r"(?i)https?://[^/\s:@]+:[^@\s/]+@[A-Za-z0-9.-]+"
    ),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "bearer_token": re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    "password_assignment": re.compile(
        r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]{8,}"
    ),
    "windows_absolute_path": re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/][^\s'\"<>]+"),
    "unix_absolute_path": re.compile(
        r"(?<![.A-Za-z0-9_])" + "/" + r"(?!/)(?:[A-Za-z0-9._-]+/)+[^\s'\"<>]+"
    ),
    "ssh_endpoint": re.compile(r"\b[A-Za-z_][A-Za-z0-9_.-]*@[A-Za-z0-9.-]+\b"),
    "ipv4_address": re.compile(
        r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])"
    ),
    "workspace_drive": re.compile("(?i)" + "H:" + r"[\\/]"),
    "server_home": re.compile("/" + "home1/"),
    "server_data": re.compile("/" + "data1/"),
}


def _strings(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _strings(key)
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)
    elif isinstance(value, str):
        yield value


def _scan_tensor_metadata(path: Path, relative: str) -> list[dict[str, object]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # pragma: no cover - defensive release check
        return [{"kind": "unreadable_torch_asset", "path": relative, "detail": str(error)}]
    findings = []
    for value in _strings(payload):
        for name, pattern in PATTERNS.items():
            if pattern.search(value):
                findings.append({"kind": f"binary_{name}", "path": relative})
    return findings


def audit(root: Path) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    extension_counts: Counter[str] = Counter()
    file_count = 0
    total_bytes = 0
    largest: list[tuple[int, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append({"kind": "symlink", "path": relative})
            continue
        forbidden_parts = [part for part in path.relative_to(root).parts if part in FORBIDDEN_DIRS]
        if forbidden_parts:
            if path.name in FORBIDDEN_DIRS:
                findings.append({"kind": "forbidden_directory", "path": relative})
            continue
        if path.name in FORBIDDEN_FILES:
            findings.append({"kind": "forbidden_build_file", "path": relative})
        if not path.is_file():
            continue
        file_count += 1
        size = path.stat().st_size
        total_bytes += size
        extension_counts[path.suffix.lower() or "<none>"] += 1
        largest.append((size, relative))
        if SUSPICIOUS_FILENAMES.search(path.name):
            findings.append({"kind": "suspicious_filename", "path": relative})
        if size > 150 * 1024 * 1024 and relative not in ALLOWED_BINARY_ASSETS:
            findings.append({"kind": "unexpected_large_file", "path": relative})
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "MANIFEST.sha256":
            allowed_checkpoint = relative in ALLOWED_BINARY_ASSETS
            allowed_figure = (
                any(part in {"figures", "outputs"} for part in path.parts)
                and path.suffix.lower() in FIGURE_SUFFIXES
            )
            if not allowed_checkpoint and not allowed_figure:
                findings.append({"kind": "unapproved_binary_asset", "path": relative})
            elif allowed_checkpoint:
                findings.extend(_scan_tensor_metadata(path, relative))
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            findings.append({"kind": "non_utf8_text", "path": relative})
            continue
        for line_number, line in enumerate(lines, start=1):
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append({
                        "kind": name,
                        "path": relative,
                        "line": line_number,
                    })
    return {
        "status": "PASS" if not findings else "FAIL",
        "root": ".",
        "file_count": file_count,
        "total_bytes": total_bytes,
        "extension_counts": dict(sorted(extension_counts.items())),
        "largest_files": [
            {"path": path, "bytes": size}
            for size, path in sorted(largest, reverse=True)[:10]
        ],
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan release files for disclosure risks")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root.resolve())
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
