#!/usr/bin/env python3
"""Fail if the HBC release contains credentials, private paths, or raw data."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".py", ".json", ".md", ".txt", ".csv", ".tsv", ".yml", ".yaml",
    ".toml", ".ini", ".cfg", ".gitignore",
}
FORBIDDEN_SUFFIXES = {
    ".h5ad", ".h5", ".loom", ".env", ".pem", ".key", ".p12", ".pfx",
}
PATTERNS = {
    "private_windows_path": re.compile(r"[A-Za-z]:\\(?:Users|ZACAE-deconv)\\", re.I),
    "private_unix_path": re.compile(r"/(?:home1|data1)/[A-Za-z0-9_.-]+/"),
    "private_host": re.compile(r"222\.212\.86\.164"),
    "openai_secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._~-]{20,}", re.I),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    for path in files:
        relative = path.relative_to(ROOT)
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES or path.name.lower().startswith(".env"):
            failures.append(f"forbidden file: {relative}")
        if (suffix in TEXT_SUFFIXES or path.name in {"README.md", ".gitignore"}) and relative.as_posix() != "scripts/audit_release.py":
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in PATTERNS.items():
                if pattern.search(text):
                    failures.append(f"{label}: {relative}")

    config = json.loads((ROOT / "configs/model_config.json").read_text())
    checkpoint = ROOT / config["checkpoint"]["file"]
    if sha256(checkpoint) != config["checkpoint"]["sha256"]:
        failures.append("checkpoint SHA256 mismatch")
    if failures:
        raise SystemExit("Release audit failed:\n- " + "\n- ".join(sorted(set(failures))))
    print(json.dumps({
        "status": "PASS",
        "files_scanned": len(files),
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "raw_data_files": 0,
        "credential_or_private_path_hits": 0,
    }, indent=2))


if __name__ == "__main__":
    main()

