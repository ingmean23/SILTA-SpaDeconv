"""Audit the SILTA DLPFC release for safety, portability, and loadability."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cfg", ".csv", ".json", ".md", ".py", ".rst", ".toml", ".tsv",
    ".txt", ".yaml", ".yml",
}
FORBIDDEN_SUFFIXES = {
    ".env", ".h5ad", ".key", ".pem", ".pickle", ".pkl",
}
EXPECTED_DEPENDENCIES = {
    "anndata": "BSD-3-Clause",
    "h5py": "BSD-3-Clause",
    "matplotlib": "PSF-based",
    "numpy": "BSD-3-Clause",
    "pandas": "BSD-3-Clause",
    "scanpy": "BSD-3-Clause",
    "scikit-learn": "BSD-3-Clause",
    "scipy": "BSD-3-Clause",
    "torch": "BSD-3-Clause",
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            yield path


def scan_text() -> tuple[list[str], list[str]]:
    secret_patterns = {
        "private_key": re.compile(
            "BEGIN " + "(?:RSA |OPENSSH |EC )?" + "PRIVATE" + " KEY"
        ),
        "credential_assignment": re.compile(
            r"(?i)(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*[\x22\x27]?[A-Za-z0-9_./+=-]{16,}"
        ),
        "wandb_key": re.compile("wandb" + r"[_-]?(?:api[_-]?)?key\s*[:=]"),
    }
    absolute_patterns = {
        "windows_absolute": re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"),
        "unix_home": re.compile("/" + "home/"),
        "unix_data": re.compile("/" + "data1/"),
        "known_host": re.compile(re.escape("222" + ".212.86.164")),
    }
    secret_hits: list[str] = []
    path_hits: list[str] = []
    for path in source_files():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in secret_patterns.items():
            if pattern.search(text):
                secret_hits.append(f"{relative(path)}: {name}")
        for name, pattern in absolute_patterns.items():
            if pattern.search(text):
                path_hits.append(f"{relative(path)}: {name}")
    return secret_hits, path_hits


def audit_files() -> list[str]:
    issues = []
    for path in source_files():
        lower = path.name.lower()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(f"forbidden extension: {relative(path)}")
        if lower.startswith(".env") or lower in {"id_rsa", "id_ed25519"}:
            issues.append(f"forbidden credential file: {relative(path)}")
    checkpoints = list((ROOT / "checkpoint").glob("*.model"))
    if len(checkpoints) != 1:
        issues.append(f"expected exactly one checkpoint, found {len(checkpoints)}")
    return issues


def audit_checkpoint() -> dict:
    from silta.inference import build_model, load_model_config

    checkpoint = ROOT / "checkpoint" / "silta_dlpfc_mix5_best_single_seed.model"
    config = load_model_config(ROOT / "configs" / "model.json")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in state.items()
    ):
        raise RuntimeError("checkpoint is not a tensor-only state dictionary")
    model = build_model(config)
    model.load_state_dict(state, strict=True)
    return {
        "file": relative(checkpoint),
        "sha256": sha256(checkpoint),
        "tensor_count": len(state),
        "strict_model_load": True,
        "weights_only_load": True,
    }


def audit_licenses() -> dict:
    license_path = ROOT / "LICENSE"
    third_party = ROOT / "THIRD_PARTY_LICENSES.md"
    if "MIT License" not in license_path.read_text(encoding="utf-8"):
        raise RuntimeError("release LICENSE is not MIT")
    text = third_party.read_text(encoding="utf-8")
    missing = [
        package for package in EXPECTED_DEPENDENCIES if package not in text
    ]
    if missing:
        raise RuntimeError(f"third-party license inventory misses: {missing}")
    payload = {
        "release_license": "MIT",
        "dependencies": EXPECTED_DEPENDENCIES,
        "vendored_third_party_source": False,
    }
    (ROOT / "metadata" / "third_party_licenses.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def write_hashes() -> None:
    output = ROOT / "metadata" / "SHA256SUMS.txt"
    files = [
        path for path in source_files()
        if path != output and not path.name.endswith(".pyc")
    ]
    lines = [f"{sha256(path)}  {relative(path)}" for path in sorted(files)]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    secret_hits, path_hits = scan_text()
    file_issues = audit_files()
    checkpoint = audit_checkpoint()
    licenses = audit_licenses()
    report = {
        "secret_hits": secret_hits,
        "absolute_path_hits": path_hits,
        "file_issues": file_issues,
        "checkpoint": checkpoint,
        "licenses": licenses,
        "passed": not (secret_hits or path_hits or file_issues),
    }
    report_path = ROOT / "metadata" / "release_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ROOT / "metadata" / "secret_scan_report.txt").write_text(
        "PASS: no credential-like strings detected.\n"
        if not secret_hits
        else "\n".join(secret_hits) + "\n",
        encoding="utf-8",
    )
    (ROOT / "metadata" / "absolute_path_scan_report.txt").write_text(
        "PASS: no machine-specific absolute paths detected.\n"
        if not path_hits
        else "\n".join(path_hits) + "\n",
        encoding="utf-8",
    )
    write_hashes()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
