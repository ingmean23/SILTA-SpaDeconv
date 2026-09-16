# Security and privacy audit policy

The release must contain no:

- API keys, access tokens, passwords, private keys, or `.env` files;
- SSH hosts, usernames, internal IP addresses, or cluster paths;
- Windows absolute paths or user-home paths;
- patient identifiers or restricted biological matrices;
- unlicensed third-party source or model weights.

Run:

```bash
python scripts/security_audit.py
```

The scanner rejects known credential patterns, private-key headers, internal
path forms, suspicious filenames, symlinks, and unexpected large binaries.
The only allowed model binaries are the named file below `checkpoint/` and the
named C01+C04 state below `calibration/`. Both are inspected with restricted
PyTorch loading and recorded in `MANIFEST.sha256`.

The scanner is a defense-in-depth check, not a substitute for manual legal,
privacy, and intellectual-property review.

## Release audit result

The final build procedure performs all of the following before the SHA256
manifest is written:

- scans text for credential assignments, token formats, private-key headers,
  SSH endpoints, IPv4 addresses, workspace-drive paths, and cluster paths;
- safely inspects string metadata inside the checkpoint and calibration bundle;
- rejects symlinks, suspicious credential filenames, unexpected binary types,
  and unexpected files larger than 150 MiB;
- summarizes extension counts, total bytes, and the ten largest files;
- verifies that temporary source manifests, build markers, logs, and caches are
  absent.

The completed capsule passed this scanner with zero findings. The checkpoint
contains tensors only. The C01+C04 bundle originally contained one provenance
path; the release copy replaces it with a relative checkpoint path while
preserving all calibration tensors and records both source and release hashes
in `configs/provenance.json`.

The bundled inference runtime is author-owned and contains no copied third-party
source. Its tests use synthetic arrays only; no prepared OSM bundle, raw
expression, reference matrix, target truth, or baseline prediction is included.

The remaining non-technical release blocker is legal rather than security
related: the copyright holders must select a software license before public
distribution.
