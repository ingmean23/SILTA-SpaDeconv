# Inference boundary and verification status

The former external-runtime dependency has been eliminated. This capsule now
contains the minimal author-owned SILTA modules needed to instantiate the OSM
endpoint, strictly load all 323 checkpoint keys, and produce checkpoint logits
and fractions. C01-only and C01+C04 outputs are also bundled.

The remaining boundary is data preparation. OSM matrices are not
redistributable here, and the release intentionally does not guess the exact
registered normalization and graph-building choices from arbitrary raw files.
Users must supply the prepared NPZ described in `DATA_CONTRACT.md`.

Verification performed by the package tests:

- restricted tensor-only checkpoint loading;
- exact 323/323 state-key compatibility;
- prepared NPZ save/load validation;
- a synthetic forward through the real checkpoint;
- finite `[spots, 31]` logits and row-normalized fraction outputs;
- C01 pre-C04 and final C01+C04 calibration contracts.

Thus the first private-data-dependent boundary is prepared-input
materialization. There is no unpublished Python runtime dependency after that
boundary.
