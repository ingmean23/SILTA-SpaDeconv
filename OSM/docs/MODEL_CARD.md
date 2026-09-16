# Model card: SILTA on OSM Benchmark B

## Model identity

- **Name:** SILTA
- **Expansion:** Separating composition learning from target adaptation in
  spatial transcriptomic deconvolution
- **Released endpoint:** OSM Benchmark B, split 42, model seed 42, epoch 7
- **Output:** 31 normalized cell-type fractions per target spot

## Intended use

Research reproduction and audit of the locked OSM Benchmark B endpoint. The
checkpoint is intended only for compatible 33-gene OSM inputs, the registered
31-type ontology, and the graph/data preprocessing contract described here.

## Out-of-scope use

- clinical diagnosis or treatment;
- unvalidated tissues, panels, or ontologies;
- fitting or selecting parameters with target ground-truth fractions;
- presenting the one released checkpoint as the historical nine-run mean;
- redistribution of OSM or baseline data without permission.

## Performance

Released recovery checkpoint after C01+C04:

| RMSE down | JSD down | SSIM up | PCC up |
|---:|---:|---:|---:|
| 0.064058 | 0.459200 | 0.804938 | 0.799748 |

Historical locked `3 splits x 3 model seeds` arithmetic mean:

| RMSE down | JSD down | SSIM up | PCC up |
|---:|---:|---:|---:|
| 0.064292 | 0.458655 | 0.805256 | 0.799278 |

All nine historical runs passed the recorded structural-validity guard.

## Selection and leakage controls

Stage I/II training and C01+C04 fitting used reference-side supervision only.
The representative calibration recorded `target_truth_used_for_selection=false`.
The target-evaluation choice was serialized before the final evaluator read
target truth. Ground truth is an evaluator-only input in this capsule.

## Limitations

- Model construction and checkpoint forward are bundled and all 323 state keys
  load strictly. Users must still materialize the registered prepared-input
  NPZ from authorized OSM data; arbitrary raw-data preprocessing is not bundled.
- C01+C04 is reference-only but specific to this ontology and endpoint.
- The 33-gene OSM panel is unusually small; transfer to other panels is not
  established here.
- Biological data and baseline predictions are not distributed.
- No project software license has yet been selected by the copyright holders.

## Integrity

The checkpoint, calibration bundle, readable parameters, sanitized provenance,
tests, and every source file are covered by `MANIFEST.sha256`.
