# SILTA: OSM Benchmark B release capsule

SILTA stands for **Separating composition learning from target adaptation in
spatial transcriptomic deconvolution**. This capsule contains one representative
OSM Benchmark B checkpoint, its reference-only C01+C04 calibration state, a
sanitized relative configuration, locked result evidence, and standalone
evaluation and visualization utilities.

## Scope

Included:

- the recovered split-42/model-seed-42 checkpoint;
- the fitted C01+C04 transfer state and a human-readable C04 parameter export;
- an author-owned, inference-only SILTA implementation that strictly loads all
  323 checkpoint keys without the historical training/search runners;
- checkpoint and calibration integrity checks;
- C04 application to exported SILTA logits;
- RMSE, JSD, SSIM, PCC, rare-type, and topology evaluation;
- dominant-map, spot-pie, composite, and metric figure rendering;
- the historical registered `3 splits x 3 model seeds` evidence table;
- secret/path/symlink/extension/size scanning and SHA256 manifests.

Excluded:

- OSM expression matrices, coordinates, reference cells, target fractions,
  pseudo-spots, and baseline predictions;
- credentials, server identifiers, absolute local paths, logs, caches, and Git
  metadata;
- third-party baseline source code;
- the historical training/search dependency graph.

## SILTA architecture represented by the checkpoint

**Stage I: reference composition establishment.** Ordinary labelled reference
pseudo-spots train the content expert for 80 epochs at learning rate `1e-3`.
The spatial expert is frozen and replaced by zero in this stage. The content
latent is stochastic during training with KL weight `1e-4`.

**Stage II: target adaptation.** Seven epochs at learning rate `3e-3` use
fraction replay (`0.05`), a protected content learning-rate factor (`0.02`),
additive cooperative fusion `h + 0.5*g`, fixed Gaussian distance-weighted
hypergraph attention (`2` heads, beta `1.0`, gate `0.1`), and a qNB observation
weight that increases from `0.03` to `0.20` through epoch 5. Direct mode and
bilinear residuals are disabled for this OSM endpoint.

**Reference-only calibration.** C01 performs broad-group temperature
calibration. C04 adds bounded group-centered per-type bias and temperature
corrections. Target fractions were not used for training, selection, or
calibration; the final evaluator read them only after the C04 choice was
written to a lock file.

## Representative artifact versus paper aggregate

These numbers answer different questions and must not be interchanged.

| Evidence | RMSE down | JSD down | SSIM up | PCC up |
|---|---:|---:|---:|---:|
| Released split-42/model-seed-42 recovery | 0.064058 | 0.459200 | 0.804938 | 0.799748 |
| Historical locked 3x3 arithmetic mean | 0.064292 | 0.458655 | 0.805256 | 0.799278 |

The released checkpoint is a fresh recovery of the same locked configuration.
It is not an extra member of, nor a replacement for, the nine runs used to
compute the paper aggregate. See `expected_results/` and `evidence/`.

## Layout

```text
checkpoint/       representative model state dictionary
calibration/      C01+C04 transfer state and readable parameters
configs/          relative config, sanitized provenance, source hashes
src/silta_osm/    author-owned model inference, calibration, evaluation, and figure code
scripts/          inference contract, C04, evaluation, plotting, validation tools
expected_results/ representative and 3x3 expected values
evidence/         the nine registered historical metric rows
docs/             data contract, model card, notices, and security report
tests/            synthetic and real-artifact contract tests
```

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Validate the redistributable capsule without private OSM data:

```bash
python scripts/validate_release.py --allow-missing-data
python scripts/inspect_checkpoint.py
python scripts/security_audit.py
python scripts/generate_manifest.py --check
pytest -q -p no:cacheprovider
```

## Inference contract

Prepare the authorized 33-gene endpoint inputs as specified in
`docs/DATA_CONTRACT.md`. The helper below verifies table labels and ordering and
writes the single strict input bundle expected by the model:

```bash
python scripts/prepare_inputs.py \
  --expression data/expression.tsv \
  --expression-adjacency data/expression_adjacency.tsv \
  --spatial-adjacency data/spatial_adjacency.tsv \
  --coordinates data/locations.tsv \
  --reference-signature data/reference_signature.tsv \
  --output data/osm_prepared_input.npz
```

Then run the real recovered checkpoint:

```bash
python scripts/run_inference.py \
  --config configs/silta_osm_b_fold42_seed42.json \
  --device cpu
```

The command writes checkpoint logits, raw checkpoint fractions, C01-only
pre-C04 fractions, and final C01+C04 fractions. C04 can also be reapplied to a
saved logit table:

```bash
python scripts/apply_c04.py \
  --logits outputs/checkpoint_logits.tsv.gz \
  --calibration-state calibration/C01_C04_state.pt \
  --output outputs/silta_fractions.tsv.gz
```

## Evaluation

Ground truth is evaluator-only and must never be passed to inference or
calibration:

```bash
python scripts/evaluate.py \
  --prediction outputs/silta_fractions.tsv.gz \
  --truth data/ground_truth_fractions.tsv \
  --coordinates data/locations.tsv \
  --type-counts data/type_counts.tsv \
  --output outputs/metrics.json
```

## Plotting

Prepare a manifest following `figures/example_manifest.yaml` and run:

```bash
python scripts/render_figures.py --manifest figures/manifest.yaml
```

The renderer requires user-supplied matched fractions, coordinates, and metric
tables. No OSM or baseline data are embedded here.

## Data and licensing caveat

Users must obtain OSM and every baseline output from their authorized sources
and comply with the original licenses. This directory contains no patient or
biological matrices. The parent project currently has no finalized root
software license; `LICENSE_PENDING.md` must be replaced by an approved license
before describing the capsule as open source. See `docs/THIRD_PARTY_NOTICES.md`.

The first private-data-dependent boundary is the creation of the prepared NPZ.
This release deliberately does not guess normalization or graph construction
from arbitrary raw H5AD files. Once the prepared contract is supplied, model
construction, strict checkpoint loading, forward inference, C01, and C04 are
self-contained.
