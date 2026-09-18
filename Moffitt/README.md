# SILTA: Moffitt Benchmark B code and evaluation release

This folder contains the inference, evaluation, and plotting code for the
Moffitt Benchmark B result of **SILTA** (*Separating composition learning from
target adaptation in spatial transcriptomic deconvolution*). It intentionally
does **not** contain training, hyperparameter-search, pseudo-spot-generation,
or private evaluator code.

## Contents

```text
Moffitt/
  configs/           locked public inference contract
  expected/          published aggregate and representative metrics
  silta/             inference-only model, preprocessing, metrics, and plots
  scripts/           command-line entry points
  tests/             offline smoke tests
```

## Data acquisition

Download the public MERFISH cell table
`Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv` from the
[Moffitt et al. Dryad record](https://doi.org/10.5061/dryad.8t8s248).
The associated study is Moffitt et al., *Science* (2018),
[doi:10.1126/science.aau5324](https://doi.org/10.1126/science.aau5324).

The release does not redistribute the source table or target truth. Benchmark B
uses three leave-one-animal-out folds: animals 2+4 -> 1 (B-M1), 1+4 -> 2
(B-M2), and 1+2 -> 4 (B-M3). Only female, naive, bregma -0.14 sections are
used. Target truth is required only by `scripts/evaluate.py` after predictions
have been produced.

## Preprocessing contract

The inference command accepts two tab-separated files:

1. `expression.tsv`: rows are `spot_id`; columns are genes. Values must be
   finite, non-negative, unscaled MERFISH expression. The inference
   configuration selects and orders its locked 135-gene panel, performs
   per-gene z-scoring, and does not apply library normalization, rounding, or
   `log1p`.
2. `coordinates.tsv`: columns `spot_id`, `x`, and `y`, with one row per spot.

The expression graph is mutual cosine 6-NN with a symmetric minimum-degree
fallback. The spatial graph uses a radius of 1.5 times the median nearest-spot
distance plus symmetric 4-NN edges. The spatial expert uses continuous
distance-decay attention normalized by the median spatial-edge length.

For the paper benchmark, each target animal was aggregated into non-overlapping
75-unit square bins, bins with fewer than three retained cells were removed,
and the 21 sequential-FISH tail genes were excluded. The canonical 45-type
taxonomy retains types with at least 20 cells in every benchmark animal.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

CPU inference is supported. A CUDA device may be selected explicitly when a
compatible PyTorch build is installed.

## Inference

```bash
python scripts/run_inference.py \
  --checkpoint /path/to/model.pt \
  --expression data/expression.tsv \
  --coordinates data/coordinates.tsv \
  --output results/prediction.tsv \
  --device cpu
```

The command loads the supplied model weights with
`torch.load(..., weights_only=True)`, verifies the model schema, aligns genes
and spots, and writes both the fraction table and a SHA256 manifest.

## Evaluation

Ground truth must have the same tabular layout as the prediction: rows are
spots, columns are the canonical cell types, and each row is a fraction vector.

```bash
python scripts/evaluate.py \
  --prediction results/prediction.tsv \
  --truth data/truth.tsv \
  --output results/metrics.json
```

The reported primary metrics exactly follow the paper evaluator:

- RMSE: mean of per-cell-type RMSE values;
- JSD: mean cell-type-wise Jensen-Shannon distance over active types;
- SSIM: flattened global structural similarity;
- PCC: flattened global Pearson correlation.

`expected/representative_metrics.json` records the representative B-M1/seed-42
evaluation summary. `expected/headline_metrics.csv` records the paper's
three-fold by three-seed mean.

## Plotting

```bash
python scripts/plot_results.py spatial \
  --prediction results/prediction.tsv \
  --coordinates data/coordinates.tsv \
  --output figures/spatial_maps.png

python scripts/plot_results.py metrics \
  --metrics data/method_metrics.csv \
  --output figures/method_comparison.png
```

The metrics table must contain `method`, `ST_RMSE`, `ST_JSD`, `ST_SSIM`, and
`ST_Pearson` columns. Figures are exported as PNG, PDF, and SVG.

## Offline verification

```bash
python -m pytest -q
```

See `SECURITY_AUDIT.md` for the release scan and `THIRD_PARTY_NOTICES.md` for
dependency licenses.
