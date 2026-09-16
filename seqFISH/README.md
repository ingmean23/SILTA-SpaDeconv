# SILTA seqFISH release

**SILTA: Separating composition learning from target adaptation in spatial transcriptomic deconvolution**

This dataset-scoped folder contains the code, documentation, sanitized runtime configuration, and one representative seqFISH Benchmark B checkpoint required for inference, evaluation, and figure reproduction. It contains no dataset, target ground truth, API key, authentication token, host address, or private server path.

## Representative checkpoint

```text
checkpoints/silta_seqfish_sp17_ms17_e005.model
```

The checkpoint is a tensor-only `state_dict` loaded with `torch.load(..., weights_only=True)`. Its representative single-run endpoint is:

| Field | Value |
|---|---:|
| reference split | 17 |
| model seed | 17 |
| Stage II epoch | 5 |
| calibration strength | 1.0 |
| inference temperature | 0.525 |
| RMSE | 0.0693117646 |
| JSD | 0.3853535927 |
| SSIM | 0.9061342643 |
| PCC | 0.9047498461 |

SHA256:

```text
b03f043463a4cc5e7070e4065ddd009fe5b0d6cfe99b6a607702e38bcfc2548d
```

The internal Python class remains named `DACGModel` so the original state-dict keys load without conversion. The public method name is SILTA.

## Architecture

SILTA separates composition learning from target adaptation:

1. Stage I trains the expression/content expert and mode supervision on diffusion-augmented reference pseudo-spots.
2. Stage II trains the spatial expert on the target spatial graph while isolating pseudo-space gradients from the spatial branch.
3. Cooperative fusion combines content and spatial representations using an additive spatial term and a bounded bilinear interaction.
4. qNB observation supervision and reference-fitted calibration constrain the final cell-type proportions.

The released endpoint uses sampled-h training, deterministic-g inference, `lambda_g=0.55`, `lambda_x=0.025`, and `lambda_m=0`.

## Contents

```text
checkpoints/  SILTA checkpoint
configs/      sanitized architecture and inference configuration
models/       checkpoint-compatible model implementation
runtime/      isolated seqFISH attention and sampled-h routing
utils/        losses and readout utilities
inference/    checkpoint inspection and prediction CLIs
evaluation/   RMSE, JSD, SSIM, PCC, and structural metrics
plotting/     standardized spatial and metric figure renderer
```

## Install

Tested with Python 3.11 and PyTorch 2.4:

```bash
python -m pip install -r requirements.txt
```

## Inspect the checkpoint

```bash
python inference/inspect_checkpoint.py --checkpoint checkpoints/silta_seqfish_sp17_ms17_e005.model
```

## Run inference

Prepare a non-pickle NPZ containing:

- `x`: float array `[spots, 351]`
- `expression_adjacency`: float array `[spots, spots]`
- `spatial_adjacency`: float array `[spots, spots]`
- optional `spot_ids`: Unicode array `[spots]`
- optional `cell_types`: Unicode array `[21]`

The row order of `locations.tsv` must match `x`; the table must contain numeric `x` and `y` columns.

```bash
python inference/predict.py --checkpoint checkpoints/silta_seqfish_sp17_ms17_e005.model --input-npz preprocessed_seqfish.npz --locations locations.tsv --temperature 0.525 --output predictions.tsv
```

## Evaluate

```bash
python evaluation/evaluate.py --prediction predictions.tsv --truth ground_truth_fractions.tsv --locations locations.tsv --type-counts type_counts.tsv --output-dir evaluation_output
```

The evaluator aligns spot and cell-type labels before writing `metrics.json`.

## Plot

The detailed input contract is in `plotting/input-contract.md`. After updating the paths in `plotting/example_manifest.yaml`, run:

```bash
python plotting/render_benchmark_figures.py --manifest plotting/example_manifest.yaml
```

The legacy manifest key `dacg_method` is retained for compatibility; set its value to `SILTA`.