# SILTA: HBC Reproduction Package

SILTA stands for **Separating composition learning from target adaptation in
spatial transcriptomic deconvolution**. This folder is the minimal HBC release:
model code, one locked SOTA checkpoint, inference, evaluation, and plotting.

本目录是 HBC 数据集的独立复现包。它不包含原始数据、训练日志、API key、身份令牌、
W&B 配置或本地/服务器绝对路径。

## What is included

```text
HBC/
|-- checkpoints/
|   `-- SILTA_HBC_TrackB_J0.model
|-- configs/
|   |-- model_config.json
|   |-- pathology_protocol.json
|   |-- direct_marker_config.json
|   `-- hbc_g5_panel.json
|-- src/                         # SILTA model modules needed by the checkpoint
|-- experiments/                 # Leakage-controlled HBC pathology evaluator
|-- marker_benchmark/            # Direct marker-fraction colocalization evaluator
|-- scripts/
|   |-- verify_checkpoint.py
|   |-- infer.py
|   |-- evaluate_pathology.py
|   |-- evaluate_markers.py
|   |-- plot_fractions.py
|   `-- plot_pathology_summary.py
|-- reference/
|   |-- checkpoint_manifest.json
|   `-- expected_spatial_trackb_metrics.csv
|-- requirements.txt
`-- SHA256SUMS
```

The checkpoint is the locked **J0 representative endpoint** from the HBC
Track-B search. It was selected in folds 0, 2, and 3 of the original sealed
search. Evaluated alone as one fixed model across all five folds, it obtains:

| Metric | Representative checkpoint |
|---|---:|
| Balanced accuracy | 0.766813 |
| Macro-F1 | 0.754318 |
| Macro-AUROC | 0.933683 |

The published Track-B result used leakage-controlled nested selection over the
complete candidate search. Its selected folds happened to use endpoint profile
`J0` three times and `J1` twice:

| Metric | Sealed complete-search result |
|---|---:|
| Balanced accuracy | 0.784729 |
| Macro-F1 | 0.772011 |
| Macro-AUROC | 0.938278 |

The original grid intentionally did not retain every candidate checkpoint. The
`J1` weights and the complete search pool are therefore not included. The
sealed aggregate above is retained only as provenance in
`reference/expected_spatial_trackb_metrics.csv`; it is **not** reproducible from
the single checkpoint in this package. `configs/trackb_candidate_pool.json`
records the selected endpoint profiles. `scripts/evaluate_pathology_pool.py`
can evaluate any newly regenerated fixed candidate pool, but a two-endpoint
rerun is not equivalent to the original complete-search selection. Do not
attribute the sealed complete-search score to this checkpoint alone.

The direct-marker code is included to reproduce the biological audit from any
compatible prediction. Marker evaluation is not used to select the Track-B
checkpoint, and no fraction-level ground truth claim is made for HBC.

## Architecture

```text
expression x --> E1/content encoder --> h (composition identity) ----+
                                                                    |
spatial graph --> E2/hypergraph attention --> g (target topology) --+--> cooperative fusion
                                                                    |       h - lambda_g*g
E1 --> frozen mode prior --> m (semantic auxiliary; direct m = 0) ---+       - lambda_x*(h*g)
                                                                            |
                                                                            v
                                                                    shared fraction readout
                                                                            |
                                                                            v
                                                                  13 cell-type proportions
```

The released endpoint uses:

- Stage-I endpoint 80 for composition identity.
- Stage isolation with a five-step protected spatial handoff.
- Deterministic `h` and `g` posterior means.
- Hypergraph-attention E2.
- Cooperative direct fusion with `lambda_g=1.0`, `lambda_x=0.25`.
- Direct `m` residual disabled.
- Readout adapter and C01 calibration disabled.
- qNB gradient ratio `0.05` during target adaptation.
- Raw model fractions at inference; no visual smoothing or probability refinement.

The internal class remains named `DACGModel` solely to preserve checkpoint key
compatibility. The public method name is **SILTA**.

## Data contract

Raw HBC data are not redistributed. Prepare these files yourself:

```text
data/st.h5ad
data/metadata.tsv
data/cor.txt
```

For exact checkpoint inference:

- `st.h5ad` must contain all 4,784 genes listed in `configs/model_config.json`.
- `metadata.tsv` must cover every ST spot and contain spatial coordinates.
- Pathology evaluation additionally requires `old_annot_type`,
  `old_fine_annot_type`, `scaled_x`, and `scaled_y`.
- Direct-marker evaluation reads observed ST expression from `data/st.h5ad` and
  coordinates from `data/cor.txt`.
- Prediction CSVs must contain `spot_id` plus the 13 cell-type fraction columns
  in `configs/model_config.json`.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA version when GPU inference
is desired. CPU inference is supported but slower. DGL is not required by the
released hypergraph-attention checkpoint; it is only used by a legacy block.

## 1. Verify the checkpoint

```bash
python scripts/verify_checkpoint.py
```

Expected SHA256:

```text
6226c6938db301e7c7ab3c69ce185ab80856ade477f91c24b9c1255460cb007b
```

## 2. Run inference

```bash
python scripts/infer.py \
  --st data/st.h5ad \
  --metadata data/metadata.tsv \
  --output results/silta_hbc_fractions.csv \
  --device cuda
```

Use `--device cpu` when CUDA is unavailable.

## 3. Reproduce the pathology benchmarks

This command evaluates both tracks with the locked protocol:

- Random-Direct: stratified random splits using fractions only.
- Track-B: five leakage-controlled spatial grouped folds with nested selection,
  multi-scale neighborhood composition, and no absolute coordinates as model
  features.

```bash
python scripts/evaluate_pathology.py \
  --prediction results/silta_hbc_fractions.csv \
  --metadata data/metadata.tsv \
  --output-dir results/pathology
```

To evaluate a newly regenerated fixed candidate pool under the same nested-CV`r`ncode path (this does not reconstruct the missing complete historical pool):

```bash
python scripts/evaluate_pathology_pool.py \
  --candidate J0=results/J0_prediction.csv \
  --candidate J1=results/J1_prediction.csv \
  --metadata data/metadata.tsv \
  --output-dir results/pathology_pool
```

Create a compact summary figure:

```bash
python scripts/plot_pathology_summary.py \
  --evaluation-dir results/pathology \
  --output results/pathology_summary.png
```

## 4. Reproduce direct-marker evaluation

```bash
python scripts/evaluate_markers.py \
  --prediction results/silta_hbc_fractions.csv \
  --output-dir results/direct_markers
```

The evaluator reports per-gene and per-cell-type PCC/Spearman values, strict
macro summaries, negative controls, and spatial comparison maps. Marker genes
are observed ST expression, not inferred unseen genes.

## 5. Plot fractions

```bash
python scripts/plot_fractions.py \
  --prediction results/silta_hbc_fractions.csv \
  --metadata data/metadata.tsv \
  --output-dir results/fraction_maps
```

## Reproducibility and scope

- Target pathology labels are not used to train the SILTA checkpoint.
- Track-B uses coordinates only to construct neighborhoods and spatial folds;
  absolute coordinates are not classifier features.
- The downstream pathology converter is supervised by pathology labels under a
  sealed nested-CV protocol. It evaluates pathology information retained by the
  predicted composition; it is not cell-fraction ground truth.
- `reference/expected_spatial_trackb_metrics.csv` records the original locked
  complete-search fold results; `reference/representative_checkpoint_metrics.json`
  records the independently rerun single-checkpoint result.
- Small numeric differences can occur across PyTorch/hardware builds, but the
  checkpoint hash and input ordering must match exactly.

## Security and privacy

This package intentionally excludes `.env` files, API credentials, SSH keys,
authentication tokens, W&B run metadata, private server paths, raw patient data,
and generated training logs. Run `python scripts/audit_release.py` before making
any future copy public.


