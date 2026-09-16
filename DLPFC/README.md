# SILTA: DLPFC inference and marker evaluation

**SILTA** stands for **Separating composition learning from target adaptation
in spatial transcriptomic deconvolution**.

This folder contains the inference-only code, one locked best single-seed DLPFC
checkpoint, the public direct-marker evaluator, and the plotting code required
to reproduce the reported DLPFC result. It intentionally excludes training and
search runners, raw or restricted biomedical data, private target-fraction
truth, and machine credentials.

## Locked endpoint

- Dataset: DLPFC section 151673 (ds11)
- Training arm: MALL_fixedG5_identity
- Selected epoch: 45
- Highlighted prespecified output column: Mix_5
- Direct-marker PCC: 0.0895226972
- Direct-marker Spearman: 0.1338798382
- Marker-consensus PCC: 0.1950430414
- Marker-consensus Spearman: 0.2283773259
- Locked prediction SHA256: 1d3abace20ea2892437b382608bd52c889d39b26226631608bd86976e0fa57ac
- Release checkpoint SHA256: 31f72b63d559169f784f997711dad45d74159968b5bf29c23d73fc40e289fcba

The checkpoint was re-materialized from the locked model as a tensor-only
inference state dictionary. It was not substituted with an older checkpoint.
The reproduction report in metadata/reproduction_report.json records the spot
order, cell-type order, and numerical tolerance check.

## Data acquisition

Data are not redistributed. Obtain DLPFC section 151673, a matching annotated
single-cell reference, and spatial coordinates under the original providers'
terms. The spatialLIBD project is one public entry point:
https://research.libd.org/spatialLIBD/

The supplied checkpoint requires the exact 15,844-gene order in
configs/gene_order.json. The single-cell reference is needed to reconstruct
the original preprocessing contract or train new models, but it is not read by
the released inference command.

## Input contracts

### Spatial transcriptomics H5AD

- Rows are spots and columns are genes.
- obs_names are unique spot identifiers.
- var_names are unique gene symbols.
- All genes in configs/gene_order.json must be present.
- Raw non-negative counts are recommended. Already transformed input is
  accepted only when it follows the original preprocessing convention.
- Missing genes are an error; the release does not silently impute zeros.

### Coordinate table

A tab-separated table with one row per retained ST spot. Include a spot_id,
barcode, or other string identifier column plus a recognized coordinate pair:
x/y, coor_X/coor_Y, scaled_x/scaled_y, array_row/array_col,
pxl_row/pxl_col, or imagecol/imagerow.

### Single-cell reference

For provenance and retraining outside this release:

- H5AD rows are cells and columns are genes.
- Cell labels are stored in obs["celltype"] or obs["cell_type"].
- Labels follow the 33-type order in configs/model.json.
- Gene symbols match the ST input.

### Prediction CSV

The first column is spot_id. The remaining 33 columns follow
configs/model.json exactly and contain non-negative row-normalized fractions.

### Marker panel

configs/marker_panel.json contains five fixed markers for each of the 33 cell
types. Marker genes remain ordinary model inputs. Evaluation directly compares
each predicted fraction column with observed ST marker expression; it does not
evaluate a reconstructed expression matrix.

## Installation

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

Run commands from this DLPFC folder.

## Inference

~~~bash
python -m silta.inference \
  --checkpoint checkpoint/silta_dlpfc_mix5_best_single_seed.model \
  --model-config configs/model.json \
  --st data/st.h5ad \
  --coordinates data/coordinates.tsv \
  --output results/silta_ds11.csv \
  --device cpu
~~~

Use --device cuda when a compatible GPU is available. Hypergraph reductions
can differ by approximately 2e-4 at isolated entries across device and PyTorch
versions; aggregate metrics and the selected endpoint are unchanged.

## Direct-marker evaluation

Copy configs/evaluation.json and adjust only its relative ST and coordinate
paths when using a different directory layout.

~~~bash
python -m evaluation.direct_marker_colocalization.evaluate \
  --config configs/evaluation.json \
  --slice ds11 \
  --method SILTA \
  --prediction results/silta_ds11.csv \
  --output-dir results/direct_marker \
  --no-controls
~~~

Main outputs are per_gene.csv, per_cell_type.csv, spot_marker_proxies.csv,
summary.json, and run_manifest.json. The main per-type fields are
mean_marker_pcc_strict and mean_marker_spearman_strict. Marker expression is
normalized as log1p(CPM) using the full raw ST library size.

## Marker comparison figure

For SILTA alone:

~~~bash
python -m plotting.marker_comparison.plot \
  --st data/st.h5ad \
  --coordinates data/coordinates.tsv \
  --panel configs/marker_panel.json \
  --cell-type Mix_5 \
  --prediction SILTA=results/silta_ds11.csv \
  --metrics SILTA=results/direct_marker/per_cell_type.csv \
  --output-dir results/figures
~~~

To recreate a multi-method paper figure, append one matched pair for every
baseline:

~~~bash
  --prediction STRIDE=results/stride.csv \
  --metrics STRIDE=results/stride_eval/per_cell_type.csv
~~~

SILTA must remain the first prediction. Every marker is scaled independently
with q01-q99 clipping. All method maps share one pooled q99 fraction scale.
The plotter exports PNG, PDF, SVG, TIFF, and source-data CSV files without
smoothing, interpolation, or spot removal.

## Verification

~~~bash
python tests/offline_smoke_test.py
python tools/audit_release.py
~~~

The audit checks for credential-like strings, machine-specific paths,
forbidden data and serialization files, exactly one checkpoint, safe
weights-only deserialization, strict architecture loading, the MIT license,
third-party dependency inventory, and file hashes.

## Scope

This is a single locked DLPFC inference release. It supports transparent
reproduction of the reported prediction and marker-colocalization evidence; it
does not provide fraction-level ground truth or claim that marker expression is
a direct cell-fraction measurement.

