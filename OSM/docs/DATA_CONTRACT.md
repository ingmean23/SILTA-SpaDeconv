# OSM prepared-input contract

No biological data are distributed in this folder. Users must obtain every
input from an authorized source and confirm its license.

## Private-data-dependent boundary

The bundled model consumes a prepared NPZ, not an arbitrary raw H5AD file. Raw
counts, feature selection, normalization, gene alignment, and graph building
must reproduce the registered OSM endpoint upstream. This release does not
infer those choices from private data. `scripts/prepare_inputs.py` only checks
and packs already prepared, explicitly labelled tables.

The NPZ must contain these arrays:

| Key | Shape | Meaning |
|---|---|---|
| `expression` | `[spots, 33]` | Finite, nonnegative model-scale target expression in registered gene order. |
| `expression_adjacency` | `[spots, spots]` | Finite, nonnegative expression graph. Retained for endpoint compatibility; the released E1 is dense attention. |
| `spatial_adjacency` | `[spots, spots]` | Finite, nonnegative spatial graph used by E2. |
| `coordinates` | `[spots, 2]` | Physical x/y coordinates in spot order. |
| `reference_signature` | `[31, 33]` | Nonnegative reference-only type signature in the exact ontology/gene order. Validated for provenance; it is not read by the checkpoint forward. |
| `spot_ids` | `[spots]` | Unique strings. |
| `gene_names` | `[33]` | Unique strings in expression/signature order. |
| `cell_types` | `[31]` | Unique strings matching the serialized C01+C04 ontology exactly. |

Both adjacency diagonals are set to one at tensor conversion. Physical
distances are recomputed from coordinates and divided by the median of the six
nearest positive distances, matching the recovered endpoint.

Input TSV/CSV conventions for `scripts/prepare_inputs.py`:

- the first column is the row identifier;
- expression rows are spots and columns are genes;
- both adjacency rows and columns equal the expression spot order;
- coordinates have exactly two columns and the same spot order;
- signature rows are cell types and columns equal the expression gene order.

## Prediction outputs

`scripts/run_inference.py` writes four labelled, gzip-compressed TSV tables:

- `checkpoint_logits.tsv.gz`: pre-calibration logits;
- `checkpoint_fractions.tsv.gz`: direct checkpoint softmax;
- `pre_c04_fractions.tsv.gz`: C01-only fractions;
- `silta_fractions.tsv.gz`: final C01+C04 fractions.

Fraction tables have one row per unique spot and 31 ontology columns. Values
must be finite, nonnegative, and sum to one per row.

## Evaluation-only inputs

- `ground_truth_fractions.tsv`: same fraction-table contract as prediction.
- `type_counts.tsv`: columns `celltype` and `target_cells` for rare-type AUPRC.
- `locations.tsv`: coordinates used to construct the evaluation grid.

Ground truth must remain outside preprocessing, inference, training, model
selection, and calibration. It is read only by the final evaluator.

## Figure inputs

The YAML contract is demonstrated in `figures/example_manifest.yaml`.
Ground truth, coordinates, predictions, and aggregate metrics must belong to
the same registered benchmark protocol.
