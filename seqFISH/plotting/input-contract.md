# Input Contract

## Manifest

Use one YAML manifest per benchmark figure suite.

```yaml
benchmark: "OSM Benchmark B"
ground_truth: "tables/ground_truth_fractions.tsv"
coordinates: "tables/locations.tsv"
metrics: "tables/method_summary.csv"
output_dir: "figures/standard_suite"
dacg_method: "SILTA"
spot_id_column: "spot_id"
coordinate_spot_id_column: "spot_id"
coordinate_columns: ["x", "y"]
invert_y: true
spatial_context: "Ground truth and predictions on the same 300 spots; split 42, algorithm/model seed 42."
methods:
  - name: "SILTA"
    prediction: "predictions/silta.tsv.gz"
    spot_id_column: "spot_id"
    valid: true
    n: 9
  - name: "Stereoscope"
    prediction: "predictions/stereoscope.tsv.gz"
    valid: true
    n: 9
  - name: "GMGAT"
    prediction: "predictions/gmgat.tsv.gz"
    valid: false
    invalid_reason: "structural validity"
    n: 3
```

All relative paths resolve from the manifest directory.

Optional keys:

| Key | Default | Meaning |
|---|---:|---|
| `row_sum_atol` | `1e-6` | Row-sum deviation treated as exact |
| `renormalize_atol` | `1e-3` | Largest row-sum deviation eligible for recorded normalization |
| `negative_atol` | `1e-10` | Tiny negative value that may be clipped to zero |
| `coordinate_spot_id_column` | `spot_id_column` | Coordinate ID column; set explicitly to `null` only for row-aligned coordinate files |
| `cell_type_colormap` | `turbo` | Matplotlib colormap used for the benchmark-local cell-type palette |
| `dominant_marker_size` | `20` | Square marker area in points squared |
| `dominant_render_mode` | `scatter` | Use `tiles` for edge-to-edge grid rectangles inferred from coordinate spacing |
| `pie_radius_scale` | `0.42` | Pie radius relative to median nearest-spot distance |
| `spatial_figure_width` | `16.0` | Width in inches of the four-column spatial figure |
| `spatial_panel_height` | `4.6` | Height budget in inches for each spatial panel row |
| `spatial_legend_height` | `2.4` | Height budget for a full-width legend row |
| `spatial_wspace` | `0.20` | Matplotlib horizontal spacing between spatial panels |
| `spatial_hspace` | `0.34` | Matplotlib vertical spacing between spatial panel rows |
| `method_color_overrides` | `{}` | Explicit project method colors as hex strings |
| `paired_composite` | `true` | Render paired dominant/pie tiles above the aggregate metric panels; set false only when an independent-only export is explicitly required |
| `composite_figure_width` | `19.2` | Width in inches of the paired composite |
| `composite_spatial_panel_height` | `4.45` | Height budget for each paired spatial row |
| `composite_legend_height` | `2.05` | Height budget for the shared cell-type legend |
| `composite_metric_panel_height` | `4.7` | Height budget for each metric row |
| `composite_metric_layout` | `2x2` | Aggregate metric layout: `2x2` preserves the default; `single_row` places RMSE/JSD/SSIM/PCC in one horizontal row |
| `composite_metric_wspace` | `0.32` | Horizontal gap between metric panels when `composite_metric_layout: single_row` |
| `composite_text_scale` | `1.0` | Multiplier for composite method titles, legend text, metric labels, values, and panel letters |
| `composite_bold_method_titles` | `false` | Use bold dark text for every dominant/pie method title, including invalid baselines |
| `composite_legend_fontsize` | `6.8` | Base legend font size before applying `composite_text_scale` |
| `composite_legend_ncol` | `8` | Number of shared legend columns; reduce this when enlarged labels need more horizontal room |
| `composite_legend_expand` | `false` | Expand the shared cell-type legend to the full composite content width |
| `composite_include_metrics` | `true` | Include aggregate metric panels; set `false` for a dominant/pie-only composite |
| `composite_show_title` | `true` | Show the composite figure title; set `false` when the manuscript caption provides the title |
| `composite_outer_wspace` | `0.06` | Horizontal gap between paired method tiles |
| `composite_outer_hspace` | `0.10` | Vertical gap between composite rows |
| `composite_tile_wspace` | `0.12` | Gap between dominant and pie maps within a method tile; values greater than `-0.5` are accepted for deliberately tight pairing |
| `composite_title_fontsize` | `16.0` | Composite figure title size in points |
| `composite_method_fontsize` | `12.6` | Method-name size above each dominant/pie pair in points |

Each method may set `spot_id_column` when a locked prediction uses a different
identifier header from the ground-truth table. The renderer records the resolved
column in the input audit; it does not rewrite the source prediction.

## Fraction Tables

- Supported extensions: `.csv`, `.csv.gz`, `.tsv`, `.tsv.gz`, `.txt`,
  `.txt.gz`.
- Require one `spot_id` column (or the configured `spot_id_column`).
- Treat every remaining column as a cell-type fraction.
- Require unique spot IDs and unique cell-type names.
- Require finite values and positive row sums.
- Reject values below `-negative_atol`.
- Require each prediction to contain exactly the GT spot and cell-type sets.
  The renderer reorders an exact set match to GT order and records that fact.

## Coordinates

Normally require the configured spot ID, x, and y columns. Spot IDs must
exactly match GT. For legacy coordinate files containing only x/y columns,
set `coordinate_spot_id_column: null`; the row count must equal GT and rows are
assigned to the exact GT order. Coordinates must be finite and unique by spot
ID. The renderer uses equal aspect ratio and, by default, reverses the y axis
to match image/grid indexing. Provenance records whether alignment was by ID
or position.

## Aggregate Metric Table

Require:

```text
method,RMSE,JSD,SSIM,PCC
```

The table may also contain `n` and `valid`, but manifest values take
precedence. Each manifest method must occur exactly once. Extra metric rows are
ignored only after being recorded in provenance.

Every valid method must have finite values for all four metrics. An invalid
method may use an empty value when a metric is undefined after collapse; its
bar is retained at the bottom and labeled `n/a`.

The ranked figures use aggregate table values. Representative per-spot RMSE
and dominant agreement are recomputed and written to
`spatial_panel_metrics.csv` and provenance, but spatial panel titles contain
only the method name. Invalid status remains encoded in ordering, styling,
normalized tables, and provenance, not in visible figure text.

## Validity Semantics

`valid: false` means the method is shown for audit but excluded from formal
valid-method ordering. It does not permit malformed input. Invalid methods are
placed after all valid methods and use diagonal hatching, while their visible
label remains the method name only.

## Relative Improvement

For RMSE and JSD:

```text
(baseline - SILTA) / baseline * 100
```

For SSIM and PCC:

```text
(SILTA - baseline) / baseline * 100
```

Negative values must remain visible.

## Required Deliverables

```text
dominant_maps.{png,tiff,pdf,svg}
spot_pie_maps.{png,tiff,pdf,svg}
RMSE.{png,tiff,pdf,svg}
JSD.{png,tiff,pdf,svg}
SSIM.{png,tiff,pdf,svg}
PCC.{png,tiff,pdf,svg}
spatial_panel_metrics.csv
metric_values_normalized.csv
method_manifest_normalized.csv
cell_type_colors.csv
method_colors.csv
figure_provenance.json
paired_spatial_metric_composite.{png,tiff,pdf,svg}  # when enabled
```
