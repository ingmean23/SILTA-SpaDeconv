# Input schema

The renderer accepts JSON or YAML manifests. JSON has no optional dependency; YAML requires PyYAML.

## Manifest

```json
{
  "dataset": "HBC",
  "inputs": {
    "coordinates": "data/coordinates.csv",
    "markers": "data/marker_values.csv",
    "predictions": "data/predictions.csv",
    "metrics": "data/precomputed_metrics.csv"
  },
  "methods": ["DACG", "Tangram", "Stereoscope", "CARD"],
  "cell_types": [
    {
      "id": "Fibroblast",
      "label": "Fibroblast",
      "markers": ["COL1A1", "COL1A2", "DCN", "LUM", "COL3A1"]
    }
  ],
  "output_dir": "outputs/marker_spatial_comparison",
  "style": {
    "invert_y": true,
    "map_columns": 5,
    "cmap": "viridis",
    "marker_quantiles": [0.01, 0.99],
    "prediction_quantile": 0.99,
    "formats": ["png", "svg", "pdf", "tiff"],
    "dpi": 600
  }
}
```

Paths are resolved relative to the manifest. `methods[0]` must be `DACG`.

## Standard tables

Coordinates: `spot_id,x,y`.

Observed markers: `spot_id,cell_type,gene,value`.

Predicted fractions: `spot_id,method,cell_type,fraction`.

Precomputed metrics: `cell_type,method,pcc,spearman`.

Each configured marker and method must contain exactly the coordinate spot IDs. Marker values must be finite and nonnegative. Fractions must lie in `[0, 1]`; correlations must lie in `[-1, 1]`.

All tables may use `.csv`, `.tsv`, `.txt`, or `.parquet`. TSV/TXT files use tab separators.
