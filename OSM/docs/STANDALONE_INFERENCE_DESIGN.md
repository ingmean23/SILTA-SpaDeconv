# Standalone checkpoint inference design

The release uses a minimal inference-only implementation rather than copying
the historical training and search runtime.

## Boundary

The runtime starts from an explicitly prepared OSM bundle containing the
model-scale expression matrix, expression and spatial adjacency matrices,
coordinates, reference signature, and ordered identifiers. It does not read
target ground truth or regenerate training pseudo-spots.

## Compatibility

`silta_osm.model.SILTAModel` keeps the original module attribute names so the
released tensor-only checkpoint loads with `strict=True`. Release-facing
classes use SILTA names; state-dict keys remain unchanged for compatibility.
The implemented forward contains only the operations needed to obtain the
checkpoint logits and fractions:

1. vanilla dense E1 content attention;
2. fixed-Gaussian distance-aware hypergraph E2 attention;
3. structured `h`, `g`, and supervised mode representations;
4. additive cooperative `h + 0.5 g` readout.

Modules whose parameters are present in the checkpoint but are inactive in
this endpoint are instantiated for strict state compatibility and are not
executed by the inference forward.

## Verification

The release tests require all 323 checkpoint tensors to load without missing
or unexpected keys, run a finite synthetic graph forward, verify normalized
fractions, exercise the prepared-bundle contract, and run the inference CLI.
