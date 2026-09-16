# C01 + C04 state contract

The calibration bundle is a PyTorch-serialized mapping:

```text
schema_version: 1
kind: silta_osm_c01_c04
cell_types: list[str]
group_index: int64 vector of length K
c01:
  mode: group_temperature
  state: TypeLogitCalibrator state_dict
c04:
  bias_cap: 0.125
  temperature_cap: 0.25
  state: GroupCenteredShapeCalibrator state_dict
```

The cell-type order must exactly match base logits and final output columns.
The original experiment did not persist this bundle; it must be recovered by
rerunning the locked reference-only calibration once and saving both transfer
modules. CSV loss histories are insufficient to reconstruct the parameters.
