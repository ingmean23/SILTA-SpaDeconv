# SILTA architecture used for OSM Benchmark B

The locked endpoint separates reference composition learning from target
adaptation.

## Stage I: source composition establishment

- Ordinary reference pseudo-spots.
- 80 epochs at learning rate 0.001.
- The spatial `E2/g` path is frozen and overridden to zero.
- Content latent `h` is stochastic during training with KL weight `1e-4`.

## Stage II: target adaptation

- Seven epochs at learning rate 0.003.
- Fraction replay weight 0.05.
- Linear handoff over epochs 1-5 with content learning-rate factor 0.02.
- Additive cooperative fusion: `h + 0.5 g`; no bilinear or direct-mode term.
- Fixed Gaussian distance-weighted spatial hypergraph attention, two heads,
  beta 1.0, gate 0.1.
- qNB observation weight increases linearly from 0.03 to 0.20 over epochs 1-5.
- Sample loss 0.5, spatial smoothness 0.005, BATV disabled.

## Released inference graph

- E1 is the recovered four-head vanilla dense-attention content expert.
- E2 is the recovered two-head fixed-Gaussian distance-weighted hypergraph
  expert; distances are normalized by the median six-neighbor scale.
- Inference uses posterior means `h_mu` and `g_mu` and additive cooperative
  fusion `h + 0.5 g`.
- Direct mode and bilinear terms are zero for this endpoint. Mode probabilities
  are still reconstructed for checkpoint compatibility but have zero direct
  readout weight.
- qNB and fraction replay are training constraints, not extra inference inputs.
- The package emits direct-checkpoint, C01-only, and C01+C04 fractions.

## Calibration

The deployment output applies two reference-only logit transformations:

1. C01 broad-group temperature calibration fitted on all reference-validation
   pseudo-spots.
2. C04 bounded, group-centered per-type bias and temperature calibration with
   bias cap 0.125, temperature cap 0.25, anchor 0.3, 100 steps, and LR 0.003.

No target fractions are read during model training, selection, or calibration.

## Locked dimensions

- Genes: 33.
- Cell types: 31.
- Cooperative fusion dimension: 512.
- Structured `h/g/m` dimensions: 192/192/64.
- Parameter count recorded for the locked model: 9,982,983.
