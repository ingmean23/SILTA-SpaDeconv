# SILTA: Melanoma ds6 analysis code

This folder contains source code, configuration, evaluation utilities, and
plotting resources for the **Melanoma ds6 T-cell** analysis. Raw SC/ST data and
pseudo-spots are not redistributed.

## Model architecture

For the Melanoma analysis, DACG separates reference composition identity from
target spatial topology:

```text
gene expression ---- E1 ---- h (composition/content) ---+
                                                        +-- structured z
ST expression graph -- E2 ---- g (spatial topology) ----+       |
                                                                v
                                                     fraction readout
```

The provided configuration uses:

- a structured stochastic `h/g` latent with posterior means at inference;
- the mixed spatial encoder;
- a concatenation fraction readout;
- PNP3 pseudo-spots for reference composition training;
- pseudo-side `g` freezing/zeroing, because pseudo-spots have no real topology;
- direct mode residual disabled (`m_fusion_scale=0`);
- ZIG disabled;
- Stage2 domain adversarial and ST reference-reconstruction losses disabled.

Stage1 mode supervision is used as a semantic curriculum for the expression
branch, not as a direct residual in the final fraction prediction. The compact
inference contract is in
`configs/inference_ds6_T_seed42_epoch2.json`; the sanitized full experiment
provenance is in
`configs/training_provenance_ds6_T_seed42_epoch2.json`.

## Directory layout

```text
Melanoma/
├── configs/
├── dacg/                                  # exact inference dependency closure
├── expected/ds6_T/                        # evaluation inputs and tables
├── plotting/                              # publication comparison renderer
├── scripts/
│   ├── infer_ds6_t.py
│   ├── evaluate_ds6_t.py
│   ├── verify_release.py
│   └── direct_marker_colocalization/
├── DATA_MANIFEST.json
├── RESULTS.json
└── MANIFEST.sha256
```

## Environment

The exact validated environment used Python 3.11.9, PyTorch 2.4.0 and
DGL 2.4.0+cu118. Install the pinned Python packages:

```bash
python -m pip install -r requirements.txt
```

The DGL CUDA wheel may need to be installed from the DGL wheel repository for
the CUDA version on the target machine.

## Required data

Place or reference the original ds6 inputs (Melanoma 4, replicate 1):

```text
st.h5ad
sc.h5ad
pseudo_optimize_fm_pnp_pnp3.h5ad
ST_mel4_rep1_counts.tsv
ST_mel4_rep1_cor.txt
```

Their required SHA256 values are recorded in `DATA_MANIFEST.json`. The SC file
is retained as provenance for the analysis configuration.

## Evaluate predictions and generate figures

The frozen input-only panel contains 26 literature markers. These genes were
ordinary expression inputs but were excluded from dedicated marker supervision
and model selection. Evaluation directly compares each predicted cell fraction
with observed ST marker expression:

\\[
\\mathrm{PCC}_{k,g}=\\mathrm{corr}(P_{\\cdot k},Y_{\\cdot g}).
\\]

Run:

```bash
python scripts/evaluate_ds6_t.py \\
  --prediction reproduced/ds6_fractions.csv \\
  --st /path/to/st.h5ad \\
  --counts /path/to/ST_mel4_rep1_counts.tsv \\
  --coordinates /path/to/ST_mel4_rep1_cor.txt \\
  --output-dir reproduced/evaluation
```

This writes `summary.json`, `per_cell_type.csv`, `per_gene.csv`,
spot-level marker proxies, and spatial figures. The optional
`plotting/render_marker_benchmark.py` renderer creates publication panels
from a manifest containing DACG and baseline predictions.

## Integrity and publication notes

- No API keys, authentication tokens, host addresses, or private absolute
  paths are included.
- No repository license was present when this folder was assembled; see
  `LICENSE_REQUIRED.md` before public distribution.
