# DACG Melanoma ds6 T checkpoint release

This folder is a self-contained checkpoint-inference and evaluation release for
the locked **Melanoma ds6 T-cell** result. It contains only source code,
configuration, a verified model checkpoint, frozen evaluation artifacts, and
plotting utilities. Raw SC/ST data and pseudo-spots are not redistributed.

## Released result

The included checkpoint is the retained and reproducible
`C0 / epoch 2 / seed 42` endpoint:

| Metric | DACG | Best fixed baseline | Historical DACG | Status |
|---|---:|---:|---:|---|
| T-cell marker PCC | 0.069906 | 0.046709 | 0.038696 | exceeds both |
| T-cell marker Spearman | 0.065818 | 0.040183 | 0.053329 | exceeds both |

The fixed comparison winner for both baseline thresholds was SPOTlight. The
checkpoint passed the recorded structural-health gate with no collapse flags.
These are **cell-type-level direct marker colocalization** results. Melanoma
does not provide spot-level fraction ground truth, so this release does not
claim fraction-level accuracy.

Only an exact checkpoint/evidence pair is included. Exploratory endpoints whose
matching checkpoint did not survive experiment cleanup are intentionally
excluded from this release.

## Model architecture

For the locked checkpoint, DACG separates reference composition identity from
target spatial topology:

```text
gene expression ---- E1 ---- h (composition/content) ---+
                                                        +-- structured z
ST expression graph -- E2 ---- g (spatial topology) ----+       |
                                                                v
                                                     fraction readout
```

The released endpoint uses:

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
├── checkpoints/ds6_T_seed42_epoch2/model.model
├── configs/
├── dacg/                                  # exact inference dependency closure
├── expected/ds6_T/                        # locked prediction and evaluator tables
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

Their required SHA256 values are recorded in `DATA_MANIFEST.json`. The SC
file is provenance for training and is not required for checkpoint inference.

## Verify packaged artifacts

This check requires no raw dataset:

```bash
python scripts/verify_release.py
```

It verifies the checkpoint hash, locked fraction-table hash, and the frozen
T-cell PCC/Spearman values.

## Reproduce checkpoint inference

From this directory:

```bash
python scripts/infer_ds6_t.py \\
  --st /path/to/st.h5ad \\
  --pseudo /path/to/pseudo_optimize_fm_pnp_pnp3.h5ad \\
  --coordinates /path/to/ST_mel4_rep1_cor.txt \\
  --output reproduced/ds6_fractions.csv \\
  --device cuda:0
```

With the pinned environment and matching inputs, the output SHA256 is:

```text
cf7cde5149f00116c3d17ded16af2f3b00801688311b8807aa9e7ff12ecdb0c4
```

On a different numerical stack, use `--skip-hash-check` and compare the
result numerically with `expected/ds6_T/fractions.csv`.

## Reproduce evaluation and figures

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

- `MANIFEST.sha256` covers every released file except itself.
- `.gitattributes` configures the checkpoint for Git LFS.
- No API keys, authentication tokens, host addresses, or private absolute
  paths are included.
- The parent checkpoint used before the released Stage2 endpoint is recorded
  by hash for provenance but is not needed for inference and is not included.
- No repository license was present when this folder was assembled; see
  `LICENSE_REQUIRED.md` before public distribution.
