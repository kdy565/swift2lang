# RSA: SwiFT vs GPT-2 on LeBel task-fMRI

## Setup
- Subject: `UTS01`
- Story: `againstthewind`
- N TRs: `150` (190 total, 150 valid after temporal window)
- HRF delay: `5` TRs
- SwiFT temporal window: `20` TRs
- Language model: GPT-2 small (contextual, left-truncated to 200 tokens)
- Permutation iterations: `10000`

## Results

| Condition | Spearman r | p-value | Null std | Significant? |
|---|---|---|---|---|
| SwiFT vs GPT-2 | 0.0351 | 0.0002 | 0.0095 | Yes |
| Random-init SwiFT vs GPT-2 | 0.0228 | 0.0082 | 0.0095 | Yes |
| Raw BOLD voxels vs GPT-2 | 0.0303 | 0.0002 | 0.0092 | Yes |
| SwiFT vs GPT-2 (shifted +50 TRs) | 0.0562 | 0.0001 | 0.0143 | Yes |

## Decision

- [ ] **r near zero, not above random** → SwiFT representations uninformative about stimulus semantics.
- [ ] **r significantly above zero and above voxel baseline** → SwiFT features carry semantic info worth probing/fine-tuning.
- [ ] **r positive but below voxel baseline** → SwiFT compresses away useful info; ridge on raw voxels preferred.
- [x] **Time-shift control fails** → The "signal" is temporal autocorrelation, not semantics.

## Interpretation

All conditions produce significant positive Spearman r, but the time-shift control (GPT-2 shifted by +50 TRs) gives an **even higher** correlation (0.056) than the aligned condition (0.035). This is the definitive diagnostic:

- If the RSA reflected genuine semantic alignment, shifting the language embeddings by 50 TRs (where story content is completely different) should destroy the correlation.
- The fact that it persists and even increases means the RDM structure is dominated by **slow temporal autocorrelation** — BOLD has high within-block similarity (nearby TRs are similar), and GPT-2 embeddings also have a smooth structure (nearby sentences are topically related). Any two slowly varying time series will produce correlated RDMs via this mechanism.

This is a well-known confound in RSA (Kriegeskorte 2008; see also "RSA pitfall: autoregressive structure inflates correlations").

The random-init SwiFT also showing significant correlation (0.023) is consistent with this interpretation — even random spatial filtering preserves temporal autocorrelation from the input BOLD.

## Probe pipeline status
The `truncation_side` bug does NOT affect the linear probe pipeline (`scripts/01_extract_features.py`, `02_train_probe.py`, `03_eval.py`). That pipeline uses `build_gpt2_embeddings` which tokenizes each word in isolation — truncation never fires for single words. The bug was limited to the RSA contextual embedding extraction in `extract_lang.py`, which is now fixed.

## Figures
- `figures/rdm_heatmaps.png` — RDM comparison across conditions
- `figures/rsa_bar_chart.png` — Bar chart of Spearman r with significance
