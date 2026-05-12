# SwiFT → LeBel: Negative Result Report

**Project status: Closed (February 2025)**

## Summary

We tested whether a **pre-trained SwiFT** (Swin 4D Transformer, pre-trained on
resting-state fMRI from UK Biobank) carries stimulus-locked semantic
information when applied to **LeBel et al. 2023** task-fMRI (subjects listening
to naturalistic stories). Two independent sanity checks were run. Both returned
negative results — SwiFT features do **not** capture word-level semantic
structure beyond raw voxel baselines.

**The project is being closed here.** A planned third stage (fine-tuning SwiFT
on task-fMRI) was not pursued, as the pre-trained representations show no
evidence of carrying the type of stimulus information that fine-tuning would
amplify.

---

## Experiment 1: Linear Probe (word-level decoding)

**Goal:** Train a ridge regression probe on SwiFT features to predict
word-level GPT-2 embedding for each TR. Compare against a voxel-baseline
(ridge on raw BOLD).

**Methods:**
- Subject: UTS01, Stories: againstthewind (train), sweetaspie (eval)
- SwiFT window: 20 TRs, HRF delay: 5 TRs
- Target: Isolated word GPT-2 embedding (first-token hidden state)
- Probe: Ridge regression with 6-fold cross-validation on alpha grid
- Baseline: Ridge on all brain-masked voxels (~500k) with L2 regularization

**Results:**

| Model | Pearson r | Top-1 Acc | Top-5 Acc |
|---|---|---|---|
| SwiFT probe | 0.0006 | 0.0015 | 0.0059 |
| Voxel ridge baseline | 0.0079 | 0.0089 | 0.0089 |

Chance-level decoding. Neither model beats trivial baselines. The voxel
baseline marginally outperforms SwiFT, suggesting SwiFT's compression loses
information rather than denoising.

**Conclusion:** Linear probe on pre-trained SwiFT features cannot decode word
identity — the representations are not linearly separable by word-level
semantic categories.

---

## Experiment 2: RSA (Representational Similarity Analysis)

**Goal:** Test whether the *geometric structure* of SwiFT feature RDMs
correlates with GPT-2 contextual embedding RDMs. RSA can detect structure
that a linear probe might miss (e.g., non-linear, non-separable, or very weak).

**Methods:**
- Single story (againstthewind), N = 150 TRs
- SwiFT features: pooled final layer (D × H × W × T → 288-dim)
- Language embeddings: Contextual GPT-2 (left-truncated to 200 tokens,
  last-token hidden state, 768-dim)
- Cosine-distance RDMs → Spearman correlation
- Permutation test: 10,000 iterations (shuffle RDM upper-tri entries)
- Control conditions: random-init SwiFT, raw BOLD voxels, **time-shift (+50 TRs)**

**Results:**

| Condition | Spearman r | p | Decision |
|---|---|---|---|
| SwiFT vs GPT-2 | 0.0351 | 0.0002 | Significant but artefactual |
| Random SwiFT vs GPT-2 | 0.0228 | 0.0082 | Significant (autocorrelation leak) |
| Raw BOLD vs GPT-2 | 0.0303 | 0.0002 | Significant (baseline) |
| SwiFT vs GPT-2 (shift +50) | **0.0562** | 0.0001 | **Higher than aligned** |

**Critical finding:** The time-shift control (+50 TRs) produced a *higher*
correlation than the correctly aligned condition. If the RSA reflected genuine
semantic alignment, shifting the language embeddings by 50 TRs (where story
content is completely different) should destroy the correlation. The fact that
it persists and increases proves that the RDM structure is dominated by
**slow temporal autocorrelation** in both BOLD and language embeddings.

**Methodological note:** The standard permutation test (shuffle RDM entries)
is anti-conservative when the RDM has smooth structure from temporal
autocorrelation. Breaking the RDM's internal dependencies inflates the null
distribution's variance only partially, leading to false positives. A
block-permutation or within-block shuffle is needed for valid inference.

**Conclusion:** The apparent RSA signal is a temporal autocorrelation
artefact, not semantic structure. Pre-trained SwiFT features do not carry
stimulus-locked semantic information.

---

## Overall Decision

Both experiments converge on the same conclusion: **Pre-trained SwiFT
features are not useful for word-level semantic decoding on LeBel
task-fMRI.** The project is closed here, for three reasons:

1. **Linear probe** → chance-level decoding for both SwiFT and voxels.
2. **RSA** → apparent signal invalidated by time-shift control: it is
   temporal autocorrelation, not semantics.
3. **Effort-to-benefit ratio** → The signal is so weak that even if
   fine-tuning recovered something, the effect size is unlikely to be
   practically useful.

---

## Methodological contribution

The RSA time-shift control (condition 4 in `compute_rsa.py`) is a cheap
diagnostic that should be standard in fMRI RSA studies using continuous
naturalistic stimuli. It revealed that the significant-but-small RSA
correlation was entirely driven by temporal smoothness rather than
semantic alignment. This is a known confound (Kriegeskorte 2008,
Kriegeskorte & Kievit 2013) but is rarely tested directly.

---

## Repository structure

```
swift2lang/
├── REPORT.md                      ← This file
├── AGENTS.md                      # Environment and workflow rules
├── .gitignore                     # Ignores data, cache, checkpoints
├── configs/config.yaml            # Probe configuration
├── data/
│   ├── lebel/                     # Raw BOLD (gitignored)
│   ├── features/                  # Cached SwiFT/GPT-2 features (gitignored)
│   ├── probes/                    # Trained ridge models (large pkl gitignored)
│   └── results/eval_results.json  # Final probe evaluation results
├── scripts/
│   ├── 01_extract_features.py     # SwiFT + GPT-2 extraction for probe
│   ├── 02_train_probe.py          # Ridge regression training
│   ├── 03_eval.py                 # Probe evaluation on held-out story
│   └── prepare_lebel_data.py      # BOLD preprocessing pipeline
├── src/
│   ├── data/lebel_loader.py       # fMRI loading, word alignment, GPT-2 embeddings
│   ├── models/swift_wrapper.py    # SwiFTEncoder (frozen inference wrapper)
│   ├── probe/ridge.py             # RidgeRegressionProbe (sklearn pipeline)
│   └── baseline/voxel_ridge.py    # Voxel baseline (ridge on raw BOLD)
├── rsa_swift_lebel/
│   ├── extract_features.py        # SwiFT extraction (trained + random-init)
│   ├── extract_lang.py            # Contextual GPT-2 embedding extraction
│   ├── compute_rsa.py             # RDMs + permutation tests + time-shift control
│   ├── plot_results.py            # RDM heatmaps + bar chart
│   ├── report.md                  # RSA-specific report (filled)
│   ├── cache/                     # .npy feature + RDM files
│   └── figures/                   # PNG outputs
└── SwiFT/                         # Upstream submodule (not in scope)
```

## How to reproduce

```bash
# 1. Prepare data
python scripts/prepare_lebel_data.py --subject UTS01 --stories againstthewind sweetaspie

# 2. Linear probe pipeline
conda run -n py39 python scripts/01_extract_features.py --config configs/config.yaml
conda run -n py39 python scripts/02_train_probe.py --config configs/config.yaml
conda run -n py39 python scripts/03_eval.py --config configs/config.yaml

# 3. RSA pipeline
conda run -n py39 python rsa_swift_lebel/extract_features.py --story againstthewind
conda run -n py39 python rsa_swift_lebel/extract_lang.py --story againstthewind
conda run -n py39 python rsa_swift_lebel/compute_rsa.py --story againstthewind --n-perm 10000
conda run -n py39 python rsa_swift_lebel/plot_results.py
```
