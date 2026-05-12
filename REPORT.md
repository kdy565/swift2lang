# SwiFT → LeBel: Negative Result Report

**Project status: Closed (May 2026)**

**TL;DR:** Pre-trained SwiFT (contrastively pre-trained on HCP + ABCD + UK Biobank resting-state fMRI) was applied to LeBel et al. 2023 task-fMRI for word-level semantic decoding. A linear ridge probe on SwiFT features gave chance-level accuracy (Pearson r ≈ 0.0006). RSA between SwiFT feature RDMs and GPT-2 contextual RDMs showed a small but significant correlation (r = 0.035), but a time-shift control (+50 TRs) produced an *even higher* correlation (r = 0.056), proving the signal is temporal autocorrelation, not semantics. The project is closed here — pre-trained SwiFT does not carry stimulus-locked semantic information useful for decoding.

---

## Summary

We tested whether a **pre-trained SwiFT** (Swin 4D Transformer, contrastively pre-trained on resting-state fMRI from HCP, ABCD, and UK Biobank — henceforth "all three datasets") carries stimulus-locked semantic information when applied to **LeBel et al. 2023** task-fMRI (subjects listening to naturalistic stories). Two independent sanity checks were run. Both returned negative results — SwiFT features do **not** capture word-level semantic structure beyond raw voxel baselines.

**The project is being closed here.** A planned third stage (fine-tuning SwiFT on task-fMRI) was not pursued, as the pre-trained representations show no evidence of carrying the type of stimulus information that fine-tuning would amplify.

---

## Experiment 1: Linear Probe (word-level decoding)

**Goal:** Train a ridge regression probe on SwiFT features to predict word-level GPT-2 embedding for each TR. Compare against a voxel-baseline (ridge on raw BOLD).

**Methods:**
- Subject: UTS01, Stories: againstthewind (train), sweetaspie (eval)
- SwiFT window: 20 TRs, HRF delay: 5 TRs
- Target: Isolated word GPT-2 embedding (first-token hidden state); each word is embedded independently with no surrounding context [¹](#note-target-diff)
- Probe: Ridge regression with 6-fold cross-validation on alpha grid
- Baseline: Ridge on all brain-masked voxels (~500k) with L2 regularization

**Results:**

| Model | Pearson r | Top-1 Acc | Top-5 Acc |
|---|---|---|---|
| SwiFT probe | 0.0006 | 0.0015 | 0.0059 |
| Voxel ridge baseline | 0.0079 | 0.0089 | 0.0089 |

Chance-level decoding. Neither model beats trivial baselines. The voxel baseline marginally outperforms SwiFT, suggesting SwiFT's compression loses information rather than denoising.

**Conclusion:** Linear probe on pre-trained SwiFT features cannot decode word identity — the representations are not linearly separable by word-level semantic categories.

---

## Experiment 2: RSA (Representational Similarity Analysis)

**Goal:** Test whether the *geometric structure* of SwiFT feature RDMs correlates with GPT-2 contextual embedding RDMs. RSA can detect structure that a linear probe might miss (e.g., non-linear, non-separable, or very weak).

**Methods:**
- Single story (againstthewind), N = 150 TRs
- SwiFT features: pooled final layer (D × H × W × T → 288-dim)
- Language embeddings: Contextual GPT-2 (left-truncated to 200 tokens, last-token hidden state, 768-dim); each TR's embedding depends on the full preceding story context [¹](#note-target-diff)
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

**Critical finding:** The time-shift control (+50 TRs) produced a *higher* correlation than the correctly aligned condition. If the RSA reflected genuine semantic alignment, shifting the language embeddings by 50 TRs (where story content is completely different) should destroy the correlation. The fact that it persists and increases proves that the RDM structure is dominated by **slow temporal autocorrelation** in both BOLD and language embeddings.

**Methodological note:** The standard permutation test (shuffle RDM entries) is anti-conservative when the RDM has smooth structure from temporal autocorrelation. Breaking the RDM's internal dependencies inflates the null distribution's variance only partially, leading to false positives. A block-permutation or within-block shuffle is needed for valid inference.

**Conclusion:** The apparent RSA signal is a temporal autocorrelation artefact, not semantic structure. Pre-trained SwiFT features do not carry stimulus-locked semantic information.

---

<a name="note-target-diff"></a>**Note on target embedding difference across experiments:** The linear probe used *isolated* word embeddings (each word embedded alone via GPT-2's first token hidden state), because the probe learns a per-word fixed target suitable for classification. The RSA used *contextual* embeddings (running story prefix, last-token hidden state), because RSA measures representational geometry and context is essential for capturing stimulus-driven brain responses. This difference means the two experiments test related but distinct hypotheses: linear separability of word identity vs. geometric alignment of stimulus representations. Both returned negative.

---

## Overall Decision

Both experiments converge on the same conclusion: **Pre-trained SwiFT features are not useful for word-level semantic decoding on LeBel task-fMRI.** The project is closed here, for three reasons:

1. **Linear probe** → chance-level decoding for both SwiFT and voxels.
2. **RSA** → apparent signal invalidated by time-shift control: it is temporal autocorrelation, not semantics.
3. **Effort-to-benefit ratio** → The signal is so weak that even if fine-tuning recovered something, the effect size is unlikely to be practically useful.

---

## Limitations

1. **Single subject (UTS01), two stories (againstthewind, sweetaspie).** Results may not generalize to other subjects or a wider range of narrative stimuli.
2. **Frozen SwiFT only.** We tested only the pre-trained checkpoint without any fine-tuning on task-fMRI. Fine-tuning could in principle adapt SwiFT to task data, but the absence of any signal in the frozen representations makes this a low-priority direction.
3. **GPT-2 small as the language model.** Larger LMs (LLaMA-3, GPT-2-XL) or models fine-tuned on narrative comprehension might yield different RSA results, though the time-shift confound would need to be re-checked.
4. **Fixed HRF delay (5 TRs).** A single fixed delay was used; the optimal haemodynamic lag may vary across brain regions and subjects.
5. **RSA on a single story (N=150 TRs).** Longer recordings or pooling across multiple stories could increase statistical power, though the time-shift control result suggests the limiting factor is not power but confound control.

---

## Bugs found and resolved

1. **GPT-2 `truncation_side` bug (RSA only).** In `rsa_swift_lebel/extract_lang.py`, the HuggingFace tokenizer defaulted to `truncation_side="right"`, causing all TRs to receive the same embedding (truncated to the first 200 tokens regardless of position). Fixed by setting `tokenizer.truncation_side = "left"`. The linear probe pipeline was unaffected because it uses `build_gpt2_embeddings` which tokenizes single words — truncation never fires.
2. **Random-init SwiFT pooling bug.** In `rsa_swift_lebel/extract_features.py`, the `extract_raw` helper pooled feature maps over (H, W, T) but omitted the depth (D) dimension, producing a 3D array (N, C, D) instead of 2D (N, C). Fixed by pooling over all four spatial–temporal dimensions.
3. **Missing matplotlib.** Not installed in the `py39` conda environment; added via `pip install matplotlib`.

---

## Next direction

The recommended follow-up is to bypass SwiFT entirely and use the **Tang & Huth (2024) ridge stimulation / voxelwise encoding pipeline** (https://github.com/HuthLab/speechmodelfit). This approach fits separate ridge regression models per voxel to predict BOLD from GPT-2 embeddings, which:
- Avoids the compression artifacts of a single whole-brain embedding model like SwiFT.
- Naturally accounts for heterogeneous HRF delays across brain regions.
- Has proven effective for word-level encoding across multiple subjects and stories.

If a deep learning approach is preferred, fine-tuning SwiFT on the encoding objective (predicting BOLD from language features) may be more promising than the decoding approach attempted here.

---

## Methodological contribution

The RSA time-shift control (condition 4 in `compute_rsa.py`) is a cheap diagnostic that should be standard in fMRI RSA studies using continuous naturalistic stimuli. It revealed that the significant-but-small RSA correlation was entirely driven by temporal smoothness rather than semantic alignment. This is a known confound (Kriegeskorte 2008; Kriegeskorte & Kievit 2013) but is rarely tested directly.

---

## Repository structure

```
swift2lang/
├── REPORT.md                      ← This file
├── AGENTS.md                      # Environment and workflow rules
├── environment.yml                # Conda environment (py39)
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
```

## How to reproduce

```bash
# 0. Environment (see environment.yml for pinned versions)
conda env create -f environment.yml
conda activate py39

# 1. Prepare data
python scripts/prepare_lebel_data.py --subject UTS01 --stories againstthewind sweetaspie

# 2. Linear probe pipeline
python scripts/01_extract_features.py --config configs/config.yaml
python scripts/02_train_probe.py --config configs/config.yaml
python scripts/03_eval.py --config configs/config.yaml

# 3. RSA pipeline
python rsa_swift_lebel/extract_features.py --story againstthewind
python rsa_swift_lebel/extract_lang.py --story againstthewind
python rsa_swift_lebel/compute_rsa.py --story againstthewind --n-perm 10000
python rsa_swift_lebel/plot_results.py
```

## References

- LeBel, A., Jain, S., & Huth, A. G. (2023). Voxelwise encoding models show that cerebellar language representations are highly conceptual. *Journal of Neuroscience*, 43(33), 5932–5948. [OpenNeuro ds003020]
- Kim, Y., Kwon, J., et al. (2023). SwiFT: Swin 4D fMRI Transformer. *arXiv:2307.05916*.
- Caucheteux, C., & King, J.-R. (2022). Brains and algorithms partially converge in natural language processing. *Communications Biology*, 5, 134.
- Goldstein, A., et al. (2022). Shared computational principles for language processing in humans and deep language models. *Nature Neuroscience*, 25, 369–380.
- Kriegeskorte, N. (2008). Representational similarity analysis — connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2, 4.
- Kriegeskorte, N., & Kievit, R. A. (2013). Representational geometry: integrating cognition, computation, and the brain. *Trends in Cognitive Sciences*, 17(8), 401–412.
- Tang, J., & Huth, A. G. (2024). speechmodelfit: Voxelwise encoding models for speech processing. https://github.com/HuthLab/speechmodelfit
