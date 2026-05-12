# SwiFT → LeBel: Word-level semantic decoding

**Result: Negative.** Pre-trained SwiFT features do not carry stimulus-locked
semantic information on LeBel task-fMRI beyond raw voxel baselines.

See **[REPORT.md](REPORT.md)** for the full technical report.

## Quick reference

```bash
# Environment
conda activate py39

# Linear probe pipeline
python scripts/01_extract_features.py --config configs/config.yaml
python scripts/02_train_probe.py --config configs/config.yaml
python scripts/03_eval.py --config configs/config.yaml

# RSA pipeline
python rsa_swift_lebel/extract_features.py --story againstthewind
python rsa_swift_lebel/extract_lang.py --story againstthewind
python rsa_swift_lebel/compute_rsa.py --story againstthewind --n-perm 10000
python rsa_swift_lebel/plot_results.py
```

## Dependencies
- Python 3.9+ (`py39` conda env), PyTorch 2.0.1, nibabel, scikit-learn, transformers
- SwiFT checkpoint: `SwiFT/pretrained_models/contrastive_pretrained.ckpt`
