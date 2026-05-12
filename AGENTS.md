# Environment Rules (shared GPU server)

## Golden rule
**Never execute training or inference commands.** Write scripts only. The user will review and run them.

## Data
- LeBel fMRI (ds003020) must be downloaded via HuthLab/deep-fMRI-dataset datalad pipeline.
- SwiFT checkpoint is local at `SwiFT/pretrained_models/contrastive_pretrained.ckpt`.
- Intermediate features are cached to disk to avoid repeated costly SwiFT inference.
- All derived data goes under `data/` (gitignored).

## Dependencies
Assume Python 3.10+, torch 2.x, nibabel, scikit-learn, transformers, datalad, antspyx/FSL.
Do not `pip install` without asking.

## Git
Do not commit unless explicitly asked. The workspace may or may not be a git repo.

## Monitoring
`nvidia-smi` to check GPU availability is OK. No background processes.
