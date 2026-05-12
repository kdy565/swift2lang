"""
Stage 2: Train linear probe (ridge regression) from cached SwiFT features
         to GPT-2 word embeddings.

Usage:
    python scripts/02_train_probe.py --config configs/config.yaml

Output:
    data/probes/swift_ridge_probe.pkl  (sklearn Pipeline)
    data/probes/baseline_voxel_ridge.pkl
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.probe.ridge import RidgeRegressionProbe
from src.baseline.voxel_ridge import VoxelRidgeBaseline


def parse_args():
    parser = argparse.ArgumentParser(description="Train linear probe")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--output_dir", type=str, default="data/probes")
    return parser.parse_args()


def load_cached_features(cache_dir: str, subject: str, held_out_stories: list = None) -> tuple:
    """Load cached features, separating held-out stories from training."""
    if held_out_stories is None:
        held_out_stories = []
    cache_path = Path(cache_dir) / subject
    if not cache_path.exists():
        raise FileNotFoundError(f"No cached features at {cache_path}")

    all_X, all_y = [], []
    held_out_X, held_out_y = {}, {}

    for f in sorted(cache_path.glob("*_swift_features.pt")):
        story = f.stem.replace("_swift_features", "")
        features = torch.load(f)
        targets = torch.load(cache_path / f"{story}_targets.pt")

        if features.dim() == 3:
            features = features.mean(dim=-1)

        X = features.numpy()
        y = targets.numpy()

        if story in held_out_stories:
            held_out_X[story] = X
            held_out_y[story] = y
        else:
            all_X.append(X)
            all_y.append(y)

    X_train = np.concatenate(all_X, axis=0) if all_X else None
    y_train = np.concatenate(all_y, axis=0) if all_y else None

    return X_train, y_train, held_out_X, held_out_y


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    subject = cfg["data"]["subject"]
    held_out_stories = cfg["eval"]["held_out_stories"]
    probe_cfg = cfg["probe"]
    baseline_cfg = cfg["baseline"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cached features
    X_train, y_train, held_out_X, held_out_y = load_cached_features(
        probe_cfg["cache_dir"], subject, held_out_stories
    )

    if X_train is None:
        print("No training data found. Run 01_extract_features.py first.")
        return

    print(f"Training data: X {X_train.shape}, y {y_train.shape}")

    # --- SwiFT probe ---
    print("Training SwiFT ridge probe...")
    swift_probe = RidgeRegressionProbe(
        alphas=probe_cfg["alphas"],
    )
    best_alpha, val_corr = swift_probe.fit_with_grid_search(
        X_train, y_train,
        val_ratio=probe_cfg["val_ratio"],
        n_bootstrap=20,
    )
    print(f"  Best alpha: {best_alpha}, mean val Pearson: {val_corr:.4f}")

    with open(output_dir / "swift_ridge_probe.pkl", "wb") as f:
        pickle.dump(swift_probe, f)

    # --- Voxel baseline ---
    if not args.skip_baseline:
        print("Training voxel-wise ridge baseline...")
        # For baseline, we need raw BOLD data. Load it from the dataset.
        # (Requires the subject data to still be accessible.)
        from src.data.lebel_loader import load_story_fmri, load_story_transcript, align_words_to_trs

        data_root = Path(cfg["data"]["root"]) / subject
        baseline_X, baseline_y = [], []

        for story_path in data_root.iterdir():
            if not story_path.is_dir():
                continue
            fmri_path = story_path / "mni_bold.nii.gz"
            trans_path = story_path / "transcript.json"
            if not fmri_path.exists() or not trans_path.exists():
                continue
            bold, tr_times = load_story_fmri(str(fmri_path))
            words = load_story_transcript(str(trans_path))
            tr_idx, word_labels = align_words_to_trs(words, tr_times, tr_delay=cfg["data"]["tr_delay"])
            from transformers import GPT2Model, GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            gpt2 = GPT2Model.from_pretrained("gpt2")
            gpt2.eval()
            with torch.no_grad():
                word_embs = {}
                for w in set(word_labels):
                    inp = tokenizer(w, return_tensors="pt", padding=True)
                    out = gpt2(**inp)
                    word_embs[w] = out.last_hidden_state[:, 0, :].squeeze(0).numpy()
            for i, ti in enumerate(tr_idx):
                if ti < bold.shape[-1]:
                    baseline_X.append(bold[..., ti].ravel())
                    baseline_y.append(word_embs[word_labels[i]])

        if baseline_X:
            X_vox = np.stack(baseline_X, axis=0)
            y_vox = np.stack(baseline_y, axis=0)
            print(f"Baseline data: X {X_vox.shape}, y {y_vox.shape}")

            voxel_probe = VoxelRidgeBaseline(
                alphas=baseline_cfg["alphas"],
                n_voxel_samples=baseline_cfg.get("n_voxel_samples", 50000),
            )
            best_alpha_vox, val_corr_vox = voxel_probe.fit(
                X_vox, y_vox,
                val_ratio=probe_cfg["val_ratio"],
                n_bootstrap=10,
            )
            print(f"  Best alpha (voxel): {best_alpha_vox}, mean val Pearson: {val_corr_vox:.4f}")

            with open(output_dir / "baseline_voxel_ridge.pkl", "wb") as f:
                pickle.dump(voxel_probe, f)
        else:
            print("No baseline data loaded. Skipping voxel baseline.")


if __name__ == "__main__":
    main()
