"""
Stage 3: Evaluate probes on held-out story.

Computes:
- Pearson correlation per embedding dimension (vs GPT-2 target).
- Top-1 and top-5 retrieval accuracy from a candidate word pool.

Usage:
    python scripts/03_eval.py --config configs/config.yaml
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate linear probes")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--probe_path", type=str, default="data/probes/swift_ridge_probe.pkl")
    parser.add_argument("--baseline_path", type=str, default="data/probes/baseline_voxel_ridge.pkl")
    parser.add_argument("--output_dir", type=str, default="data/results")
    return parser.parse_args()


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true - y_true.mean(axis=0, keepdims=True)
    y_pred = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (y_true * y_pred).sum(axis=0)
    den = np.sqrt((y_true ** 2).sum(axis=0) * (y_pred ** 2).sum(axis=0))
    den = np.clip(den, 1e-12, None)
    return (num / den).mean()


def top_k_retrieval_accuracy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    word_candidates: np.ndarray,
    k: int = 5,
) -> float:
    """Top-k retrieval accuracy: how often the true embedding is in the
    top-k most similar candidates.

    y_pred: (n_samples, D) predicted embeddings.
    y_true: (n_samples, D) ground-truth embeddings.
    word_candidates: (n_words, D) candidate word embeddings to search over.

    Returns accuracy across all samples.
    """
    y_pred = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-12)
    y_true = y_true / (np.linalg.norm(y_true, axis=1, keepdims=True) + 1e-12)
    candidates = word_candidates / (np.linalg.norm(word_candidates, axis=1, keepdims=True) + 1e-12)

    true_sim = (y_pred[:, None, :] * y_true[:, None, :]).sum(axis=-1)  # (n, 1)

    correct = 0
    for i in range(len(y_pred)):
        sims = y_pred[i] @ candidates.T
        top_k_idx = np.argsort(sims)[-k:]
        # Check if true embedding is larger than k-th candidate
        true_val = y_pred[i] @ y_true[i]
        if np.sum(sims > true_val) < k:
            correct += 1

    return correct / len(y_pred)


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    subject = cfg["data"]["subject"]
    held_out_stories = cfg["eval"]["held_out_stories"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cached features for held-out stories
    cache_dir = Path(cfg["probe"]["cache_dir"]) / subject
    X_held, y_held = None, None

    for story in held_out_stories:
        feat_path = cache_dir / f"{story}_swift_features.pt"
        target_path = cache_dir / f"{story}_targets.pt"
        if feat_path.exists() and target_path.exists():
            feats = torch.load(feat_path)
            if feats.dim() == 3:
                feats = feats.mean(dim=-1)
            targets = torch.load(target_path)
            X_s = feats.numpy()
            y_s = targets.numpy()
            X_held = X_s if X_held is None else np.concatenate([X_held, X_s], axis=0)
            y_held = y_s if y_held is None else np.concatenate([y_held, y_s], axis=0)

    if X_held is None:
        print("No held-out features. Run 01_extract_features.py first.")
        return

    print(f"Held-out data: X {X_held.shape}, y {y_held.shape}")

    # Build candidate word pool from all training words
    pool_path = Path(cfg["eval"]["candidate_pool"])
    if pool_path.exists():
        word_pool = np.load(pool_path)
    else:
        # Collect all target embeddings as candidate pool
        all_targets = []
        for f in sorted(cache_dir.glob("*_targets.pt")):
            all_targets.append(torch.load(f).numpy())
        word_pool = np.concatenate(all_targets, axis=0)
        np.save(pool_path, word_pool)
    print(f"Word pool: {word_pool.shape[0]} candidates")

    results = {}

    # Evaluate SwiFT probe
    if Path(args.probe_path).exists():
        with open(args.probe_path, "rb") as f:
            swift_probe = pickle.load(f)
        y_pred_swift = swift_probe.predict(X_held)
        corr_swift = pearson_r(y_held, y_pred_swift)
        top1 = top_k_retrieval_accuracy(y_pred_swift, y_held, word_pool, k=1)
        top5 = top_k_retrieval_accuracy(y_pred_swift, y_held, word_pool, k=5)
        results["swift"] = {
            "pearson_r": float(corr_swift),
            "top1_acc": float(top1),
            "top5_acc": float(top5),
        }
        print(f"SwiFT probe  -> Pearson r: {corr_swift:.4f}, "
              f"Top-1: {top1:.4f}, Top-5: {top5:.4f}")

    # Evaluate voxel baseline (uses raw BOLD, not SwiFT features)
    if Path(args.baseline_path).exists():
        with open(args.baseline_path, "rb") as f:
            voxel_probe = pickle.load(f)
        # Load raw BOLD voxels for held-out stories
        X_vox_held, y_vox_held = [], []
        from src.data.lebel_loader import load_story_fmri, load_story_transcript, align_words_to_trs
        data_root = Path(cfg["data"]["root"]) / subject
        from transformers import GPT2Model, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        gpt2 = GPT2Model.from_pretrained("gpt2")
        gpt2.eval()
        for story in held_out_stories:
            fmri_path = data_root / story / "mni_bold.nii.gz"
            trans_path = data_root / story / "transcript.json"
            if not fmri_path.exists() or not trans_path.exists():
                continue
            bold, tr_times = load_story_fmri(str(fmri_path))
            words_json = load_story_transcript(str(trans_path))
            tr_idx, word_labels = align_words_to_trs(words_json, tr_times, tr_delay=cfg["data"]["tr_delay"])
            with torch.no_grad():
                word_embs = {}
                for w in set(word_labels):
                    inp = tokenizer(w, return_tensors="pt", padding=True)
                    out = gpt2(**inp)
                    word_embs[w] = out.last_hidden_state[:, 0, :].squeeze(0).numpy()
            for i, ti in enumerate(tr_idx):
                if ti < bold.shape[-1]:
                    X_vox_held.append(bold[..., ti].ravel())
                    y_vox_held.append(word_embs[word_labels[i]])
        if X_vox_held:
            X_vox_held = np.stack(X_vox_held, axis=0)
            y_vox_held = np.stack(y_vox_held, axis=0)
            print(f"Voxel held-out: X {X_vox_held.shape}, y {y_vox_held.shape}")
            y_pred_vox = voxel_probe.predict(X_vox_held)
            corr_vox = pearson_r(y_vox_held, y_pred_vox)
            top1_vox = top_k_retrieval_accuracy(y_pred_vox, y_vox_held, word_pool, k=1)
            top5_vox = top_k_retrieval_accuracy(y_pred_vox, y_vox_held, word_pool, k=5)
            results["voxel_baseline"] = {
                "pearson_r": float(corr_vox),
                "top1_acc": float(top1_vox),
                "top5_acc": float(top5_vox),
            }
            print(f"Voxel baseline -> Pearson r: {corr_vox:.4f}, "
                  f"Top-1: {top1_vox:.4f}, Top-5: {top5_vox:.4f}")

    # Save results
    import json
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
