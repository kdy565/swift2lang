"""
Extract SwiFT features from LeBel fMRI for RSA.
Selects N TRs from one story, runs frozen SwiFT, caches to .npy.

Usage:
    python rsa_swift_lebel/extract_features.py --story againstthewind
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.swift_wrapper import SwiFTEncoder
from src.data.lebel_loader import load_story_fmri, load_story_transcript, align_words_to_trs


def select_trs(tr_indices, n_trs, min_idx, n_total):
    valid = sorted(set(i for i in tr_indices if min_idx <= i < n_total))
    if len(valid) < n_trs:
        print(f"Warning: only {len(valid)} valid TRs, using all")
    step = max(1, len(valid) // n_trs)
    return sorted(set(valid[::step]))[:n_trs]


@torch.no_grad()
def extract_swift_features(bold_data, selected_trs, swift, device, seq_len):
    feats = []
    for tr_idx in selected_trs:
        start = tr_idx - seq_len + 1
        window = bold_data[..., start:tr_idx + 1]
        window = np.expand_dims(window, (0, 1))
        inp = torch.from_numpy(window).float().to(device)
        out = swift.extract_features(inp, pool_spatial=True, pool_time=False)
        feats.append(out.mean(dim=-1).cpu().numpy()[0])
    return np.stack(feats, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/lebel")
    parser.add_argument("--subject", default="UTS01")
    parser.add_argument("--story", default="againstthewind")
    parser.add_argument("--n-trs", type=int, default=200)
    parser.add_argument("--tr-delay", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default="rsa_swift_lebel/cache")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    story_dir = Path(args.data_root) / args.subject / args.story

    bold_data, tr_times = load_story_fmri(str(story_dir / "mni_bold.nii.gz"))
    n_trs_total = bold_data.shape[-1]
    print(f"BOLD shape: {bold_data.shape}, {n_trs_total} TRs")

    words = load_story_transcript(str(story_dir / "transcript.json"))
    tr_indices, word_labels = align_words_to_trs(words, tr_times, tr_delay=args.tr_delay)

    selected_trs = select_trs(tr_indices, args.n_trs, args.seq_len, n_trs_total)
    print(f"Selected {len(selected_trs)} TRs")
    np.save(cache_dir / "selected_trs.npy", np.array(selected_trs))

    swift = SwiFTEncoder(
        ckpt_path="SwiFT/pretrained_models/contrastive_pretrained.ckpt",
        img_size=(96, 96, 96, args.seq_len),
        patch_size=(6, 6, 6, 1),
        window_size=(4, 4, 4, 4),
        first_window_size=(4, 4, 4, 4),
        embed_dim=36, depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24), c_multiplier=2,
        in_chans=1, attn_drop_rate=0.0, last_layer_full_MSA=True,
    ).to(args.device)
    swift.eval()

    features = extract_swift_features(bold_data, selected_trs, swift, args.device, args.seq_len)
    np.save(cache_dir / "swift_features.npy", features)
    print(f"Saved SwiFT features: {features.shape}")

    del swift
    torch.cuda.empty_cache()

    # --- Random-init SwiFT control ---
    sys.path.insert(0, str(Path("SwiFT").resolve()))
    from project.module.models.swin4d_transformer_ver7 import SwinTransformer4D

    rand_model = SwinTransformer4D(
        img_size=(96, 96, 96, args.seq_len),
        in_chans=1, embed_dim=36,
        window_size=(4, 4, 4, 4),
        first_window_size=(4, 4, 4, 4),
        patch_size=(6, 6, 6, 1),
        depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
        c_multiplier=2, last_layer_full_MSA=True,
        drop_rate=0., drop_path_rate=0., attn_drop_rate=0.,
        to_float=True,
    ).to(args.device)
    rand_model.eval()

    def extract_raw(feats):
        return feats.mean(dim=(2, 3, 4, 5))  # pool D, H, W, T -> (B, C)

    rand_feats = []
    with torch.no_grad():
        for tr_idx in selected_trs:
            start = tr_idx - args.seq_len + 1
            window = bold_data[..., start:tr_idx + 1]
            window = np.expand_dims(window, (0, 1))
            inp = torch.from_numpy(window).float().to(args.device)
            out = rand_model(inp)
            vec = extract_raw(out).cpu().numpy()[0]
            rand_feats.append(vec)
    rand_features = np.stack(rand_feats, axis=0)
    np.save(cache_dir / "swift_features_random.npy", rand_features)
    print(f"Saved random-init SwiFT features: {rand_features.shape}")


if __name__ == "__main__":
    main()
