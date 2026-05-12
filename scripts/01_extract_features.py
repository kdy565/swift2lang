"""
Stage 1: Extract SwiFT features from preprocessed LeBel fMRI.

Memory-efficient: streams data from disk, processes in small batches,
caches to disk immediately.

Usage:
    python scripts/01_extract_features.py --config configs/config.yaml
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.lebel_loader import (
    LeBelDataset,
    build_gpt2_embeddings,
)
from src.models.swift_wrapper import SwiFTEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Extract SwiFT features from LeBel fMRI")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    subject = args.subject or cfg["data"]["subject"]
    device = torch.device(args.device)

    data_root = cfg["data"]["root"]
    cache_dir = Path(cfg["probe"]["cache_dir"]) / subject
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine stories: all available in subject dir
    subj_dir = Path(data_root) / subject
    stories = sorted([d.name for d in subj_dir.iterdir() if d.is_dir() and d.name != "T1w"])
    print(f"Found stories: {stories}")

    for story in stories:
        out_path = cache_dir / f"{story}_swift_features.pt"
        target_path = cache_dir / f"{story}_targets.pt"
        if out_path.exists() and target_path.exists():
            print(f"Found cached {story}, skipping")
            continue

        # Build per-story dataset
        dataset = LeBelDataset(
            data_root=data_root,
            subject=subject,
            stories=[story],
            sequence_length=cfg["data"]["sequence_length"],
            tr_delay=cfg["data"]["tr_delay"],
            gpt2_cache_dir=str(cache_dir),
        )
        dataset.collect_samples()
        if len(dataset) == 0:
            print(f"No samples for {story}, skipping")
            continue

        # Build GPT-2 embeddings for all words in this story
        word_set = dataset.get_word_set()
        word_embs = build_gpt2_embeddings(
            list(word_set),
            device=args.device,
            cache_path=str(cache_dir / "gpt2_embeddings.pt"),
        )

        # Load SwiFT encoder (once per story to avoid OOM from dataset accumulation)
        swift = SwiFTEncoder(
            ckpt_path=cfg["model"]["swift_ckpt"],
            img_size=tuple(cfg["model"]["img_size"]),
            patch_size=tuple(cfg["model"]["patch_size"]),
            window_size=tuple(cfg["model"]["window_size"]),
            first_window_size=tuple(cfg["model"]["first_window_size"]),
            embed_dim=cfg["model"]["embed_dim"],
            depths=tuple(cfg["model"]["depths"]),
            num_heads=tuple(cfg["model"]["num_heads"]),
            c_multiplier=cfg["model"]["c_multiplier"],
            in_chans=cfg["model"]["in_chans"],
            attn_drop_rate=cfg["model"]["attn_drop_rate"],
            last_layer_full_MSA=cfg["model"]["last_layer_full_MSA"],
        ).to(device)
        swift.eval()

        # Process in mini-batches
        all_features, all_targets = [], []

        def collate(batch):
            fmris = torch.stack([b[0] for b in batch])
            words = [b[1] for b in batch]
            return fmris, words

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
            collate_fn=collate,
        )

        with torch.no_grad():
            for batch_idx, (fmri_windows, words_batch) in enumerate(loader):
                fmri_windows = fmri_windows.to(device)
                batch_targets = torch.stack([word_embs[w] for w in words_batch])

                feats = swift.extract_features(
                    fmri_windows, pool_spatial=True, pool_time=False
                )
                feats = feats.mean(dim=-1)  # average over time -> (B, C)

                all_features.append(feats.cpu())
                all_targets.append(batch_targets)

                if (batch_idx + 1) % 50 == 0:
                    print(f"  {story}: processed {batch_idx + 1}/{len(loader)} batches")

        features = torch.cat(all_features, dim=0)
        targets = torch.cat(all_targets, dim=0)

        torch.save(features, out_path)
        torch.save(targets, target_path)

        print(f"Extracted {story}: {features.shape[0]} samples, "
              f"feat_dim={features.shape[1]}")

        del swift, dataset, loader, all_features, all_targets, features, targets
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
