"""
Extract contextual GPT-2 embeddings for selected TRs.
For each TR, feeds running story context through GPT-2,
takes last-token hidden state as the TR's language representation.

Usage:
    python rsa_swift_lebel/extract_lang.py --story againstthewind
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.lebel_loader import load_story_fmri


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/lebel")
    parser.add_argument("--subject", default="UTS01")
    parser.add_argument("--story", default="againstthewind")
    parser.add_argument("--tr-delay", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default="rsa_swift_lebel/cache")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    selected_trs = np.load(cache_dir / "selected_trs.npy")

    story_dir = Path(args.data_root) / args.subject / args.story

    # Get actual TR times from BOLD, consistent with extract_features.py
    _, tr_times = load_story_fmri(str(story_dir / "mni_bold.nii.gz"))
    delayed_times = tr_times + args.tr_delay * (tr_times[1] - tr_times[0])

    with open(story_dir / "transcript.json") as f:
        transcript = json.load(f)

    all_words = [entry["word"] for entry in transcript]
    all_onsets = np.array([entry["start"] for entry in transcript])

    # For each selected TR, find the last word whose onset ≤ delayed TR time
    selected_set = set(int(t) for t in selected_trs)
    tr_to_last_word = {}
    for tr in selected_set:
        mask = all_onsets <= delayed_times[tr]
        tr_to_last_word[tr] = int(np.where(mask)[0][-1]) if mask.any() else -1

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"  # keep latest context, drop early tokens
    model = GPT2Model.from_pretrained("gpt2").to(args.device)
    model.eval()

    embeddings = {}
    with torch.no_grad():
        for tr_idx in sorted(selected_trs):
            last_word_idx = tr_to_last_word[int(tr_idx)]
            if last_word_idx < 0:
                embeddings[int(tr_idx)] = np.zeros(model.config.n_embd, dtype=np.float32)
                continue
            context = " ".join(all_words[:last_word_idx + 1])
            tokens = tokenizer(context, return_tensors="pt", truncation=True,
                               max_length=args.max_context_tokens).to(args.device)
            outputs = model(**tokens)
            emb = outputs.last_hidden_state[0, -1, :].cpu().numpy()
            embeddings[int(tr_idx)] = emb

    emb_matrix = np.array([embeddings[int(t)] for t in selected_trs])
    np.save(cache_dir / "gpt2_embeddings.npy", emb_matrix)
    print(f"Saved GPT-2 embeddings: {emb_matrix.shape}")


if __name__ == "__main__":
    main()
