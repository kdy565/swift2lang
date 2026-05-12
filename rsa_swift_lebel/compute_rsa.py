"""
Compute RSA: RDMs + Spearman correlation + permutation tests + control conditions.

Conditions:
  1. SwiFT features vs GPT-2 embeddings (main)
  2. Random-init SwiFT vs GPT-2 (control)
  3. Raw BOLD voxels vs GPT-2 (classical baseline)
  4. Time-shifted: GPT-2 shifted by 50 TRs (temporal autocorrelation control)

Usage:
    python rsa_swift_lebel/compute_rsa.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.lebel_loader import load_story_fmri


def rdm_upper_tri(vectors):
    dist = cosine_distances(vectors)
    iu = np.triu_indices_from(dist, k=1)
    return dist[iu]


def permutation_test(rdm_a, rdm_b, n_perm=10000):
    observed, _ = spearmanr(rdm_a, rdm_b)
    null_rhos = np.empty(n_perm)
    count = 0
    for i in range(n_perm):
        perm = np.random.permutation(len(rdm_b))
        shuffled = rdm_b[perm]
        rho, _ = spearmanr(rdm_a, shuffled)
        null_rhos[i] = rho
        if rho >= observed:
            count += 1
    p_val = (count + 1) / (n_perm + 1)
    return observed, p_val, count, null_rhos


def evaluate(label, rdm_feat, rdm_lang, n_perm):
    rho, p, n_exceed, null = permutation_test(rdm_feat, rdm_lang, n_perm)
    print(f"\n=== {label} ===")
    print(f"  Spearman r = {rho:.4f}")
    print(f"  p = {p:.6f}  ({n_exceed}/{n_perm} exceed observed)")
    print(f"  null distribution: mean={null.mean():.6f}, std={null.std():.6f}, "
          f"95th={np.percentile(null, 95):.6f}")
    return {"r": rho, "p": p, "n_exceed": n_exceed, "n_perm": n_perm,
            "null_mean": float(null.mean()), "null_std": float(null.std())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/lebel")
    parser.add_argument("--subject", default="UTS01")
    parser.add_argument("--story", default="againstthewind")
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--cache-dir", default="rsa_swift_lebel/cache")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    selected_trs = np.load(cache_dir / "selected_trs.npy")
    gpt2_embs = np.load(cache_dir / "gpt2_embeddings.npy")
    swift_feats = np.load(cache_dir / "swift_features.npy")
    swift_rand_feats = np.load(cache_dir / "swift_features_random.npy")

    assert len(selected_trs) == gpt2_embs.shape[0] == swift_feats.shape[0]
    N = len(selected_trs)
    print(f"N = {N}, swift_dim = {swift_feats.shape[1]}, gpt2_dim = {gpt2_embs.shape[1]}")

    rdm_lang = rdm_upper_tri(gpt2_embs)
    rdm_swift = rdm_upper_tri(swift_feats)
    rdm_rand = rdm_upper_tri(swift_rand_feats)

    # Condition 1: Trained SwiFT vs GPT-2
    res1 = evaluate("Condition 1: Trained SwiFT vs GPT-2", rdm_swift, rdm_lang, args.n_perm)

    # Condition 2: Random-init SwiFT vs GPT-2
    res2 = evaluate("Condition 2: Random-init SwiFT vs GPT-2", rdm_rand, rdm_lang, args.n_perm)

    # Condition 3: Raw BOLD voxels vs GPT-2
    bold_data, _ = load_story_fmri(
        str(Path(args.data_root) / args.subject / args.story / "mni_bold.nii.gz")
    )
    nz = bold_data[..., 0] != 0
    bold_flat = bold_data[..., selected_trs]
    bold_mat = bold_flat[nz, :].T
    print(f"\n=== Condition 3: Raw BOLD voxels vs GPT-2 ===")
    print(f"  BOLD voxels: {bold_mat.shape[1]} (brain mask)")
    rdm_bold = rdm_upper_tri(bold_mat)
    res3 = evaluate("", rdm_bold, rdm_lang, args.n_perm)

    # Condition 4: Time-shift control (GPT-2 shifted by 50 TRs)
    print(f"\n=== Condition 4: Time-shift control (GPT-2 shifted by 50 TRs) ===")
    shift = 50
    gpt2_shifted = gpt2_embs[:-shift].copy()
    # .copy() intentional — we'll compute its RDM separately
    rdm_lang_shifted = rdm_upper_tri(gpt2_shifted)
    # But we need matching number of features. Let's use the first N-shift TRs
    # to match: we drop first `shift` entries from brain side, and last `shift`
    # from language side. Actually simpler: shift language by +50 (language
    # from TR 50..N-1 vs brain from TR 0..N-50-1).
    n_overlap = N - shift
    rdm_swift_shifted = rdm_upper_tri(swift_feats[:n_overlap])
    rdm_lang_shifted = rdm_upper_tri(gpt2_embs[shift:])
    res4 = evaluate("  SwiFT vs GPT-2 (shifted +50 TRs)", rdm_swift_shifted, rdm_lang_shifted, args.n_perm)

    results = {
        "swift_vs_gpt2": res1,
        "random_swift_vs_gpt2": res2,
        "bold_vs_gpt2": res3,
        "swift_vs_gpt2_shifted": res4,
    }
    np.save(cache_dir / "rsa_results.npy", results)
    print(f"\nResults saved to {cache_dir / 'rsa_results.npy'}")

    np.save(cache_dir / "rdm_swift.npy", rdm_swift)
    np.save(cache_dir / "rdm_random.npy", rdm_rand)
    np.save(cache_dir / "rdm_bold.npy", rdm_bold)
    np.save(cache_dir / "rdm_lang.npy", rdm_lang)
    np.save(cache_dir / "rdm_swift_shifted.npy", rdm_swift_shifted)
    np.save(cache_dir / "rdm_lang_shifted.npy", rdm_lang_shifted)

    print("\n=== Decision Rule ===")
    sig1 = res1["r"] > 0 and res1["p"] < 0.05
    print(f"  SwiFT r > 0 sig?: {sig1}  (r={res1['r']:.4f}, p={res1['p']:.6f})")
    print(f"  SwiFT > random?: {res1['r'] > res2['r']}  ({res1['r']:.4f} vs {res2['r']:.4f})")
    print(f"  SwiFT > voxel baseline?: {res1['r'] > res3['r']}  ({res1['r']:.4f} vs {res3['r']:.4f})")
    print(f"  SwiFT drops after shift?: {res1['r'] > res4['r']}  ({res1['r']:.4f} vs {res4['r']:.4f})")


if __name__ == "__main__":
    main()
