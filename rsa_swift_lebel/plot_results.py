"""
Plot RDM heatmaps + bar chart of RSA scores across conditions.

Usage:
    python rsa_swift_lebel/plot_results.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform


def plot_rdm_heatmap(rdm_upper, title, ax, vrange=None):
    n = int((1 + np.sqrt(1 + 8 * len(rdm_upper))) / 2)
    mat = squareform(rdm_upper)
    if vrange is None:
        vmin, vmax = mat.min(), mat.max()
    else:
        vmin, vmax = vrange
    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("TR")
    ax.set_ylabel("TR")
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="rsa_swift_lebel/cache")
    parser.add_argument("--output", default="rsa_swift_lebel/figures")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rdm_swift = np.load(cache_dir / "rdm_swift.npy")
    rdm_random = np.load(cache_dir / "rdm_random.npy")
    rdm_bold = np.load(cache_dir / "rdm_bold.npy")
    rdm_lang = np.load(cache_dir / "rdm_lang.npy")
    results = np.load(cache_dir / "rsa_results.npy", allow_pickle=True).item()

    # --- Figure 1: RDM heatmaps ---
    labels = ["SwiFT", "Random SwiFT", "Raw BOLD", "GPT-2"]
    rdms = [rdm_swift, rdm_random, rdm_bold, rdm_lang]
    global_min = min(r.min() for r in rdms)
    global_max = max(r.max() for r in rdms)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, rdm, label in zip(axes, rdms, labels):
        plot_rdm_heatmap(rdm, label, ax, vrange=(global_min, global_max))
    fig.colorbar(plot_rdm_heatmap(rdm_swift, "", axes[0]), ax=axes, shrink=0.6, pad=0.05)
    plt.suptitle("Representational Dissimilarity Matrices (cosine distance)")
    plt.tight_layout()
    plt.savefig(out_dir / "rdm_heatmaps.png", dpi=150)
    print(f"Saved {out_dir / 'rdm_heatmaps.png'}")
    plt.close()

    # --- Figure 2: Bar chart ---
    conditions = ["SwiFT vs GPT-2", "Random SwiFT\nvs GPT-2", "Raw BOLD\nvs GPT-2", "SwiFT vs GPT-2\n(shift +50)"]
    rhos = [results["swift_vs_gpt2"]["r"],
            results["random_swift_vs_gpt2"]["r"],
            results["bold_vs_gpt2"]["r"],
            results["swift_vs_gpt2_shifted"]["r"]]
    sigs = [results["swift_vs_gpt2"]["p"] < 0.05,
            results["random_swift_vs_gpt2"]["p"] < 0.05,
            results["bold_vs_gpt2"]["p"] < 0.05,
            results["swift_vs_gpt2_shifted"]["p"] < 0.05]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ecc71" if s else "#e74c3c" for s in sigs]
    bars = ax.bar(conditions, rhos, color=colors, edgecolor="black", linewidth=1.2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Spearman r")
    ax.set_title("RSA: Brain vs Language RDM Correlation")

    for bar, rho, sig in zip(bars, rhos, sigs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"r={rho:.3f}\n{'*' if sig else 'n.s.'}",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim(min(rhos) - 0.05, max(rhos) + 0.08)
    plt.tight_layout()
    plt.savefig(out_dir / "rsa_bar_chart.png", dpi=150)
    print(f"Saved {out_dir / 'rsa_bar_chart.png'}")
    plt.close()


if __name__ == "__main__":
    main()
