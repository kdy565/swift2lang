"""
Prepare LeBel fMRI data for SwiFT linear probe.

Downloads raw BOLD + T1w from OpenNeuro S3, resamples to MNI 96x96x96,
parses TextGrid transcripts, and aligns word timings to TRs.

Usage:
    python scripts/prepare_lebel_data.py --subject UTS01 --stories againstthewind
    python scripts/prepare_lebel_data.py --subject UTS01 --all
"""
import argparse
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import image as nimg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_textgrid_words(textgrid_path: str) -> list:
    """Extract word-level intervals from a Praat TextGrid file.

    Returns list of (word, start_sec, end_sec) tuples from the "word" tier.
    """
    with open(textgrid_path) as f:
        content = f.read()

    # Find the "word" tier
    tier_match = re.search(
        r'item\s*\[\d+\]:\s*class\s*=\s*"IntervalTier"\s*name\s*=\s*"words?"',
        content, re.IGNORECASE
    )
    if not tier_match:
        raise ValueError("Could not find 'word' tier in TextGrid")

    # Get everything after the word tier header up to the next tier or end
    tier_start = tier_match.start()
    next_tier = re.search(r'item\s*\[\d+\]:', content[tier_start + 1:])
    if next_tier:
        tier_content = content[tier_start:tier_start + 1 + next_tier.start()]
    else:
        tier_content = content[tier_start:]

    # Extract intervals
    pattern = r'intervals\s*\[\d+\]:\s*xmin\s*=\s*([\d.e+\-]+)\s*xmax\s*=\s*([\d.e+\-]+)\s*text\s*=\s*"([^"]*)"'
    words = []
    for m in re.finditer(pattern, tier_content):
        start, end, text = float(m.group(1)), float(m.group(2)), m.group(3)
        if text.strip():
            words.append((text.strip(), start, end))

    return words


def resample_to_mni96(bold_path: str, t1w_path: str, output_path: str) -> str:
    bold_img = nib.load(bold_path)

    # Create an empty MNI 2mm reference image at 96x96x96
    target_affine = np.array([
        [2., 0., 0., -96.],
        [0., 2., 0., -132.],
        [0., 0., 2., -72.],
        [0., 0., 0., 1.],
    ])
    target_img = nib.Nifti1Image(np.zeros((96, 96, 96), dtype=np.float32), target_affine)

    bold_mni = nimg.resample_to_img(
        bold_img, target_img,
        interpolation="continuous",
        force_resample=True,
    )

    bold_data = bold_mni.get_fdata()
    n_trs = bold_data.shape[-1]

    mean = bold_data.mean(axis=-1, keepdims=True)
    std = bold_data.std(axis=-1, keepdims=True)
    std = np.clip(std, 1e-8, None)
    bold_data = (bold_data - mean) / std

    bold_mni = nib.Nifti1Image(bold_data.astype(np.float32), bold_mni.affine)
    nib.save(bold_mni, output_path)
    print(f"Saved MNI-resampled BOLD: {output_path}")
    print(f"  Shape: {bold_data.shape}, TRs: {n_trs}")

    return str(output_path)


def align_and_save(subject: str, story: str, data_root: str):
    """Full pipeline for one story."""
    story_dir = Path(data_root) / subject / story
    story_dir.mkdir(parents=True, exist_ok=True)

    raw_bold = story_dir / "raw_bold.nii.gz"
    textgrid = story_dir / "transcript.TextGrid"
    t1w_path = Path(data_root) / subject / "T1w.nii.gz"

    # Step 1: Resample BOLD to MNI 96x96x96
    mni_bold = story_dir / "mni_bold.nii.gz"
    if not mni_bold.exists():
        if not raw_bold.exists():
            print(f"Download raw BOLD for {subject}/{story} first.")
            return
        resample_to_mni96(str(raw_bold), str(t1w_path), str(mni_bold))
    else:
        print(f"MNI BOLD already exists: {mni_bold}")

    # Step 2: Parse TextGrid for word timing
    transcript_json = story_dir / "transcript.json"
    if not transcript_json.exists():
        words = parse_textgrid_words(str(textgrid))
        with open(transcript_json, "w") as f:
            json.dump([{"word": w, "start": s, "end": e} for w, s, e in words], f, indent=2)
        print(f"Saved transcript: {transcript_json} ({len(words)} words)")
    else:
        print(f"Transcript already exists: {transcript_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LeBel data for SwiFT")
    parser.add_argument("--subject", default="UTS01")
    parser.add_argument("--stories", nargs="+", default=None)
    parser.add_argument("--all", action="store_true", help="Process all available stories")
    parser.add_argument("--data-root", default="data/lebel")
    args = parser.parse_args()

    if args.all:
        print("--all not yet implemented, use --stories")
        sys.exit(1)

    for story in args.stories:
        print(f"\n=== Processing {args.subject}/{story} ===")
        align_and_save(args.subject, story, args.data_root)
