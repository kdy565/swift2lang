import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


def parse_textgrid_words(textgrid_path: str) -> List[Tuple[str, float, float]]:
    """Extract word-level intervals from a Praat TextGrid file."""
    with open(textgrid_path) as f:
        content = f.read()

    tier_matches = list(re.finditer(
        r'item\s*\[\d+\]:\s*class\s*=\s*"IntervalTier"\s*name\s*=\s*"([^"]+)"(.*?)(?=item\s*\[|\Z)',
        content, re.DOTALL | re.IGNORECASE
    ))
    word_section = None
    for m in tier_matches:
        if m.group(1).lower() in ("word", "words"):
            word_section = m.group(2)
            break
    if word_section is None:
        raise ValueError("Could not find 'word' tier in TextGrid")

    pattern = r'intervals\s*\[\d+\]:\s*xmin\s*=\s*([\d.e+\-]+)\s*xmax\s*=\s*([\d.e+\-]+)\s*text\s*=\s*"([^"]*)"'
    words = []
    for m in re.finditer(pattern, word_section):
        text = m.group(3).strip()
        if text and text not in ("sp", "<sil>"):
            words.append((text, float(m.group(1)), float(m.group(2))))
    return words


def load_story_fmri(fmri_path: str, expected_tr: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    img = nib.load(fmri_path)
    bold = img.get_fdata().astype(np.float32)
    n_trs = bold.shape[-1]
    tr_times = np.arange(n_trs) * expected_tr
    return bold, tr_times


def load_story_transcript(transcript_path: str) -> List[Tuple[str, float, float]]:
    with open(transcript_path) as f:
        data = json.load(f)
    return [(entry["word"], entry["start"], entry["end"]) for entry in data]


def align_words_to_trs(
    words: List[Tuple[str, float, float]],
    tr_times: np.ndarray,
    tr_delay: int = 5,
) -> Tuple[List[int], List[str]]:
    tr_indices = []
    word_labels = []
    delayed_tr_times = tr_times + tr_delay * (tr_times[1] - tr_times[0])
    for word, onset, offset in words:
        idx = np.searchsorted(delayed_tr_times, onset, side="right") - 1
        if 0 <= idx < len(tr_times):
            tr_indices.append(idx)
            word_labels.append(word)
    return tr_indices, word_labels


def build_gpt2_embeddings(
    words: List[str],
    device: str = "cpu",
    cache_path: Optional[str] = None,
) -> dict:
    """Build and cache GPT-2 embeddings for a set of words."""
    import torch
    from transformers import GPT2Model, GPT2Tokenizer

    unique_words = sorted(set(words))

    if cache_path and Path(cache_path).exists():
        cached = torch.load(cache_path, weights_only=True)
        if all(w in cached for w in unique_words):
            return cached

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained("gpt2").to(device)
    model.eval()

    embeddings = {}
    with torch.no_grad():
        for word in unique_words:
            inputs = tokenizer(word, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
            embeddings[word] = emb

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cache_path)
    return embeddings


class LeBelDataset(Dataset):
    """Memory-efficient LeBel dataset. Loads one story at a time."""

    def __init__(
        self,
        data_root: str,
        subject: str,
        stories: List[str],
        sequence_length: int = 20,
        tr_delay: int = 5,
        gpt2_cache_dir: Optional[str] = None,
    ):
        self.data_root = Path(data_root)
        self.subject = subject
        self.stories = stories
        self.sequence_length = sequence_length
        self.tr_delay = tr_delay
        self.gpt2_cache_dir = Path(gpt2_cache_dir) if gpt2_cache_dir else None

        self._samples = []  # (story, tr_idx)
        self._bold_cache = {}  # story -> np.ndarray

    def collect_samples(self):
        """Scan stories and build sample index, caching BOLD in memory."""
        for story in self.stories:
            mni_bold = self.data_root / self.subject / story / "mni_bold.nii.gz"
            transcript = self.data_root / self.subject / story / "transcript.json"
            if not mni_bold.exists() or not transcript.exists():
                print(f"Skipping {story}: missing data")
                continue

            # Load BOLD once and cache
            self._bold_cache[story], tr_times = load_story_fmri(str(mni_bold))
            n_trs = self._bold_cache[story].shape[-1]

            words_json = load_story_transcript(str(transcript))
            tr_indices, word_labels = align_words_to_trs(
                words_json, tr_times, tr_delay=self.tr_delay
            )

            for i, tr_idx in enumerate(tr_indices):
                start = tr_idx - self.sequence_length + 1
                if start < 0:
                    continue
                self._samples.append((story, tr_idx, i, word_labels[i]))

        print(f"Collected {len(self._samples)} samples from {self.stories}")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        story, tr_idx, word_idx, word = self._samples[idx]
        bold_data = self._bold_cache[story]
        start = tr_idx - self.sequence_length + 1
        window = bold_data[..., start:tr_idx + 1]  # (X, Y, Z, T)
        window = np.expand_dims(window, 0)  # (1, X, Y, Z, T)
        return torch.from_numpy(window).float(), word, story, tr_idx

    def get_word_set(self) -> set:
        return {s[3] for s in self._samples}

    def get_story_tr_mask(self, story: str) -> List[int]:
        return [i for i, s in enumerate(self._samples) if s[0] == story]
