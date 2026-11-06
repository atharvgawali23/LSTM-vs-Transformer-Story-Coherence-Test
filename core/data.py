"""Corpus loading: TinyStories (clean children's story text)."""
from pathlib import Path
from .settings import CORPUS_PATH, TRAIN_FRACTION


def load_corpus() -> str:
    return CORPUS_PATH.read_text(encoding="utf-8")


def train_val_split(text: str | None = None):
    text = text if text is not None else load_corpus()
    n = int(len(text) * TRAIN_FRACTION)
    # cut at a story boundary
    cut = text.find("\n\n<|endoftext|>\n\n", n)
    if cut != -1:
        cut += len("\n\n<|endoftext|>\n\n")
    else:
        cut = n
    return text[:cut], text[cut:]
