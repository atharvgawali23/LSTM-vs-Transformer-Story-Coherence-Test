"""BERT next-sentence-prediction coherence scorer (U3).

Text is split into sentence-like segments. For every pair (segment i, segment i+d) we ask
BERT-NSP for P(IsNext). Local coherence = mean over d=1; long-range coherence = mean over d>=3.
Also reports the 'coherence vs distance' curve, the core evidence for the LSTM vs Transformer story.
"""
import re

import numpy as np
import torch
from transformers import BertForNextSentencePrediction, BertTokenizer

from .settings import BERT_NSP


def segment(text: str, min_words: int = 6, max_words: int = 40):
    """Split into sentence-ish units; merge very short ones (Shakespeare lines are short)."""
    raw = re.split(r"(?<=[.!?;:])\s+|\n{2,}", text.strip())
    segs, buf = [], []
    for piece in raw:
        words = piece.replace("\n", " ").split()
        if not words:
            continue
        buf.extend(words)
        if len(buf) >= min_words:
            segs.append(" ".join(buf[:max_words]))
            buf = []
    if buf and segs:
        segs[-1] += " " + " ".join(buf)
    elif buf:
        segs.append(" ".join(buf))
    return segs


class NSPScorer:
    def __init__(self, name=BERT_NSP):
        self.tok = BertTokenizer.from_pretrained(name)
        self.model = BertForNextSentencePrediction.from_pretrained(name).eval()

    @torch.no_grad()
    def p_next(self, pairs):
        if not pairs:
            return np.array([])
        a, b = zip(*pairs)
        enc = self.tok(list(a), list(b), return_tensors="pt", padding=True, truncation=True, max_length=128)
        logits = self.model(**enc).logits
        return torch.softmax(logits, -1)[:, 0].numpy()  # index 0 = "IsNext"

    def score(self, text: str, max_distance: int = 5):
        segs = segment(text)
        curve = {}
        for d in range(1, max_distance + 1):
            pairs = [(segs[i], segs[i + d]) for i in range(len(segs) - d)]
            if pairs:
                curve[d] = float(self.p_next(pairs).mean())
        local = curve.get(1, float("nan"))
        far = [v for k, v in curve.items() if k >= 3]
        return {
            "n_segments": len(segs),
            "local_coherence": local,
            "long_range_coherence": float(np.mean(far)) if far else float("nan"),
            "curve": curve,
        }
