"""U5: memorisation check — does generated text copy the training corpus verbatim?"""
import re
from functools import lru_cache

from .data import train_val_split

N = 8  # an 8-word span copied verbatim is treated as memorised


def _words(t):
    return re.findall(r"[a-z']+", t.lower())


@lru_cache(maxsize=1)
def _train_index():
    train, _ = train_val_split()
    w = _words(train)
    grams = {tuple(w[i:i + N]) for i in range(len(w) - N + 1)}
    return grams, " ".join(w)


def memorisation_report(text: str):
    grams, train_joined = _train_index()
    w = _words(text)
    gen_grams = [tuple(w[i:i + N]) for i in range(len(w) - N + 1)]
    hits = [g for g in gen_grams if g in grams]
    # longest verbatim word span present in the training text (greedy extension)
    longest, longest_span = 0, ""
    i = 0
    while i < len(w):
        j = i + 1
        while j <= len(w) and (" " + " ".join(w[i:j]) + " ") in (" " + train_joined + " "):
            j += 1
        if j - 1 - i > longest:
            longest, longest_span = j - 1 - i, " ".join(w[i:j - 1])
        i += 1
    return {
        "ngram_n": N,
        "n_ngrams": len(gen_grams),
        "copied_ngrams": len(hits),
        "copied_pct": 100.0 * len(hits) / max(1, len(gen_grams)),
        "longest_copied_words": longest,
        "longest_copied_span": longest_span,
        "flag": len(hits) > 0,
    }
