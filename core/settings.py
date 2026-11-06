"""Shared, fixed sampling + evaluation settings for TinyStories.
"""
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

CORPUS_PATH = DATA_DIR / "tinystories.txt"
TRAIN_FRACTION = 0.9          # first 90% train, last 10% held-out validation

LSTM_DIR = MODELS_DIR / "char_lstm"
GPT2_FT_DIR = MODELS_DIR / "distilgpt2_tinystories"
GPT2_BASE = "distilgpt2"
BERT_NSP = "bert-base-uncased"


@dataclass(frozen=True)
class SamplingSettings:
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.90
    max_new_chars: int = 450   # standard length for a child story continuation
    seed: int = 42


SAMPLING = SamplingSettings()


def sampling_dict():
    return asdict(SAMPLING)
