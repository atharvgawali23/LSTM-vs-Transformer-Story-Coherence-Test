"""Character-level LSTM language model in TensorFlow 2 / Keras (U2 + U4)."""
import json
import math
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .settings import LSTM_DIR, SAMPLING

EMBED, UNITS, LAYERS = 128, 512, 2


def build(vocab_size: int, with_state: bool = False) -> keras.Model:
    """Same layer stack for training (sequence in, logits out) and inference (+ carried states)."""
    tokens = keras.Input(shape=(None,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(vocab_size, EMBED, name="embed")(tokens)
    state_inputs, state_outputs = [], []
    for i in range(LAYERS):
        lstm = keras.layers.LSTM(UNITS, return_sequences=True, return_state=True,
                                 dropout=0.0 if with_state else 0.2, name=f"lstm{i}")
        if with_state:
            h = keras.Input(shape=(UNITS,), name=f"h{i}")
            c = keras.Input(shape=(UNITS,), name=f"c{i}")
            state_inputs += [h, c]
            x, h_out, c_out = lstm(x, initial_state=[h, c])
            state_outputs += [h_out, c_out]
        else:
            x, _, _ = lstm(x)
    logits = keras.layers.Dense(vocab_size, name="logits")(x)
    if with_state:
        return keras.Model([tokens] + state_inputs, [logits] + state_outputs)
    return keras.Model(tokens, logits)


class CharLSTM:
    """Loads trained weights into a stateful inference graph."""

    def __init__(self, model_dir=LSTM_DIR):
        meta = json.loads((model_dir / "vocab.json").read_text())
        self.chars = meta["chars"]
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.model = build(len(self.chars), with_state=True)
        self.model.load_weights(str(model_dir / "char_lstm.weights.h5"))
        self.n_params = int(self.model.count_params())
        sig = [tf.TensorSpec((1, 1), tf.int32)] + [tf.TensorSpec((1, UNITS), tf.float32)] * (2 * LAYERS)
        self._fast_step = tf.function(lambda *a: self.model(list(a), training=False), input_signature=sig)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def zero_state(self, batch=1):
        return [np.zeros((batch, UNITS), np.float32) for _ in range(2 * LAYERS)]

    def _step(self, ids, states):
        out = self.model([np.asarray(ids, np.int32).reshape(1, -1)] + states, training=False)
        return out[0].numpy()[0], [s.numpy() for s in out[1:]]

    def generate(self, prompt: str, n_chars: int = SAMPLING.max_new_chars, temperature=SAMPLING.temperature,
                 top_k=SAMPLING.top_k, top_p=SAMPLING.top_p, seed=SAMPLING.seed) -> str:
        rng = np.random.default_rng(seed)
        ids = self.encode(prompt) or [self.stoi["\n"]]
        logits, states = self._step(ids, self.zero_state())
        last = logits[-1]
        out = []
        for _ in range(n_chars):
            nxt = sample_logits(last, temperature, top_k, top_p, rng)
            out.append(self.chars[nxt])
            res = self._fast_step(tf.constant([[nxt]], tf.int32), *states)
            last, states = res[0].numpy()[0, -1], list(res[1:])
        return "".join(out)

    def next_char_probs(self, prompt: str, temperature: float = 1.0):
        logits, _ = self._step(self.encode(prompt) or [0], self.zero_state())
        p = softmax(logits[-1] / max(temperature, 1e-6))
        return {self.chars[i]: float(p[i]) for i in np.argsort(-p)[:10]}

    def char_nll(self, text: str, chunk: int = 2048):
        """Per-character negative log-likelihood (nats) with the state carried across the whole text."""
        ids = self.encode(text)
        states = self.zero_state()
        nlls = []
        for s in range(0, len(ids) - 1, chunk):
            inp = ids[s:s + chunk]
            tgt = ids[s + 1:s + chunk + 1]
            logits, states = self._step(inp, states)
            logits = logits[:len(tgt)]
            logp = logits - logsumexp(logits)
            nlls.extend(-logp[np.arange(len(tgt)), tgt])
        return np.asarray(nlls)

    def perplexity(self, text: str) -> float:
        return float(math.exp(self.char_nll(text).mean()))


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def logsumexp(x):
    m = x.max(axis=-1, keepdims=True)
    return m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def sample_logits(logits, temperature, top_k, top_p, rng):
    """Identical sampler used for both models (temperature -> top-k -> nucleus)."""
    logits = np.asarray(logits, np.float64) / max(temperature, 1e-6)
    if top_k and top_k < len(logits):
        kth = np.sort(logits)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    p = softmax(logits)
    if top_p and top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep = order[: int(np.searchsorted(cum, top_p) + 1)]
        mask = np.zeros_like(p)
        mask[keep] = 1
        p = p * mask
        p /= p.sum()
    return int(rng.choice(len(p), p=p))
