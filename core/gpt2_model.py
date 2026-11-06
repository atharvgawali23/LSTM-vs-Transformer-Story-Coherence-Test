"""distilGPT-2 (PyTorch, Hugging Face transformers): generation, char-normalised perplexity, attention."""
import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .settings import GPT2_BASE, GPT2_FT_DIR, SAMPLING

torch.set_num_threads(max(1, torch.get_num_threads()))


class GPT2LM:
    def __init__(self, path=None):
        path = str(path or (GPT2_FT_DIR if GPT2_FT_DIR.exists() else GPT2_BASE))
        self.name = path
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, attn_implementation="eager").eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def generate(self, prompt: str, n_chars: int = SAMPLING.max_new_chars, temperature=SAMPLING.temperature,
                 top_k=SAMPLING.top_k, top_p=SAMPLING.top_p, seed=SAMPLING.seed) -> str:
        """Sample tokens with the shared settings until >= n_chars new characters, then trim to n_chars."""
        torch.manual_seed(seed)
        enc = self.tok(prompt or "\n", return_tensors="pt")
        ids = enc.input_ids
        # ~4 chars per GPT-2 token on Shakespeare; over-generate then trim so both models emit n_chars.
        out = self.model.generate(ids, attention_mask=enc.attention_mask, do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p,
                                  max_new_tokens=int(n_chars / 3.2) + 12,
                                  pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return text[:n_chars]

    @torch.no_grad()
    def next_token_probs(self, prompt: str, temperature: float = 1.0, k: int = 10):
        ids = self.tok(prompt or "\n", return_tensors="pt").input_ids
        logits = self.model(ids).logits[0, -1] / max(temperature, 1e-6)
        p = torch.softmax(logits, -1)
        top = torch.topk(p, k)
        return {repr(self.tok.decode([int(i)])): float(v) for v, i in zip(top.values, top.indices)}

    @torch.no_grad()
    def total_nll(self, text: str, window: int = 1024, stride: int = 512):
        """Sum of token NLL (nats) over text with a sliding window; returns (nll_sum, n_tokens)."""
        ids = self.tok(text, return_tensors="pt").input_ids[0]
        nll, counted, prev_end = 0.0, 0, 0
        for begin in range(0, len(ids), stride):
            end = min(begin + window, len(ids))
            trg_len = end - prev_end
            chunk = ids[begin:end].unsqueeze(0)
            labels = chunk.clone()
            labels[:, :-trg_len] = -100
            loss = self.model(chunk, labels=labels).loss
            n = int((labels[:, 1:] != -100).sum())
            nll += float(loss) * n
            counted += n
            prev_end = end
            if end == len(ids):
                break
        return nll, counted

    def perplexity(self, text: str):
        nll, n_tok = self.total_nll(text)
        return {
            "token_ppl": math.exp(nll / n_tok),
            # per-character perplexity: same unit as the char-LSTM -> directly comparable
            "char_ppl": math.exp(nll / len(text)),
            "bpc": nll / len(text) / math.log(2),
        }

    @torch.no_grad()
    def attention(self, text: str, layer: int = -1):
        """Head-averaged attention matrix of one layer + tokens (U2 attention explanation)."""
        enc = self.tok(text, return_tensors="pt")
        out = self.model(**enc, output_attentions=True)
        att = out.attentions[layer][0].mean(0).numpy()
        toks = [self.tok.decode([int(i)]) for i in enc.input_ids[0]]
        return att, toks

    @torch.no_grad()
    def mean_attention_distance(self, text: str):
        """Average |i-j| weighted by attention, per layer — how far back the model looks."""
        enc = self.tok(text, return_tensors="pt")
        out = self.model(**enc, output_attentions=True)
        n = enc.input_ids.shape[1]
        dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        return [float((a[0].mean(0).numpy() * dist).sum(-1).mean()) for a in out.attentions]
