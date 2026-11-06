"""P16 · LSTM vs Transformer Coherence Test — TinyStories edition.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from core.coherence import NSPScorer
from core.gpt2_model import GPT2LM
from core.guardrails import check_input, screen_output
from core.lstm_model import CharLSTM
from core.memorisation import memorisation_report
from core.settings import RESULTS_DIR, SAMPLING
from core.tts import speak

ROOT = Path(__file__).parent
print("Loading TinyStories models...", flush=True)
LSTM, GPT, NSP = CharLSTM(), GPT2LM(), NSPScorer()
memorisation_report("warm up training index")
SUMMARY = json.loads((RESULTS_DIR / "summary.json").read_text()) if (RESULTS_DIR / "summary.json").exists() else None
print("Models ready!", flush=True)

EXAMPLES = [
    "Once upon a time, there was a little boy named Tim. Tim had a pet dog named",
    "Lily loved to play in the sunny garden. One afternoon, she found a little bird that",
    "Ben was very hungry. He opened the big red fridge and saw",
    "Spot the puppy ran into the woods. He looked behind a big oak tree and",
]


def curve_plot(c_lstm, c_gpt):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for c, lab, st in ((c_gpt, "distilGPT-2 (TinyStories)", "o-"), (c_lstm, "char-LSTM (Keras)", "s-")):
        if c:
            ax.plot(list(c.keys()), list(c.values()), st, label=lab)
    if SUMMARY and "real_tinystories" in SUMMARY.get("coherence", {}):
        r = SUMMARY["coherence"]["real_tinystories"]["curve"]
        ax.plot([int(k) for k in r], list(r.values()), "k--", alpha=.6, label="real story (ref)")
    ax.set_ylim(0, 1.02); ax.set_xlabel("sentence distance d"); ax.set_ylabel("BERT P(IsNext)")
    ax.set_title("Story Coherence vs Distance"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def compare(prompt, temperature, top_k, top_p, n_chars, seed):
    ok, prompt, reason = check_input(prompt)
    if not ok:
        raise gr.Error(reason)
    n_chars, seed, top_k = int(n_chars), int(seed), int(top_k)
    t = time.time(); g_l = LSTM.generate(prompt, n_chars, temperature, top_k, top_p, seed); t_l = time.time() - t
    t = time.time(); g_g = GPT.generate(prompt, n_chars, temperature, top_k, top_p, seed); t_g = time.time() - t
    g_l, _ = screen_output(g_l); g_g, _ = screen_output(g_g)
    s_l, s_g = NSP.score(g_l), NSP.score(g_g)
    m_l, m_g = memorisation_report(g_l), memorisation_report(g_g)
    p_l = LSTM.perplexity(prompt + g_l)
    p_g = GPT.perplexity(prompt + g_g)["char_ppl"]
    f = lambda x: f"{x:.3f}" if x == x else "n/a"
    table = pd.DataFrame({
        "metric": ["local coherence (d=1) ↑", "long-range coherence (d≥3) ↑", "sentences scored",
                   "char perplexity of output ↓", "copied 8-grams % ↓", "longest copied span (words)",
                   "generation latency (s)"],
        "char-LSTM": [f(s_l["local_coherence"]), f(s_l["long_range_coherence"]), s_l["n_segments"], f"{p_l:.2f}",
                      f"{m_l['copied_pct']:.1f}", m_l["longest_copied_words"], f"{t_l:.2f}"],
        "distilGPT-2": [f(s_g["local_coherence"]), f(s_g["long_range_coherence"]), s_g["n_segments"], f"{p_g:.2f}",
                        f"{m_g['copied_pct']:.1f}", m_g["longest_copied_words"], f"{t_g:.2f}"],
    })
    score = lambda s: (s["long_range_coherence"] if s["long_range_coherence"] == s["long_range_coherence"]
                       else s["local_coherence"])
    winner = "distilGPT-2" if score(s_g) >= score(s_l) else "char-LSTM"
    best = g_g if winner == "distilGPT-2" else g_l
    audio = speak(best)
    verdict = (f"**Winner: {winner}** (Story coherence: {f(score(s_g))} vs {f(score(s_l))}). "
               f"Notice how distilGPT-2 keeps the character names and plot intact, while the LSTM tends to drift or misspell words after 2–3 sentences.")
    return g_l, g_g, table, curve_plot(s_l["curve"], s_g["curve"]), verdict, audio


CSS = ".gradio-container{max-width:1200px !important}"
with gr.Blocks(title="P16 · LSTM vs Transformer Coherence Test (Simple Stories)", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown("# 🧸 P16 · LSTM vs Transformer Story Coherence Test\n"
                "**Student**: Atharv Gawali (Roll No. 16) &nbsp;|&nbsp; **Problem**: P16 &nbsp;|&nbsp; **Corpus**: Simple TinyStories (Preschool Vocabulary)\n\n"
                '<div style="background-color: #f8fafc; padding: 12px 16px; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 12px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap;">'
                '<span style="font-weight: 600; color: #1e293b;">📥 Quick Downloads:</span> '
                '<a href="/file=report/16_Atharv_P16.pdf" download style="color: #2563eb; font-weight: 500; text-decoration: none;">📄 PDF Report</a> '
                '<span style="color: #cbd5e1;">|</span> '
                '<a href="/file=report/demo/demo_walkthrough.mp4" download style="color: #2563eb; font-weight: 500; text-decoration: none;">🎥 Demo Video (MP4)</a> '
                '<span style="color: #cbd5e1;">|</span> '
                '<a href="/file=report/p16_codebase.zip" download style="color: #2563eb; font-weight: 500; text-decoration: none;">📦 Codebase (.zip)</a> '
                '<span style="color: #cbd5e1;">|</span> '
                '<a href="https://github.com/atharvgawali23" target="_blank" style="color: #2563eb; font-weight: 500; text-decoration: none;">🐙 GitHub Profile</a>'
                '</div>')
    with gr.Tab("Compare Stories"):
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(EXAMPLES[0], lines=3, label="Story Opening (Prompt)")
                gr.Examples(EXAMPLES, prompt)
            with gr.Column(scale=1):
                temp = gr.Slider(0.1, 1.5, SAMPLING.temperature, step=0.05, label="Creativity (temperature)")
                topk = gr.Slider(0, 100, SAMPLING.top_k, step=1, label="top-k")
                topp = gr.Slider(0.5, 1.0, SAMPLING.top_p, step=0.01, label="top-p")
                nchar = gr.Slider(150, 700, SAMPLING.max_new_chars, step=50, label="Story Length (chars)")
                seed = gr.Number(SAMPLING.seed, precision=0, label="seed")
        btn = gr.Button("Generate Story & Compare", variant="primary")
        verdict = gr.Markdown()
        with gr.Row():
            out_l = gr.Textbox(label="char-LSTM Story (Keras)", lines=12)
            out_g = gr.Textbox(label="distilGPT-2 Story (Transformer)", lines=12)
        with gr.Row():
            table = gr.Dataframe(label="Objective Metrics", wrap=True)
            plot = gr.Plot(label="Coherence Drop Across Sentences")
        audio = gr.Audio(label="🔊 Listen to the Best Story (Text-to-Speech in en-IN)", type="filepath")
        btn.click(compare, [prompt, temp, topk, topp, nchar, seed], [out_l, out_g, table, plot, verdict, audio])

    with gr.Tab("📥 Project Deliverables & Downloads"):
        gr.Markdown("### 📂 Project Files & Downloads\n"
                    "Download the official evaluation PDF report, the walkthrough demo video, and the full source codebase archive.")
        with gr.Row():
            dl_pdf = gr.File(value="report/16_Atharv_P16.pdf", label="📄 PDF Report (16_Atharv_P16.pdf)")
            dl_video = gr.File(value="report/demo/demo_walkthrough.mp4", label="🎥 Walkthrough Video (demo_walkthrough.mp4)")
            dl_code = gr.File(value="report/p16_codebase.zip", label="📦 Complete Codebase (p16_codebase.zip)")
        gr.Markdown("### 🎬 Watch Walkthrough Demo Video (101s · Indian English en-IN)")
        gr.Video(value="report/demo/demo_walkthrough.mp4", label="Demo Walkthrough Video Preview")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        allowed_paths=["report"]
    )
