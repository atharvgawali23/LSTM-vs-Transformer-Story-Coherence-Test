"""U4 multimodal output: read the more coherent sample aloud.

Primary: gTTS (Google Translate TTS endpoint). Fallback: offline espeak-ng if installed.
"""
import shutil
import subprocess
import tempfile


def speak(text: str, lang: str = "en") -> str | None:
    text = " ".join(text.split())[:600]
    if not text:
        return None
    out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    try:
        from gtts import gTTS
        gTTS(text=text, lang=lang, tld="co.in").save(out)
        return out
    except Exception:
        pass
    if shutil.which("espeak-ng"):
        wav = out.replace(".mp3", ".wav")
        subprocess.run(["espeak-ng", "-v", "en-in", "-w", wav, text], check=False)
        return wav
    return None
