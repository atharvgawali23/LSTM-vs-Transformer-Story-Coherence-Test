"""U5 safety guardrail: input moderation + output screening for the public demo."""
import re

MAX_PROMPT_CHARS = 400
# Small, explicit blocklist (slurs/sexual-violence/self-harm instructions). Kept generic on purpose.
_BLOCK = [r"\bkill (yourself|urself)\b", r"\bsuicide method", r"\brape\b", r"\bchild porn", r"\bterror attack plan",
          r"\bn[i1]gg", r"\bf[a@]gg[o0]t"]
_PII = [r"\b\d{3}-\d{2}-\d{4}\b", r"\b(?:\d[ -]*?){13,16}\b", r"[\w.+-]+@[\w-]+\.[\w.]+"]


def check_input(prompt: str):
    """Returns (ok, cleaned_prompt, reason)."""
    p = (prompt or "").replace("\r", "")
    if len(p) > MAX_PROMPT_CHARS:
        p = p[:MAX_PROMPT_CHARS]
    for pat in _BLOCK:
        if re.search(pat, p, re.I):
            return False, "", "Prompt blocked by safety guardrail (harmful content)."
    for pat in _PII:
        p = re.sub(pat, "[REDACTED]", p)
    return True, p, ""


def screen_output(text: str):
    """Mask blocked terms in generated text; return (text, n_masked)."""
    n = 0
    for pat in _BLOCK:
        text, k = re.subn(pat, "[filtered]", text, flags=re.I)
        n += k
    return text, n
