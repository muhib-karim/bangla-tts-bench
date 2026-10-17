"""Character error rate for Bangla round-trip ASR, with the normalisation used in the README."""
import re
import unicodedata

from .bnnorm import normalise

_PUNCT = re.compile(r"[\s\u0964\u0965.,!?;:'\"’‘“”()\-–—…/]+")


def clean(text: str, spoken: bool = True) -> str:
    """NFC, optional bnnorm spoken form, lower-case Latin, punctuation and spaces removed."""
    t = normalise(text) if spoken else unicodedata.normalize("NFC", text)
    t = unicodedata.normalize("NFC", t).lower()
    return _PUNCT.sub("", t)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str, spoken: bool = True) -> float:
    r, h = clean(ref, spoken), clean(hyp, spoken)
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


def best_cer(refs, hyp: str, spoken: bool = True) -> float:
    """Lowest CER over acceptable references (code-mixed rows accept either script)."""
    return min(cer(r, hyp, spoken) for r in refs if r)
