"""bnnorm: a small rule-based Bangla text normaliser (written form -> spoken form).

Covers what the test set needs and what ASR judges tend to emit: Bangla and Latin
digits, cardinals in the Indian grouping (hazar/lakh/koti), years, decimals,
percentages, Taka amounts, phone numbers read digit by digit, clock times and
common ordinals. Anything it does not recognise is left untouched.
"""
import re
import unicodedata

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

ONES = (
    "শূন্য এক দুই তিন চার পাঁচ ছয় সাত আট নয় দশ এগারো বারো তেরো চৌদ্দ পনেরো ষোলো "
    "সতেরো আঠারো উনিশ বিশ একুশ বাইশ তেইশ চব্বিশ পঁচিশ ছাব্বিশ সাতাশ আটাশ ঊনত্রিশ "
    "ত্রিশ একত্রিশ বত্রিশ তেত্রিশ চৌত্রিশ পঁয়ত্রিশ ছত্রিশ সাঁইত্রিশ আটত্রিশ ঊনচল্লিশ "
    "চল্লিশ একচল্লিশ বিয়াল্লিশ তেতাল্লিশ চুয়াল্লিশ পঁয়তাল্লিশ ছেচল্লিশ সাতচল্লিশ "
    "আটচল্লিশ ঊনপঞ্চাশ পঞ্চাশ একান্ন বাহান্ন তিপ্পান্ন চুয়ান্ন পঞ্চান্ন ছাপ্পান্ন সাতান্ন "
    "আটান্ন ঊনষাট ষাট একষট্টি বাষট্টি তেষট্টি চৌষট্টি পঁয়ষট্টি ছেষট্টি সাতষট্টি আটষট্টি "
    "ঊনসত্তর সত্তর একাত্তর বাহাত্তর তিয়াত্তর চুয়াত্তর পঁচাত্তর ছিয়াত্তর সাতাত্তর আটাত্তর "
    "ঊনআশি আশি একাশি বিরাশি তিরাশি চুরাশি পঁচাশি ছিয়াশি সাতাশি আটাশি ঊননব্বই নব্বই "
    "একানব্বই বিরানব্বই তিরানব্বই চুরানব্বই পঁচানব্বই ছিয়ানব্বই সাতানব্বই আটানব্বই নিরানব্বই"
).split()
assert len(ONES) == 100

ORDINALS = {
    "১ম": "প্রথম", "২য়": "দ্বিতীয়", "৩য়": "তৃতীয়", "৪র্থ": "চতুর্থ", "৫ম": "পঞ্চম",
    "৬ষ্ঠ": "ষষ্ঠ", "৭ম": "সপ্তম", "৮ম": "অষ্টম", "৯ম": "নবম", "১০ম": "দশম",
}


def cardinal(n: int) -> str:
    """Spoken Bangla for a non-negative integer (Indian grouping)."""
    if n < 100:
        return ONES[n]
    parts = []
    for size, word in ((10_000_000, "কোটি"), (100_000, "লাখ"), (1000, "হাজার")):
        if n >= size:
            parts.append(f"{cardinal(n // size)} {word}")
            n %= size
    if n >= 100:
        parts.append(ONES[n // 100] + "শো")
        n %= 100
    if n:
        parts.append(ONES[n])
    return " ".join(parts)


def year(n: int) -> str:
    """Years 1100-1999 are read as 'উনিশশো একাত্তর'; others as cardinals."""
    if 1100 <= n <= 1999:
        hi, lo = divmod(n, 100)
        return ONES[hi] + "শো" + (" " + ONES[lo] if lo else "")
    return cardinal(n)


def digits(s: str) -> str:
    return " ".join(ONES[int(c)] for c in s if c.isdigit())


def _num(s: str) -> str:
    s = s.replace(",", "")
    if "." in s:
        whole, frac = s.split(".", 1)
        return f"{cardinal(int(whole))} দশমিক {digits(frac)}"
    n = int(s)
    return year(n) if len(s) == 4 else cardinal(n)


def _taka(m):
    amount = m.group(1).replace(",", "")
    whole, _, frac = amount.partition(".")
    out = f"{cardinal(int(whole))} টাকা"
    if frac and int(frac):
        out += f" {cardinal(int(frac.ljust(2, '0')[:2]))} পয়সা"
    return out


def _time(m):
    h, mm = int(m.group(1)), int(m.group(2))
    return f"{ONES[h]}টা" + (f" {ONES[mm]}" if mm else "")


def normalise(text: str) -> str:
    """Return the spoken form of `text`."""
    t = unicodedata.normalize("NFC", text)
    for k, v in ORDINALS.items():
        t = t.replace(unicodedata.normalize("NFC", k), v)
    t = t.translate(_BN_DIGITS)
    t = re.sub(r"(?<!\d)(0\d{4})-?(\d{6})(?!\d)", lambda m: digits(m.group(1) + m.group(2)), t)
    t = re.sub(r"৳\s?([\d,]+(?:\.\d+)?)", _taka, t)
    t = re.sub(r"([\d,]+(?:\.\d+)?)\s?টাকা", lambda m: _taka(m), t)
    t = re.sub(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", _time, t)
    t = re.sub(r"([\d,]*\d(?:\.\d+)?)\s?%", lambda m: _num(m.group(1)) + " শতাংশ", t)
    t = re.sub(r"\d[\d,]*(?:\.\d+)?", lambda m: _num(m.group(0).rstrip(",")), t)
    return t
