"""Synthesise the test set with the commercial API systems.

Keys come from the environment (OPENAI_API_KEY, GEMINI_API_KEY, FAL_KEY). Base URLs
and the auth header can be overridden for a gateway. A system whose key is missing is
skipped. Writes audio/<system>/<id>.<variant>.mp3 and results/latency_<system>.csv.
"""
import argparse
import base64
import concurrent.futures as cf
import csv
import os
import pathlib
import subprocess
import tempfile
import time

import requests

from bntts.bnnorm import normalise

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _env(name, default=""):
    return os.environ.get(name, default)


def openai_tts(text):
    base = _env("OPENAI_BASE_URL", "https://api.openai.com")
    r = requests.post(f"{base}/v1/audio/speech", timeout=120, stream=True,
                      headers={"Authorization": f"Bearer {_env('OPENAI_API_KEY')}"},
                      json={"model": "gpt-4o-mini-tts", "voice": "coral", "input": text, "response_format": "mp3"})
    return r, "mp3"


def gemini_tts(text):
    base = _env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
    model = _env("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")
    hdr = _env("GEMINI_AUTH_HEADER", "x-goog-api-key")
    body = {"contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}}}
    r = requests.post(f"{base}/v1beta/models/{model}:generateContent", timeout=120, stream=True,
                      headers={hdr: _env("GEMINI_API_KEY")}, json=body)
    return r, "gemini"


def eleven_v3_fal(text):
    base = _env("FAL_BASE_URL", "https://fal.run")
    scheme = _env("FAL_AUTH_SCHEME", "Key")
    r = requests.post(f"{base}/fal-ai/elevenlabs/tts/eleven-v3", timeout=180, stream=True,
                      headers={"Authorization": f"{scheme} {_env('FAL_KEY')}"},
                      json={"text": text, "voice": "Rachel"})
    return r, "fal"


SYSTEMS = {
    "openai-gpt-4o-mini-tts": ("OPENAI_API_KEY", openai_tts),
    "gemini-3.1-flash-tts": ("GEMINI_API_KEY", gemini_tts),
    "elevenlabs-v3": ("FAL_KEY", eleven_v3_fal),
}


def to_mp3(src_bytes, kind, out):
    with tempfile.TemporaryDirectory() as td:
        if kind == "pcm24k":
            src = pathlib.Path(td, "in.raw"); src.write_bytes(src_bytes)
            inp = ["-f", "s16le", "-ar", "24000", "-ac", "1", "-i", str(src)]
        else:
            src = pathlib.Path(td, "in.bin"); src.write_bytes(src_bytes)
            inp = ["-i", str(src)]
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inp, "-ac", "1", "-ar", "24000",
                        "-c:a", "libmp3lame", "-b:a", "64k", str(out)], check=True, timeout=60)


def one(system, fn, rid, variant, text):
    out = ROOT / "audio" / system / f"{rid}.{variant}.mp3"
    if out.exists():
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        t0 = time.perf_counter()
        r, kind = fn(text)
        it = r.iter_content(8192)
        first = next(it, b"")
        ttfb = time.perf_counter() - t0
        body = first + b"".join(it)
        total = time.perf_counter() - t0
        if r.status_code == 200:
            break
        err = body[:300]
        time.sleep((15 if r.status_code == 429 else 3) * (attempt + 1))
    else:
        return dict(system=system, id=rid, variant=variant, status=f"error {r.status_code}: {err!r}")
    if kind == "gemini":
        import json
        j = json.loads(body)
        part = j["candidates"][0]["content"]["parts"][0]["inlineData"]
        to_mp3(base64.b64decode(part["data"]), "pcm24k", out)
    elif kind == "fal":
        import json
        url = json.loads(body)["audio"]["url"]
        to_mp3(requests.get(url, timeout=60).content, "mp3", out)
    else:
        to_mp3(body, "mp3", out)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                               capture_output=True, text=True, timeout=30).stdout.strip() or 0)
    return dict(system=system, id=rid, variant=variant, status="ok", chars=len(text),
                ttfb_s=round(ttfb, 3), total_s=round(total, 3), audio_s=round(dur, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(ROOT / "data/testset.tsv", encoding="utf-8"), delimiter="\t"))
    jobs = []
    for r in rows:
        jobs.append((r["id"], "raw", r["text"]))
        n = normalise(r["text"])
        if n != r["text"]:
            jobs.append((r["id"], "norm", n))
    for system in a.systems.split(","):
        keyvar, fn = SYSTEMS[system]
        if not _env(keyvar):
            print(f"skip {system}: {keyvar} not set"); continue
        log = ROOT / "results" / f"latency_{system}.csv"
        log.parent.mkdir(exist_ok=True)
        new = not log.exists()
        with open(log, "a", newline="", encoding="utf-8") as fh, cf.ThreadPoolExecutor(a.workers) as ex:
            w = csv.DictWriter(fh, ["system", "id", "variant", "status", "chars", "ttfb_s", "total_s", "audio_s"])
            if new:
                w.writeheader()
            for res in ex.map(lambda j: one(system, fn, *j), jobs):
                if res:
                    w.writerow(res); fh.flush(); print(system, res["id"], res["variant"], res["status"], flush=True)


if __name__ == "__main__":
    main()
