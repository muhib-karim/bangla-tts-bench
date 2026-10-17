"""Round-trip ASR judge: transcribe every clip with gpt-4o-transcribe (language=bn).

OpenAI is also a contestant, so this judge is reported as a second opinion next to the
open IndicConformer judge. Writes results/asr_gpt-4o-transcribe.csv (append, resumable).
"""
import concurrent.futures as cf
import csv
import os
import pathlib

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "asr_gpt-4o-transcribe.csv"


def transcribe(path):
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
    for _ in range(4):
        with open(path, "rb") as fh:
            r = requests.post(f"{base}/v1/audio/transcriptions", timeout=120,
                              headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                              files={"file": (path.name, fh, "audio/mpeg")},
                              data={"model": "gpt-4o-transcribe", "language": "bn", "temperature": "0"})
        if r.status_code == 200:
            return r.json()["text"].strip()
    raise RuntimeError(f"{r.status_code} {r.text[:200]}")


def main():
    done = set()
    if OUT.exists():
        done = {(r["system"], r["clip"]) for r in csv.DictReader(open(OUT, encoding="utf-8"))}
    todo = [(p.parent.name, p) for p in sorted((ROOT / "audio").glob("*/*.mp3")) if (p.parent.name, p.stem) not in done]
    new = not OUT.exists()
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "a", newline="", encoding="utf-8") as fh, cf.ThreadPoolExecutor(4) as ex:
        w = csv.DictWriter(fh, ["system", "clip", "hyp"])
        if new:
            w.writeheader()
        futs = {ex.submit(transcribe, p): (s, p) for s, p in todo}
        for f in cf.as_completed(futs):
            s, p = futs[f]
            try:
                w.writerow({"system": s, "clip": p.stem, "hyp": f.result()}); fh.flush()
            except Exception as e:
                print("fail", s, p.stem, e)
    print("transcribed", len(todo))


if __name__ == "__main__":
    main()
