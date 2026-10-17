"""Synthesise the test set with an open model on a CPU runner (GitHub Actions). One system per call.

python scripts/synth_open.py <system>    systems: mms, omnivoice, indic-parler
Writes audio/<system>/<id>.raw.mp3 (+ .norm.mp3 for rows the normaliser changes) and results/rtf_<system>.csv.
"""
import csv
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bntts.bnnorm import normalise  # noqa: E402

NAMES = {"mms": "meta-mms-tts-ben", "omnivoice": "omnivoice", "indic-parler": "indic-parler-tts"}


def ref_clip(accent="bd"):
    """First line of OpenSLR-37 bn_<accent> as the cloning reference (CC-BY-SA-4.0)."""
    import glob
    import zipfile
    d = ROOT / f".cache/slr37_{accent}"
    if not d.exists():
        z = ROOT / f".cache/bn_{accent}.zip"
        z.parent.mkdir(exist_ok=True)
        subprocess.run(["curl", "-sSL", "-o", str(z), f"https://www.openslr.org/resources/37/bn_{accent}.zip"], check=True)
        zipfile.ZipFile(z).extractall(d)
    idx = sorted(glob.glob(str(d / "**/line_index.tsv"), recursive=True))[0]
    first = sorted(l.rstrip("\n").split("\t") for l in open(idx, encoding="utf-8") if "\t" in l)[0]
    wav = glob.glob(str(d / f"**/{first[0].strip()}.wav"), recursive=True)[0]
    return wav, first[1].strip()


def load(system):
    if system == "mms":
        import torch
        from transformers import AutoTokenizer, VitsModel
        tok = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
        model = VitsModel.from_pretrained("facebook/mms-tts-ben").eval()
        torch.manual_seed(0)

        def gen(text):
            with torch.no_grad():
                return model(**tok(text, return_tensors="pt")).waveform[0].numpy(), model.config.sampling_rate
        return gen
    if system == "indic-parler":
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        mid = os.environ.get("INDIC_PARLER_ID", "ai4bharat/indic-parler-tts")
        model = ParlerTTSForConditionalGeneration.from_pretrained(mid).eval()
        tok = AutoTokenizer.from_pretrained(mid)
        dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
        desc = "Aditi speaks in a clear, natural voice at a moderate pace, with very high quality audio and no background noise."
        d = dtok(desc, return_tensors="pt")

        def gen(text):
            torch.manual_seed(0)
            p = tok(text, return_tensors="pt")
            with torch.no_grad():
                a = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                   prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
            return a.cpu().numpy().squeeze(), model.config.sampling_rate
        return gen
    if system == "omnivoice":
        from omnivoice import OmniVoice
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice")
        wav, txt = ref_clip("bd")

        def gen(text):
            out = model.generate(text=text, ref_audio=wav, ref_text=txt)
            return np.asarray(out[0] if isinstance(out, (list, tuple)) else out).squeeze(), 24000
        return gen
    raise SystemExit(f"unknown system {system}")


def main():
    system = sys.argv[1]
    name = NAMES[system]
    gen = load(system)
    rows = list(csv.DictReader(open(ROOT / "data/testset.tsv", encoding="utf-8"), delimiter="\t"))
    out_dir = ROOT / "audio" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = ROOT / "results" / f"rtf_{name}.csv"
    log.parent.mkdir(exist_ok=True)
    with open(log, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "id", "variant", "synth_s", "audio_s", "rtf"])
        for r in rows:
            jobs = [("raw", r["text"])]
            if normalise(r["text"]) != r["text"]:
                jobs.append(("norm", normalise(r["text"])))
            for variant, text in jobs:
                t0 = time.perf_counter()
                audio, sr = gen(text)
                dt = time.perf_counter() - t0
                dur = len(audio) / sr
                with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
                    sf.write(tf.name, np.asarray(audio, dtype="float32"), sr)
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tf.name, "-ac", "1", "-ar", "24000",
                                    "-b:a", "64k", str(out_dir / f"{r['id']}.{variant}.mp3")], check=True)
                w.writerow([name, r["id"], variant, round(dt, 2), round(dur, 2), round(dt / max(dur, 1e-6), 3)])
                fh.flush()
                print(name, r["id"], variant, f"{dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
