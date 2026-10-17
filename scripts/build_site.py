"""Copy loudness-matched raw-input clips into _site/ and write the listening page manifest."""
import csv
import json
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"
LABELS = {
    "openai-gpt-4o-mini-tts": "OpenAI gpt-4o-mini-tts", "elevenlabs-v3": "ElevenLabs v3",
    "gemini-3.1-flash-tts": "Gemini 3.1 Flash TTS (preview)", "meta-mms-tts-ben": "Meta MMS-TTS (ben)",
    "omnivoice": "OmniVoice", "indic-parler-tts": "Indic-Parler-TTS", "voxcpm2-base": "VoxCPM2 (base)",
    "voxcpm2-bn-lora": "VoxCPM2 + Bangla LoRA (ours)",
}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(ROOT / "site", OUT)
    rows = list(csv.DictReader(open(ROOT / "data/testset.tsv", encoding="utf-8"), delimiter="\t"))
    sents = []
    for r in rows:
        clips = {}
        for p in sorted((ROOT / "audio").glob(f"*/{r['id']}.raw.mp3")):
            dst = OUT / "audio" / p.parent.name / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(p), "-af", "loudnorm=I=-23:TP=-2",
                            "-ar", "24000", "-ac", "1", "-b:a", "64k", str(dst)], check=True)
            clips[p.parent.name] = f"audio/{p.parent.name}/{p.name}"
        sents.append({"id": r["id"], "category": r["category"], "text": r["text"], "clips": clips})
    (OUT / "manifest.json").write_text(json.dumps({"labels": LABELS, "sentences": sents}, ensure_ascii=False), encoding="utf-8")
    print("sentences", len(sents))


if __name__ == "__main__":
    main()
