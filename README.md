# bangla-tts-bench

[![ci](https://github.com/muhib-karim/bangla-tts-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/muhib-karim/bangla-tts-bench/actions/workflows/ci.yml)
[![listen](https://img.shields.io/badge/listen-blind%20A%2FB-2b7a78)](https://muhib-karim.github.io/bangla-tts-bench/)

A measured, reproducible comparison of Bangla text-to-speech systems on one 50-sentence test set, plus a
Bangla LoRA for an open TTS model (VoxCPM2) scored on the same sentences before and after fine-tuning.

**Listen:** https://muhib-karim.github.io/bangla-tts-bench/ (blind A/B, system names hidden until you vote)

## The problem

Bangla has around 270 million speakers, and most TTS vendors list it as supported. Nobody publishes a measured,
side-by-side check of how well they actually read it: numbers, dates, taka amounts and phone numbers, conjunct
consonants, and the Bangla-English code-mixing people really type. This repo is that check, with every clip, every
transcript and every score committed so anyone can audit or rerun it.

## Results

Clarity is measured by round trip: synthesise the sentence, transcribe it with a speech-to-text judge
(gpt-4o-transcribe, language forced to `bn`), and compute character error rate (CER) against the reference after
normalising both sides with `bntts/bnnorm.py` (digits, dates, ৳ and phone numbers expanded to spoken words). Lower
is better. Full tables, including strict CER and per-clip transcripts: [`results/summary.md`](results/summary.md).

| system | type | CER % | error-free | news | conv. | numbers | conjuncts | Banglish | judge off-script |
|---|---|---|---|---|---|---|---|---|---|
| OpenAI gpt-4o-mini-tts | paid API | **7.5** | 27/50 | 2.0 | 0.4 | 16.1 | 4.0 | 19.3 | 0 |
| OmniVoice | open, CPU | 8.3 | **30/50** | 1.0 | 1.5 | 34.2 | 1.1 | 4.1 | 0 |
| ElevenLabs v3 | paid API | 12.8 | 28/50 | 0.6 | 5.7 | 37.0 | 3.6 | 20.8 | 3 |
| Indic-Parler-TTS | open, CPU | 14.3 | 27/50 | 14.2 | 14.7 | 23.9 | 6.7 | 9.2 | 3 |
| Meta MMS-TTS (ben) | open, CPU | 27.3 | 16/50 | 6.2 | 14.1 | 55.8 | 9.0 | 61.5 | 2 |
| VoxCPM2 base (bnnorm input) | open, GPU | 55.8 | 7/50 | 52.5 | 61.0 | 36.7 | 75.6 | 57.5 | 25 |
| VoxCPM2 + Bangla LoRA (this repo) | open, GPU | _training_ | | | | | | | |
| Gemini Flash TTS | paid API | _not measured_ (free-tier cap; a labelled n=10 subset is planned) | | | | | | | |

n = 50 sentences: news 12, conversational 12, numbers 10, conjuncts 8, Banglish 8. "Judge off-script" counts clips the judge transcribed mostly outside Bengali
script (for example in Devanagari), which is a sign the voice drifted into another language.

Speed: Meta MMS runs at a real-time factor of 0.33 on a 2-core CPU runner, Indic-Parler-TTS at about 7.5 and OmniVoice at about 30 (median, 50 clips); VoxCPM2 runs at about 2.4 on a free
cloud T4 GPU (slower than real time). API latency per sentence (median total): OpenAI 1.60 s, ElevenLabs v3 2.82 s.

### Where each system loses

- **OpenAI** reads plain Bangla almost perfectly. It stumbles on raw numerals and on Banglish: digits are sometimes
  read in the wrong number word. Fed `bnnorm`-expanded text instead of digits, its numbers CER drops from 16.1 to 5.1.
- **ElevenLabs v3** is close on news and conjuncts but worst of the paid systems on numbers (37.0); in the smoke test
  it read ঢাকা as "ঠাকা". With `bnnorm` input its numbers CER falls to 14.6.
- **Meta MMS** cannot read digits or Latin script at all (numbers 55.8, Banglish 61.5). With `bnnorm` input its
  numbers CER is 2.7, so most of its gap is the text front end, not the voice.
- **Indic-Parler-TTS** is the best open model so far (14.3) and the best of all systems on Banglish (9.2), but it
  is slow on CPU (about 7.5x real time) and drops to 23.9 on numbers; `bnnorm` input brings numbers to 9.5.
- **VoxCPM2 base** does not list Bangla among its languages. Half its clips were transcribed in Devanagari: it
  drifts into a Hindi accent. On the clips that stayed in Bangla script its CER is 11.1, so the model has the
  capacity and the fine-tune targets the drift.

## The fine-tune

Recipe adapted from a published Turkish LoRA on the same base model: LoRA on the language model and the diffusion
decoder, rank 32, learning rate 1e-4, then the same 50 test sentences plus 5 held-out training lines per
checkpoint.

- Base: `openbmb/VoxCPM2` (2B parameters, Apache-2.0).
- Data: OpenSLR-37 `bn_bd` (Google crowdsourced Bangladeshi Bangla, CC-BY-SA-4.0). The corpus is small: 6
  speakers, 1,891 utterances, 2.94 h. The LoRA uses the speaker with the most audio (0.78 h, 473 utterances),
  20 utterances held out, clips longer than 4 s dropped to fit GPU memory (263 training clips). All training text goes through `bnnorm` first.
- Compute: a free cloud T4 (16 GB). Batch 1 with gradient accumulation 4, 400 steps (about 6 epochs, 51 min),
  checkpoints at 200 and 400. The first attempt at batch 2 ran out of GPU memory; the job falls back to LM-only
  LoRA, then rank 16, if the full recipe does not fit, and records which one ran in `train_cfg_used.json`.
- Status: the full recipe (LM + diffusion decoder, rank 32) trained all 400 steps in bf16; the diffusion loss
  went from about 0.68 to about 0.52. Loading the adapter for synthesis then failed: the installed `voxcpm`
  release builds a default-rank LoRA when no config is passed, which does not match a rank-32 checkpoint. The
  kernel now passes the training LoRA config explicitly and saves the checkpoint before synthesis; the run is
  being repeated, and the "after" row is filled in when it finishes.

## Method notes

- Test set: [`data/testset.tsv`](data/testset.tsv), 50 sentences in 5 categories (news 12, conversational 12,
  numbers/dates/৳/phone 10, conjuncts 8, Banglish code-mix 8), each with a hand-checked spoken form.
- Every system gets the raw sentence. Number sentences are also synthesised from the `bnnorm` spoken form, reported
  separately, to separate front-end errors from voice errors. GPU systems were synthesised from the spoken form
  only and are labelled "(bnnorm input)".
- Code-mixed rows accept either script in the transcript.
- The judge is also a contestant's vendor (OpenAI). That bias is real; an open judge (IndicConformer) is planned as
  a second opinion and is not in the table yet.

## Limitations

- One judge, one run per sentence, 50 sentences. Differences of a couple of CER points are within noise.
- CER measures intelligibility, not naturalness. The listening page exists for that; no MOS number is claimed.
- The LoRA training voice is one speaker from a small crowdsourced corpus. It is a proof of method, not a
  production voice.
- Gemini's free tier allows a handful of requests per day, so it is not in the main table.

## Reproduce

```bash
pip install -r requirements-dev.txt
pytest -q                                   # normaliser + scorer tests
python scripts/synth_api.py                 # paid APIs (keys in env; see scripts/proxyenv.sh.example)
gh workflow run synth-open.yml              # open models on GitHub Actions CPU runners
python scripts/judge_openai.py              # round-trip transcription
PYTHONPATH=. python scripts/report.py       # rebuilds results/summary.md
```

The GPU jobs (VoxCPM2 before/after) are in [`gpu/`](gpu/): `build.sh` writes one self-contained script for a
free cloud T4 notebook.

## Licences

| component | licence |
|---|---|
| this repo's code, test set, normaliser | MIT |
| OpenSLR-37 `bn_bd` training data | CC-BY-SA-4.0 |
| Bangla LoRA weights (follow the data) | CC-BY-SA-4.0 |
| VoxCPM2 base model | Apache-2.0 |
| Meta MMS-TTS (ben) | CC-BY-NC-4.0 (non-commercial) |
| OmniVoice model | CC-BY-NC (code Apache-2.0) |
| Indic-Parler-TTS | Apache-2.0 |
| OpenAI / ElevenLabs / Gemini clips | generated under each vendor's terms; committed for evaluation only |

## Cite

See [`CITATION.cff`](CITATION.cff).
