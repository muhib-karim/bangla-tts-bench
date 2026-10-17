# Kernel B: LoRA fine-tune of VoxCPM2 on OpenSLR-37 bn_bd, then synthesise the test set at chosen checkpoints.
import shutil
STEPS = int(os.environ.get("STEPS", "__STEPS__")); SYNTH_AT = [int(x) for x in "__SYNTH_AT__".split(",")]
BS, ACCUM = int("__BS__"), int("__ACCUM__")
log("start", STEPS, SYNTH_AT, BS, ACCUM); sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
mdir = setup(); rows = testset()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
spk = prep_data(max_dur=8.0)
# A 16 GB T4 OOMed at batch 2 / 8192 tokens with LoRA on LM + DiT. Try the full recipe at batch 1 with a tight
# token cap first, then fall back to LM-only LoRA, then a smaller rank. Each attempt starts from a clean ckpt dir.
ATTEMPTS = [dict(mbt=2560, lora=dict(enable_lm=True, enable_dit=True)),
            dict(mbt=2048, lora=dict(enable_lm=True, enable_dit=False)),
            dict(mbt=1024, lora=dict(enable_lm=True, enable_dit=False, r=16, alpha=16))]
for i, att in enumerate(ATTEMPTS):
    shutil.rmtree(f"{T}/ckpt", ignore_errors=True)
    cfg = write_cfg(mdir, steps=STEPS, save_every=100, outdir=f"{T}/ckpt", bs=BS, accum=ACCUM, **att)
    log("attempt", i, att)
    t0 = time.time(); train(cfg, f"train_try{i}.log"); log("train wall s", round(time.time() - t0, 1))
    tl = open(f"{W}/train_try{i}.log", errors="ignore").read()
    if "OutOfMemoryError" not in tl and "CUDA out of memory" not in tl:
        json.dump(dict(attempt=i, **att, bs=BS, accum=ACCUM, steps=STEPS, train_s=round(time.time() - t0, 1)),
                  open(f"{W}/train_cfg_used.json", "w"))
        break
    log("OOM on attempt", i)
import torch, gc
from voxcpm import VoxCPM
held = [json.loads(l) for l in open(f"{T}/heldout.jsonl")]
for step in SYNTH_AT:
    ck = f"{T}/ckpt/step_{step:07d}"
    if not os.path.isdir(ck):
        log("missing", ck); continue
    try:
        tts = VoxCPM.from_pretrained(mdir, load_denoiser=False, optimize=False, lora_weights_path=ck)
        synth_set(tts, f"{W}/after_step{step}", rows, f"after{step}")
        hrows = [dict(id=f"h{i:02d}", input=h["text"]) for i, h in enumerate(held[:5])]
        synth_set(tts, f"{W}/after_step{step}_heldout", hrows, f"held{step}")
        shutil.copytree(ck, f"{W}/lora/step_{step:07d}", dirs_exist_ok=True)
        del tts; gc.collect(); torch.cuda.empty_cache()
    except Exception:
        log("SYNTH FAILED", step, traceback.format_exc()[-1500:])
json.dump(held, open(f"{W}/heldout.json", "w"), ensure_ascii=False)
log("done")
