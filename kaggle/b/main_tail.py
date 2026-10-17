# Kernel B: LoRA fine-tune of VoxCPM2 on OpenSLR-37 bn_bd, then synthesise the test set at chosen checkpoints.
import shutil
STEPS = int(os.environ.get("STEPS", "__STEPS__")); SYNTH_AT = [int(x) for x in "__SYNTH_AT__".split(",")]
BS, ACCUM = int("__BS__"), int("__ACCUM__")
log("start", STEPS, SYNTH_AT, BS, ACCUM); sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
mdir = setup(); rows = testset()
spk = prep_data()
cfg = write_cfg(mdir, steps=STEPS, save_every=100, outdir=f"{T}/ckpt", bs=BS, accum=ACCUM)
t0 = time.time(); train(cfg, "train.log"); log("train wall s", round(time.time() - t0, 1))
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
