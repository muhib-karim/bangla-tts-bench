# Kernel A: base model "before" samples + data stats + short training probe (s/step on T4).
log("start", sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False))
mdir = setup()
rows = testset()
try:
    import torch
    log("bf16 supported", torch.cuda.is_bf16_supported())
    from voxcpm import VoxCPM
    tts = VoxCPM.from_pretrained(mdir, load_denoiser=False, optimize=False)
    synth_set(tts, f"{W}/before", rows, "before")
    del tts; import gc; gc.collect(); torch.cuda.empty_cache()
except Exception:
    log("BEFORE FAILED", traceback.format_exc()[-1500:])
try:
    prep_data()
    cfg = write_cfg(mdir, steps=int(os.environ.get("PROBE_STEPS", "20")), save_every=1000, outdir=f"{T}/probe_ckpt")
    t0 = time.time(); train(cfg, "probe_train.log"); log("probe wall s", round(time.time() - t0, 1))
except Exception:
    log("PROBE FAILED", traceback.format_exc()[-1500:])
log("done")
