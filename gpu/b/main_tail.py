def patch_vx():
    """The upstream train script keeps every frozen weight in fp32 in training mode and has no activation
    checkpointing, which is ~9 GB of weights before the first activation on a 16 GB T4. Patch the clone:
    frozen weights -> VX_CAST dtype, autocast in that dtype, checkpoint each MiniCPM layer, print GPU memory."""
    src = f"{T}/VoxCPM"
    p = f"{src}/scripts/train_voxcpm_finetune.py"; s = open(p).read()
    if "VX_CAST" not in s:
        a = "    tokenizer = base_model.text_tokenizer"
        assert a in s, "anchor tokenizer"
        s = s.replace(a, "    import torch as _t\n    _c = os.environ.get('VX_CAST', 'bf16')\n"
            "    if _c != 'none':\n        _d = _t.bfloat16 if _c == 'bf16' else _t.float16\n"
            "        for _n, _p in base_model.named_parameters():\n"
            "            if not _p.requires_grad and 'audio_vae' not in _n: _p.data = _p.data.to(_d)\n"
            "    print('VX_CAST', _c, 'trainable', sum(x.numel() for x in base_model.parameters() if x.requires_grad), file=sys.stderr)\n" + a, 1)
        b = "    unwrapped_model.train()"
        assert b in s, "anchor train"
        s = s.replace(b, b + "\n    import torch as _t\n    print('GPU_MEM after load GiB', round(_t.cuda.memory_allocated()/2**30, 2), file=sys.stderr, flush=True)", 1)
        open(p, "w").write(s)
    p = f"{src}/src/voxcpm/training/accelerator.py"; s = open(p).read()
    o = 'torch.amp.autocast("cuda", enabled=self.amp, *args, **kwargs)'
    if o in s:
        s = s.replace(o, 'torch.amp.autocast("cuda", enabled=self.amp, dtype=(torch.float16 if os.environ.get("VX_CAST") == "fp16" else torch.bfloat16))')
        if "import os" not in s: s = "import os\n" + s
        open(p, "w").write(s)
    p = f"{src}/src/voxcpm/modules/minicpm4/model.py"; s = open(p).read()
    o = ("            hidden_states, this_cache = decoder_layer(\n                hidden_states,\n"
         "                position_emb,\n                is_causal,\n            )")
    if "_ck.checkpoint" in s:
        pass
    elif o in s:
        s = s.replace(o, "            if self.training and torch.is_grad_enabled() and os.environ.get('VX_CKPT', '1') == '1':\n"
            "                import torch.utils.checkpoint as _ck\n"
            "                hidden_states, this_cache = _ck.checkpoint(decoder_layer, hidden_states, position_emb, is_causal, use_reentrant=False)\n"
            "            else:\n" + o.replace("\n            ", "\n                ").replace("            hidden_states", "                hidden_states", 1))
        if "import os" not in s: s = "import os\n" + s
        open(p, "w").write(s)
    else:
        log("WARN: ckpt anchor not found in minicpm4/model.py")
    log("patched VoxCPM clone (cast, autocast, checkpoint)")

# Kernel B: LoRA fine-tune of VoxCPM2 on OpenSLR-37 bn_bd, then synthesise the test set at chosen checkpoints.
import shutil
STEPS = int(os.environ.get("STEPS", "__STEPS__")); SYNTH_AT = [int(x) for x in "__SYNTH_AT__".split(",")]
BS, ACCUM = int("__BS__"), int("__ACCUM__")
log("start", STEPS, SYNTH_AT, BS, ACCUM); sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
mdir = setup(); rows = testset()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Run 2 OOMed on all three tries at the first forward, even LM-only LoRA r16 at batch 1: the upstream train
# script keeps every frozen weight in fp32 in training mode (~9 GB) and has no activation checkpointing.
# patch_vx() casts frozen weights to half precision, autocasts in the same dtype and checkpoints each layer.
patch_vx()
spk = prep_data(max_dur=4.0)
if json.load(open(f"{W}/data_stats.json"))["n_train"] < 150:
    spk = prep_data(max_dur=6.0)
ATTEMPTS = [dict(env=dict(VX_CAST="bf16"), lora=dict(enable_lm=True, enable_dit=True)),
            dict(env=dict(VX_CAST="bf16"), lora=dict(enable_lm=True, enable_dit=False, r=16, alpha=16)),
            dict(env=dict(VX_CAST="fp16"), lora=dict(enable_lm=True, enable_dit=False, r=16, alpha=16))]
for i, att in enumerate(ATTEMPTS):
    shutil.rmtree(f"{T}/ckpt", ignore_errors=True)
    os.environ.update(att["env"])
    cfg = write_cfg(mdir, steps=STEPS, save_every=100, outdir=f"{T}/ckpt", bs=BS, accum=ACCUM, lora=att["lora"])
    log("attempt", i, att)
    t0 = time.time(); train(cfg, f"train_try{i}.log"); log("train wall s", round(time.time() - t0, 1))
    tl = open(f"{W}/train_try{i}.log", errors="ignore").read()
    for ln in tl.splitlines():
        if ln.startswith(("GPU_MEM", "VX_CAST")): log(ln)
    done = os.path.isdir(f"{T}/ckpt/step_{STEPS:07d}") or os.path.isdir(f"{T}/ckpt/latest")
    if done and "OutOfMemoryError" not in tl:
        json.dump(dict(attempt=i, **att, bs=BS, accum=ACCUM, steps=STEPS, train_s=round(time.time() - t0, 1)),
                  open(f"{W}/train_cfg_used.json", "w"))
        break
    log("attempt failed", i, "OOM" if "OutOfMemoryError" in tl else tl[-600:])
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
