import os, subprocess, sys, json, time, csv, glob, base64, zipfile, collections, io, traceback
W = "/kaggle/working"; T = "/kaggle/temp"; os.makedirs(T, exist_ok=True)
def sh(c, check=True):
    print("+", c, flush=True); return subprocess.run(c, shell=True, check=check)
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
    with open(f"{W}/progress.log", "a") as f: print(time.strftime("%H:%M:%S"), *a, file=f)

def setup():
    sh("pip install -q 'voxcpm' soundfile 2>&1 | tail -3")
    if not os.path.exists(f"{T}/VoxCPM"):
        sh(f"git clone -q --depth 1 https://github.com/OpenBMB/VoxCPM {T}/VoxCPM")
    from huggingface_hub import snapshot_download
    return snapshot_download("openbmb/VoxCPM2", local_dir=f"{T}/VoxCPM2")

def testset():
    rows = list(csv.DictReader(io.StringIO(base64.b64decode(TESTSET_B64).decode()), delimiter="\t"))
    exec(base64.b64decode(BNNORM_B64).decode(), globals())
    for r in rows: r["input"] = normalise(r["text"])
    return rows

def synth_set(tts, outdir, rows, tag):
    import soundfile as sf, numpy as np
    os.makedirs(outdir, exist_ok=True); sr = tts.tts_model.sample_rate; stats = []
    for r in rows:
        t0 = time.time()
        try:
            wav = tts.generate(text=r["input"], cfg_value=2.0, inference_timesteps=10, normalize=False)
        except Exception as e:
            log(tag, r["id"], "ERR", repr(e)[:300]); continue
        dt = time.time() - t0; dur = len(wav) / sr
        sf.write(f"{T}/x.wav", np.asarray(wav, dtype="float32"), sr)
        sh(f"ffmpeg -loglevel error -y -i {T}/x.wav -ac 1 -ar 24000 -b:a 64k {outdir}/{r['id']}.mp3")
        stats.append(dict(id=r["id"], synth_s=round(dt, 2), audio_s=round(dur, 2)))
        log(tag, r["id"], f"{dt:.1f}s for {dur:.1f}s audio")
    json.dump(stats, open(f"{outdir}/stats.json", "w"))
    return stats

def prep_data(max_hours=6.0):
    """OpenSLR-37 bn_bd -> 16 kHz wavs + train/heldout jsonl for the single speaker with most audio."""
    import soundfile as sf
    z = f"{T}/bn_bd.zip"
    if not os.path.exists(z):
        sh(f"wget -q -O {z} https://www.openslr.org/resources/37/bn_bd.zip || wget -q -O {z} https://us.openslr.org/resources/37/bn_bd.zip")
    d = f"{T}/bn_bd"
    if not os.path.exists(d):
        sh(f"mkdir -p {d} && cd {d} && unzip -q {z}")
    idx = glob.glob(f"{d}/**/line_index.tsv", recursive=True)[0]
    wavs = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{d}/**/*.wav", recursive=True)}
    lines = [l.rstrip("\n").split("\t") for l in open(idx, encoding="utf-8") if "\t" in l]
    by = collections.defaultdict(list)
    for fid, text in lines:
        if fid.strip() in wavs: by[fid.strip().split("_")[1]].append((fid.strip(), text.strip()))
    dur = {}
    for spk, items in by.items():
        dur[spk] = sum(sf.info(wavs[f]).duration for f, _ in items) / 3600
    top = sorted(dur, key=dur.get, reverse=True)
    log("speakers", len(by), "utts", len(lines), "top hours", [(s, round(dur[s], 2), len(by[s])) for s in top[:8]])
    spk = top[0]; items = by[spk]
    exec(base64.b64decode(BNNORM_B64).decode(), globals())
    os.makedirs(f"{T}/d16", exist_ok=True)
    out = []
    for f, text in items:
        dst = f"{T}/d16/{f}.wav"
        if not os.path.exists(dst):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wavs[f], "-ac", "1", "-ar", "16000", dst], check=True)
        out.append({"audio": dst, "text": normalise(text)})
    held, train = out[:20], out[20:]
    with open(f"{T}/train.jsonl", "w") as fh:
        for x in train: fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(f"{T}/heldout.jsonl", "w") as fh:
        for x in held: fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    json.dump({"speaker": spk, "hours": dur[spk], "n_train": len(train), "n_heldout": len(held),
               "all_speakers": {s: [round(dur[s], 3), len(by[s])] for s in top}}, open(f"{W}/data_stats.json", "w"), indent=1)
    log("data ready speaker", spk, "hours", round(dur[spk], 2), "train", len(train))
    return spk

def write_cfg(mdir, steps, save_every, outdir, bs=2, accum=8):
    import yaml
    cfg = yaml.safe_load(open(f"{T}/VoxCPM/conf/voxcpm_v2/voxcpm_finetune_lora.yaml"))
    cfg.update(pretrained_path=mdir, train_manifest=f"{T}/train.jsonl", val_manifest=None, batch_size=bs,
               grad_accum_steps=accum, num_workers=2, preprocessing_num_workers=2, num_iters=steps, max_steps=steps,
               log_interval=5, valid_interval=100000, save_interval=save_every, save_path=outdir,
               tensorboard=f"{T}/tb", warmup_steps=min(100, max(1, steps // 10)))
    yaml.safe_dump(cfg, open(f"{T}/cfg.yaml", "w"), allow_unicode=True)
    return f"{T}/cfg.yaml"

def train(cfg, logname):
    return sh(f"cd {T}/VoxCPM && python scripts/train_voxcpm_finetune.py --config_path {cfg} 2>&1 | tee {W}/{logname} | grep -E -i 'step|loss|error|Traceback|memory|it/s|s/it' | tail -n 400", check=False)
