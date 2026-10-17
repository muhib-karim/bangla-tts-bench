# GPU jobs

`common.py` plus one `main_tail.py` per job; `build.sh` inlines the test set and the normaliser and writes a single
`kernel.py` that runs as a notebook script on a free 2x T4 GPU instance (internet on). Nothing here runs locally.

- `a/`: base VoxCPM2 reads the 50 test sentences ("before"), prepares OpenSLR-37 bn_bd, and runs a short training probe.
- `b/`: LoRA fine-tune (Ahenk-style recipe), then the same 50 sentences plus 5 held-out training-corpus lines per checkpoint ("after").
