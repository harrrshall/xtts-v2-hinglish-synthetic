# fp16 distribution build + no-quality-loss verification

Halves the checkpoint on disk (2079 MB → 1040 MB) with no quality loss. This is **fp16 on disk,
fp32 at runtime**: weights are stored 16-bit and upcast to fp32 when XTTS loads them, so inference
math is unchanged (the weights just carry a ~1e-6 one-time rounding). It is *not* `model.half()`
runtime half-precision, which XTTS's vocoder can't do safely.

Full results: [`../../../docs/FP16_VERIFICATION.md`](../../../docs/FP16_VERIFICATION.md).

## Files
- `convert_fp16.py` — cast float tensors to fp16, but only when their magnitude is safely below the
  fp16 finite ceiling (so the cast can never produce `inf`). Non-float tensors are left alone.
- `verify_numerical.py` — deterministic equivalence proof: conditioning-latent / speaker-embedding
  cosine + greedy-decode token identity + waveform SNR between fp32 and the fp16-upcast model.
- `gen_panel.py` — synthesize a held-out panel with one checkpoint; `--greedy` for a truly paired
  (identical-token) comparison, or default temperature-0.7 sampling with deterministic paired seeds.
- `score_panel.py` — UTMOS (naturalness) + SECS (voice fidelity), paired, with a numpy bootstrap CI.
- `xtts_patch.py` — routes XTTS audio IO through soundfile (needed only when torch/torchaudio/
  torchcodec versions are skewed, e.g. a CPU box; harmless otherwise). Imported by the scripts.

## Run
```bash
# 1. convert
python convert_fp16.py --in model.pth --out model_fp16.pth

# 2. prove numerical equivalence (needs config.json + vocab.json + refs/<voice>.wav)
python verify_numerical.py --base <model_dir> --fp32 model.pth --fp16 model_fp16.pth \
    --refs-dir <model_dir>/refs --out numerical_equivalence.json

# 3. perceptual confirmation (greedy-paired = isolates fp16 rounding)
python gen_panel.py --base <model_dir> --ckpt model.pth      --refs-dir <model_dir>/refs --out-dir gpanel_fp32 --greedy
python gen_panel.py --base <model_dir> --ckpt model_fp16.pth --refs-dir <model_dir>/refs --out-dir gpanel_fp16 --greedy
python score_panel.py --fp32-manifest gpanel_fp32/manifest.jsonl --fp16-manifest gpanel_fp16/manifest.jsonl \
    --refs-dir <model_dir>/refs --out gpanel_scores.json
```
Scripts run on CPU by default (device-agnostic); they don't require a GPU.
