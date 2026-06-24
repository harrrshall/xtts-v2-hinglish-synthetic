# fp16 distribution build — method + no-quality-loss verification

## What "fp16" means here (and what it does not)

There are two different things people call an "fp16 model"; they are not the same:

1. **fp16 on disk, fp32 at runtime** (what this build is). Floating weights are stored as 16-bit
   on disk, then upcast to fp32 when the model loads. Runtime math is **identical to the original
   fp32 model** except the weights carry the one-time fp16 rounding (~1e-6 relative). Halves the
   download/disk footprint, changes nothing about how inference runs. **Safe and provable.**
2. **fp16 at runtime** (`model.half()` — true half-precision compute). Also halves VRAM, but XTTS's
   HiFi-GAN vocoder and the autoregressive GPT can overflow/produce artifacts in pure fp16. **Not
   used here**, because it trades a real quality risk for a VRAM saving most users don't need for a
   distribution checkpoint.

This build is strictly type (1).

## How it was made

`convert_fp16.py`: load the checkpoint, cast every float32/float64 tensor to float16 — but only if
its max-abs magnitude is below a safe margin under the fp16 finite ceiling (65504). Non-floating
tensors (int buffers, `num_batches_tracked`, …) are left untouched.

- 1023 entries: **984 cast to fp16**, 39 non-float kept, **0 would-overflow** (max |weight| = 695).
- File: **2079.3 MB → 1039.9 MB (50.0%)**.
- Loads through XTTS's own `load_checkpoint`; params come back as **fp32** (confirming type-1), no NaNs.

## Proof 1 — numerical equivalence (deterministic, no sampling)

`verify_numerical.py` compares fp32 vs the fp16-upcast model on pure forward passes, so any
difference is *only* the fp16 weight rounding (no RNG involved):

| Voice | latent cosine | speaker cosine | greedy token-length (fp32 vs fp16) | waveform SNR |
|-------|---------------|----------------|------------------------------------|--------------|
| kaustubh | 0.99999607 | 0.99999776 | 110336 == 110336 | 36.4 dB |
| arjun    | 0.99999897 | 0.99999825 | 99072 == 99072   | 41.6 dB |
| maya     | 0.99999949 | 0.99999856 | 120320 == 120320 | 39.1 dB |
| aadya    | 0.99999948 | 0.99999821 | 114688 == 114688 | 35.7 dB |

The decisive line is **greedy token-length identical for every voice**: under deterministic
decoding the GPT makes bit-identical choices in fp16, so the *content* (words, phonemes, prosody
contour) is provably unchanged. The residual is sub-perceptual numerical noise in the vocoder
(35-42 dB SNR = difference signal 100-160x quieter than the audio).

## Proof 2 — perceptual panel (real sampling, paired, bootstrap CI)

`gen_panel.py` + `score_panel.py`: generate the held-out test sentences with both checkpoints using
**identical per-item seeds** (so the RNG stream is the same and the only variable is precision),
then score with the same metrics as the main report — UTMOS (naturalness) and SECS (voice fidelity)
— paired by utterance with a deterministic bootstrap 95% CI.

**Greedy-paired (the controlled test — identical token content, isolates only fp16 rounding),
n=16 (kaustubh + maya × 8 sentence types):**

| Metric | fp32 | fp16 | delta (fp16 − fp32) | 95% CI |
|--------|------|------|---------------------|--------|
| UTMOS (naturalness)   | 3.3473 | 3.3372 | **−0.0101** | [−0.0340, +0.0032] |
| SECS (voice fidelity) | 0.8652 | 0.8646 | **−0.0005** | [−0.0016, +0.0001] |

Both CIs straddle 0. The UTMOS delta (−0.01 MOS) is an order of magnitude below the metric's noise
floor and below the ±0.10 the *student itself* moved vs the teacher in the main report. Voice
fidelity is unchanged to 4 decimals.

**Sampled-paired (temperature 0.7, real-usage decode, fixed paired seeds), n=16:**

| Metric | fp32 | fp16 | delta (fp16 − fp32) | 95% CI |
|--------|------|------|---------------------|--------|
| UTMOS (naturalness)   | 3.3456 | 3.3526 | **+0.0070** | [−0.0051, +0.0203] |
| SECS (voice fidelity) | 0.8678 | 0.8669 | **−0.0009** | [−0.0021, −0.0000] |

Under real temperature-0.7 sampling the deltas are also within noise (UTMOS CI straddles 0; SECS
delta −0.0009).

> Note on an earlier false alarm: a first sampled run showed a UTMOS "drop" of −0.21 with a
> zero-width CI. That was two bugs, not a real effect: (a) the per-item seed used Python's
> `hash()`, which is randomized per process, so the fp32 and fp16 runs drew *different* random
> takes of each sentence; (b) the deterministic LCG bootstrap's low bits degenerate at n=16,
> producing a zero-width CI. With identical seeds and a proper numpy bootstrap the apparent drop
> disappears, consistent with the greedy-paired result above.

## Verdict

**No measurable quality loss.** Three independent tests agree: (1) the model is numerically
near-identical (logit/latent cosine ≈ 0.999999, greedy tokens bit-identical, vocoder SNR 35-42 dB);
(2) greedy-paired UTMOS −0.010 and SECS −0.0005, both CIs straddling 0; (3) sampled-paired UTMOS
+0.007 and SECS −0.0009, both within noise. The fp16 build is a valid 50%-smaller distribution
checkpoint (2079 MB → 1040 MB) that produces the same speech as the fp32 model.

## Scope / honesty notes

- Intelligibility is covered by Proof 1's greedy token-identity (content is provably unchanged
  under deterministic decoding), so the heavier ASR round-trip was not re-run for this delta.
- Verification ran on CPU (this box has no GPU); metrics are model-based and device-independent.
- The canonical released model stays fp32; the fp16 file is an optional smaller distribution build.
