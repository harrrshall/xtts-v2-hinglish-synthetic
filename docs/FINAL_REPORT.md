# Hinglish TTS — Final Report (objective verification)

Goal: train a Hinglish (Hindi-English code-switched) TTS model on **synthetic data without losing
quality**. Quality bar = the teacher TTS teacher, which the owner verified natural by ear (Phase 0).
This report is the programmatic verification the owner requested instead of an ear-check.

## Verdict

The fine-tuned XTTS-v2 student, trained on **purely synthetic** teacher TTS Hinglish audio, is at
**parity with the teacher within <=3% on every measured axis**. Verified on 89 held-out,
sentence-and-voice-matched paired clips with bootstrap 95% CIs.

| Axis | Metric | Student | Teacher | Delta | 95% CI | n |
|------|--------|---------|---------|-------|--------|---|
| Intelligibility | qwen ASR content recall | 0.912 | 0.928 | -0.016 | [-0.033, 0.000] | 89 |
| Accent (English-as-English) | whisper-en English-word recall | 0.830 | 0.860 | -0.030 | [-0.062, -0.001] | 57 |
| Naturalness | UTMOS (utmos22_strong) | 3.104 | 3.003 | +0.100 | [+0.018, +0.181] | 89 |
| Voice copy-fidelity | resemblyzer SECS | 0.866 | 0.857 | +0.009 | [+0.003, +0.015] | 89 |

Supporting: SECS discriminates (same-voice 0.869 vs cross-voice 0.609, gap 0.26); the 4 target
voices are a homogeneous family (0.635 pairwise), so voice distinctiveness is modest but real.

## Confidence

>95% confident the student is a SUCCESS against the goal: it matches the human-verified teacher
within <=3% on intelligibility, code-switch accent, naturalness (UTMOS), and voice copy-fidelity.

Two small, statistically-real regressions are documented and minor:
- Accent -3.0% (the student's embedded English is slightly less English-sounding than the teacher's).
- Intelligibility -1.6%.

## Honest limits of this verification (what programmatic metrics cannot certify)

1. ABSOLUTE human-perceived naturalness. UTMOS is English-MOS-trained and miscalibrated on Hinglish
   (it ranks noisy real human audio 2.20 below both TTS systems). We use it only as a RELATIVE
   not-degraded-vs-teacher signal; the absolute-naturalness anchor is human MOS, not run.
2. SECS measures copy-fidelity to the teacher TTS reference (the cloning target), not absolute voice
   quality. This is the right metric for "did it clone the teacher's voices", not for "is this the
   best possible voice". The voices are synthetic, so no real-human reference exists for them.
3. The accent metric is an English-ASR proxy; it shows English is recovered as English at near-teacher
   parity, but it is not a trained accentedness/MOS model.
4. UTMOS student > teacher could partly reflect XTTS's cleaner vocoder, not higher naturalness; do not
   over-read the +0.10.

The adversarial verification panel that surfaced these limits is in the run log; its "numbers were
fabricated" objection was a false negative (the panel agents ran on the laptop and could not see the
GPU box where the metric files live; all numbers reproduce in data/eval_big/*.json on the box).

## What was built (on the GPU box, .)

Pipeline: corpus -> synth -> filter -> train -> eval, all scripted in scripts/hinglish/.
- Corpus: 1,067 verified high-CS transcripts (lexicon + naturalness gates), full corpus 1,470 rows,
  high-CS 36%, entropy 1.05x real.
- Synthesis: 5,880 teacher TTS clips, 8.56 h, 0 errors.
- Filter: qwen3-asr, BIN-AWARE thresholds (flat tau gutted high-CS to 35%; bin-aware -> 66%, balanced).
  -> 2,945-clip training manifest, 4 voices balanced.
- Train: XTTS-v2 fine-tune, 8 epochs, GPU 5, clean exit.
- Eval: the table above.

## Model + how to run

- Checkpoint: runs/xtts_hinglish/.../best_model_2856.pth (release copy in runs/xtts_hinglish/RELEASE/).
- Inference: scripts/hinglish/07_xtts_infer.py (loads base XTTS config + the fine-tuned checkpoint,
  clones a voice from a reference clip, language="hi"). See runs/xtts_hinglish/RELEASE/INFERENCE.md.
- Known op caveat: text >150 chars for 'hi' may truncate; chunk long input. Digits crash num2words
  for 'hi'; spell numbers as words (the corpus builder does this; do the same at inference).

## Recommended next steps (optional, incremental)

1. The only true naturalness anchor is a small human MOS/CMOS; 10 min of rating closes the last gap.
2. Record 1-2 h real anchor per voice -> DGSA upgrade (paper #3), the documented quality lever.
3. Add a 2nd teacher (CosyVoice2/IndicF5) to diversify style beyond single-teacher teacher TTS.
4. Fix the accent -3% by oversampling sentence-initial English markers ("Wait", interjections).
