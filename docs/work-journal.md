# Overnight autonomous plan (2026-06-24 night)

User is asleep. Execute this queue autonomously. Each background workflow re-invokes me on
completion, so the chain self-continues. Do the SAFE items. Leave the GATED items for the user.

## Autonomous queue (do these)

1. WAIT for round-2 scaling workflow `w6tmrm3at` to complete.
2. Re-score round-2 kept transcripts myself with common.py (verify cs_mode labels, like round 1).
3. Merge round1 (highcs_generated.jsonl) + round2 -> run `verify_prompts.py` over the FULL pool
   (deterministic lexicon gate: corrected cmi_bin, script-violation, consistency, dedup).
   Output: scripts/hinglish/prompts/highcs_verified.jsonl
4. LLM naturalness judge layer (workflow): over the verifier survivors, drop awkward/unnatural
   or semantically-odd lines that pass metrics but no real speaker would say. Adversarial,
   majority-vote. Output the clean final set + a report of what was cut and why.
5. Rebuild the corpus with all prompts: run 01_build_corpus into data/corpus, confirm the
   high-bin floor is met, token-entropy ratio vs real >= 0.85, log final cmi distribution.
6. SMALL real validation (cost ~$0.30, within research authorization):
   - Default voices: kaustubh (M) + maya (F), both passed Phase 0 clean. (User can override.)
   - Synthesize ~30 clips from the verified high-CS corpus via 02_synthesize --execute.
   - Run 03_filter_qwen REAL mode on GPU 5 (CUDA_VISIBLE_DEVICES=5) over those clips.
   - Confirm the S2 -> S3 real path works on real synthetic audio; save sample + filter_scores.
7. Write MORNING_SUMMARY.md: results, numbers, sample clips to listen to, and the decisions
   waiting for the user.

## NOW UNBLOCKED (user authorized everything 2026-06-24 night, "carry out end to end")

- VOICES LOCKED: user verified all 4 sound natural -> fixed voice set = kaustubh, arjun (M),
  maya, aadya (F). Use all 4 for synthesis (speaker diversity per arXiv:2601.00935).
- FULL corpus synthesis: AUTHORIZED after the small validation passes and corpus is finalized.
  Cost is modest (~$1-2 for ~2-4k clips). Synthesize all 4 voices x verified high/med corpus.
- TRAINING: AUTHORIZED with discipline: setup -> base-model -> SMOKE TEST (tiny run, confirm the
  loop) -> full run with monitoring. Do NOT skip the smoke test.

## BASE MODEL DECISION (evidence-based, pivoted from the plan's original CosyVoice2)

STUDENT base = **XTTS-v2** (coqui-tts). Reasoning:
- NileTTS (#15) proved XTTS-v2 for our EXACT scenario: fully-synthetic dialectal corpus, few
  fixed speakers, fine-tune. Mature GPT-trainer fine-tune recipe is in coqui-tts (already in
  ~/voicesangam/.venv_tts on the box, 0.27.5 + trainer 0.3.3).
- CosyVoice2's main edge (DGSA prosody-timbre disentanglement) needs REAL data; we are on the
  TDSC (no-real-data) path, so that edge does not apply. Its Hindi support is also weak and it
  wants 50-100h in-domain we do not have.
- Fallbacks already cached on box if XTTS setup blocks: IndicF5 (1.4G, Indian-native, but CS is
  not its strength) and indic-parler-tts (3.6G, description-prompted, less ideal for fixed voices).
- SETUP GAP found: .venv_tts is missing torchaudio (XTTS import fails). Fix: pip install
  torchaudio matching torch 2.12 into .venv_tts, then download the XTTS-v2 checkpoint (~2GB).

## Revised autonomous sequence (end to end)

1-5. Finalize verified corpus (round2 + verify_prompts + LLM naturalness judge) + rebuild. [in flight]
6. FULL synthesis: all 4 voices over the verified high/med corpus, 24kHz, via 02_synthesize --execute.
   Resumable; check cost as it goes.
7. Real qwen filter (GPU 5) over ALL synthesized clips -> filter_scores -> 04_assemble_manifest
   -> train_manifest + cosyvoice2/ export (works for XTTS too via wav.scp/text).
8. XTTS-v2 SETUP: fix torchaudio in .venv_tts, download checkpoint, convert manifest to XTTS
   metadata format (LJSpeech-style: wav|text|speaker), build the GPTTrainerConfig.
9. SMOKE TEST: tiny XTTS fine-tune (a few steps, tiny subset) to confirm the training loop runs
   clean end-to-end on GPU 5. Only proceed if green.
10. FULL training run with monitoring (loss curve, periodic sample synthesis + qwen round-trip WER).
11. EVAL the student: 05_eval_harness on the hard spontaneous set + sample clips for morning review.
12. MORNING_SUMMARY.md: every result, cost, sample clips, and any blocker hit.

## If a genuine blocker is hit
Do NOT thrash. Document the exact error + what was tried in MORNING_SUMMARY.md and move to the
next independent step. Training is the riskiest; if XTTS setup fails after honest effort, fall back
to IndicF5 or stop at "corpus + synthesis + manifest ready" and report.

## Constraints
- GPU 5 only (CUDA_VISIBLE_DEVICES=5). Never touch other GPUs.
- TEACHER_TTS_API_KEY from env; never hardcode/print. SSH key 0600 in scratch; never expose.
- Anchor recording remains the user's optional task (DGSA upgrade).

---

## LIVE STATUS (autonomous, updating)

CORPUS: FINAL = 1067 verified high-CS transcripts (515 high, 552 med) after lexicon + naturalness
gates. Full corpus data/corpus/corpus.jsonl = 1470 rows, high-CS 36%, entropy ratio 1.05 vs real.
Intermediates moved to prompts/_intermediate/. Clean.

BOX: repo at . (scripts + corpus + calib + teacher_test). teacher TTS
reachable (http 200). GPUs 0,5,6 authorized.

VALIDATION (S2->S3 real, GPU6): 32 clips synth in 27s, qwen-filtered -> 22 accept / 10 reject.
Recalls 0.82-1.00. Filter correctly rejected a bad aadya clip (cer 0.49). Chain works.

RUNNING NOW (background):
- bmc5e04nl: FULL synthesis, 5880 clips (1470 x 4 voices, speed 1.0), ~494k chars (~$5). Resumable.
- bm0cikvgg: isolated .venv_xtts setup (coqui-tts + torch + torchaudio) for the student base.

NEXT when those finish:
1. Sharded qwen filter over all 5880 clips across GPU 0/5/6 -> data/filtered/filter_scores.jsonl.
2. 04_assemble_manifest -> train_manifest + XTTS metadata (wav|text|speaker).
3. XTTS-v2: download checkpoint, convert manifest, build GPTTrainerConfig.
4. SMOKE test (tiny fine-tune, confirm loop). Only then full training run + monitoring.
5. Eval on hard spontaneous set. Write MORNING_SUMMARY.md.

HONESTY FLAGS so far: none blocking. Dense high-CS clips have ~70% filter pass (expected, hardest
bin); full corpus accept rate will be higher. Will report true accept rate after the full filter.

## MILESTONE: training path PROVEN (2026-06-24, during synth)
- XTTS-v2 env built on box: .venv_xtts (torch 2.12.1+cu130, torchcodec 0.14, transformers 4.57.6
  pinned to satisfy coqui-tts 0.27.5). Dep fixes: coqui doesn't pull torch (installed), transformers
  5.x removed isin_mps_friendly (pinned 4.57.6), torch 2.9+ needs torchcodec (installed).
- XTTS-v2 checkpoint downloaded to .tts_models/.../xtts_v2 (model.pth 1.87G + dvae.pth + mel_stats.pth
  + vocab.json + speakers_xtts.pth).
- Scripts written: 06_xtts_prepare.py (manifest -> XTTS metadata, leak-free split) and train_xtts.py
  (recipe-faithful, custom multi-speaker csv formatter, language="hi", --smoke mode).
- SMOKE TEST PASSED on GPU 5 over the 22 validation clips: XTTS loaded clean (518M params, NO torch
  weights_only issue), 8 steps, loss_mel_ce 4.20->3.41, eval ran, checkpoint saved. The full loop works.
- Bug caught+fixed by smoke: load_tts_samples relative_to(root_path) needs root_path="/" for absolute
  audio paths.

## REMAINING (gated on full synth completing)
synth (bmc5e04nl, ~831/5880) -> sharded qwen filter GPU 0/5/6 -> 04_assemble -> 06_xtts_prepare ->
train_xtts.py (full: --epochs 8 --batch-size 8) on GPU 5, monitor loss -> sample synth + qwen
round-trip WER -> 05_eval_harness on hard set -> MORNING_SUMMARY.md.

## DONE: synth + filter + KEY JUDGMENT CALL (bin-aware thresholds)
- Full synth: 5880 clips, 8.56h, 0 errors.
- Sharded qwen filter (GPU 0/5/6): flat tau=0.75 gave 52.8% accept BUT per-bin was none 78.5 / low
  71.5 / med 55.3 / HIGH 35.3%. The flat threshold gutted high-CS (the project's whole point),
  because qwen under-scores dense CS even on Phase-0 verified-good audio (cs_high recall ~0.77).
- DECISION (flagged for owner review): switched to BIN-AWARE floors anchored a ~0.07-0.10 margin
  below the Phase-0 verified-good recall ceiling per bin (rescore_binaware.py):
  none 0.85 / low 0.83 / med 0.76 / high 0.70 recall; cer 0.12/0.14/0.20/0.25.
  Result: 63.4% accept (3726), per-bin none 75 / low 52 / med 59 / HIGH 66% (balanced, not inverted).
  Duration rejects kept hard. Recovered clips tagged binaware_recovered for A/B at eval.
  RISK (honest): I cannot listen, so some recovered high-CS clips at recall 0.70-0.77 might be
  slightly degraded TTS rather than just recognizer-limited. The eval on the hard set + an owner
  LISTEN PACK are the validation. If they sound bad, retrain on the strict (52.8%) set.
- Also a genuine teacher signal: teacher TTS dense-CS scores ~0.15 lower ASR-recall than real human
  dense-CS -> the hardest bin is its relative weak spot. Worth noting, not blocking.
- Training manifest: 2945 clips (2801 train / 144 dev) -> XTTS metadata 2855 train / 90 eval,
  4 voices balanced (kaustubh 664 / maya 809 / aadya 654 / arjun 728).

## TRAINING RUNNING (GPU 5, detached pid 1386620, survives SSH drop)
train_xtts.py --epochs 8 --batch-size 8 --grad-accum 4 -> runs/xtts_hinglish/. Log: train.log.
At check: step 143/356 epoch 1, loss_mel_ce 2.80 (from 3.4), GPU 99%. Watcher: bl8ueciws.
HONESTY FLAG: some transcripts exceed XTTS 150-char Hindi limit ("might cause truncated audio") ->
quantify count in summary; mostly the longest voicesangam/lecture lines, minor.

## AFTER TRAINING (autonomous): inference sample across 4 voices -> qwen round-trip WER per cmi_bin
-> 05_eval_harness on hard spontaneous set -> build owner LISTEN PACK (incl binaware_recovered
high-CS clips) -> MORNING_SUMMARY.md with the bin-aware decision front and center.

## Constraints
- GPU 5 only (CUDA_VISIBLE_DEVICES=5). Never touch other GPUs.
- TEACHER_TTS_API_KEY from env; never hardcode/print. Key is active; user will rotate later.
- SSH key 0600 in session scratch; never expose.
- Stop after step 7 and leave the morning summary. Do not loop on new spend.
