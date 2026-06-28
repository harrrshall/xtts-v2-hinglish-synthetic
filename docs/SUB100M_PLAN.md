# Sub-100M Plan — Hinglish-only, fixed-voice TTS

Research-backed, **measured** plan to take the 265M Hinglish student to **<100M without quality loss**,
exploiting the product constraints: **Hinglish-only**, a **small fixed voice set** (no general zero-shot
cloning), and **fine-tuning allowed**. Backed by 4 web-cited research passes (2026-06-26) + a real param
audit of our checkpoint. Quality is gated on the same TOST/BCa equivalence harness as every prior tier.

---

## 1. Where the parameters actually are (measured from `round1_rft/best_model.pth`)

| component | params | % | lever |
|---|---|---|---|
| **transformer blocks** (`gpt.h.*`) | **201.5M** | 76.1% | **width-prune (primary)** |
| **speaker pathway** (conditioning encoder + perceiver) | **46.3M** | 17.5% | **DROP — fixed voices** |
| text head (untied) | 6.85M | 2.6% | trim → Hinglish |
| text embedding | 6.84M | 2.6% | trim → Hinglish |
| audio mel embed + head | 2.1M | 0.8% | fixed by frozen DVAE |
| positions + norms | 1.0M | 0.4% | — |
| **TOTAL** | **264.7M** | | |

This audit (`param_breakdown.py`) turns the research estimates into facts: the speaker pathway is **46.3M**
(droppable), and the text path is **13.7M and untied** (so trimming saves on both sides).

---

## 2. The arithmetic — <100M is comfortable, not a stretch

Three levers, applied in order:

1. **Drop the speaker pathway (fixed voices): −46.3M.** With ~4 fixed voices we don't need an encoder that
   *produces* conditioning from arbitrary audio — we **pre-compute** each voice's 32 perceiver latents + the
   HiFi-GAN d-vector once and store them (4 tiny baked tensors). Quality on *seen* voices holds; we only give
   up zero-shot cloning, which we don't ship. → ~218M.
2. **Trim the text vocab 6681 → ~2500 (Hinglish): ~−8.6M.** Devanagari + ASCII + digits + punctuation needs
   far fewer tokens than a 17-language BPE. Near-zero quality cost for the retained language. → ~210M.
3. **Width-prune the blocks d=1024 → 640: 201.5M → ~78M.** Blocks scale as d²: (640/1024)² = 0.39. Keep
   16-20 layers (deep-narrow). → **~85M total** (78M blocks + ~7M trimmed-text/audio/pos + ~1M adapter).

**MEASURED** (instantiated the real GPT class at each width, 16 layers — `width_count.py`):

| d | full model | drop-speaker + trim-vocab + adapter | verdict |
|---|---|---|---|
| 1024 (now) | 264.7M | 209.8M | current |
| 768 | 152.9M | 120.4M | over 100M |
| **640** | **108.4M** | **84.6M** | ✅ **target** |
| 512 | 71.5M | 55.1M | stretch (measured-risk) |
| 448 | 55.9M | 42.7M | edge |

The **fixed-voice drop (−19M at d=640) is what pulls it under 100M** — so Hinglish-only + fixed voices is
load-bearing, not optional. Architecture is **config-driven**: `gpt_n_model_channels` sets the width; the
conditioning encoder + perceiver auto-build at the student width; the **only** custom piece is a
**640→1024 adapter on the GPT latents before the frozen HiFi-GAN** (decoder_input_dim=1024), trained by the
latent-MSE distillation loss (the SECS protector).

**Why this is safe here, when the general case wasn't:** the earlier research said *voice identity (SECS)*
is the wall at <100M, because width cuts squeeze the speaker subspace and RL can't rebuild lost capacity.
**Fixed voices dissolve that wall** — we bake the voices, so the backbone never carries general
speaker-manifold capacity. Code-switch *intelligibility* becomes the binding axis instead, which RL **can**
defend (it's behavior, and we have the reward + harness).

---

## 3. The honest floor

- **d=640 / ~85M:** target. Sits above MARS6's validated d=512/70M AR-codec floor, buying margin for the
  extra demand of code-switch (two phonotactic systems + the Hindi↔English boundary raise the intelligibility
  floor vs a clean monolingual voice).
- **d=512 / ~50-70M:** the edge. MARS6-class; needs anti-collapse sampling (repetition-aware sampling, top-p
  backoff, repetition penalty) + heavy distillation. Reachable but riskier.
- **< d=384 / ~40M:** do not go here on the pure-AR-over-DVAE path. The failure is in the **AR content stream**
  (WER blowups, skips, repetition) — *not* speaker — so fixing language/voice does not rescue it.
- **The frozen vocoder is NOT the floor.** A cheap width→1024 MLP adapter decouples backbone width from the
  vocoder, so d can be 640 and project up.

**The one lever that beats this floor (gated):** a coarser / lower-frame-rate hierarchical codec (MARS6's
12 Hz tokens → 70M total beating XTTS-460M on speaker sim). It shrinks *what the AR backbone must model* and
is the single highest-leverage move — **but it requires retraining the DVAE + HiFi-GAN** (our frozen assets),
so it's a separate, expensive R&D track, not the next step. Same for switching to NAR flow-matching
(ZipVoice-123M class) — a clean-sheet rebuild kept as the fallback if pure-AR stalls above target.

---

## 4. The sequenced recipe (reuses our existing scripts)

**Order matches NVIDIA Nemotron production practice: align(RL) → prune → distill → recover(RL).**

**Step 0 — Teacher-correct.** Fine-tune the RL'd 265M on the Hinglish corpus first so its logits/latents are
trustworthy distillation targets (Minitron: necessary for distribution shift). 1-2 epochs, existing SFT loop.

**Step 1 — Specialize + width-prune.**
- Drop the speaker pathway; bake the 4 voices' conditioning latents + d-vectors.
- Trim the text vocab to Hinglish.
- Width-prune d=1024→640 via **activation-based importance** (1024-utt calibration set, L2-over-batch /
  mean-over-sequence; score heads, FFN neurons, hidden channels). **Init the narrow student from the
  teacher's top channels** — never random. One global hidden-channel mask across all layers. Keep 16-20 layers.
- If cutting hard, **step it** (265M → ~120M → ~85M) with a recovery pass between, not one shot.

**Step 2 — Multi-signal distillation** (extend `distill_trainer.py`). Combined loss (starting weights;
the latent-MSE scale is the one coefficient to sweep — it lives on a different scale than the code-softmax):
```
L = 2·CE  +  5·KL(T=2, ×T²)                       # logit-KL on the 1026 audio codes (Minitron: logit-only)
      +  1·[ MSE + (1−cos) ]( adapter_1024 , teacher_pre_vocoder_latent )   # TIMBRE protector (TinyBERT W_h / FitNets)
      +  1·KL( MiniLMv2 Q-Q/K-K/V-V relations )   # dimension-agnostic; single upper-mid teacher layer
      +  1·[ 1 − cos(spk_emb_gen, spk_emb_ref) ]  # speaker-SV cosine (DMDSpeech; warmup ~10k steps)
```
- **Data-bound, not compute-bound:** ignore Minitron's token budget (it'd be ~1300 epochs on 2855 utts).
  Use ~2 epochs, LR 1e-4, batch 64, cosine, **aggressive early-stop** (KD on small reused data follows a
  U-curve and starts hurting once soft labels overfit).
- **Mix real ≥ 0.5** (golden-ratio weighting; avoids synthetic collapse). Use RL-winner audio but anchor on
  the real corpus. **Augment with SpecAugment + additive noise; NOT pitch/speed** (those move the speaker
  identity we're trying to lock).

**Step 3 — Re-run RL on the student (NOT optional).** Distillation reliably transfers *style* but
**under-transfers the ASR-recall capability the reward optimized** — and it regresses silently at exactly the
code-switch boundaries. So re-run our **offline RFT best-of-N + length-normalized DPO** on the student
(reuses `reward.py`, `gen_candidates.py`, `select_winners.py`, `dpo_trainer.py`). Distill on-policy where
possible. **Gate on ASR-recall + the n≥150 code-switch panel, not MOS** — MOS will pass while recall quietly drops.

**Step 4 — Quantize for deployment (byte-only, AFTER the param target).** State plainly: quantization cuts
*bytes, not parameter count*. **INT8 (W8A8): near-lossless, default.** **INT4: measured risk** — it
disproportionately hurts *rare tokens* (→ code-switch/rare-phoneme recall); if used, QAT not PTQ, keep
embeddings/LM-head/layernorm/vocoder-adjacent layers high-precision, gate on WER. BitTTS-style ternary costs
0.45-0.66 MOS — too aggressive for "no quality loss."

---

## 5. Reuse map

| step | existing asset |
|---|---|
| teacher-correct | `train_xtts.py` SFT loop |
| width-prune + multi-signal distill | extend `scripts/hinglish/distill_trainer.py` (add latent-MSE adapter + MiniLMv2 + speaker-SV losses) |
| RL recovery | `rl/{reward,gen_candidates,select_winners,dpo_trainer}.py` re-pointed at the student |
| certification | `12_equivalence_eval.py` + n≥150 panel + TOST/BCa/Holm (primary endpoint = ASR-recall + code-switch boundary) |

---

## 6. Honest caveats

1. **No source studies RL-gain survival through TTS distillation specifically** — the Step-3 numbers are
   LLM/seq-model analogies. Treat directionally; that's *why* the RL re-run is mandatory, not optional.
2. **"No quality loss" is conditional on the RL re-run.** Naive offline distillation will likely pass a MOS
   gate while losing ASR-recall at code-switch boundaries.
3. **The 85M student will probably not beat the 265M teacher on naturalness** — aim for *measured
   equivalence* (the TOST gate), and use direct reward optimization to defend intelligibility + timbre, which
   is where students provably match/exceed teachers (DMDSpeech: student beat teacher on WER + SIM via SV/CTC).
4. **Clean <100M on pure-AR-over-DVAE is at the edge of what's demonstrated** — the fixed-voice + monolingual
   specialization is exactly what makes it credible. To go materially below ~70M, the codec must change.

---

## 7. Existence proofs (the scoreboard)

- **Kokoro-82M** — #1 TTS Arena, beats XTTS-v2 467M — fixed voices (256-dim style vectors), no zero-shot.
- **MARS6-70M** — only small *AR-codec* zero-shot proof; SIM holds (beats XTTS-460M), WER pays (7.4) — and
  only survives via a 12 Hz hierarchical codec + sampling tricks.
- **Nix-TTS 5.2M / VITS ~29M / Glow-TTS 28.6M** — fixed/single-voice at tiny sizes, near-GT MOS.
- **Goldfish 125M** monolingual beats BLOOM 7.1B on-language — specialization licenses a far smaller backbone.
- **Spotify 1.3B→500M distillation**, "no perceivable quality loss" — distillation at this ratio works.

Net: small high-quality fixed-voice monolingual TTS is well-precedented; ~85M for Hinglish fixed-voice via
width-prune + multi-signal distill + RL recovery is a sound, evidence-backed target.
