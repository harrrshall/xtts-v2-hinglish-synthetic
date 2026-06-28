# RL Expressivity Plan — Hinglish XTTS-v2 student

Post-distillation reinforcement learning to **recover the code-switch accent regression and raise
expressivity, without losing the model's existing expressivity.** Priority order is explicit:
**preserve capability/expressivity > maximize reward.** Reward-hacking that flattens prosody to score
high on a learned MOS predictor is a failure, not a success.

This plan is the output of a 20-agent research workflow (model selection + 6 verified literature
streams + synthesis). Citations are AR-codec-TTS-specific where they exist; transfers from text-LLM
RLHF are marked as such.

---

## 0. Decisions (locked)

**Final model = `16L`** (264.7M GPT params, 1.67x smaller than the 443M teacher). Ship it and use a
**frozen byte-identical copy as the RL reference policy.**

| candidate | GPT params | UTMOS (Δ) | SECS (Δ) | accent (Δ) |
|---|---|---|---|---|
| 443M teacher | 443M | 3.124 | 0.867 | 0.860 |
| 12L-strong | 214M | 3.056 (−0.067) | 0.857 (−0.010) | 0.784 (−0.076) |
| **16L (chosen)** | **265M** | **3.141 (+0.017)** | **0.858 (−0.008)** | **0.805 (−0.056)** |

Why 16L over the smaller 12L: 16L is already **at teacher naturalness** and **holds voice**, so the
**only axis RL must move is accent**. One axis to repair = a smaller, narrower policy step, and
forgetting tracks how far the policy moves from the base (RL's Razor, arXiv 2509.04259). Starting from
12L would force RL to chase *two* regressions (UTMOS and accent) and structurally amplify the
UTMOS-flattening trap. Size/speed is the only axis 12L wins, and it was deprioritized.

**Technique = offline-first.** Best-of-N rejection-sampling SFT (RFT) → CE-anchored iterative DPO/RPO,
reference = frozen 16L. Online KL-anchored GRPO is **gated** behind an offline plateau.

Decisive evidence: on **Llasa-1B + XCodec2** (the closest published AR-codec analog to XTTS), naive
online GRPO on an ASR/transcription reward — *exactly* what accent recovery is — **collapsed log-F0
variability to near-monotone** (prosody ELO 1150 → 754) while CER improved (arXiv 2509.18531). The
authors' fix was preference optimization, not more ASR reward. MPO (arXiv 2509.00685) shows a CE anchor
`L = λ·L_dpo + L_ce` (λ≈10) is what prevents DPO collapse (CER 4.24 with vs 14.52 without). Koel-TTS
shows RPO is less hyperparameter-sensitive than DPO.

---

## 1. Reward (computed on the decoded 24 kHz waveform, never on DVAE tokens)

**One trusted maximizer + capped one-sided guardrail floors.** Per-component std-normalized within the
group (GRPO) / candidate set (offline), then weighted. GRPO reward scaling **OFF** (TTS-1) to kill
length/difficulty bias.

**PRIMARY (trusted, verifiable, weight 1.0):**
`R_accent = harmonic_mean(en_recall, exp(-NLL/3))` over the known reference English tokens, from
`whisper-large-v3` `language="en"` — the `english_words()` + `fuzzy_in()` recall already in
`scripts/hinglish/10_accent_eval.py`, densified with ASR confidence (GRPO-for-TTS, arXiv 2509.18798).
This is the **only** signal allowed to drive gains.

**GUARDRAILS (one-sided hinges vs the frozen-16L base on the *same prompt+seed*; penalize regression,
give ZERO credit for exceeding; never maximize):**

| floor | weight | penalty |
|---|---|---|
| UTMOS (capped learned reward) | 0.3 | `−max(0, 3.141 − UTMOS_gen)`; zero credit above 3.141 |
| SECS | 0.5 | `−max(0, 0.828 − SECS_gen)` (base 0.858 − 0.03 margin) |
| duration / anti-slowdown | 0.1 | `−|dur_gen − dur_ref| / dur_ref` vs teacher pacing |
| F0-std floor (anti-monotone) | 0.2 | `−max(0, F0std_base − F0std_gen)` in semitones, voiced frames |
| energy dynamic-range floor | 0.1 | `−max(0, energyDR_base − energyDR_gen)` |

The UTMOS cap is the move that removes the prosody-flattening incentive — and it is *only* clean because
16L already starts at teacher UTMOS, so UTMOS can be a pure floor with nothing to gain by raising it.

**Hard degenerate filter (NOT a reward term — applied BEFORE the loss/pair):** drop/zero-advantage any
rollout failing a VAD silence-ratio cap, n-gram token repetition, or no-EOS / length blow-up.

---

## 2. Expressivity preservation stack (ranked — this is the user priority)

1. **Smallest-move design (structural).** Accent is the ONLY maximizer; UTMOS/SECS/duration/F0 are
   one-sided floors. Less policy movement is the primary thing protecting prosody. Choosing 16L already
   minimizes the required move.
2. **Offline acceptance-floor filter (structural no-collapse, unique to offline).** A candidate is
   eligible as a DPO *winner* only if, on its own prompt+seed vs frozen 16L, it satisfies
   `UTMOS≥3.141 AND SECS≥0.828 AND F0-std≥base AND energy-DR≥base AND duration within ±15% of teacher`.
   You literally cannot train toward flatter samples. Biggest single expressivity lever the offline
   route buys.
3. **KL/CE anchor to frozen 16L.** Offline: DPO `beta` is the KL anchor (start 0.1) PLUS MPO-style CE
   term `λ≈10` mixing the original synthetic-Hinglish distillation batches (CE on teacher audio tokens
   is already free in `distill_trainer.py`). Online GRPO escalation: KL `beta=0.1` to 16L + PPO-ptx SFT
   replay interleaved.
4. **Early-stop on the expressivity monitor.** Select checkpoints by "accent recovered AND
   prosody-variance/UTMOS/SECS not below base", never by max reward. Hard-stop on the hacking signature
   (UTMOS up while pitch-SD down).
5. **Post-hoc model averaging (cheap insurance, transferred — mark plausible for an AR-codec generator).**
   `θ(α) = (1−α)·θ_16L + α·θ_RL`, sweep `α ∈ {0.2,0.4,0.6,0.8,1.0}`; ship the largest α that recovers
   accent while UTMOS/SECS and held-out F0-variance stay within margins. Reversible, post-hoc.
6. **LoRA-RL (optional lower-forgetting variant)** only if full-FT visibly over-moves; caveat: LoRA may
   underfit the −0.056 accent gap, so default is full-FT-then-interpolate. Transfer, not AR-codec-demonstrated.
7. **Degenerate-rollout filtering + low-entropy discipline.** Drop repeated-token / no-EOS / zero-advantage
   rollouts before the loss (Explore-RL diverged past ~1500 steps without it). Do NOT reward high entropy.

---

## 3. Expressivity monitoring panel (paired vs frozen 16L, stable FNV seeds)

**TIER-1 (cheap, CPU, every eval step):**
- **Pitch-SD (semitones)** = `std(12·log2(f0/median_f0))` over voiced frames (librosa.pyin fmin=65
  fmax=400, or pyworld). PRIMARY flattening detector. WARN at >10–15% relative drop in median pitch-SD;
  **HARD-STOP at >25% or >0.5 semitone absolute** (Llasa collapse signature, arXiv 2509.18531 Fig 2).
- **RMS-energy SD (dB)** and dynamic range (p95−p5) over voiced frames; peak-normalize first.
- **Articulation rate** (words / voiced time) + **pause-duration histogram** (gaps >150–200 ms) from
  whisper word timestamps or Silero-VAD; watch for shrinking rate variance and flattened pause hist.
- **Token-softmax entropy** (free, from rollout logits) + cross-sample prosodic variance (N=4–8 per
  prompt at temp 0.7); shrinking SD of clip-level pitch-SD/duration = mode collapse.

**TIER-2 (learned, at checkpoints only, RELATIVE drift vs 16L — never absolute):**
- **DS-WED prosody-diversity** (weighted edit distance over HuBERT/WavLM k-means tokens; human corr 0.77;
  this is the metric that measured the 18.8% AR-codec DPO collapse).
- **NISQA** as the reward-hacking tripwire opposite UTMOS (DLPO): alarm if UTMOS rises while NISQA falls.
- **emotion2vec+ neutral-class drift** (rising neutral mass = flattening); Audiobox-Aesthetics CE+PC.
- **2-Wasserstein** on pooled {pitch-SD, energy-SD, articulation-rate, pause-duration} vs 16L (read
  alongside the signed pitch-SD trend to distinguish flattening from healthy diversification).

Wire the panel into `09_objective_eval.py`, which already loads each wav and computes UTMOS/SECS.

---

## 4. Reward-hacking guards

UTMOS one-sided cap on the WAVEFORM (exposes vocoder artifacts) · duration-window penalty (kills the
best-documented ASR hack = slow over-articulation) · F0-std + energy-DR floors (the only thing between
the accent reward and validated collapse) · held-out NISQA/pitch-SD divergence tripwire · degenerate
rollout rejection before the loss · SECS as a floor not a target (a SIM maximizer destabilizes AR-codec
RL into no-EOS runaway) · small KL budget with a flat-KL monitor · GRPO reward scaling OFF · offline
acceptance filter · **mandatory code-switch-boundary CMOS / A-B human spot-check at certification** (the
English-ASR proxy can be satisfied while flattening Hindi-English boundary prosody, which no automatic
scorer in the stack can see — R4 in COMPRESSION_PLAN).

---

## 5. Hyperparameters

**Offline RFT/DPO (primary):** candidates N=8–16/prompt; sampling temp 0.9–1.0, top_k 50–80, top_p 0.9
(above the 0.7 inference temp for reward-bearing diversity). Pair build: winner = max
`harmonic_mean(en_recall, exp(−NLL/3))` among candidates passing ALL floors; loser = lower-recall
candidate or a base-16L sample; reject any pair whose winner fails a floor. DPO `beta=0.1` (or RPO
`eta=1.0`). LR 2e-7…5e-7, batch 32–64 pairs, ~2000–4000 steps/round, reference=frozen 16L. CE anchor
`λ≈10`. 2–3 iterative rounds, regenerate candidates from latest checkpoint each round. Optional prosody
pairs ~200/round (teacher/LLM-judge), moving reference, stop round 2–3.

**Online GRPO escalation (gated):** G=8 (up to 12), KL `beta=0.1` to 16L, LR 1e-6 (full-FT), batch 16
prompts, rollout temp 1.0–1.1, top_k 75, top_p 0.9, repetition_penalty 1.1. Reward weights after
per-group std-norm: accent 1.0; UTMOS floor 0.3; SECS floor 0.5; duration 0.1; F0-std 0.2; energy-DR
0.1. Reward scaling OFF. Filter zero-variance/degenerate groups. Early-stop before ~1500 steps if any
monitor trips.

**Throughput:** cache per-voice conditioning latents once (the `cond_cache` pattern in
`gen_panel_ckpt.py`), batch the G AR generations (KV cache), one batched DVAE→HiFi-GAN decode for the
group, then one batched pass each through utmos22_strong / whisper-large-v3 / resemblyzer. 265M student
+ frozen scorers fit on 1–4 H200.

**Certification:** oversample code-switch English-bearing prompts to **n≥150** scorable accent clips
(power floor 337–389 to clear δ=−m/3); `12_equivalence_eval.py` TOST/BCa/Holm; margins UTMOS 0.10 /
SECS 0.03 / accent 0.03 / intel 0.03; stable FNV seeds; K=5 multi-seed decode; alpha sweep
{0.2,0.4,0.6,0.8,1.0} for model averaging.

---

## 6. Execution plan (ordered; the smoke gate is BLOCKING)

1. **Power + freeze the eval.** Oversample code-switch prompts to ≥150 scorable accent clips; freeze the
   manifest. Generate the **frozen pre-RL 16L reference panel** (`gen_panel_ckpt.py`, stable FNV seeds;
   greedy + temp-0.7). Record per-prompt 16L baselines for every reward + monitor component.
2. **Build the reward/monitor module.** One `score_wav(wav)` reusing `10_accent_eval.py` (whisper-en
   recall + NLL), `09_objective_eval.py` (UTMOS, SECS), plus new F0-std / energy-DR / duration / VAD /
   repetition; add NISQA + DS-WED as monitors. Unit-test deltas against the frozen panel so a no-op
   model scores ~0 delta.
3. **SMOKE GATE (blocking).** Best-of-N rejection-sampling on ~24 prompts, N=8, one short epoch
   (mirror the existing `--smoke` discipline). Assert: rollouts decode to wav, reward computes on
   waveform, every floor fires on a planted bad sample, monitors log paired deltas, loss finite. Do NOT
   proceed until green.
4. **Round-1 offline RFT/anchored-DPO** on the full code-switch prompt set: CE anchor + distillation SFT
   replay, reference = frozen 16L, conservative beta. Log the full monitor panel each eval; early-stop
   on the hacking signature.
5. **Checkpoint selection** by "accent recovered AND prosody/UTMOS/SECS not below base", then the
   model-averaging α sweep; pick the largest α that holds the floors.
6. **Certify.** `gen_panel_ckpt.py` paired panels at n≥150 → `12_equivalence_eval.py` (TOST/BCa/Holm).
   Require accent PASS with UTMOS/SECS/intel non-inferior. Add the code-switch-boundary CMOS / A-B human
   spot-check.
7. **If accent not closed + monitors clean:** iterate rounds 2–3 (regenerate candidates), or escalate to
   online KL-anchored GRPO with the SAME reward; re-smoke first.
8. **If GRPO trips a monitor:** revert to the model-averaged offline checkpoint; do NOT relax the gate.
   Ship the certified checkpoint; report any residual accent gap as a TOST lower bound, not a parity claim.

---

## 7. Open questions (resolve during build)

1. Does the box's TRL expose the experimental `rollout_func`, and does it co-install with coqui-tts
   0.27.5 without a torch/transformers conflict? If not, **hand-roll the GRPO/RLOO loop.**
2. Can the XTTS GPT be wrapped so loss-side `model(input_ids).logits` aligns over the 1026 audio tokens
   WITH the perceiver speaker-latent (`cond_offset=32`) + text prefix, or is a hand-rolled loop cleaner?
3. Is the F0-std/energy floor a sufficient proxy for code-switch *boundary* prosody, or do we need
   DS-WED and/or a switch-point-aware monitor? (No literature covers RL prosody for Hinglish code-switch.)
4. Can we force-align generated Hinglish wav to the English code-switch words for word-level accent
   credit (W3AR-style), given whisper-en recall is currently sequence-level?
5. Is there REAL Hinglish reference audio for SpeechAlign-style golden pairs, or are we limited to
   teacher (443M) outputs as golden (risks inheriting the teacher's accent ceiling)?
6. Worth reviving the un-installable Qwen3-ASR as a second, architecturally different intelligibility leg?
7. Actual per-rollout wall-clock for G=8 on 1–4 H200 — decides whether online GRPO is ever practical.
8. Swap resemblyzer for WavLM-SV/ECAPA-TDNN as the SECS floor (more robust, less entangled)?
