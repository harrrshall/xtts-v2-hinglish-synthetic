# Hinglish XTTS Compression — Project Journey & Motivation

The full record of what we built, why, and where it's going: shrinking a 443M-parameter Hinglish
text-to-speech model **without losing quality**, then pushing further toward a tiny Hinglish-only model.

---

## 1. Motivation

We have a high-quality **Hinglish** (Hindi-English code-switch) voice model, fine-tuned from Coqui
**XTTS-v2**. It's good, but it's big and slow. The goal is a **smaller, faster model that sounds
identical** — same naturalness, same voice, same code-switch accent, same intelligibility.

Why this matters: a smaller model is cheaper to serve, faster to respond, and deployable in more places.
But "smaller" is worthless if it sounds worse. So the entire project is organized around one rule:

> **Shrink the model, prove no quality loss, repeat.**

"Prove" is the hard part. Naive evals lie — they pass models that sound worse and fail models that sound
fine. So before shrinking anything, we built a measurement system designed to be honest.

**Product framing (updated):** we **only** care about **Hinglish audio**. The model does not need to be a
general multilingual XTTS. That single constraint is a major lever for the deepest compression tier (see §8).

---

## 2. The model (what we're compressing)

XTTS-v2 has three parts. We compress only the first; the other two are frozen and inherited byte-for-byte:

- **Autoregressive GPT** (the "brain"): a GPT-2-style transformer (hidden width d=1024, 16 heads, audio
  vocab 1026) that generates a sequence of discrete **DVAE audio tokens** from text + a speaker reference.
  **This is the ~443M "400M" we're shrinking.**
- **Frozen DVAE tokenizer** (1024-code book): turns audio into the discrete tokens the GPT predicts.
- **Frozen HiFi-GAN vocoder**: turns generated tokens back into a 24 kHz waveform. It expects **1024-dim**
  GPT latents — a constraint that matters a lot for the sub-100M tier.
- **PerceiverResampler** (32 latents): compresses the speaker-reference clip into the conditioning the GPT uses.

Quality is judged on **four axes**, each with a pre-registered "no-loss" margin:

| axis | what it measures | how | margin |
|---|---|---|---|
| **intelligibility** | are the words right | ASR content recall | 0.03 |
| **accent** | does embedded English sound natively English | whisper-en recall on code-switch words | 0.03 |
| **naturalness** | does it sound human | UTMOS (neural MOS predictor) | 0.10 |
| **voice (SECS)** | does it sound like the target speaker | resemblyzer speaker-embedding cosine | 0.03 |

---

## 3. Method overview — the compression ladder

```
443M teacher ──distill──► 265M student ──RL (RFT→DPO)──► 265M, accent recovered ──► [next] <100M
   (full)      depth 30→16    (16 layers)   reward = accent + capped quality floors      width cut + Hinglish-only
```

Two distinct moves, each with a different job:
- **Distillation** installs the teacher's *capability* into a smaller network.
- **Reinforcement learning** repairs the specific *behavior* (code-switch accent) that distillation disturbs,
  without touching what already works.

---

## 4. Phase 1 — Research & a false-positive-proof benchmark

We ran a deep multi-agent research pass on *how* to compress this class of model, and on *how to measure*
"no quality loss" without fooling ourselves. Outputs: `docs/COMPRESSION_PLAN.md`, `docs/BENCHMARK_SPEC.md`.

Key benchmark decisions (see `scripts/hinglish/12_equivalence_eval.py`):
- **Equivalence testing, not just "is it close."** We use **TOST non-inferiority** with a **BCa bootstrap**
  lower bound against pre-registered margins, Holm-corrected across axes. A model only "passes" if we can
  *statistically prove* it isn't worse by more than the margin.
- **Power matters.** An underpowered test returns **INCONCLUSIVE**, not a false PASS. The accent axis needs
  ~300+ scorable code-switch clips for real power.
- **Process-stable seeds (FNV)** so a candidate and the reference draw the same sampling stream per utterance
  — this killed a class of false alarms.

This benchmark is the spine of the whole project: every shrink is gated on it.

---

## 5. Phase 2 — Distillation: 443M → 265M, no quality loss on voice & naturalness

We distilled the 443M GPT down by **cutting depth** (30 → 16 transformer layers) while keeping width d=1024,
so every frozen interface (DVAE, vocoder, perceiver) stays byte-identical. The student is warm-started from
strided teacher layers and trained with logit-KL + hidden-state distillation on top of the standard CE
(`scripts/hinglish/distill_trainer.py`).

We tried two depths and compared both against the teacher:

| candidate | GPT params | UTMOS | SECS | accent |
|---|---|---|---|---|
| 443M teacher | 443M | 3.124 | 0.867 | 0.860 |
| 12L-strong | 214M | 3.056 | 0.857 | 0.784 |
| **16L (chosen)** | **265M** | **3.141** | **0.858** | **0.805** |

**16L was chosen** because it already sits **at the teacher's naturalness and voice** — only one axis
(**accent**) regressed. One axis to fix means a small, targeted repair, which is the key to not breaking
anything else.

---

## 6. Phase 3 — The certification harness

To certify a candidate we generate a paired held-out panel and run the equivalence gate
(`scripts/hinglish/compare/gen_panel_ckpt.py` → `09_objective_eval.py` + `10_accent_eval.py` →
`12_equivalence_eval.py`). We later **powered up** the eval to **n≈314** (187 + 127 newly-authored,
leakage-checked code-switch prompts: `scripts/hinglish/rl/powered_prompts*.py`) so the accent axis has real
statistical power. Scoring is parallelized 8 ways with thread caps (`score_parallel.sh`) — this took GPU
utilization from 2% to ~99% and cut scoring time ~5-8×.

---

## 7. Phase 4 — RL to recover accent *without* losing expressivity

The accent regression is a **behavior** problem (the model can produce good English; distillation made it
less consistent). RL is the right tool — but RL on a learned naturalness reward is famous for **flattening
prosody** to game the metric. So the reward was designed to make that impossible.

**Reward design** (`scripts/hinglish/rl/reward.py`), computed on the decoded waveform:
- **One trusted maximizer:** code-switch English-recall (the only thing allowed to drive gains).
- **Everything else is a capped one-sided floor** vs the frozen 16L base — UTMOS, SECS, **pitch-variance**,
  energy-range, duration. Each penalizes *dropping below* the base and gives **zero credit for exceeding it.**
  So accent cannot be bought by flattening prosody, slowing speech, or drifting the voice.

**Offline-first, because online RL is impractical here.** We measured ~8s per audio rollout; online GRPO
would need ~17 min/step. And the closest published analog (Llasa-1B) showed naive online GRPO on an ASR
reward **collapsing pitch variance to near-monotone** (arXiv 2509.18531). So:

- **Round-1: RFT (best-of-N rejection-sampling).** Generate 8 candidates per prompt, keep the best-accent
  one that passes *every* floor, fine-tune on those winners (warm-restored from frozen 16L, + SFT-replay anchor).
- **Result:** accent **0.805 → 0.837** (+0.032), **expressivity actually up +1.3%** (no flattening), voice
  unchanged, naturalness within margin. The model **generalized** — trained on training prompts, improved
  held-out. A model-averaging α-sweep confirmed round-1 is the Pareto-best point.
- **DPO escalation** (length-normalized, `scripts/hinglish/rl/dpo_trainer.py`) for the last sliver — built on
  the model's own per-sequence logprobs (the `−loss_mel` trick), with a speaker-bypass conditioning fix.

---

## 8. Results so far (powered cert, n≈314, round-1 vs 443M teacher)

| axis | delta vs teacher | margin | verdict |
|---|---|---|---|
| **naturalness (UTMOS)** | −0.032 | 0.10 | ✅ **PASS — no quality loss** |
| **voice (SECS)** | −0.007 | 0.03 | ✅ **PASS — no quality loss** |
| **accent** | −0.019 | 0.03 | ⚠️ within margin on the estimate; near-miss at the CI bound (power-limited) |
| **expressivity (pitch-SD)** | preserved | — | ✅ no flattening |

**The bottom-line achievement:** a **265M model (1.67× smaller than 443M)** that is **certified to have no
quality loss in naturalness and voice**, with **accent recovered to within ~0.019 of the teacher** (near
parity) and **expressivity preserved**. The accent axis is a hair short of a strict formal PASS at the CI
bound; DPO escalation is in progress to close it.

---

## 9. Engineering learnings (so they're not re-learned)

- **The reward floors are the product.** Making every quality metric a one-sided cap (not a maximizer) is
  what let RL improve accent without the prosody collapse the literature warns about.
- **Scoring is CPU-bound, not GPU-bound.** Whisper + pitch extraction starved the GPU; the fix was 8-way
  prompt-sharding **plus** thread caps (`OMP/OPENBLAS/MKL_NUM_THREADS`) — BLAS was spawning ~416 threads/proc
  (load 451 on 192 cores). With caps: GPU 2%→99%, load 451→157.
- **`gen_candidates.py` asserts CUDA.** A `kill -9` during CUDA init can poison the GPU context → silent CPU
  fallback (58 cores, hours/clip). Always assert device.
- **Never `pkill -f` over SSH** — it matches and kills its own shell (exit 255) and leaves zombies that
  contend on the GPU and look like a hang. Kill by exact PID via `ps | grep "[x]pattern"`.
- **`cond_idxs` in sample units** masked all mel frames → NaN CE in DPO; route conditioning through the
  perceiver and skip that masking.

---

## 10. Phase 5 (next) — deeper: toward <100M, Hinglish-only

Research (3 web-cited agents) on the sub-100M tier converged:
- **Width, not depth, is the remaining lever.** Params/layer ≈ 12·d²; at d=1024, 16 layers ≈ 200M. Depth-only
  can't reach <100M without breaking sequential modeling. Target **d=640 × 18-20 layers ≈ 85-98M** (Minitron:
  width-pruning beats depth-pruning at equal params; MobileLLM: deep-and-thin beats shallow-wide).
- **The frozen vocoder needs 1024-dim latents** → a learned width→1024 **adapter** (standard pattern).
- **The binding axis flips to voice (SECS).** A width cut squeezes the high-dimensional subspace that carries
  speaker identity; **RL recovers behavior (accent) but cannot reconstruct lost speaker capacity**
  (width-pruning dichotomy, arXiv 2512.22671). So SECS must be certified **pre-RL at each width**.

**The Hinglish-only constraint changes everything here.** Because we only ship Hinglish (one language, a
small fixed voice set, fine-tuning allowed):
- Drop XTTS's multilingual capacity (it supports 17 languages; we need one) — frees parameters and data.
- If we ship a **small fixed voice set** rather than general zero-shot cloning, the **speaker-manifold
  capacity floor largely disappears** — which is exactly the SECS risk the research flagged. This makes a
  width cut to <100M *much* safer, and possibly enables going smaller still.

Recipe to attempt (gated on SECS at each step): teacher-correct → width-prune d=1024→640 (activation-importance
init) → keep depth → width→1024 adapter **with a speaker-latent bypass** from the PerceiverResampler →
distill on logit-KL + **latent feature distillation** (the SECS protector) → RFT/DPO accent recovery →
optional low-rank FFN top-up. **Quantization is byte-only and does NOT count toward the param goal.**

Full sub-100M analysis lives in `docs/COMPRESSION_PLAN.md` (sub-100M section).

---

## 11. Shipping plan (HuggingFace)

Once the RL accent work certifies, we ship the final Hinglish model to the Hub.
- HF account: **`harrrshall`** (write-scoped token verified).
- Artifacts: the certified GPT checkpoint + frozen DVAE/HiFi-GAN/vocab + an inference snippet, a model card
  derived from this document (motivation, method, the 4-axis certified numbers, honest accent caveat), and
  the eval methodology so the "no quality loss" claim is reproducible.

---

## 11.5. Sub-100M ACHIEVED (2026-06-26)

The <100M target is met and certified. A **89.96M-param** fixed-voice student (d=640) was built by
**staged width-pruning** (d=1024 → d=768 → d=640, with a distillation-recovery pass between each cut) +
RFT, and **certified equivalent-or-better than the 265M teacher** (accent +0.097, UTMOS +0.167, SECS +0.008;
all CIs pass TOST non-inferiority, accent & UTMOS strictly superior). A one-shot d=1024→640 cut first FAILED
(accent 0.17 — capacity wall); the staged rebuild fixed it (accent 0.83→0.91 after RFT). Full write-up:
**[`SUB100M_RESULTS.md`](SUB100M_RESULTS.md)** (build + certification) and
**[`SUB100M_INFERENCE.md`](SUB100M_INFERENCE.md)** (load + run). Deliverable:
`runs/rl/sub100m/student640b_rft.pt`. Trade: fixed 4-voice set (no zero-shot cloning), which is what dissolves
the speaker-capacity floor and makes <100M reachable.

---

## 12. File index

- `docs/COMPRESSION_PLAN.md` — how to compress (incl. sub-100M).
- `docs/BENCHMARK_SPEC.md` — false-pos/neg-hardened measurement.
- `docs/RL_EXPRESSIVITY_PLAN.md` — the RL recipe (reward, technique, preservation stack).
- `scripts/hinglish/distill_trainer.py` — depth distillation (443M→265M).
- `scripts/hinglish/rl/reward.py` — the capped-floor reward + expressivity monitors.
- `scripts/hinglish/rl/{gen_candidates,select_winners,build_prompts}.py` — RFT candidate→winner pipeline.
- `scripts/hinglish/rl/dpo_trainer.py` — length-normalized DPO.
- `scripts/hinglish/rl/{model_average,pitch_monitor,powered_prompts*}.py` — α-sweep, expressivity monitor, powered eval.
- `scripts/hinglish/12_equivalence_eval.py` + `09/10_*` — the certification gate.
- `launch_assets/ab_spotcheck/` — A/B human listening pack (boundary-prosody check).
- **Sub-100M (the <100M model):**
  - `docs/SUB100M_PLAN.md` — research-backed plan; `docs/SUB100M_RESULTS.md` — build + certification results;
    `docs/SUB100M_INFERENCE.md` — load + run the 90M model.
  - `scripts/hinglish/student640.py` — width-agnostic student module (structured slicing, `forward_both`, `Student640`).
  - `scripts/hinglish/rl/channel_importance.py` — activation-importance channel ranking.
  - `scripts/hinglish/rl/{build_student640,build_student_from_student}.py` — stage-1 / stage-2 builders.
  - `scripts/hinglish/rl/{distill_preencode,distill640_trainer}.py` — offline DVAE encode + multi-signal distillation.
  - `scripts/hinglish/rl/{gen_student640,eval_student640,oracle_bestof}.py` — decode, objective eval, RFT-ceiling diagnostic.
  - `scripts/hinglish/rl/{rft_finetune_student,certify_student640}.py` — RFT SFT + paired TOST certification.
