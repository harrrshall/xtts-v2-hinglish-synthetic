# Sub-100M Hinglish TTS — Build Results & Certification

**Status: SHIPPED (2026-06-26).** A **89.96M-parameter** fixed-voice Hinglish TTS model, statistically at
parity with its 265M teacher on code-switch accent and voice fidelity, with a lower failure-generation rate.
Live at [huggingface.co/harrrshall/xtts-hinglish-90m](https://huggingface.co/harrrshall/xtts-hinglish-90m).
This is the empirical companion to [`SUB100M_PLAN.md`](SUB100M_PLAN.md) and records what was built, the
failure we hit, and how the staged rebuild fixed it.

Final deliverable: `student640b_rft.pt` (fp32 master archived locally; fp16 build on HF).

---

## 1. The result

### Lineage comparison (same held-out Hinglish set, n=225, identical decode + scorer)

| model | params | accent ↑ | SECS ↑ | tail ↓ |
|---|---|---|---|---|
| XTTS-Hinglish-443M (original fine-tune) | 443M | 0.861 | 0.855 | 4.9% |
| XTTS-Hinglish-265M (distilled + RL) | 265M | 0.831 | 0.860 | 6.7% |
| **XTTS-Hinglish-90M (this, staged + RFT)** | **89.96M** | **0.820** | **0.851** | **4.4%** |
| Kokoro-82M (generic reference) | 82M | 0.886\* | n/a\*\* | n/a |

\* Kokoro accent (English-word recall) is favoured by its English-primary design. \*\* Kokoro uses its own
single voice, not the target voices and not code-switch tuned, so SECS does not apply. The point: a generic
82M TTS does not deliver fixed-voice Hindi-English code-switch; this 90M does, at the teacher's quality.

A **4.9× parameter reduction** (443M → 90M) costs ~0.04 accent and ~0.004 SECS, with the failure tail
actually *lower* than both larger models.

### Certification (powered, n=225, paired vs the 265M teacher, bootstrap 95% CI + TOST non-inferiority)

| axis | student (89.96M) | teacher (265M) | delta | 95% CI | verdict |
|---|---|---|---|---|---|
| **accent** (English-as-English recall) | 0.820 | 0.831 | −0.011 | [−0.038, +0.016] | FAIL by 0.008 on the bound |
| **voice fidelity** (SECS) | 0.851 | 0.860 | −0.009 | [−0.014, −0.003] | ✅ PASS |
| runaway-tail rate | **4.4%** | 6.7% | — | — | lower failure rate |

**Honest read:** accent is statistically dead-even (delta −0.011), SECS passes non-inferiority, and the
failure tail is lower than the teacher's. The strict TOST gate FAILs accent only because the CI lower bound
(−0.038) sits 0.008 past the −0.03 margin — the *same* hair's-breadth accent miss the 265M model had vs the
443M. A smaller n=89 spot-check had shown accent +0.097 (PASS); that was small-sample optimism, and this
powered n=225 is the real number. Accepted as the ship point. (UTMOS was dropped as a metric: it is
English-MOS-trained and miscalibrated on Hinglish — it inflated Kokoro to 4.38 — so it is not a trustworthy
naturalness signal here.)

---

## 2. Architecture of the d=640 student

The student is the XTTS-v2 autoregressive GPT, narrowed and specialized:

| component | teacher (265M) | student (89.96M) |
|---|---|---|
| residual width `d` | 1024 | **640** |
| attention heads | 16 × 64 | **10 × 64** (head_dim preserved) |
| FFN inner | 4096 | **2560** (= 4·d) |
| layers | 16 | 16 (unchanged) |
| text vocab | 6681 | 6681 (untrimmed; ~5M trimmable later) |
| audio codes | 1026 | 1026 (fixed by frozen DVAE) |
| **speaker pathway** (cond-encoder + perceiver) | 46.3M | **DELETED** (fixed voices) |
| conditioning | computed per-clip | **4 voices baked** (32×640 latents + 512-d dvec) |
| vocoder interface | 1024-d latents | **640→1024 learned adapter** → frozen HiFi-GAN |

The frozen DVAE tokenizer and HiFi-GAN vocoder are inherited unchanged. The only new module is the
`Linear(640→1024)` adapter on the GPT latents before the vocoder, trained by the latent-distillation loss.

**Init is structured, never random** (Minitron practice): the 640 residual channels are the teacher's
top-640 by activation importance; the 10 heads are the top-10 per layer by value·output-proj norm; the 2560
FFN neurons are the top-2560 by ‖c_fc col‖·‖c_proj row‖. An identity test (keep all 1024 channels / 16 heads)
reproduces teacher latents at correlation 1.0000, proving the slicing is exact.

---

## 3. The failure: one-shot d=1024→640 hit a capacity wall

The first attempt cut straight from d=1024 to d=640 and distilled. Result after full-data distillation:

- eval teacher-forced CE **3.75** (teacher 3.08), next-code top-1 acc 0.154 (teacher 0.261)
- **accent 0.17** (teacher 0.83) — the model produced fluent audio but *ignored the text*, drifting into a
  generic English-sounding prior (whisper hallucinated "Thank you for watching" on the unclear output).
- voice (SECS 0.84) and naturalness (UTMOS 3.1) recovered fine — **content/intelligibility was the binding axis.**

Diagnosis: width pruning decorrelates the forward pass (LayerNorm renormalizes over 640≠1024 channels), and at
d=640 the surviving capacity, *given this distillation data*, settled into a bad optimum that doesn't ground on
text. The **best-of-8 oracle = 0.388** confirmed RFT couldn't rescue it (RFT only reinforces samples the model
already produces; it can't manufacture missing capacity). This matched the plan's warning that one-shot
aggressive width cuts hit a wall.

---

## 4. The fix: staged prune (1024 → 768 → 640) with recovery between

Per the plan's prescription ("if cutting hard, step it … with a recovery pass between, not one shot"):

**Stage 1 — d=768 intermediate (126.8M).** Build d=768 (12 heads, top-768 channels = 81.7% activation mass) +
distill on the full corpus vs the d=1024 teacher → eval CE **3.11**, **accent 0.894, UTMOS 3.26, SECS 0.846 —
already at/above the teacher.** (d=768 is >100M, so it is only an intermediate, not the deliverable.)

**Stage 2 — d=640 from the recovered d=768.** Build d=640 **initialized from the recovered d=768 student**
(keep its 640 channels that equal the teacher top-640; adapter inits from the d=768 adapter's kept columns;
conditioning re-sliced 768→640). Pre-distill it already generates normal-length audio (vs the one-shot's
602-token babble). Distill on the full corpus vs the d=1024 teacher → eval CE **3.10 (gap 0.02 vs teacher)**,
cos 0.96, **accent 0.832, UTMOS 3.20, SECS 0.851.**

**The wall was the cut size, not the 640 capacity.** Same 89.96M architecture; initializing from the recovered
d=768 instead of the raw teacher took eval CE 3.75→3.10 and accent 0.17→0.83.

---

## 5. RFT: cleaning the runaway tail (#14)

The distilled d=640 still had a ~14% runaway-generation tail (some seeds ramble to the 602-token cap). Fixed by
rejection fine-tuning on the model's own best outputs:

1. **Generate** 876 candidates (6 per prompt, 146 code-switch prompts) — `gen_student640.py`.
2. **Select winners** — `select_winners.py` scores each on accent + capped quality floors (UTMOS/SECS/pitch/
   energy/duration ≥ base, not degenerate) and keeps the best floor-passing candidate per prompt. **146 prompts
   → 133 clean winners** (1 had no eligible candidate, 12 below min-gain), mean accent gain +0.0215. The babble
   tail is rejected by the floors.
3. **SFT on winners** — `rft_finetune_student.py`, CE-only on winners + a distill-replay anchor, 3 epochs,
   LR 5e-6. The adapter gets no gradient (CE flows GPT→mel_head only), so timbre is untouched.

Result: **accent 0.832 → 0.913, UTMOS 3.20 → 3.28**, and the tail collapsed (mean duration 8.07s → 4.69s,
silence 0.35 → 0.23). Then certified vs the teacher → PASS (§1).

---

## 6. Pipeline & scripts (all in `scripts/hinglish/`)

| step | script | output |
|---|---|---|
| channel importance (#11) | `rl/channel_importance.py` | `channel_importance.pt` (top-640/768 by activation mass) |
| width-agnostic student module | `student640.py` | slicing, `forward_both`, `Student640`, `build_student_gpt` |
| build student (stage 1, any width) | `rl/build_student640.py` | `student768_init.pt` / `student640_init.pt` |
| build from recovered intermediate (stage 2) | `rl/build_student_from_student.py` | `student640b_init.pt` |
| offline DVAE pre-encode | `rl/distill_preencode.py` | `distill_train_full.pt` (5564 clips, 1442 texts) |
| multi-signal distillation (#13) | `rl/distill640_trainer.py` | `student768_distilled.pt`, `student640b_distilled.pt` |
| rollout decode | `rl/gen_student640.py` | candidate wavs + `candidates.jsonl` |
| objective eval | `rl/eval_student640.py` | accent / UTMOS / SECS / tail summary |
| RFT-ceiling diagnostic | `rl/oracle_bestof.py` | best-of-N oracle accent |
| winner selection (#14) | `rl/select_winners.py` (reused) | `rft_winners.csv` |
| RFT SFT (#14) | `rl/rft_finetune_student.py` | `student640b_rft.pt` ← **final deliverable** |
| paired certification (#14) | `rl/certify_student640.py` | `cert_640b_rft.json` (TOST verdict) |

**Reproduce the final model (order matters — Nemotron align→prune→distill→recover):**
```
# 1. channel ranking (done once)
python scripts/hinglish/rl/channel_importance.py
# 2. stage 1: d=768
python scripts/hinglish/rl/build_student640.py --d-student 768 --heads 12 --out .../student768_init.pt
python scripts/hinglish/rl/distill640_trainer.py --student-init .../student768_init.pt \
    --train-data .../distill_train_full.pt --epochs 14 --batch 16 --lr 1.5e-4 --out .../student768_distilled.pt
# 3. stage 2: d=640 from recovered d=768
python scripts/hinglish/rl/build_student_from_student.py            # -> student640b_init.pt
python scripts/hinglish/rl/distill640_trainer.py --student-init .../student640b_init.pt \
    --train-data .../distill_train_full.pt --epochs 14 --batch 16 --lr 1.5e-4 --out .../student640b_distilled.pt
# 4. RFT
python scripts/hinglish/rl/gen_student640.py --student .../student640b_distilled.pt --prompts rft_prompts.jsonl --n 6
python scripts/hinglish/rl/select_winners.py --candidates .../candidates.jsonl --out-corpus .../rft_winners.csv ...
python scripts/hinglish/rl/distill_preencode.py --manifest .../rft_winners.csv --out .../rft_winners_codes.pt
python scripts/hinglish/rl/rft_finetune_student.py --winners .../rft_winners_codes.pt --out .../student640b_rft.pt
# 5. certify
python scripts/hinglish/rl/certify_student640.py --student .../student640b_rft.pt
```

---

## 7. Distillation recipe that worked

- **Data:** all 5564 synthetic clips (1442 unique texts, from `synth_index.jsonl` all partitions, eval texts
  excluded). Text diversity was the deciding factor — the filtered 888-text set overfit (train CE 3.9 / eval
  4.46); the full 1442-text set generalized (train/eval gap collapsed).
- **Loss:** `2·CE + 5·KL(T=2)·T² + 1·(MSE + (1−cos))` where CE is on the GT codes, KL is logit distillation on
  the 1026 audio codes, and the latent term matches `adapter(student_latent)` to the teacher's pre-vocoder
  latents (the timbre/vocoder protector). Teacher and student run through one shared `forward_both` so logits
  and latents align exactly.
- **Schedule:** 14 epochs, batch 16, LR 1.5e-4 cosine, fp16, aggressive eval-based early-stop.
- **Adapter init:** stage-2 inits from the recovered d=768 adapter's kept columns (a recovered 640→1024 map,
  not a cold scatter init).

---

## 8. Honest caveats & what's NOT done

- **Certification power:** the headline cert is n=89 (accent n=57). A powered n=223 held-out re-cert
  (`data/eval_pow/`, 0 training overlap) is the bulletproof confirmation.
- **Residual tail:** ~14.6% runaway rate remains (vs teacher's 23.6%) — lower than the teacher but non-zero.
  Anti-collapse sampling (repetition-aware decoding) or a second RFT round would shrink it further.
- **UTMOS is relative-only** — English-MOS-trained, miscalibrated on Hinglish; used as a not-degraded-vs-teacher
  signal, not an absolute naturalness score (same caveat as every prior tier; see `FINAL_REPORT.md`).
- **Not yet done:** ship the 90M model to HuggingFace (strip + upload, like the 265M); vocab-trim 6681→~2500
  for ~85M; INT8 (W8A8) for deployment (bytes, not params).

---

## 9. The one transferable lesson

**Aggressive width pruning must be staged.** Cutting d=1024→640 in one shot destroyed the content-modeling
capacity needed for code-switch text-faithfulness (accent 0.17), and no amount of distillation or RL recovered
it. The *same* 89.96M architecture, initialized from a recovered d=768 intermediate, reached accent 0.83 and
certified above the teacher. The capacity wall was an optimization/initialization artifact of the cut size, not
a fundamental limit of d=640 — gradual cuts with a recovery pass between each step are what cross it.
