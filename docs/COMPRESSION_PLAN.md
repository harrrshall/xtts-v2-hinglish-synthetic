# XTTS-v2 Hinglish AR GPT Compression Plan: 400M to 200M to 100M

Status: engineering plan, not yet executed. Author target: take the fine-tuned XTTS-v2 Hinglish autoregressive GPT from ~443M parameters down the ladder 443M -> ~200M -> ~100M without quality loss against the current 400M student, on the existing 4-axis paired eval, gated by the same rule the student passed against the teacher (delta-CI upper bound >= -0.03 on every axis).

---

## 1. TL;DR and the recommended path

**Primary method: warm-start knowledge distillation from the existing 443M Hinglish GPT into a smaller same-vocab GPT, combining four loss signals (1026-class logit KL, sequence-level CE on teacher audio tokens, per-layer hidden-state plus attention matching, and decoder-input latent-cosine), run as a progressive 443M -> 200M -> 100M ladder against the frozen DVAE and HiFi-GAN.** This family wins because the project already owns every expensive input it needs: a same-architecture teacher, a free synthetic-data pipeline, and a frozen audio tokenizer and vocoder that pin the output contract. Structured pruning is the same recipe with a smarter initialization (prune-then-heal); it is folded in as the init strategy, not a competing path. Quantization is a byte-only shipping stage at the very end and does not count toward the parameter goal.

**Decisive recommendation on the target:**

- **Ship ~200M as the no-quality-loss deliverable.** 443M -> ~200M is depth-only at d=1024 (30 layers down to 12), every frozen interface stays byte-exact, no output adapter is needed, and the closest published analogue (SPADE on CosyVoice 2, a frozen-tokenizer frozen-vocoder AR codec-token TTS) held UTMOS and speaker similarity flat at a 50% depth cut. This is a high-confidence parity win and should be validated first as proof the distillation rig works.
- **Treat 100M as a measured-risk stretch goal, not a parity guarantee.** Reaching ~100M forces a width cut to d=768 plus a learned 768->1024 latent adapter into the frozen HiFi-GAN. Three independent adversarial reviews and the strongest 2025-2026 TTS-distillation evidence converge: voice fidelity (SECS) and naturalness (UTMOS) are the casualties of parameter reduction, they do **not** saturate at small scale, and the project's accent axis already sits exactly on the gate (delta -0.030, CI upper bound -0.001). The honest expectation at 100M is a small but measurable regression on the accent and/or SECS axis that breaches the 3% gate. **The safe no-loss floor is ~150-200M.**

If a single number is required for "no quality loss," it is **~200M**. Pursue 100M as an experiment with 200M as the guaranteed-parity fallback, and report the 100M deltas in full rather than relaxing the gate to declare a pass.

---

## 2. Why this setup is unusually favorable

Three assets stack to make this far easier than generic LLM or TTS compression:

1. **Free, unlimited distillation data.** The `scripts/hinglish/01-04` pipeline already turns Hinglish text into teacher audio. The same pipeline emits, for each item, the teacher GPT's per-layer hidden states, attention maps, 1026-way audio-token logits, and the 1024-dim decoder-input latent. Distillation data (the thing the scaling law says a small model needs) is effectively a one-time forward pass over a corpus you can regenerate at will. Speech-LM scaling exponents are near-equal (alpha ~ 0.25, beta ~ 0.24), so a smaller backbone can be partly compensated by more data, and here data is free.

2. **Frozen DVAE codebook and frozen HiFi-GAN.** The 1026-token audio vocabulary and the vocoder (which consumes a 1024-dim latent plus a 512-dim d_vector) are identical for any GPT size. That collapses TTS compression quality to one question: does the smaller GPT reproduce the same 1026-way audio-token distribution and the same pre-vocoder latent? That is exactly what logit KL and latent-cosine optimize, and exactly what `docs/FP16_VERIFICATION.md` already measures (greedy token identity, latent cosine 0.9999+). The decoder-retraining risk that dominates most TTS compression is removed entirely.

3. **Narrow domain.** This model covers 4 voices and one code-switched register, not XTTS's 17 languages. The parameters a general model spends on coverage are free headroom here. GMM-LM (ICLR 2025) trains AR speech LMs natively at 51.5M-315M and the 51.5M model already beats 500M VALL-E on content; the content and intelligibility quality knee for AR speech LMs sits below 100M. That argument carries axes 1 and 2 (intelligibility, accent) at 100M.

**The honest counterweight (load-bearing, do not skip):** narrow domain helps content, not speaker fidelity. A Dec 2025 XTTS-v2-family distillation paper (arXiv 2512.17356) found speaker similarity is **capacity-bound, not data-bound** ("model capacity constraints fundamentally limit this capability beyond data quality alone"). GMM-LM's only scaling gain across 51.5M->315M was speaker similarity (SIM +0.05, a gradient larger than the 0.03 tolerance, landing exactly in the 100-200M band this plan cuts through). So the free-data advantage does **not** rescue SECS at the width cut. SECS and UTMOS are the binding axes at 100M and no amount of free data fixes a width-induced capacity loss.

---

## 3. The parameter budget and the target configs

### 3.1 Where the ~443M actually lives (computed from first principles, d=1024, L=30)

| Component | Params | Scales with | Prunable | Notes |
|---|---|---|---|---|
| GPT-2 backbone (30 layers + final ln) | ~377.9M | L and d^2 | yes | 86% of the model. Per layer = attn 4*d^2 (4.19M) + FFN 8*d^2 (8.39M) + 2 LayerNorms = 12.60M; x30. **The compression target.** |
| ConditioningEncoder (mel speaker/prosody) | 25.2M | d^2, attn_blocks(6) | cautiously | Conv1d(80->d) + 6 AttentionBlocks. Part of the small-scale floor: invisible to layer-dropping. |
| PerceiverResampler (32 latents, depth 2) | 21.0M | d^2, depth, ff_mult | cautiously | The "32 conditioning embeddings". Part of the small-scale floor. |
| Text embedding (6681 x d) | 6.84M | d | yes (in d) | Vocab fixed by BPE. |
| Text output head (d -> 6681) | 6.85M | d | yes / droppable | Used for the text auxiliary loss; unused at inference. Tie or drop. |
| Mel positional embedding | 0.62M | d | minor | Keep audio length budget. |
| Text positional embedding | 0.41M | d | minor | |
| Mel embedding (1026 x d) | 1.05M | d only | **row count frozen** | 1026 rows pinned to the DVAE codebook. |
| Mel output head (d -> 1026) | 1.05M | d only | **row count frozen** | The load-bearing distillation head. |
| Final LayerNorm | ~0.002M | d | no | Negligible. |
| **Total** | **~440.97M (~441M)** | | | Within ~2M of the published 443M GPT-2 prior. |

**The structural insight that drives every config decision:** the compressibility story is not vocab-bound (all four vocab-coupled tables together are 16.8M, ~3.8% of the model, and only ~11-15% even at 100M). It is bounded by (a) the 378M backbone, cleanly depth/width prunable and ideally distilled from the user's own 30L teacher, and (b) a ~46M d^2 **conditioning floor** (ConditioningEncoder 25.2M + Perceiver 21.0M) that is invisible to layer-dropping and only yields to width (d). At d=1024 that floor falls to 26.8M at d=768 and 19.1M at d=640. Consequently:

- The **200M target is reachable by depth alone at d=1024** (zero decoder-compat risk).
- The **100M target forces d down**, because at d=1024 the fixed 46M conditioning floor plus 16.8M tables already floor you at ~113M with only a 4-layer backbone (too shallow for stable AR generation). d is the only knob that shrinks the floor, and it is also the most dangerous knob.

### 3.2 Target config: ~200M (the deliverable)

```
d (gpt_n_model_channels)        = 1024   (UNCHANGED)
gpt_layers                      = 12      (from 30)
gpt_n_heads                     = 16      (head_dim 64, unchanged)
ffn_mult                        = 4       (HF GPT2 default, unchanged)
ConditioningEncoder attn_blocks = 6       (UNCHANGED)
PerceiverResampler depth        = 2, num_latents = 32   (UNCHANGED)
gpt_num_audio_tokens            = 1026, start 1024, stop 1025  (frozen contract)
output adapter                  = NONE (d stays 1024)
```

Param estimate: ~214M (backbone 151.2M + conditioning 46.2M + tables 16.8M). Decoder-compat risk: **zero**. Every frozen interface (1026 vocab, decoder_input_dim=1024, d_vector=512, 32 perceiver latents) stays byte-for-byte. Distillation reduces to per-layer hidden-state plus 1026-logit plus latent-cosine matching from the 30L teacher.

### 3.3 Target config: ~100M (the stretch)

```
d (gpt_n_model_channels)        = 768     (from 1024)  <- the dangerous move
gpt_layers                      = 10
gpt_n_heads                     = 12      (head_dim stays 64)
ffn_mult                        = 4
ConditioningEncoder attn_blocks = 6       (KEEP UNREDUCED, protects SECS)
PerceiverResampler num_latents  = 32      (KEEP UNREDUCED, protects SECS)
output adapter                  = learned Linear(768 -> 1024), ~0.79M  <- feeds frozen HiFi-GAN
d_vector path                   = 512 preserved
```

Param estimate: ~111M (backbone 70.9M + conditioning 26.8M + tables 12.6M + adapter ~0.8M). The d=768 cut shrinks the conditioning floor from 46.2M to 26.8M, freeing budget for a healthier L=10 backbone instead of the unstable d=1024/L=4.

Live alternatives if 100M misses parity (pre-register these):
- **d=1024 / L=8 (~155M):** the safe parity floor if the width cut fails. Depth-only, no adapter.
- **d=896 / L=14 (~155M) or d=896 backbone with d=1024 conditioning (asymmetric width):** partial width step that keeps speaker capacity.
- **d=640 / L=14 (~99M):** smaller floor, deeper backbone, but the 640-wide head may be under-capacity for the 1026-way prediction and could make WER worse. A/B it, never assume it.
- **Avoid d=1024 / L=4 (~113M):** the conditioning floor forces a backbone too shallow for stable AR audio-token generation.

---

## 4. The compression toolkit, ranked

### Rank 1: Knowledge distillation (warm-start, multi-signal), the primary method

**What it buys:** the parameter reduction itself, recovered to near-teacher quality. Score 7/10 for hitting 100M at parity; 9/10 for hitting 200M at parity.

**Evidence.** Spotify SSW 2025 distilled an AR token-prediction TTS 1.3B -> 500M (2.6x) via logit KL on the teacher's CFG token distribution, holding speaker similarity (59.66 -> 58.72) and improving naturalness and WER; warm-start was load-bearing (random-init 180M gave WER 0.462 vs 0.047 warm-started). SPADE (arXiv 2509.20802) halved depth on CosyVoice 2 (24->12 layers, frozen tokenizer and vocoder) at UTMOS 4.41->4.41, SIM 0.81->0.82. TinyWave distilled a 7B speech LM to 2B with hidden+attention+logit matching at 93-97% retention. DMOSpeech initialized a 450M student from teacher params and surpassed the teacher on WER and SIM.

**When to use:** always, as the spine. The four signals: (1) **sequence-level CE** on teacher audio tokens (free, the cheapest leg); (2) **temperature-softened KL** on the 1026-way next-audio-token distribution (the primary signal, the one that defends SECS and WER); (3) **per-layer hidden-state plus attention-map matching** from the 30L teacher (defends naturalness and long-range coherence); (4) **latent-cosine/MSE** on the decoder-input latent (defends the frozen-vocoder contract; required once the adapter exists at 100M).

### Rank 2: Structured pruning + distillation recovery (prune-then-heal), the initialization

**What it buys:** a far better starting point than random init for Rank 1. Score 7/10. This is not a separate path; it is **how you initialize the distilled student.**

**Evidence.** SPADE is itself prune-then-heal. Minitron pruned Llama-3.1 8B->4B and recovered 93% of teacher MMLU with 94B distillation tokens (40x fewer than from-scratch). Gromov et al. (angular-distance layer selection) removed 45-55% of Llama-2 layers with brief QLoRA healing. LLM-Pruner recovered to ~95% at 20% width pruning with light LoRA.

**When to use:** at every ladder step, to choose which teacher layers seed the student. Rank the 30 layers by angular cosine distance between layer input and output on Hinglish audio-token sequences (Gromov's arccos metric) and by a SPADE-style WER-based importance index (ablate each layer, measure Qwen3-ASR content-word recall delta). Never drop the final layer or the first 2-3 shallow layers. Initialize the 12 (or 10) student layers from the surviving most-important teacher layers, copy embeddings, heads, conditioning, and perceiver verbatim.

### Rank 3: Low-rank factorization and cross-layer weight sharing, a complementary top-up

**What it buys:** an extra ~15-20% on the surviving unique-layer matrices, or a parameter-sharing route to the same targets. Score 6.5/10. **Standalone (training-free) it cannot reach 100M:** ASVD/SVD lose parity past ~20% reduction (ASVD LLaMA-2-7B: 5.47 ppl at 10% but 8.91 at 20%, cliff below 0.85 retain). Only the distillation-recovered form works.

**When to use:** as a top-up inside the distillation run, not as the main cut. Relaxed Recursive Transformers (tie layers, relax with per-loop LoRA, distill) recovered a 2x-shared Gemma to full-size parity and beat same-size-from-scratch by 13.5 points, which is the keystone result for "share-and-distill beats small-from-scratch." Useful if the depth-only 200M needs to shed a few more M without a width cut. Treat LoRA rank as part of the param budget, not free. GQA and embedding factorization look attractive from LLM literature but pay little here (KV is a small slice of 4*d^2; tables are 3.8%); do not over-invest.

### Rank 4: Quantization, byte-only shipping stage, scores ~0 on the goal

**What it buys:** smaller bytes at a fixed parameter count. It removes zero parameters, so on the literal "fewer parameters" goal it scores ~0. Score 3/10 and only as a complement.

**When to use:** as the **last** stage, applied to an already-distilled 100M-parameter student, to ship it small. int8 W8A8 (GPTQ/AWQ) on the backbone roughly halves bytes at >99% retention (LLM evidence). Quantization-Aware Distillation can fold int8-robustness into the distillation run for free. **Hard rule:** the benchmark must count `sum(p.numel())` for the GPT submodule, never file size in MB; a quantized 443M model is still 443M parameters and fails the goal. **Never** aggressively quantize the 1026 mel head or mel embedding (BitTTS shows sub-2-bit costs -0.45 to -0.66 MOS, which would fail the gate on UTMOS).

### How they combine

The recommended stack is **prune-to-init (Rank 2) -> distill (Rank 1) -> optional low-rank top-up (Rank 3) -> quantize to ship (Rank 4)**, run as a progressive ladder:

```
443M teacher
   |  depth-prune-init (layers 0,3,5,...,29) + 4-signal distill, d=1024
   v
~200M student   <- GATE A (all 4 axes). SHIP THIS. Becomes the Stage-B teacher.
   |  width-prune-init (d 1024->768, activation-importance) + 5-signal distill
   |  (use the 200M as teacher, NOT 443M, assistant distillation narrows the capacity gap)
   v
~100M student   <- GATE B (re-weighted eval). Stretch, measured, not assumed parity.
   |  optional int8 W8A8 PTQ
   v
~55MB shipped artifact at 100M params
```

The assistant-distillation step (200M as the Stage-B teacher, never the 443M directly for the width cut) is not optional: width reduction from a too-large teacher "leads to worse results" (Minitron), because the capacity gap is too wide for a single jump.

---

## 5. The end-to-end recipe, mapped onto this repo

The current `scripts/hinglish/train_xtts.py` builds `GPTArgs` **without pinning `gpt_layers` or `gpt_n_model_channels`**, so it inherits the coqui defaults (30 layers, d=1024). That single block is the insertion point for every config change below.

### Step 0: instrument and build the distillation corpus

1. Add a teacher-forward caching script (`scripts/hinglish/13_distill_cache.py`, new) that loads the 443M checkpoint and, for each item from the `01-04` pipeline, caches: per-layer hidden states, attention maps, the 1026-way audio logits, and the 1024-dim decoder-input latent, on the held-out-disjoint train sentences and 4 voices.
2. Target ~50-150 hours-equivalent of teacher audio tokens to start (SPADE healed with <5% of pretraining data; the Dec 2025 result shows SECS is capacity-bound, so do **not** burn compute past diminishing returns expecting data to fix a width gap). Scale only reactively if an axis fails.
3. Add a `param_count` helper that prints `params: X M` (GPT submodule, excluding frozen DVAE/HiFi-GAN/speaker encoder) and `size: Y MB` as two distinct numbers, so quantization is never mistaken for parameter reduction.

### Step 1: free parameter win, no training cost

4. Tie `text_embedding` (6681 x 1024) to the `text_head` weight (standard weight tying) to drop ~6.85M params. The text head is unused at inference; verify with the FP16 greedy-token-identity check that the text auxiliary loss is unaffected. Validate it does not regress the audio-token LM, not just the text head.

### Step 2 (Stage A): 443M -> ~200M, depth-only, d=1024 (the lower-risk validation of the method)

5. In `train_xtts.py`, extend `GPTArgs` to set `gpt_layers=12` and add a `--teacher-ckpt` argument; everything else (d=1024, heads=16, num_audio_tokens=1026, perceiver depth 2, conditioning attn_blocks 6) stays at defaults. Initialize the 12 layers from the SPADE/angular-distance-ranked teacher layers (Step 0 ranking, or evenly strided as a forgiving fallback; the layer-selection paper shows only a 1-3 point spread across strategies). Copy embeddings, both heads, ConditioningEncoder, and PerceiverResampler verbatim.
6. Subclass `GPTTrainer` (new `scripts/hinglish/distill_trainer.py`) to add the distillation loss to the existing CE:
   `L = lambda_seq * CE(teacher tokens) + lambda_kl * KL(student||teacher, T~2-4 on the 1026 logits) + lambda_hid * (1 - cos) on mapped hidden states + lambda_attn * KL on attention maps + lambda_lat * MSE/cosine on the decoder-input latent`.
   Wire the cached teacher tensors (Step 0) in as targets. Keep the existing AdamW recipe (lr 5e-6, betas [0.9, 0.96], wd 1e-2, batch 8, grad-accum 4) but train for more steps than the 8-epoch fine-tune since this is a capacity transfer, not adaptation. No output adapter (d=1024).
7. Use **deep-supervision** (match teacher hidden states at multiple depths), because the depth cut most threatens long-range AR coherence over the up-to-605-token generations.

### Step 3 (Gate A): validate 200M, ship it

8. Run `scripts/hinglish/08-11` (the 4-axis paired eval, n=89, bootstrap 95% CI, paired by utt_id) plus the FP16-style greedy-token-identity and latent-cosine numerical checks (`fp16/verify_numerical.py` structure; reuse its deterministic-seed discipline to avoid the documented hash()-seed false alarm).
9. **Re-weight the held-out set toward LONG Hinglish utterances** (the depth cut's failure mode is long-range prosody, which n=89 may under-sample) and report SECS per-voice across all 4 voices. The deeper-layers QA result does **not** transfer to long-horizon generation; validate on long-form before declaring parity even at 200M.
10. Gate: every axis delta-CI upper bound >= -0.03 vs the 400M student. If it passes, **200M is the no-quality-loss deliverable.** Ship it.

### Step 4 (Stage B): ~200M -> ~100M, width cut (the stretch)

11. Build the student with `gpt_n_model_channels=768`, `gpt_n_heads=12` (head_dim 64), `gpt_layers=10`, **keeping** `ConditioningEncoder` attn_blocks=6 and perceiver num_latents=32 unreduced. Add a learned `Linear(768->1024)` output adapter before the frozen HiFi-GAN latent input, initialized to approximate identity on the shared subspace. Preserve the 512-dim d_vector path byte-for-byte. Initialize backbone width via Minitron-style activation-importance selection on a ~1024-sample Hinglish calibration set, never random.
12. Distill with the **200M model as teacher** (assistant distillation). Same five-term loss, but up-weight `lambda_lat` and `lambda_kl`: the adapter must reproduce the exact latent distribution the frozen vocoder consumes (use `fp16/verify_numerical.py`'s latent-cosine as the training objective and the gate), and WER is the first axis to break so the 1026-logit KL gets extra weight. If SECS drops, keep the conditioning stack at d=1024 with a 1024->768 input projection (asymmetric width) rather than narrowing the speaker path.

### Step 5 (Gate B): measure 100M without relaxing the gate

13. Rerun the 4-axis eval, re-weighted toward long utterances, SECS reported per-voice, plus a code-switch-boundary CMOS or A/B spot-check (the accent metric is an English-ASR proxy a width-narrowed model can game by keeping English-word recall while flattening Hindi-English boundary prosody).
14. Do **not** relax the gate. If 100M misses on accent or SECS by a small margin, ship the fallback (d=1024/L=8 ~155M, or d=896 asymmetric) and report the 100M deltas as a measured tradeoff, not a parity claim.

### Step 6 (ship): optional quantization

15. Apply int8 W8A8 (GPTQ/AWQ) PTQ to the distilled student backbone, calibrated on teacher Hinglish audio-token sequences, target >99% retention. Exclude the 1026 head and mel embedding. Skip int4 unless the 4-axis gate still passes.

### GPU-hour estimate

Each ladder stage is fine-tune-scale, not pretraining. Distillation data is ~free (one teacher-forward caching pass, a few GPU-days for ~100h-equivalent). Stage A (depth, 200M) is the cheapest and highest-certainty: single-digit GPU-days on one modern GPU for a 12-layer student on a 4-voice domain. Stage B (100M) is more expensive because the adapter and narrower conditioning need more steps to converge. Two ladder stages plus eval reruns and 2-4 candidate-config ablations: roughly **1-3 GPU-weeks total wall-clock**. The eval and verification harness already exists, so the gate is near-zero marginal cost. The larger cost is engineering: the 5-term distillation loss, the teacher-tensor wiring, and the d->1024 adapter in the coqui `GPTTrainer`.

---

## 6. Risk register: what could prevent true parity

| # | Risk | Axis hit | Evidence | Mitigation |
|---|---|---|---|---|
| R1 | **The width cut (d 1024->768) is capacity loss, not just compression.** It shrinks audio-token prediction capacity and conditioning capacity at once. | SECS, UTMOS | TinyWave lost 8-10 pts speaker consistency under width compression; Dec 2025 XTTS-family paper: speaker similarity capacity-bound (SIM 0.544->0.404, ~26% relative), explicitly not fixed by data; GMM-LM SIM +0.05 across 51.5M->315M proving SECS scales with params. | Keep conditioning unreduced; asymmetric width (d=1024 conditioning, d=768 backbone) if SECS drops; latent-cosine training target. |
| R2 | **The d->1024 adapter feeds a frozen HiFi-GAN a distribution it never saw.** Under-distilled, it produces vocoder artifacts. | UTMOS, SECS | Neural vocoders degrade audibly under latent distribution shift (cyclical post-filter, latent-burden findings); no precedent that frozen-decoder adapters preserve SIM within 3% under simultaneous depth+width cuts. | Hard latent-cosine gate (>=0.999) before any subjective eval; train adapter jointly with the dedicated latent loss. |
| R3 | **Long-range AR coherence collapses before short-form metrics catch it.** Generative free-form tasks degrade far earlier than the QA benchmarks the deeper-layers paper used; XSum/C3 generative scores fell to near zero at 25% layer removal and were not fully recoverable. AR audio over 605 tokens is the worst case. | UTMOS (long), intelligibility | On the Limits of Layer Pruning for Generative Reasoning (arXiv 2602.01997); compounding errors invisible to QA evals. | Deep-supervision hidden-state matching; eval re-weighted toward long utterances; validate long-form even at 200M. |
| R4 | **The project's accent axis has near-zero error budget.** The 400M student is already at accent delta -0.030, CI upper bound -0.001, sitting exactly on the gate; intelligibility -0.016, CI upper bound 0.000. | accent, intelligibility | `docs/FINAL_REPORT.md`. A fraction of a percent more loss pushes the CI past the gate. Width pruning specifically degrades the fine code-switch acoustic precision that lives in "fragile" representations (Width Pruning Dichotomy, arXiv 2512.22671). | Up-weight 1026-logit KL; code-switch-boundary CMOS probe beyond English-word recall; oversample sentence-initial English markers. |
| R5 | **The headline 100M precedent is a misread.** Spotify's true no-perceivable-loss point was the 500M (2.6x) student; their 180M (~7x) student dropped noticeably on speaker fidelity and naturalness. SPADE tops out near 2x with a +0.79 WER regression on a 62.5% cut; no published AR codec-token TTS result hits 4x including width at parity. | all, esp SECS/WER | Spotify SSW 2025; SPADE; TinyWave (3x, -10pp speaker). The supported parity band is 2-3x. | Treat 200M (2.2x) as the deliverable; 100M (4.4x) as research at the edge of the evidence. |
| R6 | **Prune-then-heal has a measured recovery ceiling of ~93%, not zero loss.** Minitron's best width-pruned 4B reached ~93% of teacher MMLU at just 2x on a knowledge metric. | all | Minitron; self-data-distillation tops out near 93% (block size 6). | Two-stage assistant distillation; activation-importance init; greedy-token-identity vs the 400M student as a hard content-shift detector. |
| R7 | **Measuring the wrong thing.** Reporting a quantized 443M model as "100M" because the file is small. | (goal integrity) | BitTTS: 1.58-bit cuts 25.66->4.39 MB at 0% parameter reduction. | Count `p.numel()`, report bytes separately. |
| R8 | **Toolchain and seed bugs masquerade as quality loss.** | (false alarms) | The documented hash()-seed false alarm in `docs/FP16_VERIFICATION.md`. | Reuse the deterministic-bootstrap discipline; numerical checks first, subjective second. |

**Net assessment (the three adversarial reviews agree):** 200M depth-only is supported and shippable at parity. 100M is refuted as a no-quality-loss claim on current evidence. The most likely 100M outcome is WER up a few tenths of a point and SECS down 0.03-0.05, breaching the project's own 3% gate on at least one axis. The favorable setup (frozen vocoder, same-arch teacher, free data, narrow domain) makes 100M **less catastrophic** than the generic literature predicts, but less catastrophic is not no loss.

---

## 7. Open questions and decisions for the user

1. **Is 200M an acceptable deliverable, or is 100M a hard requirement?** This is the single decision that shapes everything. If 200M is acceptable, the plan is low-risk and well-precedented. If 100M is mandatory, you are accepting a likely small measured regression and should decide now whether you will (a) relax the gate for the 100M model and document the deltas, or (b) hold the gate and ship the largest config that passes (likely ~150-200M).

2. **For the 100M config, which lever do you spend first if it misses?** Options: asymmetric width (keep conditioning at d=1024), fall back to d=1024/L=8 (~155M), or try d=640/L=14. Pre-register the ladder before training so a near-miss is a planned fallback, not a scramble.

3. **Do you want a human CMOS panel as the final naturalness anchor?** `docs/FINAL_REPORT.md` already flags UTMOS as miscalibrated on Hinglish and human MOS as the only true anchor. At 100M, where UTMOS is a binding axis, a 10-minute CMOS panel on the worst voices is the only way to certify "no perceived loss." Decide whether programmatic parity is sufficient or whether a human pass gates the ship.

4. **How much distillation data before you accept diminishing returns?** The Dec 2025 result says SECS is capacity-bound, so data past ~50-150h-equivalent will not fix a width-induced SECS gap. Set a stop rule so you do not burn compute expecting data to rescue capacity.

5. **Do you tie or fully drop the text head?** Tying saves ~6.85M for free; dropping it post-distillation (the head is unused at inference) saves the same and removes the text auxiliary loss path entirely. Decide whether the text auxiliary loss is worth keeping during distillation.

6. **Quantization scope at ship time.** int8 W8A8 is safe and roughly halves bytes at >99% retention. int4 is only worth it if the full 4-axis gate still passes. Decide whether bytes matter enough to risk the extra quantization step, given the parameter goal is already met by the distillation.

---

Relevant files: `scripts/hinglish/train_xtts.py` (the `GPTArgs` block, currently unpinned at L=30/d=1024, is the config insertion point), `scripts/hinglish/08-11` (the 4-axis paired eval gate), `scripts/hinglish/fp16/verify_numerical.py` (latent-cosine and greedy-token-identity primitives to reuse as distillation targets and the numerical gate), `docs/FP16_VERIFICATION.md` (the no-quality-loss report template), `docs/FINAL_REPORT.md` (the parity bar and the accent-axis headroom this plan must respect).

---

## 8. Stage A implementation (200M, committed): files and commands

Decision: ship ~200M first, quality first. Stage A (depth-only, 30 to 12 layers, d=1024 unchanged) is
implemented. Teacher choice (answering "use Smallest.ai instead?"): the 443M XTTS GPT is the white-box
KD teacher for logit-KL and hidden-state matching (Smallest.ai is a closed API, so it cannot supply
logits or hidden states); the Smallest.ai audio corpus stays the ground-truth cross-entropy anchor (the
existing `metadata_train.csv`, DVAE-encoded by the trainer for free). This keeps the 200M pointed at the
true quality bar rather than only copying the 443M's small drift.

Files:
- `scripts/hinglish/distill_trainer.py` (new): `DistillGPTTrainer` subclasses coqui `GPTTrainer`. Per
  step it runs the 12-layer student (grad) and the frozen 30-layer teacher (no grad) on the same batch,
  and adds `loss_kd_logit` (temperature-softened KL over the 1026 audio classes, masked to valid mel
  positions), `loss_kd_hidden` (1 - cosine on the teacher layer each student layer was initialized from,
  identity projection since d=1024), and an optional `loss_kd_attn`, on top of coqui's text-CE + mel-CE.
  The teacher is hidden in a non-module container so it is never optimized, never `.to()`-moved, and
  never saved into checkpoints. Hidden states come from a forward hook that injects
  `output_hidden_states=True` into the inner HF `GPT2Model`, with no edit to coqui source. The student
  GPT is warm-started from a strided copy of the teacher layers (KEEP = 0,2,5,8,11,14,17,20,23,26,28,29).
- `scripts/hinglish/train_xtts.py` (edited): `--distill --teacher-ckpt --student-layers --kd-temp
  --kd-logit-w --kd-hidden-w --kd-attn-w --layer-map`. The non-distill path is unchanged.

Smoke first (validates the loop, shapes, finiteness, and that the hook fired; 24 train samples, 1 epoch):
```
CUDA_VISIBLE_DEVICES=5 .venv_xtts/bin/python scripts/hinglish/train_xtts.py \
    --distill --smoke --teacher-ckpt runs/xtts_hinglish/RELEASE/model.pth
```
Full Stage-A run (batch halved + grad-accum doubled to fit two GPTs at the fine-tune's effective batch 32):
```
CUDA_VISIBLE_DEVICES=5 .venv_xtts/bin/python scripts/hinglish/train_xtts.py \
    --distill --teacher-ckpt runs/xtts_hinglish/RELEASE/model.pth \
    --epochs 24 --batch-size 4 --grad-accum 8 --lr 5e-6 \
    --kd-temp 2.0 --kd-logit-w 1.0 --kd-hidden-w 1.0 \
    --out-path runs/xtts_hinglish_distill12
```

Three spots to confirm on the first smoke run (cannot be checked off-box; the smoke path prints what you
need): (1) coqui loading the 30-layer base `xtts_checkpoint` into the 12-layer student uses `strict=False`
and must not error on the dropped layers; if it does, set `xtts_checkpoint=""` for the distill branch and
load HiFi-GAN separately. (2) `mask_frac` printed by the smoke step should match the average valid mel
fraction (the mask is derived from `wav_lengths // gpt_code_stride_len + 1`; adjust the `+1` if it looks
off by a position). (3) `hs(student=13 teacher=31)` confirms the hidden-state hook fired.

After the run: set `gpt_layers=12` in the inference config, generate the golden panel with
`compare/gen_panel_ckpt.py`, score with `08`/`09`/`10`, and certify against the 443M with
`compare/certify.py --tier 200m` (the `BENCHMARK_SPEC.md` gate). Stage B (100M) is deferred per the
decision above.