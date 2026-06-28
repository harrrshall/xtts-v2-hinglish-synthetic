# BENCHMARK_SPEC: no-quality-loss certification for the compressed Hinglish XTTS GPT

Authoritative benchmark for two jobs:

1. **Measure** the current 400M Hinglish GPT student rigorously on a frozen golden set, and
2. **Certify** whether a compressed candidate (200M, then 100M) has **NO QUALITY LOSS** relative to
   that 400M student, using a test engineered so that neither a degraded model nor a fine one is
   mislabeled.

The reference for the no-loss decision is the **current 400M student**, not the teacher TTS. The goal
is: "the 100M model matches the 400M within the same tolerance the 400M matched the teacher" (<= 3%
per axis on the 4-axis paired eval). Only the autoregressive GPT is compressed; the DVAE tokenizer
and HiFi-GAN vocoder are frozen and byte-identical across every tier, so quality reduces almost
entirely to "does the smaller GPT predict the same 1026-way audio-token distribution".

This spec supersedes the verdict logic in `scripts/hinglish/11_aggregate_eval.py`. That script stays
only as a descriptive mean/CI reporter; the certification decision is owned by
`scripts/hinglish/12_equivalence_eval.py` and the one-page runner `scripts/hinglish/compare/certify.py`.

---

## 1. Purpose: false positives vs false negatives

A no-quality-loss claim has two ways to be wrong, and they are not symmetric. The old gate fails on
both.

**False positive (the headline danger): passing a degraded model.** The legacy rule in
`11_aggregate_eval.py` declares "not degraded" when the paired-delta bootstrap CI **upper** bound is
>= -0.03 (`boot_ci` line ~22, verdict line ~68). That is the wrong object. A wide, underpowered CI
whose upper end clears a near-zero tolerance passes even when the model is badly degraded; absence of
evidence is being read as evidence of equivalence, and the **noisier** the eval, the **easier** it
passes. This is exactly backwards from what certification needs.

> Guard: switch to a one-sided **lower** bound versus a pre-registered margin (non-inferiority / TOST).
> A real drop near the margin pushes the lower bound below -m and **FAILS**. Pair this with an
> explicit **INCONCLUSIVE** verdict whenever an axis is underpowered, so noise can never manufacture a
> PASS. An underpowered axis is never rubber-stamped as "no quality loss".

**False negative: failing a model that is actually fine.** Two sources. (a) Underpowered axes:
at the current held-out sizes the TOST power is only ~0.60 for accent and ~0.79 for naturalness at
true delta=0, so a fine model flags or a real-but-tiny regression is missed. (b) The
**hash()-seeded false alarm** documented in `docs/FP16_VERIFICATION.md`: a `-0.21` UTMOS "drop" that
was an artifact of the fp32 and fp16 runs drawing **different** temperature-0.7 takes because the
seed used Python `hash()` (salted per process), compounded by a degenerate bootstrap.

> Guard: a power analysis that fixes held-out n per axis (oversampling code-switched clips so accent
> leaves n=57), deterministic non-`hash()` paired seeds shared across reference and candidate, and a
> multi-seed protocol that separates temperature-0.7 sampling variance from true model variance.

**Two more structural defects the old gate has, both guarded below:**

- **Degenerate bootstrap.** `boot_ci` is a hand-rolled LCG whose low bits collapse at small n. At
  n=16 it visits only 16 distinct resample vectors out of 4000 and returns a **zero-width** CI
  ([-0.0195, -0.0195] vs numpy 0.0403), the second half of the FP16 false alarm. It is acceptable
  only at n=57/89 where it happens to reach 4000 distinct vectors, and it is unusable for the small
  per-cs-bin slices. Replaced by `numpy.random.default_rng(20260625)` BCa.
- **No multiplicity control, no sampling/model split.** The 4 axes and 8 cs-bins are read
  independently with no family-wise correction, and each utterance is decoded once at temp 0.7 so
  per-utterance sampling variance is folded into the model-difference signal. Replaced by Holm across
  the 4 axes, Benjamini-Hochberg FDR for the slices, and K-seed decoding with an ANOVA variance split.

---

## 2. Metric suite

Every axis is **paired by `utt_id`** (same sentence + same voice synthesized by reference and
candidate), which is the unit of replication for the bootstrap. Quality axes 1 through 4 already exist
in `scripts/hinglish/08-10`; the efficiency axes are new and prove the compression actually paid off.

### Quality axes (gating)

| # | Axis | What it measures | Tool / model (pinned) | Native unit | Failure mode it catches |
|---|------|------------------|------------------------|-------------|--------------------------|
| 1 | `intelligibility_recall` | Convention-robust content-word recall of the intended Hinglish text after ASR round-trip | Qwen3-ASR `Qwen/Qwen3-ASR-1.7B`, bf16, `language="Hindi"`, `max_new_tokens=256`; recall via `common.content_word_recall` (`08_student_eval.py`) | recall in [0,1] | Words dropped / mangled; the logit-KL and seq-KD signal most directly defends this |
| 2 | `accent_en_recall` | Whether embedded English actually sounds like English (code-switch quality), measured as English-orthography recall under an **English** ASR | `openai/whisper-large-v3`, fp16, `language="en"`, `task="transcribe"`, fuzzy match in `10_accent_eval.py` | recall in [0,1] | Hindi-accented English; only scorable on clips that contain English words, which is why its n is naturally smaller and must be oversampled |
| 3 | `naturalness_utmos` | Neural MOS naturalness proxy | UTMOS `tarepan/SpeechMOS:utmos22_strong` via `torch.hub`, 16 kHz (`09_objective_eval.py`) | MOS (~1 to 5) | Prosody / long-range AR coherence damage, the axis most threatened by depth cuts |
| 4 | `voice_secs` | Speaker-embedding cosine between a clip and its target-voice reference (cloning fidelity) | resemblyzer `VoiceEncoder` + `preprocess_wav`, cosine of embeddings (`09_objective_eval.py`) | cosine in [-1,1] | Voice drift, the axis most threatened by the d=1024->768 **width** cut and the d->1024 adapter |

Pinned versions, the four golden voice reference wavs (one per voice for SECS), the Qwen3-ASR /
Whisper / UTMOS / resemblyzer commit hashes, and the venv lock are all recorded in the frozen baseline
artifact (Section 4). The judges run on CPU or GPU identically; all four metrics are model-based and
device-independent (confirmed in `FP16_VERIFICATION.md`).

### Content-equivalence pre-screen (gate-0, cheap, deterministic)

Before any perceptual panel, run a numerical pre-screen analogous to
`scripts/hinglish/fp16/verify_numerical.py`. For distillation the greedy audio tokens will **not** be
bit-identical (unlike fp16, where they were), so the operative metric is **not** token identity:

- **decoder-input latent cosine vs the 400M** >= **0.95** on a fixed >= 32-prompt panel (4 voices x 8
  sentence types), AND
- **greedy token-length / edit-distance agreement** under deterministic (`top_k=1`) decode. In the
  runner this is implemented as greedy `n_audio_tokens` agreement within 5% on >= 90% of paired prompts
  (`certify.py` Gate 3).

A fail here predicts a perceptual fail and stops the run before the expensive panel.

### Efficiency axes (the compression WIN; reported, and gating where noted)

These do not measure quality; they prove the candidate is genuinely smaller and faster. Counting
**parameters** (not bytes) is load-bearing: a quantized 443M model is still 443M parameters and must
**fail** the goal regardless of file size.

| Axis | What it measures | Tool | Gates? | Note |
|------|------------------|------|--------|------|
| `gpt_params` | torch `sum(p.numel())` on the AR GPT only (backbone + heads + embeddings + conditioning) | `compare/count_params.py` | **YES** (per-tier budget) | The actual goal. Quantization cannot move this number, so int8/int4 alone fails the gate by construction |
| `conditioning_floor_params` | ConditioningEncoder + PerceiverResampler (~46M at d=1024) | `count_params.py` | reported | The small-scale floor; shrinks only with d, not with layer count |
| `frozen_params` | DVAE + HiFi-GAN + speaker encoder | `count_params.py` | **YES** (must be byte-identical to 400M) | Any change here breaks the frozen-vocoder contract and invalidates the whole comparison |
| `disk_mb` | checkpoint file size | `count_params.py` | reported, NEVER gates | bytes != params; this is where fp16/int8 belong, as a complementary final stage |
| `median_rtf` | real-time factor (decode wall-clock / audio seconds), median over the held-out panel | `gen_panel_ckpt.py` -> `efficiency.json` | **YES** (must beat reference by the tier speedup) | Compression must buy speed |
| `peak_vram` / `mean_decode_s` | peak decode VRAM and mean per-utt decode time | `gen_panel_ckpt.py` (add `torch.cuda.max_memory_allocated`) | reported | Resource win for shipping |

RTF and decode time are recorded per utterance during generation with `torch.cuda.synchronize()`
around the decode call, so the timing is honest on GPU.

---

## 3. Statistical method

### 3.1 Non-inferiority (one-sided TOST) per axis

For each axis compute the paired delta `d_u = score_cand(u) - score_ref(u)` over the common `utt_id`
set. Test, one-sided:

```
H0: mean(d) <= -m      (degraded by at least the margin)
H1: mean(d) >  -m       (non-inferior: no quality loss within tolerance)
```

PASS iff the **BCa one-sided lower confidence bound** of `mean(d)` is **strictly above -m** at the
axis's Holm-adjusted alpha. The decision is driven by the lower bound, never the point estimate and
never the upper bound.

**Pre-registered margins** (native units, frozen in `AXES` of `12_equivalence_eval.py` and `MARGINS`
of `certify.py` BEFORE any candidate exists):

| Axis | margin m | Rationale |
|------|----------|-----------|
| `intelligibility_recall` | 0.03 | 3% absolute content-word recall |
| `accent_en_recall` | 0.03 | 3% absolute English recall |
| `voice_secs` | 0.03 | matches the legacy 0.03 tolerance, ~3% of the ~0.86 operating point |
| `naturalness_utmos` | 0.10 MOS | ~3% of the ~3.1 operating point; sits **above** the fp16 greedy-paired UTMOS noise of -0.010 and below the 0.10 MOS the 400M already moved vs the teacher |

### 3.2 BCa bootstrap

Replace the LCG with `numpy.random.default_rng(seed=20260625)`, **20000** i.i.d. resamples of the
utterance-level deltas, and a **BCa** interval (bias correction z0 + jackknife acceleration a) for the
one-sided lower bound. BCa is used because recall and SECS deltas are skewed and bounded. Implemented
as `bca_bootstrap_lb` in `12_equivalence_eval.py` and `bca_lb` in `certify.py`. Verified that BCa
matches numpy within ~1% at n=57/89 while fixing the small-n collapse the LCG suffers (needed for the
per-bin slices).

### 3.3 Power analysis and required n

Required n per axis from `n = ((z_alpha + z_beta) * sd / m)^2` at one-sided alpha and power 0.80,
using the **observed paired-delta SDs** from `data/eval_big`:

| Axis | observed SD | margin | n @ delta=0 | n @ delta=-m/3 |
|------|-------------|--------|-------------|----------------|
| `intelligibility_recall` | 0.0780 | 0.03 | 65 | 145 |
| `accent_en_recall` | 0.1190 (at n=57) | 0.03 | 150 | 337 |
| `naturalness_utmos` | 0.3842 | 0.10 | 141 | 316 |
| `voice_secs` | 0.0301 | 0.03 | 10 | 22 |

(These use the Bonferroni-conservative alpha 0.0125, z=2.241. The runner `certify.py` enforces
slightly larger floors, `MIN_N = {intel 168, accent 389, naturalness 367, secs 25}`, computed at the
Holm-worst per-axis alpha rather than flat Bonferroni; use the runner values as the binding gate.)

**Recommended held-out tiers:**

- **n = 150 minimum** clears every axis at true delta=0 (accent is the binding axis).
- **n = 320 preferred** clears every axis even at a real-but-tiny drop of delta=-m/3.

The current set is **underpowered** and must grow: at n=89 (and n=57 for accent), TOST power at
delta=0 is intelligibility 0.98, accent **0.60**, naturalness **0.79**, SECS 1.00. Accent and
naturalness fall below the 0.80 bar. Accent additionally must reach parity n with the other axes; it
is only 57 because many clips have no English words to score, so the held-out manifest must
**oversample code-switched clips** until accent has >= 150 scorable English-bearing clips.

Any axis whose TOST power at delta=-m/3 is below 0.80 returns **INCONCLUSIVE** (collect more n), never
PASS. This is the single most important guard against the headline false positive.

### 3.4 Multi-seed protocol (split sampling variance from model variance)

Each held-out utterance is decoded with **K independent seeds** at temperature 0.7, using the
deterministic, process-independent seed scheme (FNV-1a over `utt_id` plus a base, in
`gen_panel_ckpt.py:stable_seed`; the same idea as `fp16/gen_panel.py` line 62
`torch.manual_seed(base + i*1000 + vi*131)`). **Never Python `hash()`** (per-process salted; it caused
the -0.21 UTMOS false alarm). The candidate and reference share the same per-utt seed stream so the
only variable is the model.

Decompose `Var(axis mean) = sigma_model^2 / n + sigma_sample^2 / (n*K)` via a one-way random-effects
ANOVA on the K-per-utt scores (`13_variance_decomp.py`). Report the intraclass correlation (ICC) and
set `K >= 9 * (sigma_sample / sigma_model)^2` so within-utterance sampling SD contributes < 10% of the
SE. **Default K=5**, then re-estimate `sigma_sample` on the first run and tune. **Average the K seeds
per utterance BEFORE forming the paired delta**, so the bootstrap resamples utterances (the true
replication unit), not seed draws.

### 3.5 Multiple-comparison handling

- **Primary family (gating): the 4 axes.** Holm step-down at family alpha 0.05 (one-sided per-axis
  pre-test alpha 0.025). Certification requires **all four** axes to pass at their Holm-adjusted alpha.
- **Secondary family (diagnostic, non-gating): the 8 per-cs-bin slices** (`cs_none`, `cs_low`,
  `cs_med`, `cs_high`, `tech`, `question`, `emotion`, `numbers`). Reported under Benjamini-Hochberg
  FDR at **q=0.10**. Per-bin n is only ~7-11 and per-bin TOST power is 0.06-0.17 for
  recall/accent/UTMOS, so a bin may **FLAG** for investigation but cannot fail certification on its
  own. **SECS is the exception**: per-bin SECS power is ~0.86, so SECS per-bin must keep its lower
  bound > -0.03 and is allowed to gate (SECS is the width-cut watch axis at d=768).

Without correction, 4 axes x 8 bins is up to 36 tests with ~1.4 expected false flags at uncorrected
0.05; Holm plus BH brings that under control while keeping the gating decision on the four pooled,
adequately powered axes.

---

## 4. Baseline lock-in protocol

Everything that could be tuned after seeing candidate results is frozen and committed **before** the
candidate is trained. This prevents margin gaming and post-hoc set selection.

1. **Frozen golden held-out set.** Build a manifest of n >= 150 (target 320) utterances balanced
   across the 4 voices and the 8 cs-bins, **oversampling code-switched bins** so `accent_en_recall`
   reaches >= 150 scorable English-bearing clips, and weighting toward `cs_high`, `numbers`, and
   `tech` (the bins most threatened by the depth cut). Freeze the exact `utt_id` list. This is the
   golden set; it never changes between tiers.
2. **Pinned judges and seeds.** Pin Qwen3-ASR `Qwen/Qwen3-ASR-1.7B`, Whisper `openai/whisper-large-v3`,
   UTMOS `tarepan/SpeechMOS:utmos22_strong`, resemblyzer `VoiceEncoder` (record commit hashes and the
   venv lock). Pin the generation seed base (`20260625`) and the deterministic `stable_seed` scheme.
   Pin the bootstrap seed (`20260625`), `N_BOOT=20000`, K (default 5), and the four golden voice
   reference wavs used for SECS.
3. **Two anchors.**
   - **Teacher TTS** anchor defines the original quality bar and the tolerance the 400M earned (this
     is the historical <= 3% / 0.10 MOS the 400M cleared vs the teacher; recorded for context).
   - **400M student** anchor is the **operative reference** for every certification. Run the existing
     `08-10` scorers on the 400M outputs into `data/eval_400m`, and on each candidate into
     `data/eval_200m` / `data/eval_100m`.
4. **Baseline artifact checked into the repo.** A frozen config committed before any candidate exists,
   holding: the golden `utt_id` list, the pre-registered margins, the per-axis required-n table, the
   pinned judge versions/hashes, the seeds, and the 400M anchor scores (per-axis mean and SD, plus the
   `params.json` from `count_params.py`). Mirror `docs/FP16_VERIFICATION.md` into a new
   `docs/EQUIVALENCE_VERIFICATION.md` that records this plan and states explicitly that the reference
   is the 400M model and that gate-0 uses latent cosine >= 0.95 (not fp16 token identity, which
   distillation will not satisfy).

---

## 5. Acceptance gates

Overall verdict per tier: **CERTIFIED no quality loss** iff **every** quality axis is PASS (not FAIL,
not INCONCLUSIVE) under Holm, AND the parameter, content, and efficiency gates pass. Any INCONCLUSIVE
forces collecting more held-out n rather than declaring equivalence.

### Quality axes (identical decision rule at both tiers; only n must satisfy power)

| Axis | Gate (200M and 100M) | Powered-n floor |
|------|----------------------|-----------------|
| `intelligibility_recall` | BCa one-sided LB of mean(cand - 400M) > **-0.03** at Holm-adjusted alpha | n >= 168 (INCONCLUSIVE below power 0.80 at delta=-0.01) |
| `accent_en_recall` | BCa LB > **-0.03**; requires **n >= 150 code-switched** scorable clips (oversample; do not inherit n=57) | n >= 389 |
| `naturalness_utmos` | BCa LB > **-0.10 MOS** | n >= 367 (141 gives power 0.80 at delta=0; 316/367 stays powered at -0.033 MOS) |
| `voice_secs` | BCa LB > **-0.03** cosine; also gates **per-cs-bin** (the width-cut watch axis) | n >= 25 (trivially powered) |
| **all 4 jointly** | **CERTIFIED** iff every axis PASS under Holm; any INCONCLUSIVE -> collect more n | n meeting all floors |

### Content + efficiency gates (tier-specific budgets)

| Gate | 200M tier | 100M tier | Source |
|------|-----------|-----------|--------|
| gate-0 latent cosine | >= 0.95 on >= 32 prompts | >= 0.95 on >= 32 prompts | `verify_numerical.py`-style |
| greedy content agreement | >= 90% of paired prompts within 5% token-length | same | `certify.py` Gate 3 |
| `gpt_params` budget | <= **215M** | <= **115M** | `count_params.py` |
| `frozen_params` | byte-identical to 400M | byte-identical to 400M | `count_params.py` |
| `median_rtf` speedup vs 400M | >= **1.30x** | >= **1.60x** | `efficiency.json` |
| `disk_mb` | reported only | reported only | never gates |

### Secondary slice diagnostics (BH FDR q=0.10, non-gating except SECS)

Report per-bin deltas across the 8 cs-bins with BH-FDR-adjusted flags at q=0.10. Bins may **FLAG**
for investigation but do not fail certification (per-bin power 0.06-0.17), **EXCEPT SECS per-bin**
(power ~0.86), which must keep its lower bound > -0.03. Weight the held-out set toward
`cs_high` / `numbers` / `tech`, the bins most threatened by the depth cut.

---

## 6. Implementation: file-by-file

| File | Status | Change |
|------|--------|--------|
| `scripts/hinglish/12_equivalence_eval.py` | created, smoke-tested | Per-axis TOST non-inferiority with pre-registered margins (`AXES`), BCa lower bound (`bca_bootstrap_lb`, `default_rng(20260625)`, 20000 resamples, jackknife acceleration), Holm step-down across the 4 axes, a power report (`tost_power` at delta=0 and delta=-m/3), and a PASS / FAIL / INCONCLUSIVE verdict with an overall CERTIFIED flag. Reads `--ref` and `--cand` dirs of per-utt scores; writes `equivalence_report.json`; exits nonzero unless certified. Smoke-tested on `eval_big` (teacher as ref, student as cand): correctly returns INCONCLUSIVE for accent/intelligibility/naturalness (underpowered) and PASS for SECS. |
| `scripts/hinglish/compare/certify.py` | created | One-page accept/reject runner composing four gates: (1) parameter gate from `count_params.py`, (2) quality TOST (BCa + Holm, reusing the `12_*` math), (3) greedy content-equivalence, (4) RTF efficiency. Tier budgets and margins frozen in `TIERS` / `MARGINS`; `MIN_N` enforces the powered-n floor. Writes `certification.json` + `certification.md`; exits nonzero unless certified. |
| `scripts/hinglish/compare/count_params.py` | created | Counts torch params split into `gpt_params` (the compression target), `conditioning_floor_params`, and `frozen_params` (must be unchanged), plus `disk_mb` reported but never gating. A quantization-only "shrink" cannot pass because param count is unchanged. Writes `params.json`. |
| `scripts/hinglish/compare/gen_panel_ckpt.py` | created | Generalizes `fp16/gen_panel.py` from "fp32-vs-fp16" to **any checkpoint** over the full held-out manifest, with deterministic per-utt paired seeds (`stable_seed`, FNV-1a, **no `hash()`**) so candidate and 400M draw the same sampling stream per `utt_id`. Records per-utt decode time + audio duration -> `efficiency.json` (median RTF). `--greedy` produces the controlled content-equivalence panel; default temp 0.7 is the perceptual panel. Run BOTH. |
| `scripts/hinglish/11_aggregate_eval.py` | deprecate decision logic | Delete the `boot_ci` LCG (lines ~22-36) and the -0.03 **upper-bound** verdict (lines ~68-74), or at minimum replace the LCG resample with `rng = numpy.random.default_rng(20260625); idx = rng.integers(0, n, (iters, n))`. Keep this script only as a descriptive mean/CI reporter; route the no-loss DECISION to `12_*` / `certify.py`. Remove the upper-bound rule entirely. |
| `scripts/hinglish/09_objective_eval.py` + `10_accent_eval.py` + `08_student_eval.py` | extend | Score whatever clips the generation step produced. Add a `--seed`/`--n-seeds K` loop (in the generator that feeds their manifests) so each utterance is scored across K temp-0.7 decodes; emit per-seed rows tagged `(utt_id, seed)`. Use `stable_seed` / the `fp16/gen_panel.py` line-62 scheme; never `hash()`. |
| `scripts/hinglish/13_variance_decomp.py` | NEW | One-way random-effects ANOVA on the K-per-utt scores per axis: estimate `sigma_model^2` (between-utt) and `sigma_sample^2` (within-utt across seeds), report ICC and recommended `K = ceil(9 * sigma_sample^2 / sigma_model^2)`. Feeds K back into the generator and confirms whether averaging seeds is needed before the bootstrap. |
| `scripts/hinglish/gen_eval_set.py` (or the existing manifest builder) | NEW / extend | Build the frozen golden held-out manifest of n >= 150 (target 320) balanced across the 4 voices and 8 cs-bins, oversampling code-switched bins so accent reaches >= 150 scorable English-bearing clips and `cs_high`/`numbers`/`tech` are well represented. Commit the `utt_id` list and margins in a frozen config before the candidate is trained. |
| `docs/EQUIVALENCE_VERIFICATION.md` | NEW | Mirror `docs/FP16_VERIFICATION.md`: pre-registered margins, the required-n power table, the multi-seed protocol, the Holm + BH FDR plan, the explicit INCONCLUSIVE rule, that the reference is the 400M model, and that gate-0 uses latent cosine >= 0.95 (not fp16 token identity). |

**Wiring reconciliation (do this before the first run).** Two entrypoints currently expect different
score-file names: `compare/certify.py` reads `recall_{sys}.json` / `accent_{sys}.json` / `obj_{sys}.json`
(one shared dir, system in the filename), while the standalone `12_equivalence_eval.py` reads
`recall.json` / `accent.json` / `obj.json` (one dir per system). Pick one convention when wiring the
scorers on the GPU box and make `08`/`09`/`10` write that name. `certify.py` is the single runner the
user invokes; `12_equivalence_eval.py` is the standalone equivalent of its Gate 2. Also note
`gen_panel_ckpt.py` derives `n_audio_tokens` as `len(wav)/256` (vocoder-frame count, a proxy for code
count); it is used only as a relative greedy-length agreement check, so the proxy is fine, but do not
read it as the literal DVAE code count.

### Report format

Two artifacts per certification, written to the candidate's `--out-dir`:

- **`certification.json`** machine-readable: `{tier, certified, param_gate{ref_gpt_M, cand_gpt_M,
  budget_M, reduction_x, gpt_under_budget, frozen_unchanged, pass}, quality{<axis>: {n, delta, margin,
  bca_lb, p_ni, power, min_n, verdict}}, content_gate{...}, efficiency_gate{speedup_x, ...}, margins,
  n_boot, boot_seed}`.
- **`certification.md`** one page: VERDICT line, then Gate 1 (parameters), Gate 2 (per-axis quality
  table with delta / margin / BCa LB / power / verdict), Gate 3 (content), Gate 4 (efficiency).

`12_equivalence_eval.py` additionally writes `equivalence_report.json` with the per-axis TOST detail,
and `13_variance_decomp.py` writes the ICC + recommended K.

---

## 7. How to run it, and how to read accept/reject

Pin the held-out manifest and refs first; then for **each** system (ref = 400M, cand = candidate) run
generation with the same `--eval-manifest` and `--seed-base`, score, count params, and certify.

```bash
# 0) GENERATE both panels (same manifest + same seed-base -> paired by utt_id).
#    Run BOTH the temp-0.7 panel (perceptual) and the --greedy panel (content-equivalence).
python scripts/hinglish/compare/gen_panel_ckpt.py --base <model_dir> --ckpt ref_400m.pth \
    --eval-manifest data/eval_big/heldout.jsonl --refs-dir <model_dir>/refs \
    --out-dir data/eval_400m/wav --label ref
python scripts/hinglish/compare/gen_panel_ckpt.py --base <model_dir> --ckpt ref_400m.pth \
    --eval-manifest data/eval_big/heldout.jsonl --refs-dir <model_dir>/refs \
    --out-dir data/eval_400m/wav_greedy --label ref --greedy
# ...same two commands for the candidate with --ckpt cand_200m.pth and --out-dir data/eval_200m/...

# 1) SCORE both panels with the pinned judges (writes recall_/accent_/obj_ json per system).
#    08 = intelligibility (Qwen3-ASR), 10 = accent (Whisper-en), 09 = UTMOS + SECS.

# 2) COUNT PARAMETERS for both (the actual goal; bytes do not gate).
python scripts/hinglish/compare/count_params.py --base <model_dir> --ckpt ref_400m.pth \
    --label ref_400m --out data/eval_400m/params.json
python scripts/hinglish/compare/count_params.py --base <model_dir> --ckpt cand_200m.pth \
    --label cand_200m --out data/eval_200m/params.json

# 3) (optional) VARIANCE DECOMP to confirm K, after the multi-seed score rows exist.
python scripts/hinglish/13_variance_decomp.py --dir data/eval_200m

# 4) DESCRIPTIVE means/CIs (reporting only, no decision).
python scripts/hinglish/11_aggregate_eval.py --dir data/eval_200m

# 5) CERTIFY (the accept/reject decision). Exits 0 = CERTIFIED, nonzero = NOT.
python scripts/hinglish/compare/certify.py --tier 200m \
    --ref-dir data/eval_400m --cand-dir data/eval_200m \
    --ref-params data/eval_400m/params.json --cand-params data/eval_200m/params.json \
    --greedy-ref data/eval_400m/wav_greedy/manifest.jsonl \
    --greedy-cand data/eval_200m/wav_greedy/manifest.jsonl \
    --out-dir data/eval_200m
# Repeat step 5 with --tier 100m and the 100M dirs for the second milestone.
```

**Reading the output.** The runner prints a per-axis table and a single VERDICT line. Per axis:

- **PASS** = BCa lower bound is above -margin at the Holm alpha, and the axis is adequately powered. No
  quality loss on that axis.
- **FAIL (degraded)** = the lower bound dropped below -margin. A real regression; reject.
- **INCONCLUSIVE (underpowered; add n)** = the axis cannot be certified at the current n (power at
  delta=-m/3 below 0.80, or n below the `MIN_N` floor). Do **not** read this as a pass; collect more
  held-out utterances (oversample the code-switched bins for accent) and re-run. This is the rule that
  blocks the false positive.

**Overall:** `CERTIFIED no quality loss` requires all four quality axes PASS under Holm **and** the
parameter gate (GPT under the tier budget, frozen stack unchanged) **and** the content gate (greedy
agreement >= 90%, latent cosine >= 0.95) **and** the efficiency gate (RTF speedup >= tier threshold).
Anything less prints `NOT CERTIFIED` and the script exits nonzero. A candidate that only shrank bytes
(quantization) fails the parameter gate immediately, because its `gpt_params` count is unchanged.
