# Hinglish synthetic-data TTS pipeline, end-to-end runbook

This is the operating manual for the pipeline under `scripts/hinglish/`. It documents the
exact run order, which stages run locally versus on GPU 5, where the teacher TTS key is read,
the convention-robust filter rationale, and the gaps the orchestrator must still close.

All commands assume the repo root as the working directory:
`/home/cybernovas/Desktop/2026/experiments/syntts`

## Stage map (what reads what, where it runs)

```
                         LOCAL (no GPU, no network)              GPU box (orchestrator)
S0  common.py            shared library, imported by all          -
S-calib 00_calibrate     eval_manifest + HYP_qwen3asr_deva  ->     (local; offline)
                         data/filtered/calib_report.json
S1  01_build_corpus      eval_manifest + prompts/*.jsonl    ->     (local; offline)
                         data/corpus/corpus.jsonl + corpus_stats.json
S2  02_synthesize        corpus.jsonl -> synth_plan.jsonl   ->     EXECUTE needs teacher TTS
                         + synth_index.jsonl + wav/<utt>.wav        (network, not GPU)
S3  03_filter_qwen       synth_index + corpus + WAVs        ->     REAL mode runs on GPU 5
                         data/filtered/filter_scores.jsonl
S4  04_assemble_manifest corpus + synth_index + filter      ->     (local; offline)
                         train_manifest.json + drop_report.json + cosyvoice2/
(train the student)      cosyvoice2/ export                 ->     GPU (outside this pipeline)
(student inference)      eval text -> student WAV -> qwen   ->     GPU 5 (orchestrator)
S5  05_eval_harness      eval_manifest + student hyp sidecar ->    (local; offline scoring)
                         data/eval/eval_report.json + eval_scores.jsonl + eval_hard_cs.jsonl
```

The chain is a linear sequence of single-purpose CLI scripts. Each reads a manifest and writes a
manifest. Every stage imports exactly one shared module, `common.py`, which owns every cross-stage
contract (schema, the filter, identity, diagnostics, the teacher TTS client). Rows are built only
through `common.new_row`, so the schema (`corpus_manifest_v1`) cannot drift between stages.

## Prerequisites (verified present this audit)

- Python 3 only for every local stage. No third-party packages are needed for S0, S1, S2 (dry-run),
  S4, S5, or S-calib. `indic_transliteration` and `datasketch` are used only if importable; the
  pure-Python fallbacks run otherwise.
- Data assets the pipeline reads (all present):
  - `data/spontaneous_hinglish/eval_spontaneous_combined_manifest.json` (1497 rows, frozen gold)
  - `data/spontaneous_hinglish/HYP_qwen3asr_deva.json` (1497 real qwen Devanagari pairs, the calibration asset)
  - `data/spontaneous_hinglish/ws4_anchor_manifest.json` (153-row anchor split, optional S4 input)
  - `data/teacher_test/qwen_roundtrip.json` (32-row Phase 0 teacher round-trip, the score-only proof)
- GPU box for S3 real mode and student inference:
  - `ssh <user>@<gpu-host>` (key path passed separately by the orchestrator; never embed or print it)
  - interpreter `<qwen-venv>/bin/python`
  - model `Qwen/Qwen3-ASR-1.7B`, `language="Hindi"` (Devanagari out)
  - `CUDA_VISIBLE_DEVICES=5`; the script uses `cuda:0` inside that masked view.

## Where TEACHER_TTS_API_KEY is read

The key is read in exactly one place: `common.synth_request`, via `os.environ.get("TEACHER_TTS_API_KEY")`.
It is never hardcoded, never logged, and never printed by any stage. Only S2 in `--execute` mode
touches it. A live S2 run without the key set fails inside `synth_request` with a clear message and
leaves the plan and partial index intact, so you set the key and resume.

Set it for the live synthesis run only:

```
export TEACHER_TTS_API_KEY=...        # in the orchestrator's environment, not committed
```

S2 `--dry-run`, `--plan-only`, and every other stage run with no key.

## Run order

### Step 0. Calibrate the filter FIRST (local, offline)

This is the mitigation for the number-one correctness trap. Run it before any clip is gated, because
the deterministic romanizer in `common.py` is the single point of failure for every accept/reject
decision, and its thresholds must come from the real known-good distribution, not intuition.

```
python3 scripts/hinglish/00_calibrate_filter.py
```

Writes `data/filtered/calib_report.json`. `common.load_config` and the no-config fallbacks in S3 and
S4 all read `tau_recall` and `tau_cer` from this exact path automatically when it exists. Do not move
the file. Observed this audit: mean recall 0.908, mean cer_roman 0.079, mean wer_raw 0.216, chosen
`tau_recall=0.750` and `tau_cer=0.183`, keeping 85.4 percent of the real known-good set.

Proof variant (no write), re-scores the Phase 0 teacher round-trip:

```
python3 scripts/hinglish/00_calibrate_filter.py --score-only
```

### Step 1. Build the text corpus (local, offline)

Supply LLM-generated transcripts as `scripts/hinglish/prompts/*.jsonl` (one JSON object per line with
a non-empty `text` field; optional `domain`, `source`, `ref_surface`). Then:

```
python3 scripts/hinglish/01_build_corpus.py --target-size 5000
```

Writes `data/corpus/corpus.jsonl` and `data/corpus/corpus_stats.json`. The stage seeds the
code-switch distribution from the real eval set, cleans and enforces per-word script, normalizes
numbers, tags languages, computes `cs_density`/`cmi_bin`, MinHash-dedups, balances by `cmi_bin`,
over-generates ~1.4x, and reports `token_entropy` versus the real baseline as the first erosion
checkpoint.

The `prompts/` directory is empty right now. With no prompt files you can prove the plumbing offline:

```
python3 scripts/hinglish/01_build_corpus.py --target-size 40 --synthesize-stub
```

`--synthesize-stub` mints placeholder code-switched transcripts. It is plumbing only, not a quality
text generator. A real run requires real prompt files (see Gaps).

### Step 2. Synthesize teacher audio

Two phases by design so spend is reviewable before money is spent.

Plan and review the spend first (local, no key):

```
python3 scripts/hinglish/02_synthesize.py --plan-only
```

Writes `data/synth/synth_plan.jsonl` and prints clip count, chunk API-call count, total characters,
and per-voice/per-speed breakdown. Chunks are the billed unit (one POST per <=250-char chunk).

Offline preview with silent WAVs so the rest of the chain can run without the API (local, no key):

```
python3 scripts/hinglish/02_synthesize.py --dry-run
```

Live synthesis against teacher TTS (network, not GPU; needs the key):

```
export TEACHER_TTS_API_KEY=...
python3 scripts/hinglish/02_synthesize.py --execute
```

Writes `data/synth/synth_index.jsonl` and `data/synth/wav/<utt_id>.wav` at 24 kHz mono. Idempotent:
utt_ids already in the index are skipped, so an interrupted run resumes without re-spending. Default
mode with no flag is `--dry-run`, so an accidental run never spends money.

### Step 3. Filter with Qwen3-ASR (GPU 5, orchestrator)

Local proof of the gate logic on real qwen output, no GPU:

```
python3 scripts/hinglish/03_filter_qwen.py --score-only
```

Offline chain test with a hypothesis sidecar (utt_id -> asr_hyp JSONL), no GPU:

```
python3 scripts/hinglish/03_filter_qwen.py --stub data/synth/hyp_sidecar.jsonl
```

Real run on the GPU box (the orchestrator runs this; it mirrors `scripts/qwen3asr_roundtrip.py`
loading exactly):

```
CUDA_VISIBLE_DEVICES=5 <qwen-venv>/bin/python \
    scripts/hinglish/03_filter_qwen.py \
    --synth-index data/synth/synth_index.jsonl \
    --corpus data/corpus/corpus.jsonl \
    --out data/filtered/filter_scores.jsonl
```

Writes `data/filtered/filter_scores.jsonl` with `asr_hyp`, `filter_recall` (primary), `filter_cer_roman`
(secondary), `filter_wer_raw` (diagnostic only), `accept`, `reject_reason`. Resumable by utt_id. The
WAV path in `synth_index.jsonl` is repo-relative; S3 resolves it against the repo root, so run from
the repo root on the box.

### Step 4. Assemble the training manifest (local, offline)

```
python3 scripts/hinglish/04_assemble_manifest.py
```

Joins corpus + synth_index + filter_scores on `corpus_id`/`utt_id`, applies the non-ASR gates
(duration in [3,30]s, sample_rate==24000, optional DNSMOS), keeps `accept==true` rows, enforces the
synthetic-reliance policy, splits train/dev_synth by `corpus_id` (no text leakage, seed 20260617), and
writes:

- `data/filtered/train_manifest.json` (JSON array, `corpus_manifest_v1`, drop-in for the ws4 loader)
- `data/filtered/drop_report.json` (per-gate drops, regen distribution, per-voice/per-cmi balance,
  token entropy and 4-gram repetition versus the real baseline = the second erosion checkpoint)
- `data/filtered/cosyvoice2/{train,dev}/{wav.scp,text,utt2spk,spk2utt,domain.scp,manifest.jsonl}`

With a real anchor manifest the true 0.5 synthetic cap is enforced:

```
python3 scripts/hinglish/04_assemble_manifest.py \
    --anchor-manifest data/spontaneous_hinglish/ws4_anchor_manifest.json
```

Note: the anchor is 16 kHz single-speaker, so blending it changes the sample-rate mix; treat anchor
blending as a deliberate experiment, not a default.

Fail-closed behavior to know about: a synth clip with no matching filter row (or a null
`filter_recall`) is rejected with `reject_reason="asr_missing"` rather than accepted, so a coverage
gap between S2 and S3 surfaces instead of leaking unfiltered audio. Every kept synthetic row is also
re-validated against the `filtered` schema profile as a backstop.

### Step 5. Train the student (GPU, outside this pipeline)

Train CosyVoice2 (or your chosen student) on `data/filtered/cosyvoice2/`. `text` maps utt_id ->
ref_orig (Devanagari/Latin training transcript), `utt2spk` uses the teacher voice as the speaker
label, `domain.scp` tags synthetic versus real so a trainer can weight or hold out by domain. This
step is not scripted here.

### Step 6. Evaluate the student (GPU for inference, local for scoring)

The orchestrator produces a hypothesis sidecar on the box: run the trained student on the eval text,
transcribe the student audio with Qwen3-ASR, and emit a `utt_id -> Devanagari hyp` file. Then score
locally:

```
python3 scripts/hinglish/05_eval_harness.py --student-hyp data/eval/student_hyp.jsonl
```

Writes `data/eval/eval_report.json`, `data/eval/eval_scores.jsonl`, and `data/eval/eval_hard_cs.jsonl`.
Scores use the same `common.py` metrics and the same `accept_clip` gate as S3, broken out per
`cmi_bin` (the high-CS tail is the headline number) and per `speaker_id`, with token entropy and
4-gram repetition on student outputs as the third erosion checkpoint. UTMOS and speaker similarity
stay stubbed with a `needs_gpu` note for the orchestrator to fill.

Two offline checks need no GPU and no student:

```
python3 scripts/hinglish/05_eval_harness.py --score-only   # real qwen pairs vs gold, metric proof
python3 scripts/hinglish/05_eval_harness.py --self-hyp     # gold vs itself, recall must be 1.0
```

## Local end-to-end smoke (no GPU, no key, no spend)

This is the sequence the audit ran to confirm the manifests actually chain. Use a scratch out-dir to
keep `data/` clean.

```
python3 scripts/hinglish/common.py                      # self-test: all green
python3 scripts/hinglish/02_synthesize.py --smoke-test  # synth self-test: all green
python3 scripts/hinglish/01_build_corpus.py --out-dir /tmp/ws/corpus --target-size 40 --synthesize-stub
python3 scripts/hinglish/02_synthesize.py --corpus /tmp/ws/corpus/corpus.jsonl --synth-dir /tmp/ws/synth --voices maya arjun --speeds 1.0 --dry-run
# build a sidecar (utt_id -> asr_hyp) from the synth index, then:
python3 scripts/hinglish/03_filter_qwen.py --synth-index /tmp/ws/synth/synth_index.jsonl --corpus /tmp/ws/corpus/corpus.jsonl --out /tmp/ws/filtered/filter_scores.jsonl --stub /tmp/ws/synth/hyp_sidecar.jsonl
python3 scripts/hinglish/04_assemble_manifest.py --corpus /tmp/ws/corpus/corpus.jsonl --synth-index /tmp/ws/synth/synth_index.jsonl --filter-scores /tmp/ws/filtered/filter_scores.jsonl --out-dir /tmp/ws/filtered
python3 scripts/hinglish/05_eval_harness.py --train-manifest /tmp/ws/filtered/train_manifest.json --out-dir /tmp/ws/eval --self-hyp
```

Dry-run note: dry-run WAVs are short silent clips (~`len(text)/15` seconds), so many will be rejected
as `too_short` by the [3,30]s duration gate at S3/S4. That is an artifact of silent dry-run audio, not
a logic bug; real teacher audio has real durations.

## The convention-robust filter rationale (why recall, not WER)

This is the project's number-one correctness trap and the reason the filter is shaped the way it is.

The teacher writes English in Latin (the intended text says "coaching center"). Qwen3-ASR runs with
`language="Hindi"` and emits Devanagari, so it transcribes the same English loanword phonetically as
Devanagari ("कोचिंग सेंटर"). A naive token WER between "coaching center" and "कोचिंग सेंटर" counts every word as a
substitution and reports a near-total error even though the audio is perfectly intelligible. The
Phase 0 verdict confirmed this, and an attempted romanized re-score made it worse. Raw token WER is
convention-confounded and is the wrong tool across scripts.

The fix lives only in `common.py` and both the train filter (S3) and the eval harness (S5) call it, so
a clip is judged by the identical rule in training and evaluation:

- `to_compare_space` maps any mixed string to one comparison space: romanize the Devanagari spans
  (deterministic ISO-15919-style, pure Python, falls back to a bundled map when `indic_transliteration`
  is absent), lowercase the Latin spans, strip punctuation and diacritics, NFC, collapse whitespace and
  spelling variants (v/w, doubled letters, long vowels, single-digit number words), delete the
  word-final schwa Hindi drops. Both the intended text and the ASR hypothesis pass through this
  identically, so a correct English word cannot be scored as a substitution just because the recognizer
  wrote it in Devanagari.
- `content_word_recall` is the PRIMARY accept signal: the fraction of intended content words recovered,
  stopwords dropped, English and switch-point words weighted higher (they are the hard part of
  Hinglish). A word counts as recovered on a fuzzy edit-ratio match plus a phonetic fold plus a
  consonant-skeleton match, which bridges English orthography against the recognizer's Devanagari
  spelling of a loanword.
- `cer_roman` is the SECONDARY signal: character error rate in compare-space, robust to segmentation
  noise.
- `wer_raw` is DIAGNOSTIC ONLY and never gates. It is kept in every report so the confound stays
  visible: high `wer_raw` next to high `recall` is the expected, correct signature.

`accept_clip` combines recall, cer, duration and sample rate into one `(accept, reason)` decision. Its
`tau_recall` and `tau_cer` come from the S-calib report, not from guessing. The calibration over the
1497 real pairs validates the scorer before any gate is trusted (the audit saw mean recall 0.908 while
mean `wer_raw` was 0.216, which is exactly the recovery the gate depends on).

## Schema and join keys (single source of truth)

- Schema is `corpus_manifest_v1` and never bumped. The 17 v1 fields stay byte-compatible with the ws4
  loader and the frozen eval set; all new fields are additive and optional.
- `corpus_id = "c" + sha1(normalized ref_orig)[:8]` groups every voice/speed/temp/regen variant of one
  transcript. It is the dedup group key and the leak-free train/dev split key.
- `utt_id = {corpus_id}__{speaker_id}__sp{speed_x10}__t{tier_or_x}__v{attempt}` is the per-clip id,
  deterministic so synth and filter skip done work on restart.
- `speaker_id` holds the teacher VOICE (kaustubh|arjun|maya|aadya) and becomes utt2spk; `teacher` holds
  the engine and is distinct.
- Join keys across stages: corpus joins to synth and filter by `corpus_id`; the per-clip filter verdict
  joins by `utt_id`; `sha256` is the audio dedup key.

## Erosion guardrails (three checkpoints)

With essentially no real Hinglish at 24 kHz, the synthetic-reliance cap degrades to a diversity-spread
target and the token-entropy and 4-gram-repetition alarms are the primary guardrail. They are computed
at three points: S1 (corpus build, versus the real baseline), S4 (kept manifest, deduped by corpus_id,
with an `erosion_alarm` flag when synth entropy drops below 0.85x real), and S5 (student outputs,
student-minus-reference deltas). Watch `token_entropy_ratio_vs_real` and the rep_4gram deltas across
all three.

## Gaps the orchestrator must close

1. Real prompt files. `scripts/hinglish/prompts/` is empty. A production S1 run needs real
   LLM-generated `*.jsonl` transcripts; `--synthesize-stub` only proves the plumbing and cannot reach a
   rich high-CS tail (S1 warns when the high-bin floor is not met). This is the single biggest input
   gap before any spend.
2. The GPU stages are written but never executed locally. The orchestrator runs S-calib confirmation,
   S3 real mode, student training, student inference, and the student-hyp sidecar generation on the box
   with the qwen venv. Confirm `Qwen/Qwen3-ASR-1.7B` and `soundfile`/`torch`/`qwen_asr` are importable
   there.
3. Student inference and the eval hypothesis sidecar are not scripted here. S5 consumes a
   `utt_id -> Devanagari hyp` file the orchestrator must produce (run the trained TTS on eval text, then
   transcribe). UTMOS and speaker-similarity are stubbed `needs_gpu`; wire a perceptual model on the box
   to fill them.
4. CosyVoice2 `wav.scp` paths follow the row's `audio_path`. When the WAVs live under the repo
   (`data/synth/wav/`), S2 stores repo-relative paths and the export carries them, so the trainer must
   run from the repo root, or the paths must be absolutized before training. WAVs outside the repo (e.g.
   a scratch dir) are exported as absolute paths.
5. The student trainer itself (Step 5) and the Phase-5 TDSC/DPO expressivity pass are out of scope for
   these scripts. The schema reserves `temp_tier` and `teacher` for the TDSC tiers and the optional
   CosyVoice2 second-teacher blend with no schema change; `run_synth` is the one place a second backend
   slots in (it dispatches on the row's `teacher` and currently raises NotImplementedError for anything
   but teacher TTS).
6. Naming: the architecture doc calls the shared module `hinglish_common.py`; the actual file is
   `common.py`. Every stage imports `common`, so this is consistent within the codebase, but anyone
   reading the spec should know the on-disk name.
