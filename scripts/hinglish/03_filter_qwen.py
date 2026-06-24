#!/usr/bin/env python3
"""S3 filter: score synthetic clips with Qwen3-ASR and the convention-robust gate.

This is the GPU stage of the Hinglish pipeline. The orchestrator runs the real
mode on the box; the local build only ever exercises the offline paths. Every
metric and the accept policy come from scripts/hinglish/common.py so the train
filter (here) and the eval harness (S5) judge a clip by exactly the same rule.

What it does
------------
Reads data/synth/synth_index.jsonl (one row per synthesized clip), joins the
intended transcript and its language tags from data/corpus/corpus.jsonl by
corpus_id, transcribes each WAV with Qwen3-ASR-1.7B (language=Hindi, Devanagari
out), and scores each clip in the shared compare-space:

  filter_recall     PRIMARY  content_word_recall, the accept gate
  filter_cer_roman  SECONDARY character error rate in compare-space
  filter_wer_raw    DIAGNOSTIC the convention-confounded token WER, never gates
  rep_4gram         per-row 4-gram repetition (Synthetic-Erosion alarm)

accept_clip combines recall, cer, duration and sample rate into one
(accept, reject_reason) decision. Output rows go to
data/filtered/filter_scores.jsonl, resumable by utt_id.

Three modes
-----------
real (default)  Loads Qwen3-ASR on cuda:0 (mirrors scripts/qwen3asr_roundtrip.py
                exactly) and transcribes each WAV. Needs a GPU. Run by the
                orchestrator on the box.

--stub PATH     No GPU. Reads a sidecar hypothesis JSONL (utt_id -> asr_hyp) and
                scores those instead of transcribing. Any clip with no sidecar
                hyp is written with accept=null and reject_reason="needs_asr" so
                the rest of the chain stays runnable end to end offline. If PATH
                is omitted every clip is marked needs_asr.

--score-only    No GPU and no synth/corpus inputs. Re-scores the real Phase 0
                round-trip file data/teacher_test/qwen_roundtrip.json with the
                convention-robust matcher and prints recall / cer / wer per clip.
                This proves the filter logic on real qwen Devanagari output and
                shows the WER confound collapsing under recall. This is the
                local smoke test for the gate.

Usage
-----
  # local proof on real data, no GPU
  python3 scripts/hinglish/03_filter_qwen.py --score-only

  # offline chain test with a hypothesis sidecar
  python3 scripts/hinglish/03_filter_qwen.py --stub data/synth/hyp_sidecar.jsonl

  # real run on the GPU box (orchestrator)
  CUDA_VISIBLE_DEVICES=5 <qwen-venv>/bin/python \
      scripts/hinglish/03_filter_qwen.py \
      --synth-index data/synth/synth_index.jsonl \
      --corpus data/corpus/corpus.jsonl \
      --out data/filtered/filter_scores.jsonl

Secrets and dependencies. No API key and no SSH key are touched here. torch,
soundfile and qwen_asr are imported lazily, only when the real transcriber is
needed, so the offline modes run on a plain Python 3 with no extra packages.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

# Import the shared module regardless of the entry directory. The file name
# starts with a digit, so this script is run, never imported, and we add its
# own directory to sys.path to pull in common.py.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import common  # noqa: E402  (path set above)

_REPO_ROOT = _HERE.parent.parent
_DEFAULT_SYNTH_INDEX = _REPO_ROOT / "data" / "synth" / "synth_index.jsonl"
_DEFAULT_CORPUS = _REPO_ROOT / "data" / "corpus" / "corpus.jsonl"
_DEFAULT_OUT = _REPO_ROOT / "data" / "filtered" / "filter_scores.jsonl"
_DEFAULT_ROUNDTRIP = (_REPO_ROOT / "data" / "teacher_test"
                      / "qwen_roundtrip.json")


# ----------------------------------------------------------------------------
# Scoring (shared by every mode)
# ----------------------------------------------------------------------------

def score_pair(ref_orig: str, asr_hyp: str, lang_tags=None) -> dict:
    """Score one (intended text, ASR hypothesis) pair in the compare-space.

    Returns the four metric fields plus rep_4gram. Both sides are mapped to the
    shared compare-space inside the metric functions, so an English word the
    recognizer wrote in Devanagari is not penalised as a substitution. recall is
    the gate, cer is secondary, wer_raw is the diagnostic confound, rep_4gram is
    the per-row erosion alarm computed on the intended text.
    """
    hyp = asr_hyp or ""
    return {
        "asr_hyp": asr_hyp,
        "filter_recall": round(
            common.content_word_recall(ref_orig, hyp, lang_tags), 4),
        "filter_cer_roman": round(common.cer_roman(ref_orig, hyp), 4),
        "filter_wer_raw": round(common.wer_raw(ref_orig, hyp), 4),
        "rep_4gram": round(common.ngram_repetition(ref_orig, n=4), 4),
    }


def _wav_meta(path: Path):
    """Return (duration_s, sample_rate) for a WAV, or (None, None) if unreadable.

    Stdlib wave only so the local stages keep needing no audio packages.
    """
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            sr = w.getframerate()
            dur = frames / float(sr) if sr else None
            return dur, sr
    except Exception:
        return None, None


def _decision_row(synth_row: dict, corpus_row: dict, scores: dict,
                  dur_s, sr, cfg: dict, asr_done: bool) -> dict:
    """Build the schema-valid filtered output row for one clip.

    Joins intended text and lang_tags from the corpus row, stamps the metric
    fields and the accept/reject decision. When asr_done is False (stub mode
    with no sidecar hyp) the row is written with accept=null and
    reject_reason="needs_asr" so the clip can be re-scored later without losing
    its place in the manifest.
    """
    # duration and sample rate fall back to whatever the synth row recorded.
    dur_s = dur_s if dur_s is not None else synth_row.get("duration_s")
    if sr is None:
        sr = cfg.get("sample_rate", 24000)

    if not asr_done:
        accept, reason = None, "needs_asr"
        score_fields = {
            "asr_hyp": None, "filter_recall": None,
            "filter_cer_roman": None, "filter_wer_raw": None,
            "rep_4gram": scores.get("rep_4gram"),
        }
    else:
        score_fields = scores
        accept, reason = common.accept_clip(
            {"filter_recall": score_fields["filter_recall"],
             "filter_cer_roman": score_fields["filter_cer_roman"],
             "asr_hyp": score_fields["asr_hyp"]},
            dur_s=dur_s, sr=sr, cfg=cfg)

    return common.new_row(
        utt_id=synth_row["utt_id"],
        audio_path=synth_row.get("audio_path"),
        ref_orig=corpus_row.get("ref_orig", synth_row.get("ref_orig")),
        ref_surface=corpus_row.get("ref_surface"),
        ref_iso15919=corpus_row.get("ref_iso15919"),
        cmi_bin=corpus_row.get("cmi_bin", synth_row.get("cmi_bin")),
        cs_density=corpus_row.get("cs_density", synth_row.get("cs_density")),
        lang_tags=corpus_row.get("lang_tags", synth_row.get("lang_tags")),
        speaker_id=synth_row.get("speaker_id"),
        duration_s=dur_s,
        sha256=synth_row.get("sha256"),
        dataset=synth_row.get("dataset", "synthetic_hinglish"),
        partition=synth_row.get("partition", "train"),
        is_synthetic=synth_row.get("is_synthetic", True),
        license=synth_row.get("license", "synthetic_teacher_tts"),
        flags=list(synth_row.get("flags") or []),
        corpus_id=synth_row.get("corpus_id"),
        speed=synth_row.get("speed"),
        temp_tier=synth_row.get("temp_tier"),
        teacher=synth_row.get("teacher", common.DEFAULT_TEACHER),
        chunks=synth_row.get("chunks"),
        asr_hyp=score_fields["asr_hyp"],
        filter_recall=score_fields["filter_recall"],
        filter_cer_roman=score_fields["filter_cer_roman"],
        filter_wer_raw=score_fields["filter_wer_raw"],
        rep_4gram=score_fields["rep_4gram"],
        accept=accept,
        reject_reason=reason,
        regen_attempt=synth_row.get("regen_attempt", 0),
    )


# ----------------------------------------------------------------------------
# Mode: score-only (prove the gate on real qwen output, no GPU)
# ----------------------------------------------------------------------------

def run_score_only(args) -> int:
    """Re-score data/teacher_test/qwen_roundtrip.json with the robust matcher.

    This is the local proof that the convention-robust filter recovers the
    English content qwen wrote in Devanagari, where raw WER cannot. Prints a per
    clip line and the accept rate at the configured thresholds, and writes a
    small report next to the round-trip file if --out points somewhere.
    """
    rt_path = Path(args.roundtrip)
    if not rt_path.exists():
        print("score-only: round-trip file not found: %s" % rt_path)
        return 1
    data = json.loads(rt_path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if not rows:
        print("score-only: no rows in %s" % rt_path)
        return 1

    cfg = _resolve_cfg(args)
    tau_r = cfg.get("tau_recall", 0.80)
    tau_c = cfg.get("tau_cer", 0.30)
    print("score-only over %d real qwen pairs (tau_recall=%.2f tau_cer=%.2f)"
          % (len(rows), tau_r, tau_c))
    print("  the point: recall recovers the English content that raw WER buries.")
    print()

    out_rows = []
    n_accept = 0
    sum_recall = sum_cer = sum_wer = 0.0
    by_mode = {}
    for r in rows:
        ref = r.get("ref_text", "")
        hyp = r.get("hyp", "")
        lang_tags = common.tag_languages(ref)
        sc = score_pair(ref, hyp, lang_tags)
        # accept on the metric gate only (these clips have no duration here).
        accept, reason = common.accept_clip(
            {"filter_recall": sc["filter_recall"],
             "filter_cer_roman": sc["filter_cer_roman"],
             "asr_hyp": sc["asr_hyp"]},
            dur_s=None, sr=24000, cfg=cfg)
        n_accept += 1 if accept else 0
        sum_recall += sc["filter_recall"]
        sum_cer += sc["filter_cer_roman"]
        sum_wer += sc["filter_wer_raw"]
        mode = r.get("cs_mode", "?")
        by_mode.setdefault(mode, []).append(sc["filter_recall"])
        verdict = "ACCEPT" if accept else ("REJECT:" + str(reason))
        print("  %-22s recall=%.2f cer=%.2f wer_raw=%.2f  %s"
              % (r.get("id", "?"), sc["filter_recall"],
                 sc["filter_cer_roman"], sc["filter_wer_raw"], verdict))
        out_rows.append({
            "id": r.get("id"), "cs_mode": mode, "voice": r.get("voice"),
            "ref_text": ref, "hyp": hyp,
            "filter_recall": sc["filter_recall"],
            "filter_cer_roman": sc["filter_cer_roman"],
            "filter_wer_raw": sc["filter_wer_raw"],
            "accept": accept, "reject_reason": reason,
        })

    n = len(rows)
    print()
    print("  mean recall=%.3f  mean cer_roman=%.3f  mean wer_raw=%.3f"
          % (sum_recall / n, sum_cer / n, sum_wer / n))
    print("  accept rate at gate: %d/%d = %.1f%%"
          % (n_accept, n, 100.0 * n_accept / n))
    print("  recall by cs_mode (higher is better even where WER is high):")
    for mode in sorted(by_mode):
        vals = by_mode[mode]
        print("    %-10s recall=%.3f (n=%d)"
              % (mode, sum(vals) / len(vals), len(vals)))

    if args.out_score_only:
        outp = Path(args.out_score_only)
        outp.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "source": str(rt_path),
            "tau_recall": tau_r, "tau_cer": tau_c,
            "mean_recall": round(sum_recall / n, 4),
            "mean_cer_roman": round(sum_cer / n, 4),
            "mean_wer_raw": round(sum_wer / n, 4),
            "accept_rate": round(n_accept / n, 4),
            "rows": out_rows,
        }
        outp.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print("\n  wrote %s" % outp)
    return 0


# ----------------------------------------------------------------------------
# Mode: filter (stub or real), the production path
# ----------------------------------------------------------------------------

def _load_corpus_by_id(corpus_path: Path) -> dict:
    """Map corpus_id -> corpus row so the filter can join the intended text."""
    by_id = {}
    for row in common.read_manifest(str(corpus_path)):
        cid = row.get("corpus_id")
        if cid:
            by_id[cid] = row
    return by_id


def _load_sidecar_hyps(path) -> dict:
    """Map utt_id -> asr_hyp from a sidecar JSONL or JSON array for --stub.

    Accepts rows shaped like {"utt_id":..., "asr_hyp":...} or {"utt_id":...,
    "hyp":...}. Missing file returns an empty map (every clip becomes needs_asr).
    """
    if not path:
        return {}
    hyps = {}
    for row in common.read_manifest(str(path)):
        uid = row.get("utt_id")
        if not uid:
            continue
        hyps[uid] = row.get("asr_hyp", row.get("hyp"))
    return hyps


def _make_transcriber(args):
    """Load Qwen3-ASR and return a callable wav_path -> Devanagari hypothesis.

    Mirrors scripts/qwen3asr_roundtrip.py exactly: Qwen3ASRModel.from_pretrained
    with bf16 on cuda:0, soundfile read, downmix to mono, language="Hindi".
    torch, soundfile and qwen_asr are imported here so the offline modes never
    need them.
    """
    import os
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=int(os.environ.get("QWEN_BATCH", "8")),
        max_new_tokens=256,
    )

    def transcribe(wav_path: str) -> str:
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return model.transcribe(audio=(audio, sr),
                                language="Hindi")[0].text.strip()

    return transcribe


def run_filter(args) -> int:
    """Score every synth clip and write data/filtered/filter_scores.jsonl.

    Resumable: utt_ids already in the output are skipped. In --stub mode the
    transcriber is the sidecar hyp map and any clip without a hyp is written as
    needs_asr. In real mode Qwen3-ASR is loaded once and run per clip.
    """
    cfg = _resolve_cfg(args)
    synth_path = Path(args.synth_index)
    corpus_path = Path(args.corpus)
    out_path = Path(args.out)

    synth_rows = common.read_manifest(str(synth_path))
    if not synth_rows:
        print("filter: no synth rows at %s (run S2 first, or use --score-only)"
              % synth_path)
        return 1
    corpus_by_id = _load_corpus_by_id(corpus_path)
    if not corpus_by_id:
        print("filter: warning, no corpus rows at %s; joining on synth row text"
              % corpus_path)

    done = common.resume_done_ids(str(out_path))
    if done:
        print("filter: resuming, %d clip(s) already scored" % len(done))

    todo = [r for r in synth_rows if r.get("utt_id") not in done]
    print("filter: %d clip(s) to score (%d total, %d done)"
          % (len(todo), len(synth_rows), len(done)))
    if not todo:
        print("filter: nothing to do.")
        return 0

    sidecar = {}
    transcribe = None
    if args.stub is not None:
        sidecar = _load_sidecar_hyps(args.stub)
        print("filter: STUB mode, %d sidecar hypotheses loaded" % len(sidecar))
    else:
        print("filter: REAL mode, loading Qwen3-ASR on cuda:0 ...")
        transcribe = _make_transcriber(args)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_accept = n_needs = 0
    batch = []
    for sr_row in todo:
        uid = sr_row["utt_id"]
        cid = sr_row.get("corpus_id")
        corpus_row = corpus_by_id.get(cid, sr_row)
        ref_orig = corpus_row.get("ref_orig", sr_row.get("ref_orig", ""))
        lang_tags = corpus_row.get("lang_tags", sr_row.get("lang_tags"))

        wav_path = sr_row.get("audio_path")
        dur_s = sr = None
        if wav_path:
            wp = Path(wav_path)
            if not wp.is_absolute():
                wp = _REPO_ROOT / wp
            if wp.exists():
                dur_s, sr = _wav_meta(wp)

        # decide the hypothesis source
        asr_hyp = None
        asr_done = False
        if transcribe is not None:
            wp = Path(wav_path) if wav_path else None
            if wp and not wp.is_absolute():
                wp = _REPO_ROOT / wp
            if wp and wp.exists():
                try:
                    asr_hyp = transcribe(str(wp))
                    asr_done = True
                except Exception as e:
                    print("  %-40s TRANSCRIBE ERROR %r" % (uid, e))
                    asr_hyp, asr_done = None, False
            else:
                print("  %-40s WAV MISSING %s" % (uid, wav_path))
        else:
            if uid in sidecar and sidecar[uid] is not None:
                asr_hyp = sidecar[uid]
                asr_done = True

        if asr_done:
            scores = score_pair(ref_orig, asr_hyp, lang_tags)
        else:
            scores = {"rep_4gram": round(
                common.ngram_repetition(ref_orig, n=4), 4)}

        row = _decision_row(sr_row, corpus_row, scores, dur_s, sr, cfg,
                            asr_done)
        problems = common.validate_row(row, profile="filtered")
        # needs_asr rows legitimately have null filter_recall/accept; allow them.
        if asr_done and problems:
            print("  %-40s SCHEMA PROBLEMS %s" % (uid, problems))
        batch.append(row)
        n_written += 1
        if row.get("accept") is True:
            n_accept += 1
        if row.get("reject_reason") == "needs_asr":
            n_needs += 1

        if asr_done:
            print("  %-40s recall=%.2f cer=%.2f wer_raw=%.2f  %s"
                  % (uid, scores["filter_recall"], scores["filter_cer_roman"],
                     scores["filter_wer_raw"],
                     "ACCEPT" if row["accept"] else
                     ("REJECT:" + str(row["reject_reason"]))))

        if len(batch) >= 50:
            common.write_manifest(str(out_path), batch, append=True)
            batch = []
    if batch:
        common.write_manifest(str(out_path), batch, append=True)

    print()
    print("filter: wrote %d row(s) -> %s" % (n_written, out_path))
    print("filter: accepted=%d  needs_asr=%d  rejected=%d"
          % (n_accept, n_needs, n_written - n_accept - n_needs))
    return 0


# ----------------------------------------------------------------------------
# Config resolution and CLI
# ----------------------------------------------------------------------------

def _resolve_cfg(args) -> dict:
    """Load the experiment config if given, else assemble defaults.

    When --config is passed, load_config validates it and pulls calibrated
    thresholds from data/filtered/calib_report.json when that exists. Without a
    config we build a minimal dict that load_config-style defaults plus any CLI
    overrides, so the gate still has tau_recall / tau_cer / duration bounds.
    """
    if args.config:
        cfg = common.load_config(args.config)
    else:
        cfg = {
            "tau_recall": 0.80, "tau_cer": 0.30,
            "min_s": 3.0, "max_s": 30.0, "dnsmos_min": None,
            "sample_rate": 24000,
        }
        # prefer calibrated thresholds when the report exists
        calib = _REPO_ROOT / "data" / "filtered" / "calib_report.json"
        if calib.exists():
            try:
                c = json.loads(calib.read_text(encoding="utf-8"))
                if "tau_recall" in c:
                    cfg["tau_recall"] = c["tau_recall"]
                if "tau_cer" in c:
                    cfg["tau_cer"] = c["tau_cer"]
            except Exception:
                pass
    # explicit CLI overrides win over both
    if args.tau_recall is not None:
        cfg["tau_recall"] = args.tau_recall
    if args.tau_cer is not None:
        cfg["tau_cer"] = args.tau_cer
    if args.min_s is not None:
        cfg["min_s"] = args.min_s
    if args.max_s is not None:
        cfg["max_s"] = args.max_s
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="S3 Qwen3-ASR filter for the Hinglish synthetic pipeline.")
    p.add_argument("--synth-index", default=str(_DEFAULT_SYNTH_INDEX),
                   help="synth_index.jsonl from S2 (default: %(default)s)")
    p.add_argument("--corpus", default=str(_DEFAULT_CORPUS),
                   help="corpus.jsonl from S1 for the join (default: %(default)s)")
    p.add_argument("--out", default=str(_DEFAULT_OUT),
                   help="output filter_scores.jsonl (default: %(default)s)")
    p.add_argument("--config", default=None,
                   help="experiment config json (calibrated thresholds)")

    p.add_argument("--stub", nargs="?", const="", default=None,
                   metavar="SIDECAR",
                   help="offline mode: read utt_id->asr_hyp from this JSONL; "
                        "clips with no hyp are marked needs_asr. No GPU.")
    p.add_argument("--score-only", action="store_true",
                   help="re-score data/teacher_test/qwen_roundtrip.json with "
                        "the robust matcher and exit. No GPU, the gate proof.")
    p.add_argument("--roundtrip", default=str(_DEFAULT_ROUNDTRIP),
                   help="round-trip file for --score-only (default: %(default)s)")
    p.add_argument("--out-score-only", default=None,
                   help="optional report path for --score-only output")

    p.add_argument("--tau-recall", dest="tau_recall", type=float, default=None,
                   help="override recall accept threshold")
    p.add_argument("--tau-cer", dest="tau_cer", type=float, default=None,
                   help="override cer reject threshold")
    p.add_argument("--min-s", dest="min_s", type=float, default=None,
                   help="override minimum duration seconds")
    p.add_argument("--max-s", dest="max_s", type=float, default=None,
                   help="override maximum duration seconds")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # argparse stores --score-only as score_only
    if getattr(args, "score_only", False):
        return run_score_only(args)
    return run_filter(args)


if __name__ == "__main__":
    raise SystemExit(main())
