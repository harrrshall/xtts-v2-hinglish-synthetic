#!/usr/bin/env python3
"""S5 eval harness for the Hinglish synthetic-data TTS pipeline.

Scores a student model against two sets, using exactly the metrics that the
train filter (S3) uses, so a clip judged good enough to keep is judged by the
same rule at evaluation time:

  1. The HARD spontaneous set: data/spontaneous_hinglish/
     eval_spontaneous_combined_manifest.json (1497 real rows, the frozen gold).
  2. The dev_synth split carried in data/filtered/train_manifest.json
     (rows whose partition is dev_synth), when that manifest exists.

For every scored row it computes content_word_recall (primary), cer_roman
(secondary), and wer_raw (diagnostic only, the convention-confounded number we
keep visible). Results are broken out per cmi_bin (the high code-mix tail is the
real test) and per speaker_id. It computes token_entropy and the mean
ngram_repetition over the student transcripts as Synthetic-Erosion collapse
alarms, comparing student entropy against the gold reference entropy on the same
set. A teacher-vs-student A/B hook scores an optional second hypothesis source
side by side.

It carves data/eval/eval_hard_cs.jsonl (rows where cmi_bin == high or
cs_density >= --cs-threshold) so the hard code-switch slice can be re-scored on
its own. It writes the per-row scores to data/eval/eval_scores.jsonl and the
aggregate report to data/eval/eval_report.json.

Student inference (running the trained TTS on the eval text, then transcribing
with Qwen3-ASR) and the perceptual metrics UTMOS / speaker similarity are GPU
work. They are not done here. This script consumes a hypothesis source that the
orchestrator produces on the GPU box:

  --student-hyp PATH   a sidecar mapping utt_id -> Devanagari ASR hypothesis,
                       accepted as a JSON list of rows, a JSON object
                       {utt_id: hyp}, or a JSONL of rows. Rows may name the
                       hypothesis field hyp, hyp_deva, asr_hyp, or text.

Two offline modes need no GPU and no student at all:

  --score-only        re-score the 1497 real qwen pairs in
                      data/spontaneous_hinglish/HYP_qwen3asr_deva.json against
                      the gold manifest with the convention-robust matcher. This
                      proves the eval metric logic on real recognizer output,
                      the same proof S3 runs against qwen_roundtrip.json. It is
                      the recommended smoke test.

  --self-hyp          score the gold reference against itself (perfect student),
                      a wiring sanity check that recall == 1.0 everywhere.

The UTMOS / SIM block in the report stays null with a "needs_gpu" note so the
orchestrator can fill it later without changing the schema of the report.

Run from the repo root. All metric logic is imported from common.py; this file
only wires inputs, grouping, and reporting.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Import the shared contracts. The script lives in scripts/hinglish/ next to
# common.py; add that directory to the path so it runs from the repo root.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import common  # noqa: E402

_REPO = _HERE.parent.parent
_DEFAULT_EVAL_MANIFEST = (
    _REPO / "data" / "spontaneous_hinglish"
    / "eval_spontaneous_combined_manifest.json"
)
_DEFAULT_QWEN_PAIRS = (
    _REPO / "data" / "spontaneous_hinglish" / "HYP_qwen3asr_deva.json"
)
_DEFAULT_TRAIN_MANIFEST = _REPO / "data" / "filtered" / "train_manifest.json"
_DEFAULT_OUT_DIR = _REPO / "data" / "eval"

# Fields the report keeps stubbed for the GPU stage. UTMOS is naturalness,
# sim is speaker similarity to the target voice. Both need a model on a GPU.
_GPU_METRIC_STUB = {
    "utmos_mean": None,
    "sim_mean": None,
    "note": "needs_gpu: run student inference + UTMOS/SIM on the GPU box",
}


# ----------------------------------------------------------------------------
# Hypothesis loading
# ----------------------------------------------------------------------------

def _extract_hyp(row: dict):
    """Pull the Devanagari hypothesis out of a row under any known field name.

    Different upstream tools name the field differently (qwen roundtrip uses
    hyp_deva, the manifest schema uses asr_hyp, ad-hoc dumps use hyp or text).
    We accept all of them so the harness does not care which producer ran.
    """
    for key in ("hyp_deva", "asr_hyp", "hyp", "text"):
        if key in row and row[key] is not None:
            return row[key]
    return None


def load_hyp_map(path: str) -> dict:
    """Load a utt_id -> hypothesis map from a list, object, or JSONL file.

    Returns a dict. A list of rows is keyed by each row's utt_id; a plain object
    is taken as the map directly; a JSONL is read line by line. Rows without a
    utt_id or without any hypothesis field are skipped.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("hypothesis file not found: %s" % path)
    raw = p.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    out = {}

    # A whole-file JSON object is the {utt_id: ...} map form. We must parse the
    # ENTIRE file as one object to distinguish it from a JSONL whose first line
    # also begins with a brace; a JSONL fails the whole-file parse and falls
    # through to the line-by-line reader.
    if stripped[:1] == "{":
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    h = _extract_hyp(v)
                    if h is not None:
                        out[k] = h
                elif isinstance(v, str):
                    out[k] = v
            return out

    # List (JSON array) or JSONL of rows; read_manifest sniffs which.
    rows = common.read_manifest(path)
    for r in rows:
        uid = r.get("utt_id")
        h = _extract_hyp(r)
        if uid and h is not None:
            out[uid] = h
    return out


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def score_pair(ref: str, hyp: str, lang_tags=None) -> dict:
    """Score one ref/hyp pair with the shared metrics.

    recall is primary, cer_roman secondary, wer_raw diagnostic only. Both sides
    pass through to_compare_space inside common, so an English word the
    recognizer wrote in Devanagari cannot count as a substitution.
    """
    return {
        "filter_recall": common.content_word_recall(ref, hyp, lang_tags),
        "filter_cer_roman": common.cer_roman(ref, hyp),
        "filter_wer_raw": common.wer_raw(ref, hyp),
    }


def build_scored_rows(eval_rows, hyp_map, cfg, source_label):
    """Score every eval row that has a hypothesis; return the scored rows.

    Each output row carries the join key, the grouping keys (cmi_bin,
    speaker_id), the three metrics, the accept verdict from the shared
    accept_clip (so eval applies the same gate as the filter), the per-utterance
    rep_4gram on the student transcript, and the texts for auditing. Rows with no
    matching hypothesis are returned separately as the missing list.
    """
    scored = []
    missing = []
    for row in eval_rows:
        uid = row.get("utt_id")
        ref = row.get("ref_orig") or ""
        hyp = hyp_map.get(uid)
        if hyp is None:
            missing.append(uid)
            continue
        metrics = score_pair(ref, hyp, row.get("lang_tags"))
        dur = row.get("duration_s")
        sr = cfg.get("sample_rate", 24000)
        # eval accept uses the same policy as the filter; duration/sr come from
        # the gold row (real audio), sr is assumed at the target rate for the
        # student. accept_clip needs asr_hyp present to run the empty check.
        scores_for_gate = dict(metrics)
        scores_for_gate["asr_hyp"] = hyp
        accept, reason = common.accept_clip(scores_for_gate, dur, sr, cfg)
        scored.append({
            "utt_id": uid,
            "source": source_label,
            "cmi_bin": row.get("cmi_bin"),
            "cs_density": row.get("cs_density"),
            "speaker_id": row.get("speaker_id"),
            "ref_orig": ref,
            "asr_hyp": hyp,
            "filter_recall": metrics["filter_recall"],
            "filter_cer_roman": metrics["filter_cer_roman"],
            "filter_wer_raw": metrics["filter_wer_raw"],
            "rep_4gram": common.ngram_repetition(hyp, n=4),
            "accept": accept,
            "reject_reason": reason,
        })
    return scored, missing


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def _mean(xs):
    """Mean of a list, or None when the list is empty."""
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _aggregate(scored):
    """Aggregate a list of scored rows into overall and grouped summaries.

    Reports recall/cer/wer means, the pass rate (fraction with accept True), and
    the rep_4gram mean overall, then broken out by cmi_bin and by speaker_id.
    The high cmi_bin slice is the headline number for Hinglish quality.
    """
    def block(rows):
        return {
            "n": len(rows),
            "recall_mean": _mean([r["filter_recall"] for r in rows]),
            "cer_roman_mean": _mean([r["filter_cer_roman"] for r in rows]),
            "wer_raw_mean_diagnostic": _mean(
                [r["filter_wer_raw"] for r in rows]),
            "pass_rate": (_mean([1.0 if r["accept"] else 0.0 for r in rows])
                          if rows else None),
            "rep_4gram_mean": _mean([r["rep_4gram"] for r in rows]),
        }

    by_cmi = defaultdict(list)
    by_spk = defaultdict(list)
    for r in scored:
        by_cmi[r.get("cmi_bin")].append(r)
        by_spk[r.get("speaker_id")].append(r)

    return {
        "overall": block(scored),
        "by_cmi_bin": {k: block(v) for k, v in sorted(
            by_cmi.items(), key=lambda kv: str(kv[0]))},
        "by_speaker_id": {k: block(v) for k, v in sorted(
            by_spk.items(), key=lambda kv: str(kv[0]))},
    }


def _collapse_alarms(scored, ref_rows):
    """Synthetic-Erosion alarms: student vs gold token entropy and repetition.

    A drop in student token entropy versus the gold reference entropy on the same
    set flags vocabulary narrowing; a rise in mean 4-gram repetition flags
    looping. We report both the absolute numbers and the student-minus-gold
    deltas so the orchestrator can alarm on a threshold.
    """
    student_texts = [r["asr_hyp"] for r in scored]
    ref_texts = [r["ref_orig"] for r in ref_rows]
    student_entropy = common.token_entropy(student_texts)
    ref_entropy = common.token_entropy(ref_texts)
    student_rep = _mean([r["rep_4gram"] for r in scored]) or 0.0
    ref_rep = _mean([common.ngram_repetition(t, n=4) for t in ref_texts]) or 0.0
    return {
        "student_token_entropy": student_entropy,
        "reference_token_entropy": ref_entropy,
        "entropy_delta_student_minus_ref": student_entropy - ref_entropy,
        "student_rep_4gram_mean": student_rep,
        "reference_rep_4gram_mean": ref_rep,
        "rep_4gram_delta_student_minus_ref": student_rep - ref_rep,
        "interpretation": (
            "negative entropy_delta = student narrowed vocabulary; "
            "positive rep_4gram_delta = student looping; both are "
            "Synthetic-Erosion (paper #3) collapse signals."
        ),
    }


# ----------------------------------------------------------------------------
# Set carving
# ----------------------------------------------------------------------------

def carve_hard_cs(eval_rows, cs_threshold: float):
    """Return the hard code-switch slice: cmi_bin high or cs_density >= thr."""
    out = []
    for row in eval_rows:
        cmi = row.get("cmi_bin")
        csd = row.get("cs_density")
        if cmi == "high" or (csd is not None and csd >= cs_threshold):
            out.append(row)
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def run(args) -> int:
    """Wire inputs, score, aggregate, and write the eval artifacts."""
    cfg = common.load_config(args.config) if args.config else {
        "sample_rate": 24000, "tau_recall": 0.80, "tau_cer": 0.30,
        "min_s": 3.0, "max_s": 30.0, "dnsmos_min": None, "cs_threshold": 0.4,
    }
    cs_threshold = (args.cs_threshold if args.cs_threshold is not None
                    else cfg.get("cs_threshold", 0.4))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = common.read_manifest(args.eval_manifest)
    if not eval_rows:
        print("ERROR: empty or missing eval manifest: %s" % args.eval_manifest)
        return 1
    print("loaded %d hard-set rows from %s"
          % (len(eval_rows), args.eval_manifest))

    # Carve and persist the hard code-switch slice for standalone re-scoring.
    hard_cs = carve_hard_cs(eval_rows, cs_threshold)
    hard_path = out_dir / "eval_hard_cs.jsonl"
    common.write_manifest(str(hard_path), hard_cs)
    print("carved %d hard code-switch rows (cmi_bin==high or cs_density>=%.2f)"
          " -> %s" % (len(hard_cs), cs_threshold, hard_path))

    # Resolve the hypothesis source for the hard set.
    if args.self_hyp:
        hyp_map = {r["utt_id"]: r.get("ref_orig") or "" for r in eval_rows}
        source_label = "self_hyp"
    elif args.score_only:
        pairs = common.read_manifest(args.qwen_pairs)
        hyp_map = {}
        for r in pairs:
            uid = r.get("utt_id")
            h = _extract_hyp(r)
            if uid and h is not None:
                hyp_map[uid] = h
        source_label = "qwen3asr_score_only"
        print("score-only: loaded %d real qwen pairs from %s"
              % (len(hyp_map), args.qwen_pairs))
    elif args.student_hyp:
        hyp_map = load_hyp_map(args.student_hyp)
        source_label = "student"
        print("loaded %d student hypotheses from %s"
              % (len(hyp_map), args.student_hyp))
    else:
        print("ERROR: no hypothesis source. Pass one of --student-hyp PATH, "
              "--score-only, or --self-hyp.")
        return 2

    scored, missing = build_scored_rows(eval_rows, hyp_map, cfg, source_label)
    print("scored %d hard-set rows (%d had no hypothesis and were skipped)"
          % (len(scored), len(missing)))
    if not scored:
        print("ERROR: nothing scored; check that hypothesis utt_ids match the "
              "eval manifest utt_ids.")
        return 3

    # rows that were actually scored, for entropy comparison against their gold
    scored_ids = {r["utt_id"] for r in scored}
    scored_ref_rows = [r for r in eval_rows if r["utt_id"] in scored_ids]

    report = {
        "eval_manifest": str(args.eval_manifest),
        "hypothesis_source": source_label,
        "cs_threshold": cs_threshold,
        "thresholds": {
            "tau_recall": cfg.get("tau_recall"),
            "tau_cer": cfg.get("tau_cer"),
            "min_s": cfg.get("min_s"),
            "max_s": cfg.get("max_s"),
        },
        "hard_set": _aggregate(scored),
        "hard_set_missing_hyp": len(missing),
        "hard_cs_slice": _aggregate(
            [r for r in scored if r["utt_id"] in
             {x["utt_id"] for x in hard_cs}]),
        "collapse_alarms": _collapse_alarms(scored, scored_ref_rows),
        "perceptual_metrics": dict(_GPU_METRIC_STUB),
    }

    all_scored = list(scored)

    # Optional dev_synth split from the train manifest, scored the same way.
    dev_rows = []
    if args.train_manifest and Path(args.train_manifest).exists():
        train = common.read_manifest(args.train_manifest)
        dev_rows = [r for r in train if r.get("partition") == "dev_synth"]
        if dev_rows and not args.self_hyp and not args.score_only:
            # dev_synth needs its own student hyps; reuse the same map by utt_id
            dev_scored, dev_missing = build_scored_rows(
                dev_rows, hyp_map, cfg, "dev_synth")
            if dev_scored:
                dev_ids = {r["utt_id"] for r in dev_scored}
                dev_ref = [r for r in dev_rows if r["utt_id"] in dev_ids]
                report["dev_synth"] = _aggregate(dev_scored)
                report["dev_synth_missing_hyp"] = len(dev_missing)
                report["dev_synth_collapse_alarms"] = _collapse_alarms(
                    dev_scored, dev_ref)
                all_scored.extend(dev_scored)
                print("scored %d dev_synth rows (%d missing hyp)"
                      % (len(dev_scored), len(dev_missing)))
        elif dev_rows:
            report["dev_synth"] = {
                "n": len(dev_rows),
                "note": "dev_synth present but skipped under "
                        "--score-only/--self-hyp (those use the qwen/gold "
                        "hyps, which are keyed to the hard set, not dev rows).",
            }
            print("dev_synth: %d rows present, skipped in this mode"
                  % len(dev_rows))

    # Teacher-vs-student A/B hook. Score a second hypothesis source over the
    # SAME utt_ids and report the per-cmi_bin delta so a regression shows up.
    if args.teacher_hyp:
        teacher_map = load_hyp_map(args.teacher_hyp)
        t_scored, t_missing = build_scored_rows(
            eval_rows, teacher_map, cfg, "teacher")
        if t_scored:
            t_agg = _aggregate(t_scored)
            s_agg = report["hard_set"]
            ab = {
                "teacher": t_agg,
                "student": s_agg,
                "student_minus_teacher": {
                    "overall_recall_delta": _delta(
                        s_agg["overall"]["recall_mean"],
                        t_agg["overall"]["recall_mean"]),
                    "by_cmi_bin_recall_delta": {
                        k: _delta(
                            s_agg["by_cmi_bin"].get(k, {}).get("recall_mean"),
                            t_agg["by_cmi_bin"].get(k, {}).get("recall_mean"))
                        for k in sorted(set(s_agg["by_cmi_bin"])
                                        | set(t_agg["by_cmi_bin"]))
                    },
                },
                "teacher_missing_hyp": len(t_missing),
            }
            report["ab_teacher_vs_student"] = ab
            print("A/B: scored %d teacher rows for comparison" % len(t_scored))

    # Write per-row scores (JSONL) and the aggregate report (JSON).
    scores_path = out_dir / "eval_scores.jsonl"
    common.write_manifest(str(scores_path), all_scored)
    report_path = out_dir / "eval_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Console summary so the smoke test is readable at a glance.
    _print_summary(report)
    print("\nwrote %s" % scores_path)
    print("wrote %s" % report_path)
    print("wrote %s" % hard_path)
    return 0


def _delta(a, b):
    """Difference a - b, or None when either is missing."""
    if a is None or b is None:
        return None
    return a - b


def _print_summary(report: dict) -> None:
    """Print the headline numbers for a human reading the smoke output."""
    hs = report["hard_set"]["overall"]
    print("\n=== hard set overall ===")
    print("  n=%d  recall=%s  cer_roman=%s  wer_raw(diag)=%s  pass_rate=%s"
          % (hs["n"], _fmt(hs["recall_mean"]), _fmt(hs["cer_roman_mean"]),
             _fmt(hs["wer_raw_mean_diagnostic"]), _fmt(hs["pass_rate"])))
    print("=== by cmi_bin (recall) ===")
    for k, b in report["hard_set"]["by_cmi_bin"].items():
        print("  %-5s n=%-4d recall=%s cer=%s wer_diag=%s"
              % (k, b["n"], _fmt(b["recall_mean"]),
                 _fmt(b["cer_roman_mean"]),
                 _fmt(b["wer_raw_mean_diagnostic"])))
    ca = report["collapse_alarms"]
    print("=== collapse alarms ===")
    print("  student_entropy=%s  ref_entropy=%s  delta=%s"
          % (_fmt(ca["student_token_entropy"]),
             _fmt(ca["reference_token_entropy"]),
             _fmt(ca["entropy_delta_student_minus_ref"])))
    print("  student_rep4=%s  ref_rep4=%s  delta=%s"
          % (_fmt(ca["student_rep_4gram_mean"]),
             _fmt(ca["reference_rep_4gram_mean"]),
             _fmt(ca["rep_4gram_delta_student_minus_ref"])))


def _fmt(x):
    """Format a float to 4 places, or the literal None."""
    return ("%.4f" % x) if isinstance(x, (int, float)) else str(x)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="S5 eval harness: score the student against the hard "
                    "spontaneous Hinglish set and the dev_synth split with the "
                    "shared convention-robust metrics.")
    p.add_argument("--eval-manifest", default=str(_DEFAULT_EVAL_MANIFEST),
                   help="the frozen hard gold set (default: the 1497-row "
                        "combined spontaneous manifest).")
    p.add_argument("--train-manifest", default=str(_DEFAULT_TRAIN_MANIFEST),
                   help="train manifest holding the dev_synth split (optional).")
    p.add_argument("--config", default=None,
                   help="experiment config; thresholds default to the "
                        "calibrated values when present.")
    p.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR),
                   help="where to write eval_report.json, eval_scores.jsonl, "
                        "eval_hard_cs.jsonl.")
    p.add_argument("--cs-threshold", type=float, default=None,
                   help="cs_density cutoff for the hard code-switch slice "
                        "(default from config, else 0.4).")

    src = p.add_argument_group("hypothesis source (pick one)")
    src.add_argument("--student-hyp", default=None,
                     help="utt_id -> Devanagari ASR hyp for the student "
                          "(list / object / JSONL).")
    src.add_argument("--score-only", action="store_true",
                     help="offline smoke: re-score the real qwen pairs in "
                          "HYP_qwen3asr_deva.json. No GPU, no student.")
    src.add_argument("--self-hyp", action="store_true",
                     help="offline wiring check: score gold against itself "
                          "(recall should be 1.0 everywhere).")

    p.add_argument("--qwen-pairs", default=str(_DEFAULT_QWEN_PAIRS),
                   help="real qwen pairs used by --score-only.")
    p.add_argument("--teacher-hyp", default=None,
                   help="optional second hyp source for the teacher-vs-student "
                        "A/B comparison.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
