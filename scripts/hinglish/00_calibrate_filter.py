#!/usr/bin/env python3
"""Calibrate the convention-robust filter against real qwen3-asr pairs.

This is stage S-calib and the orchestrator runs it FIRST, before any clip is
gated. It is the mitigation for the project's number-one correctness trap: the
deterministic romanizer in common.py is the single point of failure for the
whole accept/reject decision, so its thresholds must be set against a known-good
distribution of real (gold transcript, recognizer output) pairs rather than
guessed.

What it does
------------
1. Joins the frozen 1497-row eval set (gold ref_orig, plus lang_tags and
   cmi_bin) with HYP_qwen3asr_deva.json (real qwen3-asr Devanagari output) by
   utt_id. These are real recordings, so a working filter should recover most of
   the intended content on most of them; the few low-recall pairs are genuine
   recognizer misses (heavy code-switch, numerals, named entities), which is
   exactly the tail the gate must learn to tolerate without opening the door to
   garbage.
2. Runs the shared scorer (content_word_recall primary, cer_roman secondary,
   wer_raw diagnostic only) on every pair via common.py. No metric logic is
   reimplemented here.
3. Sweeps tau_recall and tau_cer over the observed distribution and reports, for
   each candidate threshold, how much of the real known-good set would survive.
   It then picks tau_recall at a configurable lower percentile of the real
   recall distribution (keep the bulk of genuine clips) and tau_cer at an upper
   percentile of the real cer distribution, so accept_clip starts from data, not
   intuition.
4. Writes data/filtered/calib_report.json. load_config in common.py reads
   tau_recall and tau_cer from this file automatically when it exists, so every
   downstream stage inherits the calibrated thresholds.

It also keeps the convention confound visible: it reports the mean wer_raw over
the same pairs next to the recall it never gates on, so a reader can see that the
raw token WER is high while the content recall is high, which is the whole point.

Modes
-----
default      : the calibration join described above. Fully offline (reads two
               local JSON files). This is what the orchestrator runs.
--score-only : re-score the existing data/teacher_test/qwen_roundtrip.json
               (the Phase 0 teacher round-trip) with the convention-robust
               matcher, to prove the logic on real synthesized-then-recognized
               audio without any GPU or live call. Prints per-row and aggregate
               recall/cer next to the convention-confounded wer that the Phase 0
               report flagged. Does not write the calib report.

Run from the repo root:
  python3 scripts/hinglish/00_calibrate_filter.py
  python3 scripts/hinglish/00_calibrate_filter.py --score-only
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# Import the shared module. The file lives next to this script; we add the
# script directory to sys.path so "import common" works from the repo root.
import sys
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import common  # noqa: E402  (path set above)

_REPO = _HERE.parent.parent
_DEFAULT_EVAL = _REPO / "data" / "spontaneous_hinglish" / \
    "eval_spontaneous_combined_manifest.json"
_DEFAULT_HYP = _REPO / "data" / "spontaneous_hinglish" / \
    "HYP_qwen3asr_deva.json"
_DEFAULT_OUT = _REPO / "data" / "filtered" / "calib_report.json"
_DEFAULT_ROUNDTRIP = _REPO / "data" / "teacher_test" / "qwen_roundtrip.json"


# ----------------------------------------------------------------------------
# Scoring a single pair (uses only common.py metrics)
# ----------------------------------------------------------------------------

def _score_pair(ref: str, hyp: str, lang_tags=None) -> dict:
    """Score one (gold, recognizer) pair with the shared metrics.

    recall is the primary gate signal, cer_roman the secondary, wer_raw the
    diagnostic that keeps the convention confound visible. No thresholds applied
    here; this just produces the numbers the sweep consumes.
    """
    return {
        "recall": common.content_word_recall(ref, hyp, lang_tags),
        "cer_roman": common.cer_roman(ref, hyp),
        "wer_raw": common.wer_raw(ref, hyp),
    }


def _percentile(values, q: float) -> float:
    """Linear-interpolated percentile q in [0,100] over a value list.

    Stdlib only. Returns 0.0 for an empty list. Used to anchor the chosen
    thresholds to the real distribution instead of round numbers.
    """
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


def _dist_summary(values) -> dict:
    """Compact distribution summary (count, mean, median, key percentiles)."""
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "p05": round(_percentile(values, 5), 4),
        "p10": round(_percentile(values, 10), 4),
        "p25": round(_percentile(values, 25), 4),
        "p75": round(_percentile(values, 75), 4),
        "p90": round(_percentile(values, 90), 4),
        "p95": round(_percentile(values, 95), 4),
    }


# ----------------------------------------------------------------------------
# The calibration join and sweep
# ----------------------------------------------------------------------------

def _load_join(eval_path: Path, hyp_path: Path):
    """Join the eval gold set with the qwen hypotheses by utt_id.

    Returns (pairs, missing) where pairs is a list of dicts carrying utt_id,
    ref_orig (gold), hyp_deva, lang_tags and cmi_bin, and missing reports how
    many ids on each side had no partner. We score against ref_orig (the gold
    Devanagari/Latin mixed transcript) because that is what a real clip was
    supposed to say; ref_surface is the romanized gold and is kept only for
    auditing.
    """
    ev = common.read_manifest(str(eval_path))
    hyp = common.read_manifest(str(hyp_path))

    ev_by_id = {r["utt_id"]: r for r in ev if r.get("utt_id")}
    hyp_by_id = {r["utt_id"]: r for r in hyp if r.get("utt_id")}

    common_ids = sorted(set(ev_by_id) & set(hyp_by_id))
    pairs = []
    for uid in common_ids:
        e = ev_by_id[uid]
        h = hyp_by_id[uid]
        pairs.append({
            "utt_id": uid,
            "ref_orig": e.get("ref_orig", ""),
            "ref_surface": e.get("ref_surface"),
            "hyp_deva": h.get("hyp_deva", ""),
            "lang_tags": e.get("lang_tags"),
            "cmi_bin": e.get("cmi_bin", "unknown"),
        })
    missing = {
        "eval_only": len(set(ev_by_id) - set(hyp_by_id)),
        "hyp_only": len(set(hyp_by_id) - set(ev_by_id)),
        "joined": len(common_ids),
    }
    return pairs, missing


def _sweep(scored, tau_recall_grid, tau_cer_grid) -> list:
    """For each (tau_recall, tau_cer) report the keep rate on the real set.

    Every pair in the join is a genuine recording, so the keep rate here is the
    fraction of known-good clips a given threshold pair would NOT throw away.
    A good operating point keeps most of the real distribution (high keep rate)
    while still sitting above the obvious-garbage floor. The sweep is what lets a
    human see the trade rather than trust a single picked number.
    """
    recalls = [s["recall"] for s in scored]
    cers = [s["cer_roman"] for s in scored]
    n = len(scored)
    grid = []
    for tr in tau_recall_grid:
        for tc in tau_cer_grid:
            kept = sum(1 for r, c in zip(recalls, cers) if r >= tr and c <= tc)
            grid.append({
                "tau_recall": round(tr, 3),
                "tau_cer": round(tc, 3),
                "keep_rate": round(kept / n, 4) if n else 0.0,
                "kept": kept,
            })
    return grid


def _per_bin(scored) -> dict:
    """Recall/cer distribution broken out per cmi_bin.

    The high code-mixing tail is where the romanizer is most stressed and where
    a single global threshold is most likely to be wrong, so the report shows the
    per-bin recall so the orchestrator can sanity-check that the chosen
    tau_recall is not silently nuking the high-CS bin.
    """
    bins = {}
    for s in scored:
        bins.setdefault(s["cmi_bin"], []).append(s)
    out = {}
    for b, rows in sorted(bins.items()):
        out[b] = {
            "recall": _dist_summary([r["recall"] for r in rows]),
            "cer_roman": _dist_summary([r["cer_roman"] for r in rows]),
            "wer_raw_diagnostic": _dist_summary([r["wer_raw"] for r in rows]),
        }
    return out


def calibrate(eval_path: Path, hyp_path: Path, out_path: Path,
              recall_pctile: float, cer_pctile: float,
              recall_floor: float, cer_ceiling: float,
              dump_pairs: Path | None = None) -> dict:
    """Run the full calibration and write calib_report.json.

    tau_recall is anchored at the lower recall_pctile of the real recall
    distribution but never set below recall_floor (so a pathologically lenient
    set cannot drop the gate to nothing). tau_cer is anchored at the upper
    cer_pctile of the real cer distribution but never above cer_ceiling. Both
    clamps keep the calibrated gate inside a sane envelope while still letting the
    data move it.
    """
    pairs, missing = _load_join(eval_path, hyp_path)
    if not pairs:
        raise SystemExit(
            "no joined pairs; check eval and hyp paths and the utt_id keys")

    scored = []
    for p in pairs:
        sc = _score_pair(p["ref_orig"], p["hyp_deva"], p["lang_tags"])
        sc["utt_id"] = p["utt_id"]
        sc["cmi_bin"] = p["cmi_bin"]
        scored.append(sc)

    recalls = [s["recall"] for s in scored]
    cers = [s["cer_roman"] for s in scored]
    wers = [s["wer_raw"] for s in scored]

    # Anchor thresholds to the real distribution, then clamp to a sane envelope.
    tau_recall_raw = _percentile(recalls, recall_pctile)
    tau_cer_raw = _percentile(cers, cer_pctile)
    tau_recall = round(max(recall_floor, tau_recall_raw), 3)
    tau_cer = round(min(cer_ceiling, tau_cer_raw), 3)

    # Sweep a grid around the chosen point so the trade-off is visible.
    recall_grid = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    cer_grid = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    sweep = _sweep(scored, recall_grid, cer_grid)

    # Keep rate at the chosen operating point.
    kept_at_choice = sum(
        1 for s in scored
        if s["recall"] >= tau_recall and s["cer_roman"] <= tau_cer)
    keep_rate_choice = round(kept_at_choice / len(scored), 4)

    report = {
        "purpose": (
            "Thresholds for accept_clip, calibrated against real qwen3-asr "
            "output over the frozen 1497-row eval set. tau_recall/tau_cer are "
            "read by common.load_config when this file exists."),
        "inputs": {
            "eval_manifest": str(eval_path),
            "hyp_manifest": str(hyp_path),
        },
        "join": missing,
        "n_pairs": len(scored),
        "metric_distributions": {
            "recall_primary": _dist_summary(recalls),
            "cer_roman_secondary": _dist_summary(cers),
            "wer_raw_diagnostic": _dist_summary(wers),
        },
        "convention_confound_note": (
            "mean recall %.3f is high while mean wer_raw %.3f is high too: the "
            "raw token WER is convention-confounded and is never gated on. This "
            "side-by-side is the evidence the romanizer-based recall recovers "
            "content the raw WER appears to miss."
            % (statistics.fmean(recalls), statistics.fmean(wers))),
        "per_cmi_bin": _per_bin(scored),
        "threshold_selection": {
            "recall_percentile": recall_pctile,
            "cer_percentile": cer_pctile,
            "recall_floor": recall_floor,
            "cer_ceiling": cer_ceiling,
            "tau_recall_raw": round(tau_recall_raw, 4),
            "tau_cer_raw": round(tau_cer_raw, 4),
        },
        "sweep": sweep,
        # The two values downstream stages actually read:
        "tau_recall": tau_recall,
        "tau_cer": tau_cer,
        "keep_rate_at_chosen_thresholds": keep_rate_choice,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if dump_pairs is not None:
        common.write_manifest(str(dump_pairs), [
            {"utt_id": s["utt_id"], "cmi_bin": s["cmi_bin"],
             "recall": round(s["recall"], 4),
             "cer_roman": round(s["cer_roman"], 4),
             "wer_raw": round(s["wer_raw"], 4)}
            for s in scored])

    return report


# ----------------------------------------------------------------------------
# score-only mode over the Phase 0 teacher round-trip
# ----------------------------------------------------------------------------

def score_only(roundtrip_path: Path) -> dict:
    """Re-score the Phase 0 qwen_roundtrip.json with the convention-robust matcher.

    The Phase 0 report scored this set with raw token WER and saw it look bad on
    code-switched rows. Here we re-score the SAME rows (ref_text vs hyp) with
    content_word_recall and cer_roman to prove the convention-robust logic on
    real synthesized-then-recognized audio, with no GPU and no live call.
    Returns an aggregate summary and prints a per-row table.
    """
    data = json.loads(roundtrip_path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if not rows:
        raise SystemExit("qwen_roundtrip.json has no rows to score")

    print("score-only re-scoring of %s (%d rows)\n" % (roundtrip_path.name,
                                                       len(rows)))
    header = ("%-22s %-9s %7s %7s %7s" %
              ("id", "cs_mode", "recall", "cer", "wer*"))
    print(header)
    print("-" * len(header))

    scored = []
    for r in rows:
        ref = r.get("ref_text", "")
        hyp = r.get("hyp", "")
        sc = _score_pair(ref, hyp)
        sc.update({"id": r.get("id", ""), "cs_mode": r.get("cs_mode", "")})
        scored.append(sc)
        print("%-22s %-9s %7.3f %7.3f %7.3f" %
              (sc["id"][:22], sc["cs_mode"][:9], sc["recall"],
               sc["cer_roman"], sc["wer_raw"]))

    recalls = [s["recall"] for s in scored]
    cers = [s["cer_roman"] for s in scored]
    wers = [s["wer_raw"] for s in scored]

    # per cs_mode aggregate
    by_mode = {}
    for s in scored:
        by_mode.setdefault(s["cs_mode"], []).append(s)

    print("\nper cs_mode (mean recall / mean cer / mean wer*):")
    for mode, group in sorted(by_mode.items()):
        print("  %-9s recall=%.3f  cer=%.3f  wer*=%.3f" % (
            mode,
            statistics.fmean([g["recall"] for g in group]),
            statistics.fmean([g["cer_roman"] for g in group]),
            statistics.fmean([g["wer_raw"] for g in group])))

    summary = {
        "n_rows": len(scored),
        "mean_recall": round(statistics.fmean(recalls), 4),
        "mean_cer_roman": round(statistics.fmean(cers), 4),
        "mean_wer_raw_diagnostic": round(statistics.fmean(wers), 4),
    }
    print("\noverall: mean recall=%.3f  mean cer=%.3f  mean wer*=%.3f" % (
        summary["mean_recall"], summary["mean_cer_roman"],
        summary["mean_wer_raw_diagnostic"]))
    print("\n(* wer_raw is the convention-confounded diagnostic, never a gate. "
          "High wer next to high recall is the expected, correct signature.)")
    return summary


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate the convention-robust filter thresholds against "
                    "real qwen3-asr pairs, or re-score the Phase 0 round-trip.")
    ap.add_argument("--eval-manifest", default=str(_DEFAULT_EVAL),
                    help="gold eval set (JSON array). Default: the 1497-row set.")
    ap.add_argument("--hyp-manifest", default=str(_DEFAULT_HYP),
                    help="real qwen Devanagari output joinable by utt_id.")
    ap.add_argument("--out", default=str(_DEFAULT_OUT),
                    help="where to write calib_report.json.")
    ap.add_argument("--recall-percentile", type=float, default=10.0,
                    help="tau_recall is anchored at this lower percentile of "
                         "the real recall distribution (default 10).")
    ap.add_argument("--cer-percentile", type=float, default=90.0,
                    help="tau_cer is anchored at this upper percentile of the "
                         "real cer distribution (default 90).")
    ap.add_argument("--recall-floor", type=float, default=0.60,
                    help="tau_recall is never set below this clamp.")
    ap.add_argument("--cer-ceiling", type=float, default=0.45,
                    help="tau_cer is never set above this clamp.")
    ap.add_argument("--dump-pairs", default=None,
                    help="optional JSONL of per-pair scores for auditing.")
    ap.add_argument("--score-only", action="store_true",
                    help="re-score data/teacher_test/qwen_roundtrip.json with "
                         "the convention-robust matcher and exit (no GPU, no "
                         "write).")
    ap.add_argument("--roundtrip", default=str(_DEFAULT_ROUNDTRIP),
                    help="round-trip JSON for --score-only.")
    args = ap.parse_args(argv)

    if args.score_only:
        score_only(Path(args.roundtrip))
        return 0

    report = calibrate(
        Path(args.eval_manifest), Path(args.hyp_manifest), Path(args.out),
        recall_pctile=args.recall_percentile, cer_pctile=args.cer_percentile,
        recall_floor=args.recall_floor, cer_ceiling=args.cer_ceiling,
        dump_pairs=Path(args.dump_pairs) if args.dump_pairs else None)

    md = report["metric_distributions"]
    print("calibration over %d real pairs (join: %s)" % (
        report["n_pairs"], report["join"]))
    print("  recall   (primary)   : mean %.3f  median %.3f  p10 %.3f" % (
        md["recall_primary"]["mean"], md["recall_primary"]["median"],
        md["recall_primary"]["p10"]))
    print("  cer_roman(secondary) : mean %.3f  median %.3f  p90 %.3f" % (
        md["cer_roman_secondary"]["mean"], md["cer_roman_secondary"]["median"],
        md["cer_roman_secondary"]["p90"]))
    print("  wer_raw  (diagnostic): mean %.3f  (never gated)" % (
        md["wer_raw_diagnostic"]["mean"]))
    print("  CHOSEN tau_recall=%.3f  tau_cer=%.3f  -> keeps %.1f%% of the "
          "real known-good set" % (
              report["tau_recall"], report["tau_cer"],
              100.0 * report["keep_rate_at_chosen_thresholds"]))
    print("  per cmi_bin recall mean:")
    for b, d in report["per_cmi_bin"].items():
        print("    %-7s n=%-4d recall=%.3f cer=%.3f" % (
            b, d["recall"]["count"], d["recall"]["mean"], d["cer_roman"]["mean"]))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
