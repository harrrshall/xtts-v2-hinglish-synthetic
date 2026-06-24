#!/usr/bin/env python3
"""Re-accept filter scores with BIN-AWARE thresholds (no re-running of qwen).

Why: qwen3-asr's recall ceiling drops with code-switch density even on KNOWN-GOOD audio
(Phase 0 verified: teacher TTS clips the owner confirmed natural scored cs_high recall ~0.77,
cs_med ~0.89, cs_none ~0.97). A single flat tau therefore over-rejects good dense-CS clips.
This re-derives accept/reject from the EXISTING filter_recall / filter_cer_roman using per-bin
floors anchored to those verified-good recognizer scores (a margin below them).

Duration rejects (too_short / too_long) are objective and preserved as hard rejects.
Recovered clips are tagged binaware_recovered=true so eval can A/B them.

Usage:
  python3 scripts/hinglish/rescore_binaware.py \
      --filter data/filtered/filter_scores.jsonl --corpus data/corpus/corpus.jsonl \
      --out data/filtered/filter_scores_binaware.jsonl
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

# recall floor / cer ceiling per cmi_bin, anchored to Phase-0 verified-good recognizer scores.
# Floors set a defensible margin (~0.07-0.10) below the Phase-0 verified-good recall ceiling
# per bin (none 0.97, low 1.00, med 0.89, high 0.77). Tight enough to cut degraded clips,
# lenient enough not to penalize a clip for the recognizer's own dense-CS ceiling.
BIN_TAU = {
    "none": (0.85, 0.12),
    "low":  (0.83, 0.14),
    "med":  (0.76, 0.20),
    "high": (0.70, 0.25),
    "?":    (0.75, 0.18),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="data/filtered/filter_scores.jsonl")
    ap.add_argument("--corpus", default="data/corpus/corpus.jsonl")
    ap.add_argument("--out", default="data/filtered/filter_scores_binaware.jsonl")
    args = ap.parse_args()

    corpus = {r["corpus_id"]: r for r in (json.loads(l) for l in open(args.corpus, encoding="utf-8"))}
    rows = [json.loads(l) for l in open(args.filter, encoding="utf-8")]

    binstat = defaultdict(lambda: [0, 0])  # total, accepted
    recovered = 0
    out = []
    for r in rows:
        cid = r["utt_id"].split("__")[0]
        cb = corpus.get(cid, {}).get("cmi_bin", "?")
        tau_r, tau_c = BIN_TAU.get(cb, BIN_TAU["?"])
        was_accept = bool(r.get("accept"))
        prev_reason = r.get("reject_reason")
        rec = r.get("filter_recall")
        cer = r.get("filter_cer_roman")
        # hard objective rejects stay rejected regardless of bin
        if prev_reason in ("too_short", "too_long", "asr_missing") or rec is None:
            accept, reason = False, (prev_reason or "asr_missing")
        else:
            ok = (rec >= tau_r) and (cer is None or cer <= tau_c)
            accept = ok
            reason = None if ok else ("recall_low" if rec < tau_r else "cer_high")
        if accept and not was_accept:
            recovered += 1
        nr = {**r, "accept": accept, "reject_reason": reason,
              "cmi_bin": cb, "tau_recall": tau_r, "tau_cer": tau_c,
              "binaware_recovered": bool(accept and not was_accept)}
        out.append(nr)
        binstat[cb][0] += 1
        binstat[cb][1] += 1 if accept else 0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(out); acc = sum(1 for r in out if r["accept"])
    print(f"BIN-AWARE re-accept -> {args.out}")
    print(f"  accepted {acc}/{total} = {100*acc/total:.1f}%  (recovered {recovered} previously-rejected)")
    print("  per-bin accept:")
    for cb in ["none", "low", "med", "high", "?"]:
        if cb in binstat:
            t, a = binstat[cb]
            print(f"    {cb:5s}: {a}/{t} = {100*a/t:.1f}%   (floors recall>={BIN_TAU[cb][0]}, cer<={BIN_TAU[cb][1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
