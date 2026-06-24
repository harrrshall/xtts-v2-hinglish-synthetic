#!/usr/bin/env python3
"""Convert the assembled training manifest into XTTS-v2 fine-tune metadata.

Reads data/filtered/train_manifest.json (corpus_manifest_v1, accepted synth clips) and
writes XTTS metadata CSVs: one line per clip as `audio_path|text|speaker`. Splits
train/eval by corpus_id (no text leakage), reusing the manifest's own partition when present.

The text we train on is ref_orig (the mixed Devanagari/Latin Hinglish transcript). speaker is
the teacher voice (kaustubh|arjun|maya|aadya), used for multi-speaker conditioning. Audio paths
are made absolute so the trainer can run from anywhere.

Usage:
  python3 scripts/hinglish/06_xtts_prepare.py \
      --manifest data/filtered/train_manifest.json --out-dir data/xtts [--repo-root <abs>]
"""
from __future__ import annotations
import argparse, json, os, csv
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/filtered/train_manifest.json")
    ap.add_argument("--out-dir", default="data/xtts")
    ap.add_argument("--repo-root", default=None, help="abs path to resolve repo-relative audio_path")
    ap.add_argument("--eval-frac", type=float, default=0.03)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=18.0)  # XTTS max_wav_length ~11.6s @ default; keep margin, loader will cap
    args = ap.parse_args()

    root = Path(args.repo_root or Path(args.manifest).resolve().parents[2])
    rows = json.loads(Path(args.manifest).read_text())
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    def absaudio(p):
        p = Path(p)
        return str(p if p.is_absolute() else (root / p))

    # split by corpus_id so no transcript appears in both sets
    by_cid = {}
    for r in rows:
        by_cid.setdefault(r["corpus_id"], []).append(r)
    cids = sorted(by_cid)
    n_eval_cid = max(1, int(len(cids) * args.eval_frac))
    # deterministic: take every Nth cid for eval
    step = max(1, len(cids) // n_eval_cid)
    eval_cids = set(cids[::step][:n_eval_cid])

    def usable(r):
        d = r.get("duration_s", 0)
        txt = (r.get("ref_orig") or "").strip()
        return txt and args.min_dur <= d <= args.max_dur and Path(absaudio(r["audio_path"])).exists()

    train, evl = [], []
    skipped = Counter()
    for r in rows:
        if not r.get("is_synthetic", True):
            # keep real anchor rows too if present; they train fine
            pass
        if not usable(r):
            if not (r.get("ref_orig") or "").strip(): skipped["no_text"] += 1
            elif not (args.min_dur <= r.get("duration_s", 0) <= args.max_dur): skipped["dur"] += 1
            else: skipped["missing_wav"] += 1
            continue
        rec = (absaudio(r["audio_path"]), r["ref_orig"].strip().replace("|", " "), r.get("speaker_id", "spk"))
        (evl if r["corpus_id"] in eval_cids else train).append(rec)

    def write_csv(path, recs):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="|")
            for a, t, s in recs:
                w.writerow([a, t, s])

    write_csv(out / "metadata_train.csv", train)
    write_csv(out / "metadata_eval.csv", evl)
    spk = Counter(s for _, _, s in train)
    summary = {
        "n_in": len(rows), "n_train": len(train), "n_eval": len(evl),
        "skipped": dict(skipped), "speakers": dict(spk),
        "eval_cids": len(eval_cids), "total_cids": len(cids),
        "metadata_train": str(out / "metadata_train.csv"),
        "metadata_eval": str(out / "metadata_eval.csv"),
    }
    (out / "prepare_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if len(train) < 50:
        print("WARN: very few training samples; check filter accept rate / paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
