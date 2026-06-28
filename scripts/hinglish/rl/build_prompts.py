#!/usr/bin/env python3
"""Build an RFT prompt set from the training corpus: code-switch (English-bearing) rows, voice-balanced.

These are the prompts we roll out candidates for; accent matters most on code-switch text. Held-out eval
prompts are NOT used here (those are reserved for certification). Emits jsonl: utt_id, ref_text, voice, cs_mode.
"""
import argparse, csv, json, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import english_words  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/xtts/metadata_train.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500, help="total prompts")
    ap.add_argument("--min-en", type=int, default=2, help="min English words to count as code-switch")
    ap.add_argument("--seed", type=int, default=20260625)
    args = ap.parse_args()

    rows = []
    for r in csv.reader(open(args.metadata, encoding="utf-8"), delimiter="|"):
        if len(r) < 3 or not r[1].strip():
            continue
        ew = english_words(r[1])
        if len(ew) >= args.min_en:
            cs = "high" if len(ew) >= 4 else "med"
            rows.append({"ref_text": r[1], "voice": r[2], "cs_mode": cs, "n_en": len(ew)})

    # deterministic voice-balanced round-robin (no Math.random; stable by index)
    by_voice = defaultdict(list)
    for i, r in enumerate(rows):
        by_voice[r["voice"]].append(r)
    voices = sorted(by_voice)
    # stride each voice list so we don't take only the first contiguous block
    for v in voices:
        lst = by_voice[v]
        by_voice[v] = lst[:: max(1, len(lst) // max(1, args.n // len(voices) + 1))] or lst

    out, vi, idx = [], 0, defaultdict(int)
    while len(out) < args.n and any(idx[v] < len(by_voice[v]) for v in voices):
        v = voices[vi % len(voices)]; vi += 1
        if idx[v] < len(by_voice[v]):
            r = by_voice[v][idx[v]]; idx[v] += 1
            out.append({"utt_id": f"rft{len(out):05d}__{v}", **r})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out))
    from collections import Counter
    print(f"[build_prompts] {len(out)} prompts -> {args.out}  voices={Counter(o['voice'] for o in out)}  "
          f"cs={Counter(o['cs_mode'] for o in out)}")


if __name__ == "__main__":
    main()
