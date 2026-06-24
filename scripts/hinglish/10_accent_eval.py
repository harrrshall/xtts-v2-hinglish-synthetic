#!/usr/bin/env python3
"""Code-switch ACCENT metric: does the embedded English actually sound like English?

The Hindi-forced ASR transliterates English to Devanagari, so recall can't judge accent. Here we
run an ENGLISH ASR (whisper-large-v3, language='en') on each clip and measure how well the intended
English words are recovered IN ENGLISH ORTHOGRAPHY. Native-sounding English -> the English ASR
transcribes the words correctly. Hindi-accented English -> the English ASR mis-hears them -> low
English-recall. Compare student vs teacher (paired, same sentences/voices).

Intended English words = the Latin-script tokens in the reference text.

Run (needs whisper + transformers; use the qwen venv):
  CUDA_VISIBLE_DEVICES=6 <qwen-venv>/bin/python \
      scripts/hinglish/10_accent_eval.py --manifest data/student_eval/student_manifest.jsonl --label student
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
import soundfile as sf, librosa, torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

EN_TOK = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")
STOP = set("a an the to of in on at is are am be been being and or but so if it i you we he she they "
           "my your his her our their this that these those for with as it's i'm".split())


def english_words(text):
    return [w.lower().strip("'-") for w in text.split() if EN_TOK.match(w) and w.lower() not in STOP]


def fuzzy_in(word, hypwords):
    for h in hypwords:
        if h == word:
            return True
        # short edit-distance tolerance for ASR spelling
        if abs(len(h) - len(word)) <= 2:
            dp = list(range(len(h) + 1))
            for i, a in enumerate(word, 1):
                prev, dp[0] = dp[0], i
                for j, b in enumerate(h, 1):
                    cur = dp[j]; dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a != b)); prev = cur
            if dp[-1] <= max(1, len(word) // 4):
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    args = ap.parse_args()

    man = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    dev = "cuda"
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, torch_dtype=torch.float16).to(dev).eval()

    rows = []
    for m in man:
        wav, _ = librosa.load(m["wav"], sr=16000, mono=True)
        feat = proc(wav, sampling_rate=16000, return_tensors="pt").input_features.to(dev, torch.float16)
        with torch.no_grad():
            ids = model.generate(feat, language="en", task="transcribe", max_new_tokens=128)
        hyp = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        intended = english_words(m["ref_text"])
        hypw = [w.lower().strip(".,!?'\"-") for w in hyp.split()]
        if intended:
            rec = sum(1 for w in intended if fuzzy_in(w, hypw)) / len(intended)
        else:
            rec = None
        rows.append({**m, "en_hyp": hyp, "n_en_words": len(intended),
                     "en_recall": (round(rec, 3) if rec is not None else None),
                     "intended_en": intended})
        print(f"  {m['utt_id']:24s} en_recall={rec if rec is None else round(rec,2)}  [{','.join(intended[:6])}] -> {hyp[:55]}")

    vals = [r["en_recall"] for r in rows if r["en_recall"] is not None]
    bybin = defaultdict(list)
    for r in rows:
        if r["en_recall"] is not None:
            bybin[r["cs_mode"]].append(r["en_recall"])
    summary = {"label": args.label, "n_clips": len(rows), "n_scored": len(vals),
               "english_recall_mean": round(sum(vals) / len(vals), 3) if vals else None,
               "by_cs_mode": {b: round(sum(v) / len(v), 3) for b, v in bybin.items()},
               "rows": rows}
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[{args.label}] English-recall (accent proxy) mean={summary['english_recall_mean']} "
          f"over {len(vals)} clips; by bin={summary['by_cs_mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
