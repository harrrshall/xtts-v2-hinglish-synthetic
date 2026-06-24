#!/usr/bin/env python3
"""Industry-standard objective TTS metrics: UTMOS (naturalness) + speaker similarity (SECS).

UTMOS: neural MOS predictor (tarepan/SpeechMOS utmos22_strong), the standard programmatic
naturalness proxy in TTS papers. SECS: cosine of resemblyzer speaker embeddings between a clip and
its target-voice reference (voice-fidelity / cloning quality).

Run per set (student / teacher / real) and compare:
  CUDA_VISIBLE_DEVICES=6 .venv_eval/bin/python scripts/hinglish/09_objective_eval.py \
      --manifest data/student_eval/student_manifest.jsonl --label student \
      --voice-refs '{"kaustubh":"data/synth/wav/c549e8a39__kaustubh__sp10__tx__v0.wav", ...}' \
      --out data/student_eval/objmetrics_student.json
  # real set has no target voice; pass --clips-glob and omit --voice-refs
"""
from __future__ import annotations
import argparse, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np, librosa, torch
from resemblyzer import VoiceEncoder, preprocess_wav

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--clips-glob", default=None)
    ap.add_argument("--label", required=True)
    ap.add_argument("--voice-refs", default=None, help="JSON dict voice->ref wav")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.manifest:
        rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    else:
        rows = [{"utt_id": Path(p).stem, "wav": p, "voice": None, "cs_mode": "?"}
                for p in sorted(glob.glob(args.clips_glob))]
    if args.max:
        rows = rows[:args.max]

    utmos = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True).to(DEV).eval()
    enc = VoiceEncoder(device=DEV)

    refs = json.loads(args.voice_refs) if args.voice_refs else {}
    ref_emb = {v: enc.embed_utterance(preprocess_wav(p)) for v, p in refs.items() if Path(p).exists()}

    out = []
    for r in rows:
        wav, _ = librosa.load(r["wav"], sr=16000, mono=True)
        with torch.no_grad():
            mos = float(utmos(torch.from_numpy(wav).unsqueeze(0).to(DEV), 16000))
        secs = None
        v = r.get("voice")
        if v in ref_emb:
            emb = enc.embed_utterance(preprocess_wav(r["wav"]))
            secs = float(np.dot(emb, ref_emb[v]) / (np.linalg.norm(emb) * np.linalg.norm(ref_emb[v])))
        out.append({**r, "utmos": round(mos, 3), "secs": (round(secs, 3) if secs is not None else None)})

    def agg(vals):
        vals = [x for x in vals if x is not None]
        return round(sum(vals) / len(vals), 3) if vals else None
    mos_all = [r["utmos"] for r in out]
    secs_all = [r["secs"] for r in out]
    by_voice = defaultdict(lambda: {"utmos": [], "secs": []})
    for r in out:
        by_voice[r.get("voice")]["utmos"].append(r["utmos"])
        by_voice[r.get("voice")]["secs"].append(r["secs"])

    summary = {
        "label": args.label, "n": len(out),
        "utmos_mean": agg(mos_all), "utmos_min": round(min(mos_all), 3), "utmos_max": round(max(mos_all), 3),
        "secs_mean": agg(secs_all),
        "by_voice": {str(v): {"utmos": agg(d["utmos"]), "secs": agg(d["secs"])} for v, d in by_voice.items()},
        "rows": out,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[{args.label}] n={len(out)}  UTMOS mean={summary['utmos_mean']} "
          f"(min {summary['utmos_min']}, max {summary['utmos_max']})  SECS mean={summary['secs_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
