#!/usr/bin/env python3
"""Score two panels (fp32 vs fp16) with UTMOS (naturalness) + SECS (voice fidelity), paired.

Reports each system's mean and the paired delta (fp16 - fp32) with a deterministic bootstrap
95% CI (same method as scripts/hinglish/11_aggregate_eval.py). A delta CI bracketing 0 means the
fp16 build is statistically indistinguishable from fp32 on that axis.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, librosa, torch
from resemblyzer import VoiceEncoder, preprocess_wav

DEV = "cpu"


def boot_ci(deltas, iters=20000, seed=0):
    # proper bootstrap via numpy (deterministic given seed); valid at any n, unlike the
    # LCG-modulo resampler whose low bits degenerate for small/power-of-two n.
    d = np.asarray(deltas, dtype=np.float64)
    n = len(d)
    if n == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, n, size=(iters, n))].mean(axis=1)
    return (round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4))


def score_manifest(rows, refs_dir, utmos, enc, ref_emb):
    out = {}
    for r in rows:
        wav, _ = librosa.load(r["wav"], sr=16000, mono=True)
        with torch.no_grad():
            mos = float(utmos(torch.from_numpy(wav).unsqueeze(0), 16000))
        v = r["voice"]
        emb = enc.embed_utterance(preprocess_wav(r["wav"]))
        secs = float(np.dot(emb, ref_emb[v]) / (np.linalg.norm(emb) * np.linalg.norm(ref_emb[v])))
        out[r["utt_id"]] = {"utmos": mos, "secs": secs, "voice": v, "cs_mode": r["cs_mode"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32-manifest", required=True)
    ap.add_argument("--fp16-manifest", required=True)
    ap.add_argument("--refs-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r32 = [json.loads(l) for l in open(args.fp32_manifest, encoding="utf-8") if l.strip()]
    r16 = [json.loads(l) for l in open(args.fp16_manifest, encoding="utf-8") if l.strip()]

    utmos = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True).to(DEV).eval()
    enc = VoiceEncoder(device=DEV)
    voices = sorted({r["voice"] for r in r32})
    ref_emb = {v: enc.embed_utterance(preprocess_wav(str(Path(args.refs_dir) / f"{v}.wav")))
               for v in voices}

    s32 = score_manifest(r32, args.refs_dir, utmos, enc, ref_emb)
    s16 = score_manifest(r16, args.refs_dir, utmos, enc, ref_emb)
    ids = [u for u in s32 if u in s16]

    report = {"n": len(ids), "metrics": {}}
    print(f"paired n={len(ids)}")
    print(f"{'metric':18s} {'fp32':>8} {'fp16':>8} {'delta':>9} {'95% CI (fp16-fp32)':>22}")
    for key in ("utmos", "secs"):
        a = np.array([s32[u][key] for u in ids])
        b = np.array([s16[u][key] for u in ids])
        deltas = list(b - a)
        ci = boot_ci(deltas)
        report["metrics"][key] = {"fp32_mean": round(float(a.mean()), 4),
                                  "fp16_mean": round(float(b.mean()), 4),
                                  "delta_mean": round(float(np.mean(deltas)), 4),
                                  "delta_max_abs": round(float(np.max(np.abs(deltas))), 4),
                                  "ci95": ci}
        print(f"{key:18s} {a.mean():8.4f} {b.mean():8.4f} {np.mean(deltas):+9.4f}   "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]")
    report["rows"] = {u: {"fp32": s32[u], "fp16": s16[u]} for u in ids}
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
