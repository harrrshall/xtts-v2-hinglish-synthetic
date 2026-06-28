#!/usr/bin/env python3
"""Benchmark Kokoro-82M on the SAME Hinglish prompts (accent + UTMOS; SECS N/A — Kokoro uses its own
fixed voice, not our refs). Shows how a generic 82M TTS handles code-switch vs our specialized 90M."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper
EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
from kokoro import KPipeline
from rl.reward import RewardScorer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice", default="hf_alpha")
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    if args.max:
        rows = rows[:args.max]

    pipe = KPipeline(lang_code="h", repo_id="hexgrad/Kokoro-82M")
    sc = RewardScorer(device=dev)

    A, U, n = [], [], 0
    for r in rows:
        text = r.get("ref_text") or r.get("text")
        if any(c.isdigit() for c in text):
            continue
        try:
            audio = np.concatenate([chunk.audio.cpu().numpy() for chunk in pipe(text, voice=args.voice)])
        except Exception:
            continue
        if audio.size < 2400:
            continue
        # Kokoro is 24 kHz
        c = sc.components(np.asarray(audio, np.float32), 24000, text, "_kokoro", wav_path=None)
        if c["en_recall"] is not None:
            A.append(c["en_recall"])
        U.append(c["utmos"]); n += 1
        if n % 25 == 0:
            print(f"  {n} done", flush=True)

    rep = {"label": "Kokoro-82M", "gpt_params_M": 82.0, "n": n, "n_accent": len(A),
           "accent": float(np.mean(A)) if A else None, "utmos": float(np.mean(U)) if U else None,
           "secs": None, "note": "own fixed voice (SECS N/A), not code-switch tuned"}
    print(f"\n=== Kokoro-82M (n={n}) ===  accent={rep['accent']}  UTMOS={rep['utmos']}  SECS=N/A")
    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
