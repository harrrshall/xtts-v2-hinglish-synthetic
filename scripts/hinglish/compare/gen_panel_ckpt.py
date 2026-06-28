#!/usr/bin/env python3
"""Generalizes fp16/gen_panel.py from "fp32-vs-fp16" to "any checkpoint", over the FULL held-out
manifest (not the 8-sentence smoke panel), with per-item paired seeds so a candidate and the 400M
reference draw the SAME sampling stream per utterance. Also records per-utterance wall-clock decode
time + audio duration so an RTF (real-time factor) efficiency gate can be computed downstream.

Pair contract: utt_id is identical across the ref and candidate runs (same text+voice+seed), so
11_aggregate_eval.py / 12_equivalence_eval.py pair by utt_id with zero changes.

Usage (run once per system, same --eval-manifest + same --seed-base):
  python gen_panel_ckpt.py --base <dir> --ckpt ref_400m.pth   --eval-manifest data/eval_big/heldout.jsonl \
      --refs-dir <dir>/refs --out-dir data/eval_400m/wav --label ref   [--greedy]
  python gen_panel_ckpt.py --base <dir> --ckpt cand_200m.pth  --eval-manifest data/eval_big/heldout.jsonl \
      --refs-dir <dir>/refs --out-dir data/eval_200m/wav --label cand  [--greedy]

--greedy (top_k=1) gives the controlled numerical-equivalence panel (isolates the architecture
change with identical token CONTENT, the analogue of the fp16 greedy-paired test). Default
temperature-0.7 is the real-usage panel. Run BOTH; the gates require greedy for the
content-equivalence check and sampled for the perceptual gate (see FP16_VERIFICATION.md).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, soundfile as sf, torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fp16"))
import xtts_patch  # noqa: F401  (routes XTTS audio IO through soundfile; harmless if not needed)
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def stable_seed(seed_base: int, utt_id: str) -> int:
    """Process-independent per-utt seed (NOT Python hash(): that is salted per process, which was
    the documented FP16 false-alarm). Identical utt_id -> identical seed across both runs."""
    h = 1469598103934665603
    for b in utt_id.encode("utf-8"):
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return (seed_base + (h & 0x7FFFFFFF)) & 0x7FFFFFFF


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-manifest", required=True,
                    help="held-out manifest: rows with utt_id, ref_text, voice, cs_mode")
    ap.add_argument("--refs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--seed-base", type=int, default=20260625)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--gpt-layers", type=int, default=0,
                    help="override config gpt_layers (e.g. 12 for the distilled student); 0 = use config")
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.eval_manifest, encoding="utf-8") if l.strip()]
    if args.max:
        rows = rows[:args.max]

    config = XttsConfig(); config.load_json(str(base / "config.json"))
    if args.gpt_layers:
        config.model_args.gpt_layers = args.gpt_layers
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=args.ckpt,
                          vocab_path=str(base / "vocab.json"), use_deepspeed=False)
    model.eval()

    cond_cache = {}
    def cond_for(v):
        if v not in cond_cache:
            cond_cache[v] = model.get_conditioning_latents(audio_path=[str(Path(args.refs_dir) / f"{v}.wav")])
        return cond_cache[v]

    use_cuda = torch.cuda.is_available()
    manifest = []
    for r in rows:
        uid, v, text, tag = r["utt_id"], r["voice"], r.get("ref_text") or r.get("text"), r.get("cs_mode", "?")
        gpt_cond, spk = cond_for(v)
        torch.manual_seed(stable_seed(args.seed_base, uid))
        kw = dict(temperature=args.temperature, enable_text_splitting=False)
        if args.greedy:
            kw.update(top_k=1, top_p=1.0)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            o = model.inference(text, "hi", gpt_cond, spk, **kw)
        except Exception as e:
            print(f"  SKIP {uid}: {type(e).__name__} {str(e)[:60]}"); continue
        if use_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        wav = np.asarray(o["wav"], dtype=np.float32)
        dur = len(wav) / 24000.0
        wpath = out / f"{uid}.wav"
        sf.write(str(wpath), wav, 24000)
        manifest.append({"utt_id": uid, "wav": str(wpath), "ref_text": text, "voice": v,
                         "cs_mode": tag, "decode_s": round(dt, 4), "audio_s": round(dur, 4),
                         "rtf": round(dt / dur, 4) if dur > 0 else None,
                         "n_audio_tokens": int(len(wav) / 256)})
    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest))
    rtfs = [m["rtf"] for m in manifest if m["rtf"]]
    med_rtf = float(np.median(rtfs)) if rtfs else None
    print(f"[gen:{args.label}] {len(manifest)} clips -> {out}  median RTF={med_rtf}")
    (out / "efficiency.json").write_text(json.dumps({
        "label": args.label, "n": len(manifest), "device": "cuda" if use_cuda else "cpu",
        "median_rtf": round(med_rtf, 4) if med_rtf else None,
        "mean_decode_s": round(float(np.mean([m["decode_s"] for m in manifest])), 4) if manifest else None,
        "greedy": bool(args.greedy)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
