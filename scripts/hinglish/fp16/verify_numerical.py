#!/usr/bin/env python3
"""Numerical-equivalence proof: fp32 checkpoint vs fp16-upcast checkpoint.

Two deterministic tests (no sampling RNG involved), so any delta is ONLY fp16 weight rounding:
  (1) conditioning latents + speaker embedding from get_conditioning_latents() per voice ref
      -> cosine similarity + max-abs-diff. This is a pure forward pass (the encoder/perceiver).
  (2) GREEDY decode (top_k=1) of a fixed text per voice -> compare generated waveforms.
      With greedy decoding there is no sampling; identical weights => identical tokens => identical
      audio. Any divergence is the weight-rounding flipping a borderline argmax. We report
      token-length match and waveform SNR up to the common length.

Loads each model sequentially (peak RAM ~ one model) and keeps only the small result tensors.
"""
from __future__ import annotations
import argparse, gc, json
from pathlib import Path
import numpy as np, torch

import xtts_patch  # noqa: F401  (routes XTTS audio IO through soundfile)
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

BASE = None  # set in main


def load_model(ckpt_path):
    config = XttsConfig(); config.load_json(str(BASE / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=str(ckpt_path),
                          vocab_path=str(BASE / "vocab.json"), use_deepspeed=False)
    model.eval()
    return model


def run_one(ckpt_path, refs, texts):
    model = load_model(ckpt_path)
    res = {"latents": {}, "spk": {}, "wav": {}}
    for v, ref in refs.items():
        gpt_cond, spk = model.get_conditioning_latents(audio_path=[ref])
        res["latents"][v] = gpt_cond.detach().cpu().float().numpy()
        res["spk"][v] = spk.detach().cpu().float().numpy()
        # greedy (top_k=1) deterministic decode
        torch.manual_seed(1234)
        text = texts[v]
        try:
            o = model.inference(text, "hi", gpt_cond, spk,
                                temperature=0.7, top_k=1, top_p=1.0,
                                enable_text_splitting=False)
        except TypeError:
            torch.manual_seed(1234)
            o = model.inference(text, "hi", gpt_cond, spk, temperature=0.01)
        res["wav"][v] = np.asarray(o["wav"], dtype=np.float32)
    del model; gc.collect()
    return res


def cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def maxabs(a, b):
    return float(np.max(np.abs(a.ravel() - b.ravel())))


def snr_db(ref, test):
    n = min(len(ref), len(test))
    if n == 0:
        return None
    r = ref[:n].astype(np.float64); t = test[:n].astype(np.float64)
    noise = np.sum((r - t) ** 2)
    sig = np.sum(r ** 2)
    if noise <= 0:
        return float("inf")
    return float(10 * np.log10(sig / noise))


def main() -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="dir with config.json + vocab.json")
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--fp16", required=True)
    ap.add_argument("--refs-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    BASE = Path(args.base)

    voices = ["kaustubh", "arjun", "maya", "aadya"]
    refs = {v: str(Path(args.refs_dir) / f"{v}.wav") for v in voices}
    refs = {v: p for v, p in refs.items() if Path(p).exists()}
    texts = {v: t for v, t in zip(refs, [
        "यार ये नया phone का camera बिल्कुल insane है, low light में भी photos crisp आती हैं।",
        "Honestly बोलूँ तो उस meeting में जो presentation थी वो totally next level थी।",
        "मुझे कल office जल्दी पहुँचना है इसलिए alarm subah छह बजे का लगा देना।",
        "Please इस bug का fix deploy कर दो और मुझे pull request का link भेज देना।",
    ])}

    print("=== running fp32 ===");  r32 = run_one(args.fp32, refs, texts)
    print("=== running fp16 ===");  r16 = run_one(args.fp16, refs, texts)

    rows = []
    for v in refs:
        lat_cos = cos(r32["latents"][v], r16["latents"][v])
        lat_max = maxabs(r32["latents"][v], r16["latents"][v])
        spk_cos = cos(r32["spk"][v], r16["spk"][v])
        spk_max = maxabs(r32["spk"][v], r16["spk"][v])
        n32, n16 = len(r32["wav"][v]), len(r16["wav"][v])
        snr = snr_db(r32["wav"][v], r16["wav"][v])
        rows.append({"voice": v,
                     "latent_cosine": round(lat_cos, 8), "latent_maxabs": round(lat_max, 6),
                     "spk_cosine": round(spk_cos, 8), "spk_maxabs": round(spk_max, 6),
                     "greedy_len_fp32": n32, "greedy_len_fp16": n16,
                     "greedy_len_match": n32 == n16,
                     "greedy_snr_db": (round(snr, 2) if snr not in (None, float("inf")) else snr)})
        print(f"  {v:9s} lat_cos={lat_cos:.8f} lat_max={lat_max:.2e}  "
              f"spk_cos={spk_cos:.8f}  greedy len {n32} vs {n16} "
              f"({'==' if n32==n16 else '!='})  SNR={snr}")

    summary = {
        "latent_cosine_min": round(min(r["latent_cosine"] for r in rows), 8),
        "spk_cosine_min": round(min(r["spk_cosine"] for r in rows), 8),
        "greedy_all_len_match": all(r["greedy_len_match"] for r in rows),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nlatent cosine (min over voices): {summary['latent_cosine_min']}")
    print(f"speaker  cosine (min over voices): {summary['spk_cosine_min']}")
    print(f"greedy token-length identical for all voices: {summary['greedy_all_len_match']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
