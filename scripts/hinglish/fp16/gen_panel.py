#!/usr/bin/env python3
"""Generate a held-out audio panel with ONE checkpoint (fp32 or fp16), fixed per-item seed.

Same (text, voice) pairs + same seed are used for both checkpoints by the caller, so the RNG
stream is identical and the only difference between the two runs is the weight precision.
Writes wavs + a manifest (utt_id, wav, ref_text, voice, cs_mode) for scoring.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, soundfile as sf, torch

import xtts_patch  # noqa: F401  (routes XTTS audio IO through soundfile)
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

TEST = [
    ("cs_high", "यार ये नया phone का camera बिल्कुल insane है, low light में भी photos crisp आती हैं।"),
    ("cs_high", "Honestly बोलूँ तो उस meeting में जो presentation थी वो totally next level थी।"),
    ("cs_med",  "मुझे कल office जल्दी पहुँचना है इसलिए alarm subah छह बजे का लगा देना।"),
    ("cs_med",  "ये recipe try करना, इसमें थोड़ा सा butter और garlic डालने से taste double हो जाता है।"),
    ("tech",    "Please इस bug का fix deploy कर दो और मुझे pull request का link भेज देना।"),
    ("question","Wait, तुमने सच में वो concert के tickets book कर लिए?"),
    ("cs_none", "आज शाम को बारिश होने की पूरी संभावना है, छाता ज़रूर साथ रखना।"),
    ("emotion", "Oh my god ये तो बहुत amazing news है, मैं बहुत खुश हूँ तुम्हारे लिए!"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--refs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--voices", nargs="+", default=["kaustubh", "maya"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--greedy", action="store_true",
                    help="top_k=1 deterministic decode -> truly paired (isolates fp16 rounding)")
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    config = XttsConfig(); config.load_json(str(base / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=args.ckpt,
                          vocab_path=str(base / "vocab.json"), use_deepspeed=False)
    model.eval()

    cond = {}
    for v in args.voices:
        ref = Path(args.refs_dir) / f"{v}.wav"
        cond[v] = model.get_conditioning_latents(audio_path=[str(ref)])

    manifest = []
    for vi, v in enumerate(args.voices):
        gpt_cond, spk = cond[v]
        for i, (tag, text) in enumerate(TEST):
            uid = f"{tag}_{i}__{v}"
            # deterministic, process-independent seed (no hash()): identical for the SAME item
            # across both checkpoints, so the only difference is weight precision.
            torch.manual_seed(args.seed + i * 1000 + vi * 131)
            kw = dict(temperature=0.7, enable_text_splitting=False)
            if args.greedy:
                kw.update(top_k=1, top_p=1.0)  # deterministic -> truly paired content
            try:
                o = model.inference(text, "hi", gpt_cond, spk, **kw)
            except Exception as e:
                print(f"  SKIP {uid}: {type(e).__name__} {str(e)[:60]}"); continue
            wav = np.asarray(o["wav"], dtype=np.float32)
            sf.write(str(out / f"{uid}.wav"), wav, 24000)
            manifest.append({"utt_id": uid, "wav": str(out / f"{uid}.wav"),
                             "ref_text": text, "voice": v, "cs_mode": tag})
            print(f"  wrote {uid}.wav ({len(wav)/24000:.1f}s)")
    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest))
    print(f"[gen] {len(manifest)} clips -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
