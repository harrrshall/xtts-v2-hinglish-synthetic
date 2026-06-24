#!/usr/bin/env python3
"""Inference with the fine-tuned XTTS-v2 Hinglish model across the fixed voices.

Loads the base XTTS config/vocab/speakers but the FINE-TUNED checkpoint, clones each fixed voice
from a reference clip, and synthesizes a held-out Hinglish test set. Output WAVs (24 kHz) go to
--out-dir for listening and for the qwen round-trip WER.

Run on the box (GPU pinned by caller):
  CUDA_VISIBLE_DEVICES=5 .venv_xtts/bin/python scripts/hinglish/07_xtts_infer.py
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import torch, soundfile as sf

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

BASE = Path(".tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2")


def find_ckpt(run_dir: Path) -> Path:
    cands = sorted(run_dir.glob("**/best_model*.pth"))
    if not cands:
        cands = sorted(run_dir.glob("**/*.pth"))
    if not cands:
        raise SystemExit(f"no checkpoint under {run_dir}")
    return cands[-1]


def pick_refs(synth_index, voices):
    """One clean reference clip per voice from the synth index (first match per voice)."""
    refs = {}
    for line in open(synth_index, encoding="utf-8"):
        r = json.loads(line)
        v = r.get("speaker_id") or (r["utt_id"].split("__")[1] if "__" in r.get("utt_id", "") else None)
        ap = r.get("audio_path")
        if v in voices and v not in refs and ap and Path(ap).exists():
            refs[v] = ap
    return refs


TEST = [
    ("cs_high", "यार ये नया phone का camera बिल्कुल insane है, low light में भी photos ekdam crisp आती हैं।"),
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
    ap.add_argument("--run-dir", default="runs/xtts_hinglish")
    ap.add_argument("--synth-index", default="data/synth/synth_index.jsonl")
    ap.add_argument("--out-dir", default="data/student_eval")
    ap.add_argument("--voices", nargs="+", default=["kaustubh", "arjun", "maya", "aadya"])
    ap.add_argument("--language", default="hi")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--texts-manifest", default=None,
                    help="jsonl {utt_id, ref_text, voice, cs_mode}; synth those exact (text,voice) pairs")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ckpt = find_ckpt(Path(args.run_dir))
    print(f"[infer] checkpoint: {ckpt}")

    config = XttsConfig(); config.load_json(str(BASE / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=str(ckpt), vocab_path=str(BASE / "vocab.json"),
                          use_deepspeed=False)
    model.cuda().eval()

    refs = pick_refs(args.synth_index, set(args.voices))
    print(f"[infer] refs: { {k: Path(v).name for k, v in refs.items()} }")

    # build the work list: either the built-in TEST x voices, or an external texts manifest
    if args.texts_manifest:
        items = [(json.loads(l)) for l in open(args.texts_manifest, encoding="utf-8") if l.strip()]
        work = [(it["utt_id"], it["ref_text"], it["voice"], it.get("cs_mode", "?")) for it in items]
    else:
        work = [(f"{tag}_{i}__{v}", text, v, tag) for v in args.voices for i, (tag, text) in enumerate(TEST)]

    cond_cache = {}
    manifest = []
    for uid, text, v, tag in work:
        if v not in refs:
            print(f"[infer] WARN no ref for {v}, skipping {uid}"); continue
        if v not in cond_cache:
            cond_cache[v] = model.get_conditioning_latents(audio_path=[refs[v]])
        gpt_cond, spk_emb = cond_cache[v]
        try:
            o = model.inference(text, args.language, gpt_cond, spk_emb, temperature=args.temperature)
        except Exception as e:
            print(f"  SKIP {uid}: {type(e).__name__} {str(e)[:60]}"); continue
        name = f"{uid}.wav"
        sf.write(str(out / name), o["wav"], 24000)
        manifest.append({"utt_id": uid, "wav": str(out / name),
                         "ref_text": text, "voice": v, "cs_mode": tag})
        print(f"  wrote {name} ({len(o['wav'])/24000:.1f}s)")
    (out / "student_manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest))
    print(f"[infer] wrote {len(manifest)} clips -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
