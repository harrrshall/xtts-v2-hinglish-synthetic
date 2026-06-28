#!/usr/bin/env python3
"""Score ONE full XTTS model (any gpt_layers) on a prompt set with the same greedy decode + RewardScorer
used by certify_student640, so the 400M/265M/sub-100M numbers are apples-to-apples on the same prompts."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.gen_student640 import stable_seed
from rl.reward import RewardScorer
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="dir with config.json + model.pth + vocab.json")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--refs-dir", default="runs/xtts_hinglish/RELEASE/refs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="model")
    args = ap.parse_args()
    assert torch.cuda.is_available(); dev = "cuda"
    md = Path(args.model_dir)

    cfg = XttsConfig(); cfg.load_json(str(md / "config.json"))
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_path=str(md / "model.pth"), vocab_path=str(md / "vocab.json"), use_deepspeed=False)
    model.eval(); model.cuda()
    gpt_p = sum(p.numel() for p in model.gpt.parameters())

    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    voices = sorted({r["voice"] for r in rows})
    cond = {v: model.get_conditioning_latents(audio_path=[str(Path(args.refs_dir) / f"{v}.wav")]) for v in voices}

    sc = RewardScorer(device=dev)
    for v in voices:
        sc.register_voice(v, str(Path(args.refs_dir) / f"{v}.wav"))

    A, U, S, n, tail = [], [], [], 0, 0
    for r in rows:
        v = r["voice"]; text = r.get("ref_text") or r.get("text")
        if any(c.isdigit() for c in text):
            continue
        gptc, spk = cond[v]
        torch.manual_seed(stable_seed(7, r["utt_id"]))
        try:
            o = model.inference(text, "hi", gptc, spk, temperature=0.7, enable_text_splitting=False,
                                repetition_penalty=1.3, do_sample=False, top_k=1, top_p=1.0)
        except Exception:
            continue
        wav = np.asarray(o["wav"], np.float32)
        c = sc.components(wav, 24000, text, v)
        if c["en_recall"] is not None:
            A.append(c["en_recall"])
        U.append(c["utmos"]); S.append(c["secs"] or 0.0)
        n += 1; tail += int(len(wav) / 24000 > 25 or c["silence_ratio"] > 0.6)
        if n % 25 == 0:
            print(f"  {n} done", flush=True)

    rep = {"label": args.label, "gpt_params_M": round(gpt_p / 1e6, 1), "n": n, "n_accent": len(A),
           "accent": float(np.mean(A)), "utmos": float(np.mean(U)), "secs": float(np.mean(S)),
           "tail_rate": tail / max(n, 1)}
    print(f"\n=== {args.label} ({rep['gpt_params_M']}M GPT, n={n}) ===")
    print(f"  accent={rep['accent']:.3f}  UTMOS={rep['utmos']:.3f}  SECS={rep['secs']:.3f}  tail={rep['tail_rate']:.1%}")
    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
