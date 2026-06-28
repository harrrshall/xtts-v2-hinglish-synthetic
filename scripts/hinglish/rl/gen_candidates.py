#!/usr/bin/env python3
"""Generate rollout candidates (or a single frozen-base sample) for offline RFT.

Two modes, same loader:
  --n 1 --greedy   -> ONE deterministic frozen-16L base sample per prompt (for per-prompt reward floors)
  --n 8            -> N stochastic candidates per prompt from the current policy (temp 0.9-1.0)

Reuses the XTTS inference path + process-independent per-utt seeds (stable_seed) from gen_panel_ckpt.py,
so candidate j of prompt P always draws the same stream across re-runs (resumable / reproducible).
Writes <out-dir>/<utt>__c<j>.wav and a candidates.jsonl (utt_id, cand_id, wav, ref_text, voice, cs_mode).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fp16"))
import xtts_patch  # noqa: F401
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def stable_seed(seed_base: int, utt_id: str) -> int:
    h = 1469598103934665603
    for b in utt_id.encode("utf-8"):
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return (seed_base + (h & 0x7FFFFFFF)) & 0x7FFFFFFF


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="xtts model dir (config.json, vocab.json)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpt-layers", type=int, default=16)
    ap.add_argument("--prompts", required=True, help="jsonl: utt_id, ref_text, voice, cs_mode")
    ap.add_argument("--refs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=8, help="candidates per prompt")
    ap.add_argument("--temp", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=75)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--greedy", action="store_true", help="deterministic base sample (top_k=1); forces n=1")
    ap.add_argument("--seed-base", type=int, default=20260625)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    n = 1 if args.greedy else args.n
    base = Path(args.base)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    if args.max:
        rows = rows[:args.max]

    cfg = XttsConfig(); cfg.load_json(str(base / "config.json"))
    if args.gpt_layers:
        cfg.model_args.gpt_layers = args.gpt_layers
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_path=args.ckpt, vocab_path=str(base / "vocab.json"), use_deepspeed=False)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        dev = next(model.gpt.parameters()).device
        assert dev.type == "cuda", f"model on {dev} not CUDA — refusing to run inference on CPU"
        print(f"[gen_candidates] model on {dev} ({torch.cuda.get_device_name(0)})")
    else:
        raise SystemExit("CUDA not available — refusing CPU rollout (would take hours/clip)")

    cond_cache = {}
    def cond_for(v):
        if v not in cond_cache:
            cond_cache[v] = model.get_conditioning_latents(audio_path=[str(Path(args.refs_dir) / f"{v}.wav")])
        return cond_cache[v]

    use_cuda = torch.cuda.is_available()
    man, t_start = [], time.perf_counter()
    for r in rows:
        uid, v = r["utt_id"], r["voice"]
        text = r.get("ref_text") or r.get("text")
        tag = r.get("cs_mode", "?")
        gpt_cond, spk = cond_for(v)
        for j in range(n):
            torch.manual_seed(stable_seed(args.seed_base + j * 7919, uid))
            kw = dict(temperature=(0.7 if args.greedy else args.temp), enable_text_splitting=False)
            kw.update(top_k=1, top_p=1.0) if args.greedy else kw.update(top_k=args.top_k, top_p=args.top_p)
            try:
                o = model.inference(text, "hi", gpt_cond, spk, **kw)
            except Exception as e:
                print(f"  SKIP {uid} c{j}: {type(e).__name__} {str(e)[:50]}"); continue
            wav = np.asarray(o["wav"], dtype=np.float32)
            wpath = out / f"{uid}__c{j}.wav"
            sf.write(str(wpath), wav, 24000)
            man.append({"utt_id": uid, "cand_id": j, "wav": str(wpath), "ref_text": text,
                        "voice": v, "cs_mode": tag, "audio_s": round(len(wav) / 24000, 3)})
    (out / "candidates.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in man))
    dt = time.perf_counter() - t_start
    print(f"[gen_candidates] {len(man)} clips ({len(rows)} prompts x {n}) -> {out}  "
          f"{dt:.0f}s ({dt/max(len(man),1):.2f}s/clip)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
