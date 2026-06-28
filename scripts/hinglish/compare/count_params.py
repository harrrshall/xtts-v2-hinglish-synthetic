#!/usr/bin/env python3
"""Parameter-count + size gate. The user's goal is FEWER PARAMETERS, not fewer bytes, so the
primary gate counts torch params (sum of p.numel()) on the GPT autoregressive component only;
disk MB and the frozen DVAE/HiFi-GAN counts are reported but never gate.

Counts are split so a quantization-only "shrink" (which leaves param COUNT unchanged) cannot pass:
  gpt_params  -> the compression target (must hit the tier budget)
  cond_params -> ConditioningEncoder + PerceiverResampler (the ~46M d^2 floor at d=1024)
  frozen_params -> DVAE + HiFi-GAN + speaker encoder (must be UNCHANGED vs the 400M ref)

Usage:
  python count_params.py --base <model_dir> --ckpt model.pth --label cand_200m \
      --out data/eval_200m/params.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def numel(module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="dir with config.json + vocab.json")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpt-layers", type=int, default=0,
                    help="override config gpt_layers (e.g. 12 for the distilled student); 0 = use config")
    args = ap.parse_args()
    base = Path(args.base)

    config = XttsConfig(); config.load_json(str(base / "config.json"))
    if args.gpt_layers:
        config.model_args.gpt_layers = args.gpt_layers
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=args.ckpt,
                          vocab_path=str(base / "vocab.json"), use_deepspeed=False)
    model.eval()

    gpt = model.gpt
    # the AR backbone + heads + embeddings (the compression target)
    gpt_total = numel(gpt)
    # the conditioning floor (does NOT shrink with gpt_layers, only with d)
    cond = 0
    for attr in ("conditioning_encoder", "perceiver_encoder"):
        m = getattr(gpt, attr, None)
        if m is not None:
            cond += numel(m)
    # frozen shared components (must be identical to the 400M reference)
    frozen = 0
    for attr in ("hifigan_decoder", "dvae", "speaker_manager"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "parameters"):
            frozen += numel(m)

    # disk bytes (reported, NEVER gates: bytes != params)
    disk_mb = Path(args.ckpt).stat().st_size / 1e6

    out = {
        "label": args.label,
        "gpt_params": int(gpt_total),
        "gpt_params_millions": round(gpt_total / 1e6, 2),
        "conditioning_floor_params": int(cond),
        "conditioning_floor_millions": round(cond / 1e6, 2),
        "frozen_params": int(frozen),
        "frozen_params_millions": round(frozen / 1e6, 2),
        "disk_mb": round(disk_mb, 1),
        "gpt_model_dim": int(getattr(gpt, "model_dim", config.model_args.gpt_n_model_channels)),
        "gpt_layers": int(getattr(config.model_args, "gpt_layers", -1)),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[{args.label}] GPT={out['gpt_params_millions']}M  "
          f"cond_floor={out['conditioning_floor_millions']}M  "
          f"frozen={out['frozen_params_millions']}M  disk={out['disk_mb']}MB  "
          f"d={out['gpt_model_dim']} L={out['gpt_layers']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
