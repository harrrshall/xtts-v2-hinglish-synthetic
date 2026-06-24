#!/usr/bin/env python3
"""Cast the XTTS-v2 checkpoint's floating weights to fp16 for distribution.

Strategy: fp16 ON DISK, upcast to fp32 AT LOAD (XTTS runs fp32 inference; load_state_dict
copies fp16 values into the fp32 modules, so runtime numerics stay fp32 with only the
weight-rounding perturbation). This halves download/disk with no runtime dtype change.

Safety: only float32/float64 tensors are cast, and ONLY if their max-abs magnitude is below
the fp16 finite ceiling (65504). Anything larger stays fp32 so the cast can never make inf.
Non-floating tensors (int buffers, num_batches_tracked, etc.) are left untouched.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

FP16_MAX = 65504.0
SAFETY = 60000.0  # keep a margin below the fp16 ceiling


def find_state_dict(obj):
    """Return (container, key_path) where the tensor state_dict lives. Handles flat sd or {'model': sd}."""
    if isinstance(obj, dict):
        # a state_dict is a dict whose values are mostly tensors
        tens = sum(1 for v in obj.values() if torch.is_tensor(v))
        if tens > 0 and tens >= 0.5 * len(obj):
            return obj, None
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"], "model"
    raise SystemExit(f"could not locate a tensor state_dict in checkpoint (top keys: "
                     f"{list(obj.keys())[:10] if isinstance(obj, dict) else type(obj)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.inp, map_location="cpu", weights_only=False)
    sd, key = find_state_dict(ckpt)
    print(f"[convert] checkpoint top-level: {type(ckpt).__name__}; "
          f"state_dict at: {key or '<root>'}; {len(sd)} entries")

    n_cast = n_skip_nonfloat = n_skip_overflow = 0
    bytes_before = bytes_after = 0
    global_max = 0.0
    overflow_keys = []
    for k, v in list(sd.items()):
        if not torch.is_tensor(v):
            n_skip_nonfloat += 1
            continue
        bytes_before += v.numel() * v.element_size()
        if v.dtype in (torch.float32, torch.float64):
            mx = float(v.abs().max()) if v.numel() else 0.0
            global_max = max(global_max, mx)
            if mx < SAFETY:
                sd[k] = v.to(torch.float16)
                n_cast += 1
            else:
                n_skip_overflow += 1
                overflow_keys.append((k, mx))
        else:
            n_skip_nonfloat += 1
        bytes_after += sd[k].numel() * sd[k].element_size()

    if key:
        ckpt[key] = sd
    else:
        ckpt = sd
    torch.save(ckpt, args.out)

    print(f"[convert] cast to fp16:     {n_cast} tensors")
    print(f"[convert] kept (non-float): {n_skip_nonfloat} tensors")
    print(f"[convert] kept fp32 (would-overflow fp16): {n_skip_overflow} tensors")
    if overflow_keys:
        for k, mx in overflow_keys[:10]:
            print(f"           !! {k}: max|w|={mx:.1f} (>{SAFETY}) kept fp32")
    print(f"[convert] max |weight| across all float tensors: {global_max:.3f} "
          f"(fp16 finite ceiling {FP16_MAX})")
    in_mb = Path(args.inp).stat().st_size / 1e6
    out_mb = Path(args.out).stat().st_size / 1e6
    print(f"[convert] file size: {in_mb:.1f} MB -> {out_mb:.1f} MB  ({out_mb/in_mb*100:.1f}%)")
    print(f"[convert] tensor bytes: {bytes_before/1e6:.1f} MB -> {bytes_after/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
