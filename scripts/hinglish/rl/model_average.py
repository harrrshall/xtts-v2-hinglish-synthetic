#!/usr/bin/env python3
"""Post-hoc weight interpolation: theta(alpha) = (1-alpha)*16L + alpha*RL.
The documented alignment-tax mitigation: recover capability lost to RL by moving back toward the base,
trading a little of the accent gain for naturalness. Interpolates every matching float tensor."""
import argparse, torch
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="frozen 16L checkpoint (alpha=0 end)")
    ap.add_argument("--rl", required=True, help="round-1 RL checkpoint (alpha=1 end)")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A = torch.load(args.base, map_location="cpu"); A = A.get("model", A)
    Bfull = torch.load(args.rl, map_location="cpu"); B = Bfull.get("model", Bfull)
    a = args.alpha
    n = 0
    for k in B:
        if k in A and torch.is_tensor(B[k]) and B[k].dtype.is_floating_point and A[k].shape == B[k].shape:
            B[k] = (1 - a) * A[k].float() + a * B[k].float()
            B[k] = B[k].to(torch.float32)
            n += 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(Bfull, args.out)
    print(f"[model_average] alpha={a} interpolated {n} tensors -> {args.out}")


if __name__ == "__main__":
    main()
