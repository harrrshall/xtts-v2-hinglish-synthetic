#!/usr/bin/env python3
"""Expressivity monitor over a panel: median pitch-SD (semitones) + energy dynamic-range (dB).
Paired comparison vs a reference panel by utt_id tells us if RL flattened prosody."""
import argparse, json, sys
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import f0_std_semitones, energy_dr_db  # noqa: E402


def panel_stats(manifest):
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    out = {}
    for r in rows:
        wav, sr = sf.read(r["wav"])
        out[r["utt_id"]] = (f0_std_semitones(np.asarray(wav, np.float32), sr),
                            energy_dr_db(np.asarray(wav, np.float32), sr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, help="candidate panel manifest.jsonl")
    ap.add_argument("--ref", default=None, help="reference (frozen 16L) panel manifest.jsonl for paired delta")
    ap.add_argument("--label", default="cand")
    args = ap.parse_args()

    c = panel_stats(args.cand)
    cf0 = np.array([v[0] for v in c.values()]); cdr = np.array([v[1] for v in c.values()])
    print(f"[{args.label}] n={len(c)}  pitch_SD median={np.median(cf0):.3f} semitones  energy_DR median={np.median(cdr):.2f} dB")
    if args.ref:
        r = panel_stats(args.ref)
        paired = [(c[u][0], r[u][0], c[u][1], r[u][1]) for u in c if u in r]
        cf, rf, cd, rd = map(np.array, zip(*paired))
        d_f0 = float(np.median(cf - rf)); d_dr = float(np.median(cd - rd))
        rel_f0 = 100 * d_f0 / max(np.median(rf), 1e-6)
        print(f"[paired vs ref] n={len(paired)}  d_pitchSD={d_f0:+.3f} st ({rel_f0:+.1f}%)  d_energyDR={d_dr:+.2f} dB")
        # the flattening hard-stop from the plan: >25% relative drop or >0.5 st absolute
        flag = (rel_f0 < -25) or (d_f0 < -0.5)
        print(f"[expressivity] {'FLATTENING ALARM' if flag else 'OK (pitch variance held)'}")


if __name__ == "__main__":
    main()
