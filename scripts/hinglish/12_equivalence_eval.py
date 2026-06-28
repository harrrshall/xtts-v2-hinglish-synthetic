#!/usr/bin/env python3
"""Equivalence (no-quality-loss) certification for a compressed student vs the 400M reference.

Replaces the "delta-CI upper bound >= -0.03" heuristic in 11_aggregate_eval.py with a statistically
correct non-inferiority test: a one-sided TOST lower bound against a PRE-REGISTERED per-axis margin,
on a proper i.i.d. bootstrap, with Holm correction across the 4 axes and a separate power report.

Reference system here is the CURRENT 400M model (not the teacher TTS): the goal is "the 100M model
matches the 400M student within the same tolerance the 400M matched the teacher". Pass two run dirs:
  --ref   data/eval_400m   (the 400M model's per-utt scores)
  --cand  data/eval_100m   (the compressed model's per-utt scores)
Each dir holds the same four files 11_aggregate_eval.py reads (recall_*.json, accent_*.json,
obj_*.json) but produced by the candidate / reference model.

Pre-registered margins (native units, frozen BEFORE looking at candidate data):
  intelligibility_recall  m = 0.03   (3% absolute content-word recall)
  accent_en_recall        m = 0.03   (3% absolute English-recall)
  naturalness_utmos       m = 0.10   (MOS; ~3% of the ~3.1 operating point, > fp16 greedy noise)
  voice_secs              m = 0.03   (cosine; matches the legacy -0.03 tolerance, ~3% of ~0.86)

Decision rule (NON-INFERIORITY, one-sided): the candidate is "not degraded" on an axis iff the
Holm-adjusted lower bound of the paired delta (cand - ref) is strictly above -margin:
    LB_holm(delta) > -m  ->  PASS (no quality loss on that axis)
Certification PASSES overall iff ALL FOUR axes pass. This controls the FALSE-POSITIVE (declaring
"no loss" when truly degraded): a real drop near the margin will, with the powered n, push the lower
bound below -m and FAIL. It also bounds FALSE-NEGATIVES via the power report: if an axis is
underpowered (TOST power < 0.80 at the margin), the axis is flagged INCONCLUSIVE, not PASS/FAIL.

Usage:
  python3 scripts/hinglish/12_equivalence_eval.py --ref data/eval_400m --cand data/eval_100m
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from statistics import NormalDist

import numpy as np

ND = NormalDist()

# axis -> (ref_file, cand_file, key, pre-registered equivalence margin)
AXES = {
    "intelligibility_recall": ("recall.json", "recall.json", "recall", 0.03),
    "accent_en_recall":       ("accent.json", "accent.json", "en_recall", 0.03),
    "naturalness_utmos":      ("obj.json",    "obj.json",    "utmos", 0.10),
    "voice_secs":             ("obj.json",    "obj.json",    "secs", 0.03),
}
FAMILY_ALPHA = 0.05          # two-sided family error
PER_TEST_ALPHA = 0.025       # one-sided per axis BEFORE multiplicity correction
POWER_TARGET = 0.80
N_BOOT = 20000
BOOT_SEED = 20260625         # fixed integer seed -> reproducible, NOT Python hash()


def load_rows(path: Path, key: str) -> dict:
    d = json.loads(path.read_text())
    return {r["utt_id"]: r[key] for r in d["rows"] if r.get(key) is not None}


def paired_deltas(ref_dir: Path, cand_dir: Path, rf: str, cf: str, key: str):
    r = load_rows(ref_dir / rf, key)
    c = load_rows(cand_dir / cf, key)
    ids = sorted(set(r) & set(c))
    return np.array([c[u] - r[u] for u in ids], dtype=float), ids


def bca_bootstrap_lb(deltas: np.ndarray, alpha_one_sided: float, n_boot=N_BOOT, seed=BOOT_SEED):
    """BCa one-sided lower bound on the mean paired delta. Proper i.i.d. resampling with a numpy
    Generator (the LCG in 11_aggregate_eval.py degenerates to a zero-width CI at small n)."""
    n = len(deltas)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = deltas[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    theta = deltas.mean()
    # bias correction
    z0 = ND.inv_cdf((np.sum(boot < theta) + 0.5) / (n_boot + 1))
    # acceleration via jackknife
    jk = (deltas.sum() - deltas) / (n - 1)
    jbar = jk.mean()
    num = np.sum((jbar - jk) ** 3)
    den = 6.0 * (np.sum((jbar - jk) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    z = ND.inv_cdf(alpha_one_sided)
    adj = z0 + (z0 + z) / (1 - a * (z0 + z))
    lo = float(np.percentile(boot, 100 * ND.cdf(adj)))
    return lo, theta


def tost_power(sd_delta: float, n: int, margin: float, alpha_one_sided: float):
    """Power of the non-inferiority test at true delta=0 (and at delta=-margin/3, the tiny-but-real
    drop we still want to clear)."""
    if n < 2 or sd_delta <= 0:
        return float("nan"), float("nan")
    se = sd_delta / math.sqrt(n)
    za = ND.inv_cdf(1 - alpha_one_sided)
    p0 = ND.cdf(margin / se - za)                 # true delta = 0
    p3 = ND.cdf((margin - margin / 3) / se - za)  # true delta = -margin/3
    return p0, p3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="dir with the 400M reference per-utt scores")
    ap.add_argument("--cand", required=True, help="dir with the compressed candidate per-utt scores")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ref_dir, cand_dir = Path(args.ref), Path(args.cand)

    # 1) compute per-axis deltas, p-values for Holm, and raw BCa lower bounds
    per_axis = {}
    skipped = []
    for name, (rf, cf, key, margin) in AXES.items():
        try:
            deltas, ids = paired_deltas(ref_dir, cand_dir, rf, cf, key)
        except FileNotFoundError:
            skipped.append(name); continue
        n = len(deltas)
        if n == 0:
            skipped.append(name); continue
        sd = float(deltas.std(ddof=1)) if n > 1 else float("nan")
        se = sd / math.sqrt(n) if n > 1 else float("nan")
        # non-inferiority z and one-sided p: H0 delta <= -m  vs  H1 delta > -m
        z_ni = (deltas.mean() + margin) / se if se > 0 else float("nan")
        p_ni = 1 - ND.cdf(z_ni)
        p0, p3 = tost_power(sd, n, margin, PER_TEST_ALPHA)
        per_axis[name] = dict(n=n, mean=float(deltas.mean()), sd=sd, se=se, margin=margin,
                              p_ni=p_ni, power_d0=p0, power_d3=p3, deltas=deltas)

    # 2) Holm step-down across the 4 axes (controls family-wise false-positive at 0.05)
    order = sorted(per_axis, key=lambda a: per_axis[a]["p_ni"])
    k = len(order)
    holm_reject = {}
    prev_ok = True
    for i, name in enumerate(order):
        thr = PER_TEST_ALPHA / (k - i)   # one-sided Holm threshold
        rej = prev_ok and (per_axis[name]["p_ni"] <= thr)
        holm_reject[name] = rej
        per_axis[name]["holm_alpha"] = thr
        prev_ok = rej

    # 3) Holm-consistent BCa lower bound at each axis's Holm alpha, then verdict
    print(f"{'axis':24s} {'n':>4} {'delta':>8} {'margin':>7} {'BCa_LB':>8} "
          f"{'p_ni':>8} {'pow@0':>6} {'pow@-m/3':>8}  verdict")
    report = {}
    overall_pass = True
    for name in per_axis:
        a = per_axis[name]
        lb, _ = bca_bootstrap_lb(a["deltas"], a["holm_alpha"])
        powered = (not math.isnan(a["power_d3"])) and a["power_d3"] >= POWER_TARGET
        if not powered:
            verdict = "INCONCLUSIVE (underpowered: add n)"
            overall_pass = False
        elif lb > -a["margin"]:
            verdict = "PASS (no quality loss)"
        else:
            verdict = "FAIL (degraded)"
            overall_pass = False
        report[name] = dict(n=a["n"], delta=round(a["mean"], 4), margin=a["margin"],
                            bca_lb=round(lb, 4), p_ni=round(a["p_ni"], 5),
                            power_d0=round(a["power_d0"], 3), power_d3=round(a["power_d3"], 3),
                            holm_alpha=round(a["holm_alpha"], 5), verdict=verdict)
        print(f"{name:24s} {a['n']:4d} {a['mean']:+8.4f} {a['margin']:7.3f} {lb:+8.4f} "
              f"{a['p_ni']:8.5f} {a['power_d0']:6.3f} {a['power_d3']:8.3f}  {verdict}")

    if skipped:
        print(f"SKIPPED (no scores for these axes): {', '.join(skipped)}")
        overall_pass = False  # cannot fully certify with axes missing
    print(f"\nOVERALL: {'CERTIFIED no quality loss' if overall_pass else 'NOT FULLY CERTIFIED (see per-axis + skipped)'}")
    out = Path(args.out) if args.out else (cand_dir / "equivalence_report.json")
    out.write_text(json.dumps({"axes": report, "certified": overall_pass, "skipped_axes": skipped,
                               "family_alpha": FAMILY_ALPHA, "n_boot": N_BOOT}, indent=2))
    print(f"wrote {out}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
