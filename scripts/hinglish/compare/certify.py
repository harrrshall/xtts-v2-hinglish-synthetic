#!/usr/bin/env python3
"""ONE-PAGE accept/reject certification: is the candidate (200M/100M) "no quality loss" vs the
400M reference? Emits a single report.md + report.json with per-axis CIs, the param gate, the
content-equivalence gate, the efficiency gate, and an overall CERTIFIED / NOT CERTIFIED verdict.

This is the single runner the user invokes after generating both panels. It composes:
  (1) PARAMETER GATE     from compare/count_params.py outputs (the actual "fewer params" goal)
  (2) QUALITY GATE       per-axis non-inferiority TOST (reuses 12_equivalence_eval.py's BCa+Holm)
  (3) CONTENT GATE       greedy-panel agreement (analogue of fp16 greedy token identity)
  (4) EFFICIENCY GATE    candidate median RTF must beat the reference (compression must buy speed)

Tiers and pre-registered margins are frozen here (see TIERS). "No quality loss" = the candidate
clears the SAME tolerance the 400M student cleared vs the teacher: <=3% on recall/accent/secs and
<=0.10 MOS on UTMOS, certified by a non-inferiority lower bound, NOT a point estimate.

Usage:
  python certify.py --tier 200m \
      --ref-dir data/eval_400m --cand-dir data/eval_200m \
      --ref-params data/eval_400m/params.json --cand-params data/eval_200m/params.json \
      --greedy-ref data/eval_400m/wav_greedy/manifest.jsonl \
      --greedy-cand data/eval_200m/wav_greedy/manifest.jsonl \
      --out-dir data/eval_200m
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from statistics import NormalDist
import numpy as np

ND = NormalDist()

# ---- pre-registered tier budgets + margins (FROZEN before looking at candidate data) ----
# margins in native units; recall/accent/secs are absolute (cosine/recall), utmos is MOS.
MARGINS = {"intelligibility_recall": 0.03, "accent_en_recall": 0.03,
           "naturalness_utmos": 0.10, "voice_secs": 0.03}
# axis -> (per-system filename stem, key). Files are written per system by 08-10/09.
AXIS_FILES = {
    "intelligibility_recall": ("recall_{sys}.json", "recall"),
    "accent_en_recall":       ("accent_{sys}.json", "en_recall"),
    "naturalness_utmos":      ("obj_{sys}.json",    "utmos"),
    "voice_secs":             ("obj_{sys}.json",    "secs"),
}
TIERS = {
    "200m": {"gpt_max_params": 215_000_000, "min_speedup": 1.30, "label": "200M"},
    "100m": {"gpt_max_params": 115_000_000, "min_speedup": 1.60, "label": "100M"},
}
PER_TEST_ALPHA = 0.025
POWER_TARGET = 0.80
N_BOOT = 20000
BOOT_SEED = 20260625
# minimum paired-n required per axis for an adequately powered certification (from the power
# analysis at the observed delta SDs; Holm-worst @ true delta=-margin/3, power 0.80):
MIN_N = {"intelligibility_recall": 168, "accent_en_recall": 389,
         "naturalness_utmos": 367, "voice_secs": 25}


def load_rows(path: Path, key: str) -> dict:
    d = json.loads(path.read_text())
    return {r["utt_id"]: r[key] for r in d["rows"] if r.get(key) is not None}


def paired(ref_dir, cand_dir, sysmap, stem, key):
    r = load_rows(ref_dir / stem.format(sys=sysmap["ref"]), key)
    c = load_rows(cand_dir / stem.format(sys=sysmap["cand"]), key)
    ids = sorted(set(r) & set(c))
    return np.array([c[u] - r[u] for u in ids], dtype=float), ids


def bca_lb(deltas, alpha_one_sided, n_boot=N_BOOT, seed=BOOT_SEED):
    n = len(deltas)
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    boot = deltas[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    theta = deltas.mean()
    z0 = ND.inv_cdf((np.sum(boot < theta) + 0.5) / (n_boot + 1))
    jk = (deltas.sum() - deltas) / (n - 1)
    jbar = jk.mean()
    num = np.sum((jbar - jk) ** 3); den = 6.0 * (np.sum((jbar - jk) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    z = ND.inv_cdf(alpha_one_sided)
    adj = z0 + (z0 + z) / (1 - a * (z0 + z))
    return float(np.percentile(boot, 100 * ND.cdf(adj)))


def tost_power_at_third(sd, n, margin, alpha):
    if n < 2 or sd <= 0:
        return float("nan")
    se = sd / math.sqrt(n); za = ND.inv_cdf(1 - alpha)
    return ND.cdf((margin - margin / 3) / se - za)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=list(TIERS))
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--cand-dir", required=True)
    ap.add_argument("--ref-params", required=True)
    ap.add_argument("--cand-params", required=True)
    ap.add_argument("--greedy-ref", default=None)
    ap.add_argument("--greedy-cand", default=None)
    ap.add_argument("--ref-sys", default="ref")
    ap.add_argument("--cand-sys", default="cand")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    tier = TIERS[args.tier]
    ref_dir, cand_dir = Path(args.ref_dir), Path(args.cand_dir)
    sysmap = {"ref": args.ref_sys, "cand": args.cand_sys}

    # ---- GATE 1: parameter count (the actual goal) ----
    rp = json.loads(Path(args.ref_params).read_text())
    cp = json.loads(Path(args.cand_params).read_text())
    gpt_ok = cp["gpt_params"] <= tier["gpt_max_params"]
    frozen_ok = cp["frozen_params"] == rp["frozen_params"]   # frozen stack must be byte-identical
    param_gate = {
        "ref_gpt_M": rp["gpt_params_millions"], "cand_gpt_M": cp["gpt_params_millions"],
        "budget_M": round(tier["gpt_max_params"] / 1e6, 1),
        "reduction_x": round(rp["gpt_params"] / cp["gpt_params"], 2),
        "gpt_under_budget": gpt_ok,
        "frozen_unchanged": frozen_ok,
        "pass": gpt_ok and frozen_ok,
    }

    # ---- GATE 2: per-axis non-inferiority TOST (BCa lower bound, Holm-corrected) ----
    per_axis = {}
    for name, (stem, key) in AXIS_FILES.items():
        deltas, ids = paired(ref_dir, cand_dir, sysmap, stem, key)
        n = len(deltas)
        sd = float(deltas.std(ddof=1)) if n > 1 else float("nan")
        se = sd / math.sqrt(n) if n > 1 else float("nan")
        m = MARGINS[name]
        z_ni = (deltas.mean() + m) / se if se and se > 0 else float("nan")
        p_ni = 1 - ND.cdf(z_ni)
        per_axis[name] = dict(n=n, mean=float(deltas.mean()), sd=sd, margin=m, p_ni=p_ni,
                              power=tost_power_at_third(sd, n, m, PER_TEST_ALPHA), deltas=deltas)
    # Holm step-down across the 4 axes
    order = sorted(per_axis, key=lambda a: per_axis[a]["p_ni"])
    k = len(order)
    for i, name in enumerate(order):
        per_axis[name]["holm_alpha"] = PER_TEST_ALPHA / (k - i)

    quality = {}
    quality_pass = True
    for name in AXIS_FILES:
        a = per_axis[name]
        lb = bca_lb(a["deltas"], a["holm_alpha"])
        powered = a["n"] >= MIN_N[name] and (not math.isnan(a["power"])) and a["power"] >= POWER_TARGET
        if not powered:
            verdict, ok = "INCONCLUSIVE (underpowered; add n)", False
        elif lb > -a["margin"]:
            verdict, ok = "PASS", True
        else:
            verdict, ok = "FAIL (degraded)", False
        quality_pass = quality_pass and ok
        quality[name] = dict(n=a["n"], delta=round(a["mean"], 4), margin=a["margin"],
                             bca_lb=round(lb, 4), p_ni=round(a["p_ni"], 5),
                             power=round(a["power"], 3) if not math.isnan(a["power"]) else None,
                             min_n=MIN_N[name], verdict=verdict)

    # ---- GATE 3: content-equivalence on the greedy panel (analogue of fp16 token identity) ----
    content = {"checked": False, "pass": True}
    if args.greedy_ref and args.greedy_cand:
        gr = {json.loads(l)["utt_id"]: json.loads(l) for l in open(args.greedy_ref) if l.strip()}
        gc = {json.loads(l)["utt_id"]: json.loads(l) for l in open(args.greedy_cand) if l.strip()}
        ids = sorted(set(gr) & set(gc))
        # greedy token-length agreement within 5%: divergent content under deterministic decode
        # means the architecture changed the audio-token distribution, not just sampling noise.
        agree = [abs(gr[u]["n_audio_tokens"] - gc[u]["n_audio_tokens"]) /
                 max(1, gr[u]["n_audio_tokens"]) <= 0.05 for u in ids]
        frac = sum(agree) / len(agree) if agree else 0.0
        content = {"checked": True, "n": len(ids), "frac_len_agree_within_5pct": round(frac, 3),
                   "pass": frac >= 0.90}

    # ---- GATE 4: efficiency (compression must buy speed; RTF lower is better) ----
    re_ = json.loads((ref_dir / "wav" / "efficiency.json").read_text()) if (ref_dir / "wav" / "efficiency.json").exists() else None
    ce_ = json.loads((cand_dir / "wav" / "efficiency.json").read_text()) if (cand_dir / "wav" / "efficiency.json").exists() else None
    eff = {"checked": False, "pass": True}
    if re_ and ce_ and re_.get("median_rtf") and ce_.get("median_rtf"):
        speedup = re_["median_rtf"] / ce_["median_rtf"]
        eff = {"checked": True, "ref_rtf": re_["median_rtf"], "cand_rtf": ce_["median_rtf"],
               "speedup_x": round(speedup, 2), "min_speedup": tier["min_speedup"],
               "pass": speedup >= tier["min_speedup"]}

    certified = param_gate["pass"] and quality_pass and content["pass"] and eff["pass"]

    report = {"tier": tier["label"], "certified": certified,
              "param_gate": param_gate, "quality": quality,
              "content_gate": content, "efficiency_gate": eff,
              "margins": MARGINS, "n_boot": N_BOOT, "boot_seed": BOOT_SEED}
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    (outd / "certification.json").write_text(json.dumps(report, indent=2))

    # ---- one-page markdown ----
    L = []
    L.append(f"# Certification: {tier['label']} candidate vs 400M reference")
    L.append(f"\n**VERDICT: {'CERTIFIED no quality loss' if certified else 'NOT CERTIFIED'}**\n")
    L.append("## Gate 1 - parameters (the goal: fewer PARAMS, not bytes)")
    L.append(f"- GPT {param_gate['ref_gpt_M']}M -> {param_gate['cand_gpt_M']}M "
             f"({param_gate['reduction_x']}x), budget <= {param_gate['budget_M']}M "
             f"-> {'PASS' if param_gate['gpt_under_budget'] else 'FAIL'}")
    L.append(f"- frozen DVAE+HiFi-GAN unchanged -> {'PASS' if param_gate['frozen_unchanged'] else 'FAIL'}")
    L.append("\n## Gate 2 - quality (paired by utt_id, non-inferiority BCa LB, Holm-corrected)")
    L.append(f"| axis | n | delta | margin | BCa LB | power@-m/3 | verdict |")
    L.append(f"|---|---|---|---|---|---|---|")
    for name, q in quality.items():
        L.append(f"| {name} | {q['n']} | {q['delta']:+.4f} | {q['margin']:.3f} | "
                 f"{q['bca_lb']:+.4f} | {q['power']} | {q['verdict']} |")
    L.append("\n## Gate 3 - content equivalence (greedy panel)")
    L.append(f"- {content}")
    L.append("\n## Gate 4 - efficiency")
    L.append(f"- {eff}")
    (outd / "certification.md").write_text("\n".join(L))

    print("\n".join(L))
    print(f"\nwrote {outd/'certification.md'} and {outd/'certification.json'}")
    return 0 if certified else 1


if __name__ == "__main__":
    raise SystemExit(main())
