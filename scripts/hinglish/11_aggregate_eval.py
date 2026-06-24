#!/usr/bin/env python3
"""Aggregate the expanded paired eval: student vs teacher, per metric, with bootstrap 95% CIs.

Pairs every metric by utt_id (same sentence + voice synthesized by both systems) and reports the
mean for each system plus the paired delta (student - teacher) with a bootstrap CI. A delta CI that
sits at/above ~0 means "not degraded vs the (human-verified) teacher" for that axis.

Deterministic bootstrap (fixed seed via index hashing, no RNG) so results are reproducible.

Usage: python3 scripts/hinglish/11_aggregate_eval.py --dir data/eval_big
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def load_rows(p, key):
    d = json.loads(Path(p).read_text())
    return {r["utt_id"]: r.get(key) for r in d["rows"] if r.get(key) is not None}


def boot_ci(deltas, iters=4000):
    n = len(deltas)
    if n == 0:
        return (None, None)
    means = []
    for b in range(iters):
        # deterministic LCG resample (no RNG dependency)
        s = 0.0; seed = (b * 2654435761 + 12345) & 0xFFFFFFFF
        for _ in range(n):
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            s += deltas[seed % n]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * iters)]; hi = means[int(0.975 * iters)]
    return (round(lo, 3), round(hi, 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/eval_big")
    args = ap.parse_args()
    D = Path(args.dir)

    metrics = {
        "intelligibility_recall": ("recall_student.json", "recall_teacher.json", "recall"),
        "accent_en_recall": ("accent_student.json", "accent_teacher.json", "en_recall"),
        "naturalness_utmos": ("obj_student.json", "obj_teacher.json", "utmos"),
        "voice_secs": ("obj_student.json", "obj_teacher.json", "secs"),
    }
    report = {}
    print(f"{'metric':24s} {'student':>9} {'teacher':>9} {'delta':>8} {'95% CI (student-teacher)':>26} {'n':>4}")
    for name, (sf_, tf_, key) in metrics.items():
        try:
            s = load_rows(D / sf_, key); t = load_rows(D / tf_, key)
        except FileNotFoundError:
            print(f"{name:24s}  (missing files)"); continue
        common_ids = [u for u in s if u in t]
        deltas = [s[u] - t[u] for u in common_ids]
        smean = sum(s[u] for u in common_ids) / len(common_ids) if common_ids else None
        tmean = sum(t[u] for u in common_ids) / len(common_ids) if common_ids else None
        dmean = sum(deltas) / len(deltas) if deltas else None
        ci = boot_ci(deltas)
        report[name] = {"student": round(smean, 3), "teacher": round(tmean, 3),
                        "delta": round(dmean, 3), "ci95": ci, "n": len(common_ids)}
        print(f"{name:24s} {smean:9.3f} {tmean:9.3f} {dmean:+8.3f}   [{ci[0]:+.3f}, {ci[1]:+.3f}]       {len(common_ids):4d}")

    # verdict heuristic per axis: "not degraded" if CI upper bound >= a small negative tolerance
    print("\nPer-axis read (not-degraded vs teacher if delta CI upper bound >= -0.03):")
    verdict = {}
    for name, r in report.items():
        ok = r["ci95"][1] is not None and r["ci95"][1] >= -0.03
        verdict[name] = ok
        print(f"  {name:24s} {'NOT DEGRADED vs teacher' if ok else 'POSSIBLE REGRESSION'}  (delta {r['delta']:+.3f}, CI {r['ci95']})")
    (D / "aggregate_report.json").write_text(json.dumps({"metrics": report, "not_degraded": verdict}, indent=2))
    print(f"\nwrote {D/'aggregate_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
