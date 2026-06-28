#!/usr/bin/env python3
"""Score RFT candidates against per-prompt frozen-16L floors; emit the winner corpus + DPO pairs.

Winner rule (the expressivity-preserving core): a candidate is ELIGIBLE only if it passes ALL floors
(UTMOS/SECS/F0-std/energy-DR >= base, duration within tol, not degenerate). Among the eligible, the
winner = max R_accent. You cannot train toward a flatter/slower/voice-drifted sample.

Outputs:
  --out-corpus  pipe-delimited winner_wav|ref_text|voice  (drop-in for train_xtts.py CE fine-tune)
  --out-pairs   jsonl winner/loser wavs for the DPO escalation path
  --out-report  per-utt accent gain + eligibility stats
"""
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import RewardScorer, Floors, Weights  # noqa: E402


def derive_floor(scored, dur_tol):
    """Self-calibrated per-prompt floor from the frozen-16L SAMPLED distribution (this group of candidates).
    Anchors to the model's typical output, not its greedy best: floor = low percentile (don't drop below the
    typical-low end), with small tolerances on the learned/coarse metrics. Returns a Floors + base accent."""
    f0 = [s["f0"] for s in scored]
    edr = [s["edr"] for s in scored]
    utm = [s["utmos"] for s in scored]
    secs = [s["secs"] for s in scored if s["secs"] is not None]
    durs = [s["dur"] for s in scored]
    raccs = [s["racc"] for s in scored]
    fl = Floors(
        utmos=float(np.median(utm)) - 0.10,                  # learned MOS: median minus a tolerance band
        secs=(float(np.median(secs)) - 0.02) if secs else 1.0,
        f0_std=float(np.percentile(f0, 25)) * 0.95,          # don't drop below the typical-low pitch variance
        energy_dr=float(np.percentile(edr, 25)) * 0.95,
        dur=float(np.median(durs)), dur_tol=dur_tol, secs_margin=0.0)
    return fl, float(np.median(raccs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--base-stats", default=None,
                    help="frozen-16L floor table (rounds 2+). If omitted, derive floors from THIS group (round 1).")
    ap.add_argument("--refs-dir", required=True)
    ap.add_argument("--out-corpus", required=True)
    ap.add_argument("--out-pairs", default=None)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--out-floors", default=None, help="save the derived per-prompt frozen-16L floors for reuse")
    ap.add_argument("--dur-tol", type=float, default=0.25)
    ap.add_argument("--min-gain", type=float, default=0.0, help="only keep winners whose r_accent beats base by >= this")
    args = ap.parse_args()

    base_stats = json.loads(Path(args.base_stats).read_text()) if args.base_stats else {}
    cands = [json.loads(l) for l in open(args.candidates, encoding="utf-8") if l.strip()]
    by_utt = defaultdict(list)
    for c in cands:
        by_utt[c["utt_id"]].append(c)

    sc = RewardScorer()
    for v in ["kaustubh", "arjun", "maya", "aadya"]:
        rp = Path(args.refs_dir) / f"{v}.wav"
        if rp.exists():
            sc.register_voice(v, str(rp))

    corpus, pairs, report, floor_table = [], [], [], {}
    n_winner = n_noeligible = n_belowgain = 0
    for uid, group in by_utt.items():
        # pass 1: components for every candidate (no floor yet)
        scored = []
        for c in group:
            wav, sr = sf.read(c["wav"])
            comp = sc.components(np.asarray(wav, np.float32), sr, c["ref_text"], c["voice"], wav_path=c["wav"])
            scored.append({**c, "comp": comp, "racc": None, "f0": round(comp["f0_std"], 3),
                           "edr": round(comp["energy_dr"], 3), "utmos": round(comp["utmos"], 3),
                           "secs": comp["secs"], "dur": comp["dur"]})
        # provisional r_accent (floor-independent) for percentile-floor derivation
        for s in scored:
            s["racc"] = sc.score(s["comp"], Floors(0, 0, 0, 0, s["dur"]))["r_accent"]
        # pass 2: floor — frozen table (rounds 2+) or self-calibrated from this 16L group (round 1)
        if uid in base_stats:
            bs = base_stats[uid]
            fl = Floors(utmos=bs["utmos"], secs=bs["secs"] if bs["secs"] is not None else 1.0,
                        f0_std=bs["f0_std"], energy_dr=bs["energy_dr"], dur=bs["dur"], dur_tol=args.dur_tol)
            base_racc = bs.get("base_racc")
        else:
            fl, base_racc = derive_floor(scored, args.dur_tol)
        floor_table[uid] = {"utmos": round(fl.utmos, 3), "secs": (round(fl.secs, 3) if fl.secs is not None else None),
                            "f0_std": round(fl.f0_std, 3), "energy_dr": round(fl.energy_dr, 3),
                            "dur": round(fl.dur, 3), "base_racc": base_racc}
        # pass 3: eligibility + total reward against the floor
        for s in scored:
            sc_out = sc.score(s["comp"], fl)
            s["eligible"] = sc_out["eligible"]; s["total"] = sc_out["total"]
            del s["comp"]
        eligible = [s for s in scored if s["eligible"]]
        if not eligible:
            n_noeligible += 1
            report.append({"utt": uid, "status": "no_eligible", "n": len(group),
                           "best_racc": round(max(s["racc"] for s in scored), 4), "base_racc": base_racc})
            continue
        winner = max(eligible, key=lambda s: s["total"])   # accent + mild duration tiebreak (floors already held)
        gain = (winner["racc"] - base_racc) if base_racc is not None else None
        if gain is not None and gain < args.min_gain:
            n_belowgain += 1
            report.append({"utt": uid, "status": "below_gain", "winner_racc": winner["racc"],
                           "base_racc": base_racc, "gain": round(gain, 4)})
            continue
        n_winner += 1
        corpus.append(f'{os.path.abspath(winner["wav"])}|{winner["ref_text"]}|{winner["voice"]}')
        # DPO pair: winner vs the lowest-accent eligible (or any lower candidate) for the escalation path
        loser = min(scored, key=lambda s: s["racc"])
        if loser["wav"] != winner["wav"]:
            pairs.append({"utt_id": uid, "ref_text": winner["ref_text"], "voice": winner["voice"],
                          "chosen_wav": winner["wav"], "rejected_wav": loser["wav"],
                          "chosen_racc": winner["racc"], "rejected_racc": loser["racc"]})
        report.append({"utt": uid, "status": "winner", "winner_racc": winner["racc"],
                       "base_racc": base_racc, "gain": (round(gain, 4) if gain is not None else None),
                       "winner_f0": winner["f0"], "n_eligible": len(eligible), "n": len(group)})

    Path(args.out_corpus).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_corpus).write_text("\n".join(corpus))
    if args.out_pairs:
        Path(args.out_pairs).write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs))
    gains = [r["gain"] for r in report if r.get("gain") is not None]
    summary = {"n_prompts": len(by_utt), "n_winner": n_winner, "n_no_eligible": n_noeligible,
               "n_below_gain": n_belowgain, "mean_winner_gain": round(float(np.mean(gains)), 4) if gains else None,
               "rows": report}
    Path(args.out_report).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out_floors:
        Path(args.out_floors).write_text(json.dumps(floor_table, indent=2, ensure_ascii=False))
    print(f"[select] prompts={len(by_utt)} winners={n_winner} no_eligible={n_noeligible} "
          f"below_gain={n_belowgain} mean_gain={summary['mean_winner_gain']}")
    print(f"[select] corpus -> {args.out_corpus} ({len(corpus)} rows); pairs={len(pairs)}")


if __name__ == "__main__":
    main()
