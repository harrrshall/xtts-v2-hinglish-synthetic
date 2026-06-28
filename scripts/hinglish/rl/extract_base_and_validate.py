#!/usr/bin/env python3
"""Extract per-prompt frozen-16L base floors AND unit-validate the reward module.

- Computes Floors (utmos, secs, f0_std, energy_dr, dur) for each base (16L) clip -> data/rl/base_stats.json.
- Self-check: re-scoring the base clip against its OWN floor must be eligible with ~0 floor penalties (no-op ~0 delta).
- Candidate check: the known accent-degraded 12L panel should score LOWER r_accent than the 16L base on paired utts.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import RewardScorer, Floors, Weights  # noqa: E402


def load_manifest(p):
    return {json.loads(l)["utt_id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-panel", required=True)
    ap.add_argument("--cand-panel", default=None)
    ap.add_argument("--refs-dir", required=True)
    ap.add_argument("--max", type=int, default=8)
    ap.add_argument("--english-only", action="store_true", help="only score prompts that contain English words")
    ap.add_argument("--out", default="data/rl/base_stats.json")
    args = ap.parse_args()

    from reward import english_words
    base = load_manifest(args.base_panel)
    cand = load_manifest(args.cand_panel) if args.cand_panel else {}
    uids = list(base)
    if args.english_only:
        uids = [u for u in uids if english_words(base[u]["ref_text"])]
    if args.max:
        uids = uids[:args.max]

    sc = RewardScorer()
    for v in ["kaustubh", "arjun", "maya", "aadya"]:
        rp = Path(args.refs_dir) / f"{v}.wav"
        if rp.exists():
            sc.register_voice(v, str(rp))

    base_stats, rows = {}, []
    for uid in uids:
        b = base[uid]
        wav, sr = sf.read(b["wav"])
        comp = sc.components(np.asarray(wav, np.float32), sr, b["ref_text"], b["voice"], wav_path=b["wav"])
        fl = Floors(utmos=comp["utmos"], secs=comp["secs"] if comp["secs"] is not None else 1.0,
                    f0_std=comp["f0_std"], energy_dr=comp["energy_dr"], dur=comp["dur"])
        # self-check: base vs its own floor
        s_self = sc.score(comp, fl)
        base_stats[uid] = {"utmos": comp["utmos"], "secs": comp["secs"], "f0_std": round(comp["f0_std"], 4),
                           "energy_dr": round(comp["energy_dr"], 4), "dur": round(comp["dur"], 4),
                           "base_recall": comp["en_recall"], "base_racc": s_self["r_accent"]}
        line = {"utt": uid, "voice": b["voice"], "base_acc": comp["en_recall"], "base_f0": round(comp["f0_std"], 3),
                "self_eligible": s_self["eligible"], "self_pen": round(s_self["p_utmos"] + s_self["p_secs"] + s_self["p_f0"] + s_self["p_edr"], 4),
                "self_sil": round(comp["silence_ratio"], 3), "self_rep": round(comp["ngram_rep"], 3), "self_degen": s_self["degenerate"]}
        # candidate (12L) vs base floor
        if uid in cand:
            cw, csr = sf.read(cand[uid]["wav"])
            cc = sc.components(np.asarray(cw, np.float32), csr, b["ref_text"], b["voice"], wav_path=cand[uid]["wav"])
            sc_cand = sc.score(cc, fl)
            line.update({"cand_acc": cc["en_recall"], "cand_f0": round(cc["f0_std"], 3),
                         "cand_eligible": sc_cand["eligible"], "cand_racc": sc_cand["r_accent"],
                         "base_racc": s_self["r_accent"]})
        rows.append(line)
        print(json.dumps(line, ensure_ascii=False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(base_stats, indent=2))
    # summary
    self_ok = sum(1 for r in rows if r["self_eligible"])
    print(f"\n[validate] base self-eligible {self_ok}/{len(rows)} (expect all)")
    if cand:
        paired = [r for r in rows if "cand_acc" in r and r["cand_acc"] is not None and r["base_acc"] is not None]
        if paired:
            db = np.mean([r["base_acc"] for r in paired]); dc = np.mean([r["cand_acc"] for r in paired])
            print(f"[validate] paired accent: base(16L)={db:.3f} cand(12L)={dc:.3f}  (expect cand < base)")
    print(f"[validate] wrote base floors for {len(base_stats)} utts -> {args.out}")


if __name__ == "__main__":
    main()
