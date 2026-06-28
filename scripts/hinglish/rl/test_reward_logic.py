#!/usr/bin/env python3
"""Fast, model-free unit test of the reward scoring LOGIC and prosody features.

Plants deliberately-bad candidate measurements against a fixed base floor and asserts each floor /
degenerate filter fires as designed. Run: .venv_xtts/bin/python scripts/hinglish/rl/test_reward_logic.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import Floors, Weights, RewardScorer, f0_std_semitones, energy_dr_db, silence_ratio, ngram_rep

BASE = Floors(utmos=3.14, secs=0.86, f0_std=4.0, energy_dr=20.0, dur=5.0)
W = Weights()

# a clean candidate exactly at base -> eligible, ~0 floor penalties
GOOD = dict(en_recall=0.9, nll=0.6, n_en=5, hyp="ok", utmos=3.20, secs=0.87,
            f0_std=4.2, energy_dr=21.0, dur=5.0, silence_ratio=0.2, ngram_rep=0.08)


def score(**over):
    comp = {**GOOD, **over}
    # RewardScorer.score is a pure method; call it unbound (no model init needed)
    return RewardScorer.score(RewardScorer.__new__(RewardScorer), comp, BASE, W)


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


print("== scoring-logic floors ==")
g = score()
check("clean candidate eligible", g["eligible"] is True)
check("clean candidate ~0 floor penalty", abs(g["p_utmos"]) + abs(g["p_f0"]) + abs(g["p_edr"]) < 1e-6)

check("UTMOS below base -> penalty<0 and ineligible",
      (lambda s: s["p_utmos"] < 0 and not s["eligible"])(score(utmos=2.9)))
check("SECS below base-margin -> ineligible",
      not score(secs=0.80)["eligible"])           # 0.80 < 0.86-0.03
check("SECS within margin -> still eligible",
      score(secs=0.84)["eligible"])               # 0.84 >= 0.83
check("F0-std collapse -> p_f0<0 and ineligible",
      (lambda s: s["p_f0"] < 0 and not s["eligible"])(score(f0_std=2.0)))
check("energy-DR collapse -> p_edr<0 and ineligible",
      (lambda s: s["p_edr"] < 0 and not s["eligible"])(score(energy_dr=10.0)))
check("slowdown >15% -> p_dur<0 and ineligible",
      (lambda s: s["p_dur"] < 0 and not s["eligible"])(score(dur=6.5)))   # +30%
check("small pace drift within tol -> eligible", score(dur=5.3)["eligible"])  # +6%

print("== degenerate filter ==")
check("high silence -> degenerate, total sinks",
      (lambda s: s["degenerate"] and s["total"] < -4)(score(silence_ratio=0.6)))
check("looping n-grams -> degenerate",
      score(ngram_rep=0.5)["degenerate"] is True)

print("== R_accent densification ==")
hi = score(en_recall=1.0, nll=0.2)["r_accent"]
lo = score(en_recall=0.4, nll=0.2)["r_accent"]
check("higher recall -> higher r_accent", hi > lo)
check("pure-Hindi (no English) -> neutral r_accent=1.0",
      abs(score(en_recall=None)["r_accent"] - 1.0) < 1e-6)

print("== prosody feature sanity ==")
sr = 22050; t = np.linspace(0, 2, 2 * sr, endpoint=False)
flat = 0.3 * np.sin(2 * np.pi * 150 * t)                          # constant pitch
vib = 0.3 * np.sin(2 * np.pi * (150 + 30 * np.sin(2 * np.pi * 3 * t)) * t)  # vibrato
check("vibrato pitch-SD > flat pitch-SD", f0_std_semitones(vib, sr) > f0_std_semitones(flat, sr))
silent = np.concatenate([0.3 * np.sin(2 * np.pi * 150 * t[:sr]), np.zeros(sr)])
check("half-silent clip -> silence_ratio>0.4", silence_ratio(silent, sr) > 0.4)
check("partial 3-gram repeat above degenerate cap (0.30)",
      ngram_rep(["a", "b", "c", "a", "b", "c", "a", "b", "c"]) > BASE.rep_cap)
check("pure babble loop -> ~1.0", ngram_rep(["uh"] * 8) > 0.9)

print("\nALL REWARD-LOGIC TESTS PASSED")
