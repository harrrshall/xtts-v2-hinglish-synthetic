#!/usr/bin/env python3
"""Rigorous false-positive filter for generated Hinglish prompts.

The corpus scorer (common.tag_languages) tags by script alone: any Latin token
counts as English. So a Hindi word written in Latin (romanized) is miscounted as
English and inflates the code-switch score. This re-verifies every transcript with
real lexicons and drops the false positives.

Signals (deterministic, auditable):
  1. TRUE language tags: Latin token is English only if it is in the English
     dictionary AND not a romanized-Hindi-only form (Dakshina lexicon minus the
     English dict), with a high-frequency Hindi function-word override. Recompute
     cmi_bin from corrected tags.
  2. SCRIPT violation: count Latin tokens that are romanized Hindi. Lines that
     break the "Hindi in Devanagari" convention beyond a small tolerance are dropped.
  3. CONSISTENCY: romanize the text's Devanagari spans and compare to ref_surface;
     a low match means the two fields describe different sentences.
  4. DEGENERATE: too short, or heavy token repetition.
  5. DEDUP: global near-duplicate removal across the whole pool (common.minhash_dedup).

Keeps only rows whose CORRECTED cmi_bin is in {high, med}. Writes a clean prompts
file plus a drop report grouped by reason.

Usage:
  python3 scripts/hinglish/verify_prompts.py --in scripts/hinglish/prompts/highcs_generated.jsonl \
      --out scripts/hinglish/prompts/highcs_verified.jsonl
  (accepts multiple --in; merges then dedups)
"""
from __future__ import annotations
import argparse, difflib, json, re, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

ROOT = HERE.parent.parent
EN_PATH = Path("/usr/share/dict/words")
DAKSHINA = ROOT.parent / "VoiceSangam" / "data" / "dakshina" / "hi" / "lexicons" / "hi.romanized.lexicon.tsv"

# High-frequency Hindi function/content words that, romanized, collide with English
# or are short enough to slip through. These ALWAYS count as Hindi in a Hinglish line.
# Unambiguous Hindi function/content words that, romanized, are NOT real English
# words, so they always count as Hindi. Genuinely ambiguous English words
# (to, me, hi, so, is, even, the, do, din, use, ...) are deliberately EXCLUDED;
# the LLM judge layer disambiguates those in context.
STOPLIST_HI = set("""
hai hain hota hoti hote hua hui ka ke ki ko kaa mein se aur ya na bhi
wo woh ye yeh kya kyu kyun kyunki nahi nahin haan yaar matlab accha acha theek thik
bhai mujhe tujhe tumhe unhe unhein humein hum tum aap mera meri mere tera teri
tere uska uski unka unke apna apni apne raha rahi rahe gaya gayi gaye diya kar
karo karenge karna karne karta karti lagta lagti laga jaa jaana jaane chal
chalo abhi kal aaj subah shaam raat ghar baat baar bahut bohot bhot zyada thoda
sab kuchh koi kisi jis naya nayi naye purana purani wala wali waale waala
sahi galat phir kaise kaisa kaisi kahan yahan wahan kaun agar
warna lekin toh sirf haina honge unhone usne maine
tune humne tumne kiya liya lekar dekho suno bola bolo
""".split())


def load_en() -> set:
    s = set()
    if EN_PATH.exists():
        for w in EN_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
            w = w.strip().lower()
            if w.isalpha() and (len(w) >= 3 or w in ("a", "i", "ok", "no", "go", "hi", "do", "to", "me", "we", "us", "it", "is", "am", "be", "so", "up", "in", "on", "of", "by", "my")):
                s.add(w)
    return s


def load_hi_roman(en: set) -> set:
    roman = set()
    if DAKSHINA.exists():
        for line in DAKSHINA.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                for w in p[1].split(","):
                    w = w.strip().lower()
                    if w.isalpha():
                        roman.add(w)
    # romanized-Hindi-only = Dakshina forms that are not real English words
    return (roman - en) | STOPLIST_HI


EN = load_en()
HI_ONLY = load_hi_roman(EN)


def classify(tok: str) -> str:
    if common._DEVA_RE.search(tok):
        return "hi"
    if re.fullmatch(r"[A-Za-z'\-]+", tok):
        lw = tok.lower().strip("'-")
        if lw in STOPLIST_HI:
            return "hi"
        if lw in HI_ONLY:           # romanized Hindi (not an English word)
            return "hi"
        if lw in EN:                # real English (incl. loanwords like account)
            return "en"
        return "en"                 # unknown alpha (brands, slang): treat as English
    return "other"


def corrected_tags(text: str) -> list:
    return [classify(t) for t in text.split()]


def consistency(text: str, ref_surface: str) -> float:
    """Romanize text (Devanagari spans) and fuzzy-match against ref_surface."""
    if not ref_surface:
        return 1.0  # no reference to contradict
    a = common.to_compare_space(common.romanize_deva(text))
    b = common.to_compare_space(ref_surface)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def degenerate(text: str) -> str:
    toks = text.split()
    if len(toks) < 5:
        return "too_short"
    low = [t.lower() for t in toks]
    if max(Counter(low).values()) >= max(4, len(toks) // 2):
        return "token_repetition"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-consistency", type=float, default=0.55)
    ap.add_argument("--max-roman-hi", type=int, default=1,
                    help="max romanized-Hindi Latin tokens tolerated before script-violation drop")
    ap.add_argument("--dedup-threshold", type=float, default=0.85)
    args = ap.parse_args()

    rows = []
    for p in args.inputs:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    print(f"loaded {len(rows)} rows from {len(args.inputs)} file(s)")
    print(f"lexicons: EN={len(EN)} words, HI_ONLY={len(HI_ONLY)} romanized-Hindi forms")

    # global dedup first
    keep_idx = set(common.minhash_dedup([r["text"] for r in rows], threshold=args.dedup_threshold))
    drops = Counter()
    kept = []
    bin_before = Counter(r.get("cs_mode") for r in rows)
    bin_after = Counter()
    for i, r in enumerate(rows):
        if i not in keep_idx:
            drops["near_duplicate"] += 1
            continue
        text = r["text"]
        deg = degenerate(text)
        if deg:
            drops[deg] += 1
            continue
        tags = corrected_tags(text)
        cs_density, cmi_bin = common.lang_tags_to_cs_density(tags)
        roman_hi = sum(1 for t, g in zip(text.split(), tags)
                       if g == "hi" and re.fullmatch(r"[A-Za-z'\-]+", t))
        if roman_hi > args.max_roman_hi:
            drops["script_violation_hindi_in_latin"] += 1
            continue
        cons = consistency(text, r.get("ref_surface", ""))
        if cons < args.min_consistency:
            drops["ref_surface_mismatch"] += 1
            continue
        if cmi_bin not in ("high", "med"):
            drops[f"corrected_bin_{cmi_bin}"] += 1
            continue
        bin_after[cmi_bin] += 1
        kept.append({**r, "cs_mode": cmi_bin, "cs_density": round(cs_density, 3),
                     "verified": True, "consistency": round(cons, 2)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nbins BEFORE (agent labels): {dict(bin_before)}")
    print(f"bins AFTER  (corrected):    {dict(bin_after)}")
    print(f"\nDROPPED {sum(drops.values())} of {len(rows)}:")
    for reason, n in drops.most_common():
        print(f"  {reason}: {n}")
    print(f"\nKEPT {len(kept)} verified -> {args.out}")
    report = {"n_in": len(rows), "n_kept": len(kept), "bins_before": dict(bin_before),
              "bins_after": dict(bin_after), "drops": dict(drops),
              "lexicons": {"en": len(EN), "hi_only": len(HI_ONLY)}}
    Path(args.out).with_suffix(".report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
