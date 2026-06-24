#!/usr/bin/env python3
"""S1: build the Hinglish text corpus for synthesis.

This stage turns text into the corpus manifest that the synthesizer (S2) will
read. It does three things and nothing else:

  1. Seed the code-switch distribution from the real spontaneous eval set.
     We read ref_orig from eval_spontaneous_combined_manifest.json ONLY to learn
     the target shape (how cmi_bin and switch density are spread) and, when asked,
     to copy a held-out slice of real transcripts in as corpus text. The real
     audio is never used for training; the eval set stays the frozen hard gold.
     Seeding from real text keeps the synthetic corpus on the real manifold.

  2. Ingest LLM-generated transcripts from prompts/*.jsonl. Each line is an
     object with at least a "text" field (Devanagari Hindi + Latin English, no
     transliteration). Optional "domain" and "source" fields are carried into
     flags. No network call happens here: the LLM expansion is a pluggable,
     offline step. A --synthesize-stub mode mints placeholder transcripts by
     light recombination of the real seed text so the whole chain runs with zero
     prompt files on hand, purely to prove the plumbing.

  3. Clean, tag, score, dedup, balance, and over-generate. Every accepted row is
     run through the common.py contracts: per-word script enforcement, number
     normalization, tag_languages, cs_density/cmi_bin, per-utt rep_4gram, and a
     MinHash dedup pass. We then balance across cmi_bin so the high-code-switch
     tail is not starved, and over-generate to ~1.4x the target so the downstream
     qwen filter has slack to discard bad clips.

Outputs:
  data/corpus/corpus.jsonl       corpus_manifest_v1 rows, audio_path/accept null
  data/corpus/corpus_stats.json  diversity report (unique-word ratio, token
                                 entropy vs the real baseline, switch-points per
                                 utterance, balance table, dedup drop counts)

All schema, identity, tagging, dedup, and diagnostic logic lives in common.py;
this file only orchestrates. Runnable from the repo root, fully offline.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Import the shared module. We support being launched from the repo root or from
# inside scripts/hinglish, so we put our own directory on the path first.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import common  # noqa: E402

_REPO = _HERE.parent.parent
_DEFAULT_EVAL = (_REPO / "data" / "spontaneous_hinglish" /
                 "eval_spontaneous_combined_manifest.json")
_DEFAULT_PROMPTS = _HERE / "prompts"
_DEFAULT_OUT_DIR = _REPO / "data" / "corpus"


# ----------------------------------------------------------------------------
# Text cleaning: per-word script enforcement and number normalization
# ----------------------------------------------------------------------------

# Map ASCII digits and Devanagari digits to spelled-out forms so the synthesizer
# pronounces them and so they tokenize as words for cs_density. We spell numbers
# in the script of their surrounding word when we can tell, else default to the
# Devanagari (Hindi) reading because the teacher voices are Hindi-first.
_EN_NUM = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve",
}
_HI_NUM = {
    "0": "शून्य", "1": "एक", "2": "दो", "3": "तीन", "4": "चार",
    "5": "पाँच", "6": "छह", "7": "सात", "8": "आठ", "9": "नौ",
    "10": "दस", "11": "ग्यारह", "12": "बारह",
}
_DEVA_DIGITS = "०१२३४५६७८९"
_DEVA_TO_ASCII = {d: str(i) for i, d in enumerate(_DEVA_DIGITS)}


def _ascii_digits(tok: str) -> str:
    """Fold Devanagari digits to ASCII so one number table covers both."""
    return "".join(_DEVA_TO_ASCII.get(c, c) for c in tok)


def normalize_numbers(text: str) -> str:
    """Spell out standalone integers so they read as words, not symbols.

    A run of digits surrounded by (or attached to) Latin letters spells in
    English; a run next to Devanagari, or standalone, spells in Hindi (the
    teacher's first language). Numbers above the small table are left as digits;
    the synthesizer reads them and the filter compares in number-word space
    anyway. Mixed alphanumerics like "10am" are split into "ten am" style.
    """
    out_tokens = []
    for tok in text.split():
        a = _ascii_digits(tok)
        if not re.search(r"\d", a):
            out_tokens.append(tok)
            continue
        # decide script context from the original token's letters
        en_ctx = bool(common._LATIN_RE.search(tok))
        hi_ctx = bool(common._DEVA_RE.search(tok))
        table = _HI_NUM if (hi_ctx and not en_ctx) else (
            _EN_NUM if en_ctx else _HI_NUM)
        # split the token into digit-runs and non-digit-runs, spell the runs
        parts = re.findall(r"\d+|\D+", a)
        rebuilt = []
        for part in parts:
            if part.isdigit() and part in table:
                rebuilt.append(table[part])
            elif part.isdigit():
                # leave long numbers as digits (filter handles digit/word align)
                rebuilt.append(part)
            else:
                rebuilt.append(part)
        out_tokens.append(" ".join(p for p in rebuilt if p))
    return " ".join(out_tokens)


# Strip leading/trailing punctuation from a token before judging its script.
# The Devanagari block (ऀ-ॿ) includes the danda । and double-danda ॥, so a
# naive "keep word chars and Devanagari" strip would treat "disaster।" as a
# mixed-script word (Latin body plus a Devanagari-range mark). We therefore strip
# the danda and common terminal punctuation explicitly.
_WORD_PUNCT_RE = re.compile(r"^[^\wऀ-ॿ]+|[।॥]+|[^\wऀ-ॿ]+$")


def _strip_word_punct(tok: str) -> str:
    """Strip surrounding punctuation (incl. Devanagari danda) from a token."""
    prev = None
    s = tok
    # apply repeatedly so a trailing "।" then a "," both come off
    while s != prev:
        prev = s
        s = _WORD_PUNCT_RE.sub("", s)
    return s


def enforce_per_word_script(text: str):
    """Reject tokens that mix Devanagari and Latin inside one word.

    The project rule is Hindi in Devanagari, English in Latin, no
    transliteration. A single token containing both scripts (for example a
    half-transliterated word) is a data error: the synthesizer mispronounces it
    and tag_languages cannot label it. Returns (clean_text, problems) where
    problems is a list of the offending tokens; empty means the text is clean.
    """
    problems = []
    for tok in text.split():
        # strip surrounding punctuation before judging the word body
        body = _strip_word_punct(tok)
        if not body:
            continue
        has_deva = bool(common._DEVA_RE.search(body))
        has_latin = bool(re.search(r"[A-Za-z]", body))
        if has_deva and has_latin:
            problems.append(tok)
    return text, problems


_WS_RE = re.compile(r"\s+")


def clean_text(raw: str):
    """Normalize one raw transcript and report any per-word script violation.

    NFC, whitespace collapse, number spell-out, then the per-word script check.
    Returns (text, problems). The caller drops a row whose problems list is
    non-empty (or, with --repair, keeps the row after dropping the bad tokens).
    """
    s = unicodedata.normalize("NFC", raw).strip()
    s = _WS_RE.sub(" ", s)
    s = normalize_numbers(s)
    s = _WS_RE.sub(" ", s).strip()
    _, problems = enforce_per_word_script(s)
    return s, problems


def repair_text(text: str) -> str:
    """Drop tokens that mix scripts so a salvageable row survives --repair."""
    keep = []
    for tok in text.split():
        body = _strip_word_punct(tok)
        has_deva = bool(common._DEVA_RE.search(body))
        has_latin = bool(re.search(r"[A-Za-z]", body))
        if body and has_deva and has_latin:
            continue
        keep.append(tok)
    return " ".join(keep)


# ----------------------------------------------------------------------------
# Sources: real seed text and LLM prompt files
# ----------------------------------------------------------------------------

def load_eval_seed(path: Path):
    """Yield (ref_orig, ref_surface, source_meta) from the real eval manifest.

    Used only as the text seed and to learn the target cmi_bin distribution. The
    real audio is never referenced. source_meta records provenance for flags.

    We carry through the manifest's own lang_tags / cs_density / cmi_bin. These
    were computed against the romanized surface form (ref_surface), where an
    English loanword written in Devanagari in ref_orig appears in Latin and so
    tags 'en'. Recomputing them from ref_orig with tag_languages would collapse
    every English-in-Devanagari token to 'hi' and flatten the whole real seed to
    cmi_bin 'none', erasing the high-code-switch tail we exist to seed. The
    manifest values are the load-bearing truth (verified to reproduce the
    1497-row eval distribution exactly), so we reuse them for real-seed rows.
    """
    rows = common.read_manifest(str(path))
    for r in rows:
        ref = r.get("ref_orig")
        if not ref or not ref.strip():
            continue
        yield {
            "text": ref,
            "ref_surface": r.get("ref_surface"),
            "domain": "real_seed",
            "source": "eval_spontaneous",
            "seed_cmi_bin": r.get("cmi_bin"),
            # carry the manifest's correct (surface-aligned) tags/scores through
            "seed_lang_tags": r.get("lang_tags"),
            "seed_cs_density": r.get("cs_density"),
        }


def load_prompt_files(prompts_dir: Path):
    """Yield transcript dicts from every prompts/*.jsonl file.

    Each JSONL line must carry a non-empty "text". Optional keys: "domain",
    "source", "ref_surface". Malformed lines are skipped with a warning so a
    single bad line never aborts a long ingest.
    """
    if not prompts_dir.exists():
        return
    for fp in sorted(prompts_dir.glob("*.jsonl")):
        with open(fp, encoding="utf-8") as fh:
            for ln_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print("  warn: %s:%d not valid json, skipped"
                          % (fp.name, ln_no), file=sys.stderr)
                    continue
                text = (obj.get("text") or "").strip()
                if not text:
                    continue
                yield {
                    "text": text,
                    "ref_surface": obj.get("ref_surface"),
                    "domain": obj.get("domain", "llm"),
                    "source": obj.get("source", fp.name),
                }


# Common English nouns to splice into Hindi carrier frames for the offline stub.
_STUB_EN_WORDS = [
    "battery", "interview", "schedule", "deadline", "presentation",
    "manager", "salary", "weekend", "traffic", "subscription", "delivery",
    "appointment", "discount", "warranty", "password", "notification",
    "playlist", "headphones", "shortcut", "dashboard", "checkout",
]
# Single-slot frames yield mostly low/med cs_density (one English island in a
# Hindi clause). They cover the low/med bins.
_STUB_HI_FRAMES = [
    "मेरा {w} फिर से खराब हो गया।",
    "कल का {w} कैंसिल करना पड़ेगा।",
    "इस {w} के बारे में मुझे कुछ समझ नहीं आ रहा।",
    "{w} को लेकर थोड़ी दिक्कत चल रही है।",
    "अगले हफ़्ते का {w} पहले से तय कर लेते हैं।",
    "उसने {w} वाली बात अभी तक नहीं बताई।",
    "ये {w} सच में बहुत slow चल रहा है।",
]

# Multi-slot frames alternate Hindi and English densely so the minority-language
# fraction crosses into the high bin (cmi > 0.40). Without these the stub corpus
# cannot satisfy --min-high-frac and collapses to a single bin. These take TWO
# distinct English words ({w} and {x}) to keep the alternation real, not a
# repeated token.
_STUB_HI_FRAMES_DENSE = [
    "मेरा {w} और {x} दोनों एक साथ crash हो गए, अब क्या करूँ।",
    "उसका {w} late था, फिर {x} भी miss हो गया, पूरा गड़बड़ है।",
    "ये {w} download हुआ पर {x} upload नहीं हुआ, network issue है।",
    "पहले {w} fix करो फिर {x} test करो, warna deploy fail होगा।",
    "मेरे {w} का battery और {x} का charger दोनों dead हैं।",
]


def synthesize_stub_transcripts(seed_rows, n_target: int):
    """Make placeholder Hinglish transcripts offline, no LLM and no network.

    This is the pluggable expansion's stub. It splices English nouns into Hindi
    carrier frames so the resulting text is genuinely code-switched and exercises
    every downstream contract (script check, cs_density, dedup, balance). Single-
    slot frames cover the low/med bins; the dense multi-slot frames (English and
    Hindi alternating several times per clause) push the minority-language
    fraction into the high bin so stub mode can exercise --min-high-frac too. It
    is NOT a quality text generator; it only proves the pipeline end to end when
    no prompts/*.jsonl exist yet. Real runs supply prompt files instead.
    """
    out = []
    i = 0
    nw = len(_STUB_EN_WORDS)
    while len(out) < n_target:
        w = _STUB_EN_WORDS[i % nw]
        # Interleave dense (high-CS) frames with single-slot frames so the stub
        # corpus spans all bins. Every third item is a dense one.
        if i % 3 == 2:
            dframe = _STUB_HI_FRAMES_DENSE[
                (i // 3) % len(_STUB_HI_FRAMES_DENSE)]
            x = _STUB_EN_WORDS[(i + 7) % nw]
            text = dframe.format(w=w, x=x)
        else:
            frame = _STUB_HI_FRAMES[(i // nw) % len(_STUB_HI_FRAMES)]
            text = frame.format(w=w)
        out.append({
            "text": text,
            "ref_surface": None,
            "domain": "stub",
            "source": "synthesize_stub",
        })
        i += 1
    return out


# ----------------------------------------------------------------------------
# Row construction
# ----------------------------------------------------------------------------

def build_row(text: str, ref_surface, domain: str, source: str,
              partition: str = "train", seed_lang_tags=None,
              seed_cs_density=None, seed_cmi_bin=None):
    """Build one schema-valid corpus row via common.new_row.

    For LLM and stub rows, lang_tags / cs_density / cmi_bin are computed from the
    cleaned ref_orig with tag_languages, which is correct because those rows are
    written with English in Latin (the project text rule). For real-seed rows we
    carry through the manifest's surface-aligned tags/scores instead: ref_orig
    spells English loanwords in Devanagari, so recomputing from it would tag them
    'hi' and collapse the row to cmi_bin 'none', wiping out the high-CS tail. The
    carried-through values are passed in by load_eval_seed and used when present.

    corpus_id and rep_4gram are always derived from the cleaned text. audio_path
    and accept stay null (this stage produces text only). The text utterance id
    is corpus_id; the per-clip utt_id is minted later by the synthesizer once a
    voice/speed is chosen. We use corpus_id as a stand-in utt_id here so the row
    is addressable and resumable.
    """
    if seed_lang_tags and seed_cmi_bin in _CMI_BINS:
        # real-seed row: trust the manifest's surface-aligned tags and scores
        tags = list(seed_lang_tags)
        cmi_bin = seed_cmi_bin
        cs_density = (seed_cs_density if seed_cs_density is not None
                      else common.lang_tags_to_cs_density(tags)[0])
    else:
        tags = common.tag_languages(text)
        cs_density, cmi_bin = common.lang_tags_to_cs_density(tags)
    cid = common.corpus_id_of(text)
    rep = common.ngram_repetition(text, n=4)

    flags = ["domain:%s" % domain, "source:%s" % source]
    is_real_seed = domain == "real_seed"
    dataset = "real_seed_text" if is_real_seed else "synthetic_hinglish_text"

    row = common.new_row(
        utt_id=cid,
        audio_path=None,
        ref_orig=text,
        ref_surface=ref_surface,
        ref_iso15919=None,
        cmi_bin=cmi_bin,
        cs_density=cs_density,
        lang_tags=tags,
        speaker_id="__pending__",
        duration_s=None,
        sha256=None,
        dataset=dataset,
        partition=partition,
        is_synthetic=not is_real_seed,
        license="synthetic_teacher_tts" if not is_real_seed
        else "real_seed_text_only",
        flags=flags,
        corpus_id=cid,
        speed=None,
        temp_tier=None,
        teacher=common.DEFAULT_TEACHER,
        rep_4gram=rep,
        accept=None,
    )
    return row


# ----------------------------------------------------------------------------
# Balancing across cmi_bin
# ----------------------------------------------------------------------------

_CMI_BINS = ("none", "low", "med", "high")


def target_bin_quota(corpus_rows, total: int, min_high_frac: float):
    """Per-cmi_bin row quotas: follow the corpus distribution but floor the tail.

    Quotas are derived from the SAME cmi_bin field that balance_rows selects on
    (the row's own cmi_bin), so the two bin systems can never diverge: a quota is
    only ever asked for a bin the corpus can actually fill. Bins keep their
    observed proportions, except the high bin, which the hard test depends on, is
    floored at min_high_frac of the total even when the observed share is smaller.
    Returns {bin: quota}.
    """
    dist = Counter(r.get("cmi_bin") for r in corpus_rows
                   if r.get("cmi_bin") in _CMI_BINS)
    seed_total = sum(dist.values()) or 1
    high_floor = int(round(total * min_high_frac))

    quota = {}
    quota["high"] = max(high_floor,
                        int(round(total * dist.get("high", 0) / seed_total)))
    remaining = total - quota["high"]
    rest = [b for b in _CMI_BINS if b != "high"]
    rest_total = sum(dist.get(b, 0) for b in rest) or 1
    for b in rest:
        quota[b] = int(round(remaining * dist.get(b, 0) / rest_total))
    # fix rounding drift onto the largest non-high bin
    drift = total - sum(quota.values())
    if drift != 0:
        big = max(rest, key=lambda b: quota[b])
        quota[big] = max(0, quota[big] + drift)
    return quota


def balance_rows(rows, quota):
    """Select rows to meet per-bin quotas, preserving order within each bin.

    Two passes. First, greedily fill each bin up to its quota in arrival order.
    Then, if some bins fell short (the quota asked for more than the bin had), the
    leftover budget is back-filled from bins that still have rows beyond their own
    quota, largest surplus first. Back-filling keeps the total selection close to
    the over-generate target instead of silently shrinking it, while the per-bin
    taken counts (returned and reported) still expose where the corpus could not
    honor the requested shape. Returns (selected_rows, per_bin_taken,
    per_bin_available).
    """
    by_bin = {b: [] for b in _CMI_BINS}
    for r in rows:
        b = r.get("cmi_bin")
        if b in by_bin:
            by_bin[b].append(r)

    taken = {}
    avail = {}
    selected = []
    for b in _CMI_BINS:
        avail[b] = len(by_bin[b])
        take = min(quota.get(b, 0), avail[b])
        taken[b] = take
        selected.extend(by_bin[b][:take])

    # back-fill the unmet budget from bins with rows left over their quota
    shortfall = sum(max(0, quota.get(b, 0) - taken[b]) for b in _CMI_BINS)
    if shortfall > 0:
        surplus = sorted(
            (b for b in _CMI_BINS if avail[b] - taken[b] > 0),
            key=lambda b: avail[b] - taken[b], reverse=True)
        for b in surplus:
            if shortfall <= 0:
                break
            extra = min(shortfall, avail[b] - taken[b])
            selected.extend(by_bin[b][taken[b]:taken[b] + extra])
            taken[b] += extra
            shortfall -= extra
    return selected, taken, avail


# ----------------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------------

def unique_word_ratio(texts):
    """Type-token ratio in compare-space: unique words over total words."""
    total = 0
    types = set()
    for t in texts:
        toks = common.tokenize_compare(t)
        total += len(toks)
        types.update(toks)
    return (len(types) / total) if total else 0.0


def switch_points_per_utt(rows):
    """Mean number of hi/en switch points per utterance (raw count, not density)."""
    if not rows:
        return 0.0
    total = 0
    for r in rows:
        seq = [t for t in (r.get("lang_tags") or []) if t in ("hi", "en")]
        total += sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    return total / len(rows)


def compute_stats(selected, real_seed_texts, raw_count, after_clean,
                  after_dedup, dedup_drops, quota, taken, avail,
                  rejected_script):
    """Assemble corpus_stats.json: diversity, balance, dedup, erosion baseline."""
    sel_texts = [r["ref_orig"] for r in selected]
    real_baseline_entropy = common.token_entropy(real_seed_texts)
    corpus_entropy = common.token_entropy(sel_texts)

    rep_vals = [r.get("rep_4gram", 0.0) or 0.0 for r in selected]
    mean_rep = sum(rep_vals) / len(rep_vals) if rep_vals else 0.0
    high_rep = sum(1 for v in rep_vals if v > 0.0)

    bin_dist = Counter(r.get("cmi_bin") for r in selected)
    domain_dist = Counter(
        next((f.split(":", 1)[1] for f in r.get("flags", [])
              if f.startswith("domain:")), "unknown")
        for r in selected)

    return {
        "counts": {
            "raw_ingested": raw_count,
            "after_clean": after_clean,
            "rejected_script_violations": rejected_script,
            "after_dedup": after_dedup,
            "dedup_drops": dedup_drops,
            "selected_after_balance": len(selected),
        },
        "diversity": {
            "unique_word_ratio_compare_space": round(unique_word_ratio(sel_texts), 4),
            "switch_points_per_utt": round(switch_points_per_utt(selected), 3),
            "token_entropy_corpus_bits": round(corpus_entropy, 4),
            "token_entropy_real_baseline_bits": round(real_baseline_entropy, 4),
            "token_entropy_ratio_vs_real": round(
                corpus_entropy / real_baseline_entropy, 4)
            if real_baseline_entropy else None,
        },
        "synthetic_erosion_alarms": {
            "mean_rep_4gram": round(mean_rep, 5),
            "utterances_with_any_4gram_repeat": high_rep,
            "note": ("token_entropy_ratio_vs_real well below 1.0, or a rising "
                     "mean_rep_4gram, is the paper #3 narrowing/collapse signal. "
                     "Re-checked at S4 and on student outputs at S5."),
        },
        "balance_cmi_bin": {
            "quota": quota,
            "taken": taken,
            "available": avail,
            "final_distribution": dict(bin_dist),
        },
        "balance_domain": dict(domain_dist),
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="S1 build the Hinglish text corpus (offline).")
    ap.add_argument("--eval-manifest", default=str(_DEFAULT_EVAL),
                    help="real eval set used as the text seed and distribution.")
    ap.add_argument("--prompts-dir", default=str(_DEFAULT_PROMPTS),
                    help="directory of LLM transcript *.jsonl files.")
    ap.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR),
                    help="where corpus.jsonl and corpus_stats.json are written.")
    ap.add_argument("--target-size", type=int, default=5000,
                    help="desired number of unique text utterances.")
    ap.add_argument("--over-generate", type=float, default=1.4,
                    help="multiply target by this to leave the filter slack.")
    ap.add_argument("--dedup-threshold", type=float, default=0.85,
                    help="MinHash Jaccard at/above which texts are duplicates.")
    ap.add_argument("--min-high-frac", type=float, default=0.20,
                    help="floor on the high-code-switch bin's share of output.")
    ap.add_argument("--include-real-seed-text", action="store_true",
                    help="copy real eval ref_orig into the corpus as text "
                         "(audio never used). Off by default to keep synthetic "
                         "and real text streams separate.")
    ap.add_argument("--seed-sample", type=int, default=0,
                    help="if >0, use only the first N real rows for seeding "
                         "(speeds up smoke tests).")
    ap.add_argument("--synthesize-stub", action="store_true",
                    help="when no prompt files exist, mint offline placeholder "
                         "transcripts so the chain runs end to end.")
    ap.add_argument("--repair", action="store_true",
                    help="instead of dropping a mixed-script row, strip the "
                         "offending tokens and keep it.")
    ap.add_argument("--config", default=None,
                    help="optional experiment config; fills target/over-generate "
                         "/dedup defaults from it when given.")
    args = ap.parse_args(argv)

    if args.config:
        cfg = common.load_config(args.config)
        args.target_size = cfg.get("target_corpus_size", args.target_size)
        args.over_generate = cfg.get("over_generate_factor", args.over_generate)
        args.dedup_threshold = cfg.get("dedup_threshold", args.dedup_threshold)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"
    stats_path = out_dir / "corpus_stats.json"

    over_target = int(round(args.target_size * args.over_generate))
    print("S1 build_corpus: target=%d over-generate=%.2f -> %d rows"
          % (args.target_size, args.over_generate, over_target))

    # ---- collect sources ----
    seed_rows = list(load_eval_seed(Path(args.eval_manifest)))
    if args.seed_sample and args.seed_sample > 0:
        seed_rows = seed_rows[:args.seed_sample]
    real_seed_texts = [r["text"] for r in seed_rows]
    print("  real seed transcripts: %d" % len(seed_rows))

    ingest = []
    if args.include_real_seed_text:
        ingest.extend(seed_rows)
    prompt_rows = list(load_prompt_files(Path(args.prompts_dir)))
    print("  prompt-file transcripts: %d" % len(prompt_rows))
    ingest.extend(prompt_rows)

    if len(ingest) < over_target and args.synthesize_stub:
        need = over_target - len(ingest)
        stub = synthesize_stub_transcripts(seed_rows, need)
        print("  stub transcripts minted (offline): %d" % len(stub))
        ingest.extend(stub)

    raw_count = len(ingest)
    if raw_count == 0:
        print("ERROR: no input transcripts. Add prompts/*.jsonl, pass "
              "--include-real-seed-text, or pass --synthesize-stub.",
              file=sys.stderr)
        return 2

    # ---- clean + per-word script enforcement ----
    cleaned = []
    rejected_script = 0
    for item in ingest:
        text, problems = clean_text(item["text"])
        if problems:
            if args.repair:
                text = repair_text(text)
                if not text.strip():
                    rejected_script += 1
                    continue
            else:
                rejected_script += 1
                continue
        if not text.strip():
            continue
        item = dict(item)
        item["text"] = text
        cleaned.append(item)
    after_clean = len(cleaned)
    print("  after clean/script-check: %d (rejected %d mixed-script)"
          % (after_clean, rejected_script))

    # ---- dedup (by corpus_id first, then near-dup MinHash) ----
    seen_cid = set()
    unique_by_id = []
    for item in cleaned:
        cid = common.corpus_id_of(item["text"])
        if cid in seen_cid:
            continue
        seen_cid.add(cid)
        unique_by_id.append(item)
    exact_drops = after_clean - len(unique_by_id)

    keep_idx = common.minhash_dedup(
        [it["text"] for it in unique_by_id], threshold=args.dedup_threshold)
    deduped = [unique_by_id[i] for i in keep_idx]
    near_drops = len(unique_by_id) - len(deduped)
    dedup_drops = exact_drops + near_drops
    after_dedup = len(deduped)
    print("  after dedup: %d (dropped %d exact + %d near)"
          % (after_dedup, exact_drops, near_drops))

    # ---- build rows ----
    rows = []
    for item in deduped:
        row = build_row(item["text"], item.get("ref_surface"),
                        item.get("domain", "unknown"),
                        item.get("source", "unknown"),
                        seed_lang_tags=item.get("seed_lang_tags"),
                        seed_cs_density=item.get("seed_cs_density"),
                        seed_cmi_bin=item.get("seed_cmi_bin"))
        problems = common.validate_row(row, profile="corpus")
        if problems:
            print("  warn: row failed validation, skipped: %s" % problems,
                  file=sys.stderr)
            continue
        rows.append(row)

    # ---- balance across cmi_bin and trim to the over-generate target ----
    # Quota and balance both key on the row's own cmi_bin so the two bin systems
    # cannot diverge (the bug where quota wanted med/high but balance only had
    # none/low). target_bin_quota therefore reads the corpus rows, not seed_rows.
    quota = target_bin_quota(rows, over_target, args.min_high_frac)
    selected, taken, avail = balance_rows(rows, quota)
    print("  balanced selection: %d rows (quota %s)" % (len(selected), quota))

    # Surface bin shortfalls loudly: a near-empty high tail must never pass
    # silently, because the high-CS tail is the real test for the student model.
    sel_dist = Counter(r.get("cmi_bin") for r in selected)
    for b in _CMI_BINS:
        if quota.get(b, 0) > 0 and taken.get(b, 0) < quota[b]:
            print("  WARN: cmi_bin '%s' under-filled: wanted %d, had %d"
                  % (b, quota[b], avail.get(b, 0)), file=sys.stderr)
    high_have = sel_dist.get("high", 0)
    high_frac = (high_have / len(selected)) if selected else 0.0
    if high_frac < args.min_high_frac:
        print("  WARN: high-CS share %.3f is below --min-high-frac %.3f "
              "(have %d high of %d). The corpus cannot honor the high-bin floor; "
              "supply richer high-CS prompt files (stub mode cannot reach it)."
              % (high_frac, args.min_high_frac, high_have, len(selected)),
              file=sys.stderr)

    # ---- write outputs ----
    common.write_manifest(str(corpus_path), selected)
    stats = compute_stats(selected, real_seed_texts, raw_count, after_clean,
                          after_dedup, dedup_drops, quota, taken, avail,
                          rejected_script)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    print("  wrote %s (%d rows)" % (corpus_path, len(selected)))
    print("  wrote %s" % stats_path)
    print("  token_entropy ratio vs real: %s"
          % stats["diversity"]["token_entropy_ratio_vs_real"])
    print("  final cmi_bin distribution: %s"
          % stats["balance_cmi_bin"]["final_distribution"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
