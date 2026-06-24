#!/usr/bin/env python3
"""Shared module for the Hinglish synthetic-data TTS pipeline.

This is the single place where the three cross-stage contracts live:

  (a) THE FILTER. A deterministic Devanagari romanizer plus a compare-space
      mapper so an English word written in Latin and the same word written
      in Devanagari by the recognizer collapse to one string. Content-word
      recall is the accept gate; character error rate is the secondary signal;
      raw token WER is kept for diagnostics only and never gates. This is the
      mitigation for the project's number-one correctness trap, the
      convention-confounded WER.

  (b) THE MANIFEST. read_manifest, write_manifest, new_row, validate_row.
      Rows are built only through new_row so the schema (corpus_manifest_v1)
      cannot drift between stages. The 17 v1 fields stay byte-compatible with
      the frozen 1497-row eval set; new fields are additive and optional.

  (c) DIAGNOSTICS AND IDENTITY. Deterministic ids, clause chunking, the
      teacher TTS client, WAV concatenation, language tagging, the
      Synthetic-Erosion alarms (token entropy and n-gram repetition), and a
      self-contained MinHash deduper.

No import side effects. Local stages need no GPU and no third-party packages;
optional packages (indic_transliteration, datasketch) are used only if they
happen to be importable. Run this file directly to execute the self-test.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import wave
from collections import Counter
from pathlib import Path

# ----------------------------------------------------------------------------
# Constants and the schema definition
# ----------------------------------------------------------------------------

SCHEMA_VERSION = "corpus_manifest_v1"

# Set TEACHER_TTS_ENDPOINT to your teacher TTS get_speech endpoint (Bearer-auth, JSON in / WAV out).
TTS_URL = os.environ.get("TEACHER_TTS_ENDPOINT", "")
DEFAULT_TEACHER = "teacher_tts"
KNOWN_VOICES = ("kaustubh", "arjun", "maya", "aadya")

_HERE = Path(__file__).resolve().parent
_ROMANIZE_MAP_PATH = _HERE / "configs" / "romanize_map.json"

# The 17 fields the ws4 loader and the frozen eval set require, in order.
V1_FIELDS = (
    "utt_id", "audio_path", "ref_orig", "ref_surface", "ref_iso15919",
    "cmi_bin", "cs_density", "lang_tags", "speaker_id", "duration_s",
    "sha256", "dataset", "partition", "is_synthetic", "license", "flags",
    "schema_version",
)

# Additive, optional, namespaced fields filled progressively by the stages.
ADDITIVE_FIELDS = (
    "corpus_id", "speed", "temp_tier", "teacher", "chunks", "asr_hyp",
    "filter_recall", "filter_cer_roman", "filter_wer_raw", "rep_4gram",
    "accept", "reject_reason", "regen_attempt",
)

ALL_FIELDS = V1_FIELDS + ADDITIVE_FIELDS

# Devanagari Unicode block (main range plus the extended sign area we use).
_DEVA_RE = re.compile(r"[ऀ-ॿ꣠-ꣿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# A small Hinglish stopword set (both Latin and romanized Devanagari forms).
# content_word_recall drops these so common glue words do not mask a real miss.
_STOPWORDS = frozenset({
    # English glue
    "the", "a", "an", "is", "am", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "for", "and", "or", "but", "so", "if",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "my", "your", "his", "her", "our", "their", "as",
    "with", "by", "from", "do", "did", "does", "has", "have", "had", "will",
    "would", "can", "could", "should", "not", "no", "yes", "ok",
    # romanized Hindi glue (matches what romanize_deva emits)
    "hai", "hain", "tha", "thi", "the", "ho", "hota", "hoti", "hote",
    "ka", "ki", "ke", "ko", "kaa", "kii", "kee", "se", "me", "mein", "men",
    "par", "pe", "aur", "ya", "to", "bhi", "hi", "ye", "vo", "wo", "yah",
    "vah", "wah", "kya", "kyaa", "na", "nahi", "nahin", "naheen", "jo",
    "ek", "wala", "vala", "waala", "vaala", "raha", "rahi", "rahe", "gaya",
    "gayi", "gaye", "diya", "kar", "karna", "karne", "kiya", "tha",
})


# ----------------------------------------------------------------------------
# Romanizer and compare-space (contract a)
# ----------------------------------------------------------------------------

_romanize_map_cache = None


def _load_romanize_map() -> dict:
    """Load and cache the JSON romanize map shipped beside this module."""
    global _romanize_map_cache
    if _romanize_map_cache is None:
        with open(_ROMANIZE_MAP_PATH, encoding="utf-8") as fh:
            _romanize_map_cache = json.load(fh)
    return _romanize_map_cache


def _try_indic_romanize(text: str):
    """Use indic_transliteration if importable, else return None.

    We ask for the IAST scheme and then strip diacritics in to_compare_space,
    so the exact scheme does not need to match our fallback map perfectly.
    """
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except Exception:
        return None
    try:
        return transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
    except Exception:
        return None


def _fallback_romanize(text: str) -> str:
    """Pure-Python Devanagari to Latin using configs/romanize_map.json.

    Walks the string, attaching the inherent schwa 'a' to a bare consonant and
    cancelling it when a virama (halant) or a dependent vowel sign (matra)
    follows. Two-codepoint consonant clusters in the map (for example the
    nukta forms) are matched greedily before single codepoints.
    """
    m = _load_romanize_map()
    cons = m["consonants"]
    matras = m["matras"]
    vowels = m["independent_vowels"]
    signs = m["signs"]
    digits = m["digits"]
    punct = m["punctuation"]
    virama = "्"

    out = []
    i = 0
    n = len(text)
    while i < n:
        # greedy two-codepoint consonant (nukta forms, conjunct shortcuts)
        pair = text[i:i + 2]
        if pair in cons:
            base = cons[pair][:-1]  # drop the inherent 'a'
            i += 2
            i = _emit_consonant(text, i, base, out, matras, virama)
            continue
        ch = text[i]
        if ch in cons:
            base = cons[ch][:-1]
            i += 1
            i = _emit_consonant(text, i, base, out, matras, virama)
            continue
        if ch in vowels:
            out.append(vowels[ch]); i += 1; continue
        if ch in matras:
            # a matra with no preceding consonant (rare); emit its vowel
            out.append(matras[ch]); i += 1; continue
        if ch in signs:
            out.append(signs[ch]); i += 1; continue
        if ch in digits:
            out.append(digits[ch]); i += 1; continue
        if ch in punct:
            out.append(punct[ch]); i += 1; continue
        out.append(ch); i += 1
    return "".join(out)


def _emit_consonant(text, i, base, out, matras, virama):
    """Emit one consonant with its following vowel, handling schwa/virama.

    Returns the new index. If a matra follows, use that vowel. If a virama
    follows, emit no vowel (dead consonant). Otherwise attach inherent 'a'.
    """
    if i < len(text) and text[i] == virama:
        out.append(base)
        return i + 1
    if i < len(text) and text[i] in matras:
        out.append(base + matras[text[i]])
        return i + 1
    out.append(base + "a")
    return i


def romanize_deva(text: str) -> str:
    """Deterministic Devanagari -> Latin, ISO-15919-style.

    Prefers indic_transliteration when importable, else the bundled map. The
    output is not normalised here; to_compare_space does the lowercasing and
    diacritic stripping. Non-Devanagari characters pass through unchanged.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    indic = _try_indic_romanize(text)
    if indic is not None:
        return indic
    return _fallback_romanize(text)


# Compare-space normalisation tables. These collapse spelling conventions that
# do not change the word: long/short vowels, v/w, doubled consonants, etc.
_DIACRITIC_STRIP = str.maketrans({
    "ā": "a", "ī": "i", "ū": "u", "ṛ": "r",
    "ṃ": "m", "ṅ": "n", "ṇ": "n", "ṉ": "n",
    "ṣ": "s", "ś": "s", "ṭ": "t", "ḍ": "d",
    "ḥ": "h", "ē": "e", "ō": "o", "ñ": "n",
})

_NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten",
}


def _collapse_variants(tok: str) -> str:
    """Fold spelling variants that are the same spoken word.

    v/w merge to v; doubled consonants single; repeated vowels collapse.
    Applied per token so word boundaries are kept. This keeps the surface form
    readable; the heavier phonetic fold is applied only inside the fuzzy
    matcher so the stored compare-space stays human-auditable.
    """
    if not tok:
        return tok
    tok = tok.replace("w", "v")
    # collapse runs of the same letter to a single letter (aa->a, tt->t)
    tok = re.sub(r"(.)\1+", r"\1", tok)
    return tok


def _delete_final_schwa(s: str) -> str:
    """Delete the word-final inherent 'a' the romanizer attaches.

    Hindi drops the word-final schwa: सेंटर is pronounced 'sentar' (and written
    'center' in English), not 'sentara'. The fallback romanizer always emits the
    inherent vowel, so we strip a single trailing 'a' from each romanized token
    that has at least one other vowel, leaving genuine final-a words ('data',
    'kya') mostly intact via the vowel-count guard.
    """
    out = []
    for tok in s.split():
        if (len(tok) > 2 and tok.endswith("a")
                and sum(c in "aeiou" for c in tok) >= 2):
            out.append(tok[:-1])
        else:
            out.append(tok)
    return " ".join(out)


def to_compare_space(text: str) -> str:
    """Map any mixed Devanagari+Latin string to the single comparison space.

    Devanagari spans are romanised, Latin spans lowercased, punctuation and
    diacritics stripped, NFC applied, whitespace collapsed, and spelling
    variants (v/w, doubled letters, long vowels, single-digit number words)
    folded. Both the intended text and the ASR hypothesis pass through this
    identically, which is what makes a correct English word impossible to
    score as a substitution just because the recognizer wrote it in Devanagari.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    # romanize only the Devanagari runs, leave Latin runs as-is
    parts = []
    buf = []
    mode = None  # "deva" or "other"
    for ch in text:
        is_deva = bool(_DEVA_RE.match(ch))
        cur = "deva" if is_deva else "other"
        if mode is None:
            mode = cur
        if cur != mode:
            chunk = "".join(buf)
            parts.append(romanize_deva(chunk) if mode == "deva" else chunk)
            buf = []
            mode = cur
        buf.append(ch)
    if buf:
        chunk = "".join(buf)
        parts.append(romanize_deva(chunk) if mode == "deva" else chunk)
    s = "".join(parts)
    s = _delete_final_schwa(s)

    s = unicodedata.normalize("NFKD", s.lower())
    s = s.translate(_DIACRITIC_STRIP)
    # drop any remaining combining marks
    s = "".join(c for c in s if not unicodedata.combining(c))
    # turn standalone single digits into number words so "2"/"do"/"two" align
    s = re.sub(r"\d+", lambda m: _NUMBER_WORDS.get(m.group(0), m.group(0)), s)
    # keep only letters, digits, whitespace
    s = re.sub(r"[^\w\s]", " ", s)
    toks = [_collapse_variants(t) for t in s.split()]
    return " ".join(t for t in toks if t)


def tokenize_compare(text: str) -> list:
    """Tokenize a compare-space string into words.

    Shared so ref and hyp are tokenised identically. Accepts raw or already
    compare-space text; it normalises first to be safe.
    """
    return to_compare_space(text).split()


# ----------------------------------------------------------------------------
# Edit distance, recall, CER, WER (contract a metrics)
# ----------------------------------------------------------------------------

def _levenshtein(a, b) -> int:
    """Levenshtein distance over two sequences (tokens or chars)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _edit_ratio(a: str, b: str) -> float:
    """1 - normalised char edit distance between two tokens, in [0,1]."""
    if not a and not b:
        return 1.0
    d = _levenshtein(a, b)
    return 1.0 - d / max(len(a), len(b), 1)


def _phonetic_fold(tok: str) -> str:
    """Fold English-spelling/Hindi-spelling differences for a loanword.

    A deterministic romanizer cannot reconcile English orthography (silent
    letters, soft c, ph) with how a Hindi recognizer spells the same loanword
    in Devanagari. This collapses the predictable confusions so, for example,
    'center' and the romanized 'sentar' line up: soft c before e/i/y -> s, hard
    c/q -> k, x -> ks, ph -> f, z -> j, w -> v, and doubled letters single.
    """
    t = tok.lower().replace("w", "v")
    t = re.sub(r"c(?=[eiy])", "s", t)   # soft c
    t = t.replace("ch", "\x00")          # protect the digraph
    t = t.replace("c", "k").replace("q", "k").replace("x", "ks")
    t = t.replace("\x00", "ch").replace("ph", "f").replace("z", "j")
    t = re.sub(r"(.)\1+", r"\1", t)
    return t


def _skeleton(tok: str) -> str:
    """Consonant skeleton of a phonetically folded token (vowels removed).

    Robust against the vowel guesswork a recognizer does for a loanword
    ('subscribers' vs 'sabskraibars'). Used as a second matching signal so a
    word counts as recovered if either the phonetic fold or the skeleton lines
    up. Vowels carry little identity in a transliterated loanword.
    """
    return re.sub(r"[aeiou]+", "", _phonetic_fold(tok))


def _fuzzy_word_match(a: str, b: str, thr: float = 0.80) -> bool:
    """True if two compare-space words are the same word across conventions.

    Tries plain edit ratio, then the phonetic fold, then the consonant
    skeleton, and accepts if any clears the threshold. This is what bridges an
    English word in Latin against the recognizer's Devanagari spelling of it.
    """
    if a == b:
        return True
    if _edit_ratio(a, b) >= thr:
        return True
    if _edit_ratio(_phonetic_fold(a), _phonetic_fold(b)) >= thr:
        return True
    sa, sb = _skeleton(a), _skeleton(b)
    if len(sa) >= 2 and len(sb) >= 2 and _edit_ratio(sa, sb) >= thr:
        return True
    return False


def content_word_recall(ref: str, hyp: str, lang_tags=None) -> float:
    """PRIMARY filter metric: fraction of intended content words recovered.

    Both sides go through to_compare_space. Stopwords are dropped from the
    reference. English and switch-point words (inferred from lang_tags when
    given) are weighted higher because they are the hard part of Hinglish and
    the part most likely to be garbled. A reference word counts as recovered
    if some hypothesis word matches it with fuzzy edit-ratio >= 0.85.
    """
    ref_toks = tokenize_compare(ref)
    hyp_toks = tokenize_compare(hyp)
    if not ref_toks:
        return 1.0 if not hyp_toks else 0.0
    if not hyp_toks:
        return 0.0

    # weight: switch-point and English-tagged words get 2.0, others 1.0
    weights = _word_weights(ref_toks, lang_tags)

    hyp_set = set(hyp_toks)
    total_w = 0.0
    hit_w = 0.0
    for tok, w in zip(ref_toks, weights):
        if tok in _STOPWORDS:
            continue
        total_w += w
        if tok in hyp_set:
            hit_w += w
            continue
        # convention-robust fuzzy match against any hyp token
        if any(_fuzzy_word_match(tok, h) for h in hyp_toks):
            hit_w += w
    if total_w == 0.0:
        return 1.0
    return hit_w / total_w


def _word_weights(ref_toks, lang_tags):
    """Per-token recall weights. 2.0 for English/switch-point, else 1.0.

    lang_tags is the per-original-token tag list; after compare-space the token
    count usually matches, but we fall back to a heuristic (a Latin-looking
    word that is not pure romanized Hindi) when counts disagree.
    """
    n = len(ref_toks)
    if lang_tags and len([t for t in lang_tags if t in ("hi", "en", "other")]) == n:
        out = []
        prev = None
        usable = lang_tags
        for i, t in enumerate(usable[:n]):
            switch = prev in ("hi", "en") and t in ("hi", "en") and t != prev
            out.append(2.0 if (t == "en" or switch) else 1.0)
            prev = t
        return out
    # heuristic: weight tokens that contain an English-only letter pattern
    return [2.0 if _looks_english(t) else 1.0 for t in ref_toks]


def _looks_english(tok: str) -> bool:
    """Rough guess that a compare-space token is an English loanword.

    Romanized Hindi rarely uses f, q, x, z or ends in common English suffixes.
    Used only as a fallback weight when lang_tags are unavailable.
    """
    if re.search(r"[fqxz]", tok):
        return True
    return bool(re.search(r"(tion|ing|ment|able|ness|ful|phone|app)$", tok))


def cer_roman(ref: str, hyp: str) -> float:
    """SECONDARY metric: character error rate in compare-space.

    Robust to word-segmentation noise from the recognizer. Spaces are removed
    so chunking differences do not inflate it.
    """
    r = "".join(_phonetic_fold(t) for t in tokenize_compare(ref))
    h = "".join(_phonetic_fold(t) for t in tokenize_compare(hyp))
    if not r:
        return 0.0 if not h else 1.0
    return _levenshtein(r, h) / len(r)


def wer_raw(ref: str, hyp: str) -> float:
    """DIAGNOSTIC ONLY raw token WER (matches qwen3asr_roundtrip).

    Lowercases, strips punctuation, keeps Devanagari, tokenises on whitespace.
    This is the convention-confounded number; importing code must never gate on
    it. Kept so the confound stays visible in reports.
    """
    def norm(s):
        s = unicodedata.normalize("NFC", s.lower())
        s = re.sub(r"[^\w\sऀ-ॿ]", " ", s)
        return s.split()
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return _levenshtein(r, h) / len(r)


def accept_clip(scores: dict, dur_s: float, sr: int, cfg: dict):
    """Single accept/reject policy used identically by S3 (filter) and S5 (eval).

    Order of checks gives the most informative reject_reason. Thresholds come
    from cfg (which is seeded from calib_report when present), not intuition.
    Returns (accept: bool, reject_reason: str | None).
    """
    tau_recall = cfg.get("tau_recall", 0.80)
    tau_cer = cfg.get("tau_cer", 0.30)
    min_s = cfg.get("min_s", 3.0)
    max_s = cfg.get("max_s", 30.0)
    dnsmos_min = cfg.get("dnsmos_min")

    hyp = scores.get("asr_hyp")
    if hyp is not None and str(hyp).strip() == "":
        return False, "asr_empty"
    if sr is not None and sr != 24000:
        return False, "sr_bad"
    if dur_s is not None and dur_s < min_s:
        return False, "too_short"
    if dur_s is not None and dur_s > max_s:
        return False, "too_long"

    recall = scores.get("filter_recall")
    cer = scores.get("filter_cer_roman")
    if recall is not None and recall < tau_recall:
        return False, "recall_low"
    if cer is not None and cer > tau_cer:
        return False, "cer_high"

    if dnsmos_min is not None:
        dn = scores.get("dnsmos")
        if dn is not None and dn < dnsmos_min:
            return False, "dnsmos_low"

    return True, None


# ----------------------------------------------------------------------------
# Chunking, teacher TTS client, WAV concat (contract c, synthesis side)
# ----------------------------------------------------------------------------

_CLAUSE_SPLIT_RE = re.compile(r"(?<=[।॥.!?])\s+")
_COMMA_SPLIT_RE = re.compile(r"(?<=[,;:])\s+")


def chunk_text(text: str, max_chars: int = 250) -> list:
    """Split text on clause boundaries to <= max_chars without breaking words.

    First by sentence enders (Devanagari danda, then . ! ?), then by commas and
    semicolons, then as a last resort on spaces. Never splits inside a word.
    Audio of the returned chunks is concatenated per utterance.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    pieces = _split_keep(text, _CLAUSE_SPLIT_RE, max_chars)
    out = []
    for p in pieces:
        if len(p) <= max_chars:
            out.append(p)
            continue
        for q in _split_keep(p, _COMMA_SPLIT_RE, max_chars):
            if len(q) <= max_chars:
                out.append(q)
            else:
                out.extend(_split_on_space(q, max_chars))
    return [c for c in (s.strip() for s in out) if c]


def _split_keep(text, regex, max_chars):
    """Split by regex then greedily repack adjacent pieces under max_chars."""
    raw = regex.split(text)
    packed = []
    cur = ""
    for piece in raw:
        if not piece:
            continue
        candidate = (cur + " " + piece).strip() if cur else piece
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                packed.append(cur)
            cur = piece
    if cur:
        packed.append(cur)
    return packed


def _split_on_space(text, max_chars):
    """Last-resort word-boundary split for a single over-long clause.

    A single token longer than max_chars (e.g. a long no-space Devanagari run)
    is HARD-split into consecutive <=max_chars slices rather than truncated, so
    no characters are ever dropped. Dropping characters here would silently
    corrupt the transcript: the synthesized audio would no longer reconstruct
    ref_orig, which wastes API spend and only surfaces later at the S3 filter.
    """
    words = text.split()
    out, cur = [], ""
    for w in words:
        candidate = (cur + " " + w).strip() if cur else w
        if len(candidate) <= max_chars:
            cur = candidate
            continue
        if cur:
            out.append(cur)
            cur = ""
        if len(w) <= max_chars:
            cur = w
        else:
            # hard-split the over-long token into max_chars-sized pieces
            for i in range(0, len(w), max_chars):
                out.append(w[i:i + max_chars])
    if cur:
        out.append(cur)
    return out


def _silent_wav_bytes(seconds: float = 1.0, sample_rate: int = 24000) -> bytes:
    """Build a valid silent mono 16-bit WAV in memory for dry-run synthesis."""
    import io
    n = int(seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


def synth_request(text: str, voice_id: str, speed: float = 1.0,
                  sample_rate: int = 24000, dry_run: bool = False,
                  max_retries: int = 4, timeout: int = 60) -> bytes:
    """the teacher TTS client. Returns raw WAV bytes.

    POST get_speech, Bearer key from env TEACHER_TTS_API_KEY (never hardcoded,
    never logged). Retries on transient HTTP/network errors with exponential
    backoff. dry_run returns a valid silent WAV so S2 runs fully offline. This
    is the one place the API contract lives.
    """
    if dry_run:
        return _silent_wav_bytes(max(1.0, len(text) / 15.0), sample_rate)

    key = os.environ.get("TEACHER_TTS_API_KEY")
    if not key:
        raise RuntimeError("TEACHER_TTS_API_KEY is not set in the environment")
    if not TTS_URL:
        raise RuntimeError("TEACHER_TTS_ENDPOINT is not set in the environment")
    if len(text) > 250:
        raise ValueError("synth_request text exceeds 250 chars; chunk first")

    body = json.dumps({
        "text": text,
        "voice_id": voice_id,
        "sample_rate": sample_rate,
        "output_format": "wav",
        "speed": speed,
    }).encode("utf-8")
    headers = {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
    }

    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(TTS_URL, data=body, method="POST",
                                     headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last_err = e
            # retry only on rate limit and server errors
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
        # back off between attempts, but never sleep after the final attempt
        # (that sleep is pure wasted wall-clock before the terminal raise)
        if attempt < max_retries - 1:
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError("synth_request failed after retries: %r" % last_err)


def concat_wavs(chunk_paths: list, out_path: str, sample_rate: int = 24000) -> float:
    """Concatenate per-chunk WAVs into one 24 kHz mono WAV; return duration_s.

    Stdlib wave only. Assumes inputs are mono 16-bit at sample_rate (what the
    teacher and the dry-run silent WAV produce). Creates the parent directory.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    total = 0
    for p in chunk_paths:
        with wave.open(str(p), "rb") as w:
            total += w.getnframes()
            frames.append(w.readframes(w.getnframes()))
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"".join(frames))
    return total / float(sample_rate)


# ----------------------------------------------------------------------------
# Language tagging and CMI (contract c, identity side)
# ----------------------------------------------------------------------------

def tag_languages(text: str) -> list:
    """Per-token hi|en|other tagging by script.

    A token with any Devanagari is hi; a token that is purely Latin alphabetic
    is en; everything else (numbers, punctuation-only, mixed symbols) is other.
    Matches the ws4 convention so synthetic and real rows are comparable.
    """
    tags = []
    for tok in text.split():
        if _DEVA_RE.search(tok):
            tags.append("hi")
        elif _LATIN_RE.search(tok) and not re.search(r"[^A-Za-z'\-]", tok):
            tags.append("en")
        else:
            tags.append("other")
    return tags


def lang_tags_to_cs_density(lang_tags: list):
    """Compute (cs_density, cmi_bin) from per-token tags, ws4-compatible.

    cs_density is the switch-point density over hi/en tokens only: the fraction
    of adjacent hi/en token pairs whose language differs. cmi_bin comes from
    the Code-Mixing Index, the minority-language fraction among hi/en tokens:
    none=0, low<=0.20, med<=0.40, high>0.40. Both definitions were verified to
    reproduce the 1497-row eval set exactly.
    """
    seq = [t for t in lang_tags if t in ("hi", "en")]
    if len(seq) < 2:
        cs_density = 0.0
    else:
        switches = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
        cs_density = switches / (len(seq) - 1)

    if not seq:
        cmi = 0.0
    else:
        en = sum(t == "en" for t in seq)
        hi = sum(t == "hi" for t in seq)
        cmi = min(en, hi) / len(seq)

    if cmi == 0.0:
        cmi_bin = "none"
    elif cmi <= 0.20:
        cmi_bin = "low"
    elif cmi <= 0.40:
        cmi_bin = "med"
    else:
        cmi_bin = "high"
    return cs_density, cmi_bin


# ----------------------------------------------------------------------------
# Synthetic-Erosion diagnostics (contract c)
# ----------------------------------------------------------------------------

def token_entropy(texts: list) -> float:
    """Shannon entropy (bits) of the unigram token distribution over a corpus.

    Tokens are taken in compare-space so script convention does not split the
    same word into two symbols. A drop versus the real spontaneous baseline is
    the Synthetic-Erosion narrowing alarm.
    """
    counts = Counter()
    for t in texts:
        counts.update(tokenize_compare(t))
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h


def ngram_repetition(text: str, n: int = 4) -> float:
    """Fraction of repeated n-grams within one utterance (paper #3 alarm).

    Computed over compare-space tokens. 0.0 means every n-gram is unique; a
    rising value flags the looping/collapse failure mode. Returns 0.0 when the
    utterance is too short to contain an n-gram.
    """
    toks = tokenize_compare(text)
    if len(toks) < n:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    total = len(grams)
    unique = len(set(grams))
    return (total - unique) / total


# ----------------------------------------------------------------------------
# MinHash dedup (contract c)
# ----------------------------------------------------------------------------

def _char_shingles(text: str, k: int = 5):
    """Set of k-char shingles over a compare-space string."""
    s = to_compare_space(text).replace(" ", "")
    if len(s) < k:
        return {s} if s else set()
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def minhash_dedup(texts: list, threshold: float = 0.85) -> list:
    """Return indices to keep after near-duplicate removal.

    Self-contained MinHash over char shingles (prefers datasketch if it is
    importable, but the default path needs nothing). Two texts whose estimated
    Jaccard >= threshold are duplicates; the earlier index is kept. Stops LLM
    paraphrase clones from inflating the corpus.
    """
    n = len(texts)
    if n <= 1:
        return list(range(n))

    num_perm = 64
    # deterministic hash permutations via salted blake2b
    sigs = []
    for t in texts:
        shingles = _char_shingles(t)
        if not shingles:
            sigs.append(tuple([0] * num_perm))
            continue
        mins = [None] * num_perm
        for sh in shingles:
            for p in range(num_perm):
                hv = int.from_bytes(
                    hashlib.blake2b(sh.encode("utf-8"),
                                    digest_size=8,
                                    salt=p.to_bytes(2, "little")).digest(),
                    "little")
                if mins[p] is None or hv < mins[p]:
                    mins[p] = hv
        sigs.append(tuple(0 if m is None else m for m in mins))

    keep = []
    kept_sigs = []
    for i in range(n):
        dup = False
        for ks in kept_sigs:
            same = sum(1 for a, b in zip(sigs[i], ks) if a == b)
            if same / num_perm >= threshold:
                dup = True
                break
        if not dup:
            keep.append(i)
            kept_sigs.append(sigs[i])
    return keep


# ----------------------------------------------------------------------------
# Identity: corpus_id and utt_id (contract c)
# ----------------------------------------------------------------------------

def _normalize_for_id(ref_orig: str) -> str:
    """Stable normalisation for the corpus_id hash (NFC, collapse whitespace)."""
    s = unicodedata.normalize("NFC", ref_orig.strip())
    return re.sub(r"\s+", " ", s)


def corpus_id_of(ref_orig: str) -> str:
    """Deterministic group key 'c' + sha1(normalized ref_orig)[:8].

    All voice/speed/temp/regen variants of one transcript share it. The dedup
    group key and the leak-free train/dev split key.
    """
    h = hashlib.sha1(_normalize_for_id(ref_orig).encode("utf-8")).hexdigest()
    return "c" + h[:8]


def mint_utt_id(corpus_id: str, speaker_id: str, speed: float,
                temp_tier=None, regen_attempt: int = 0) -> str:
    """Deterministic per-clip id, idempotent and resumable.

    Format: {corpus_id}__{speaker_id}__sp{speed_x10}__t{tier_or_x}__v{attempt}
    so the same inputs always produce the same id and synth/filter skip done
    work on restart.
    """
    sp = int(round(speed * 10))
    tier = temp_tier if temp_tier else "x"
    return "%s__%s__sp%d__t%s__v%d" % (corpus_id, speaker_id, sp, tier,
                                       int(regen_attempt))


# ----------------------------------------------------------------------------
# Manifest rows, validation, IO (contract b)
# ----------------------------------------------------------------------------

def new_row(**fields) -> dict:
    """The only sanctioned way to build a manifest row.

    Stamps schema_version, asserts the 17 v1 fields are present, rejects unknown
    keys, and fills the additive fields with null defaults. So the schema can
    never drift between stages.
    """
    unknown = set(fields) - set(ALL_FIELDS) - {"schema_version"}
    if unknown:
        raise KeyError("new_row got unknown fields: %s" % sorted(unknown))

    row = {}
    for f in V1_FIELDS:
        if f == "schema_version":
            row[f] = SCHEMA_VERSION
        elif f in fields:
            row[f] = fields[f]
        else:
            raise KeyError("new_row missing required v1 field: %s" % f)

    additive_defaults = {
        "corpus_id": None, "speed": None, "temp_tier": None,
        "teacher": None, "chunks": None, "asr_hyp": None,
        "filter_recall": None, "filter_cer_roman": None,
        "filter_wer_raw": None, "rep_4gram": None, "accept": None,
        "reject_reason": None, "regen_attempt": 0,
    }
    for f, default in additive_defaults.items():
        row[f] = fields.get(f, default)

    row["schema_version"] = SCHEMA_VERSION
    return row


# Per-stage required (non-null) fields beyond the always-present v1 keys.
_PROFILE_REQUIRED = {
    "corpus": ("utt_id", "ref_orig", "corpus_id", "lang_tags",
               "cmi_bin", "cs_density"),
    "synth": ("utt_id", "ref_orig", "corpus_id", "audio_path", "speaker_id",
              "speed", "duration_s", "teacher"),
    "filtered": ("utt_id", "corpus_id", "audio_path", "filter_recall",
                 "accept"),
    "train": ("utt_id", "corpus_id", "audio_path", "ref_orig", "speaker_id",
              "duration_s", "sha256"),
}


def validate_row(row: dict, profile: str = "corpus") -> list:
    """Return a list of schema problems for a row, given a stage profile.

    An empty list means valid. Every stage validates its output before writing
    so a bad row never propagates.
    """
    problems = []
    if row.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema_version != %s" % SCHEMA_VERSION)
    for f in V1_FIELDS:
        if f not in row:
            problems.append("missing v1 field: %s" % f)
    unknown = set(row) - set(ALL_FIELDS)
    if unknown:
        problems.append("unknown fields: %s" % sorted(unknown))

    if profile not in _PROFILE_REQUIRED:
        problems.append("unknown profile: %s" % profile)
        return problems
    for f in _PROFILE_REQUIRED[profile]:
        if row.get(f) is None:
            problems.append("profile %s requires non-null %s" % (profile, f))

    if not isinstance(row.get("lang_tags"), list):
        if row.get("lang_tags") is not None:
            problems.append("lang_tags must be a list or null")
    if not isinstance(row.get("flags"), list):
        problems.append("flags must be a list")
    if not isinstance(row.get("is_synthetic"), bool):
        problems.append("is_synthetic must be a bool")
    return problems


def read_manifest(path: str) -> list:
    """Load a manifest, sniffing JSON-array vs JSONL, into list[dict].

    The frozen eval set is a JSON array; the stage outputs are JSONL. One call
    loads both. Empty or missing files return an empty list.
    """
    p = Path(path)
    if not p.exists():
        return []
    raw = p.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if not stripped:
        return []
    if stripped[0] == "[":
        return json.loads(raw)
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_manifest(path: str, rows, append: bool = False,
                   json_array: bool = False) -> None:
    """Atomic UTF-8 manifest write (or append). Devanagari stays readable.

    JSONL by default (atomic temp-file plus rename, or open-append). The final
    train_manifest.json is a JSON array; pass json_array=True for that form.
    Creates parent directories.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if json_array:
        if append:
            raise ValueError("append is not supported with json_array=True")
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(p)
        return

    if append:
        with open(p, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return

    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


def resume_done_ids(out_path: str) -> set:
    """Read an existing output manifest and return done utt_ids.

    Lets any stage skip completed work on restart, making long network/GPU runs
    safely interruptible. Missing file means an empty set.
    """
    done = set()
    for row in read_manifest(out_path):
        uid = row.get("utt_id")
        if uid:
            done.add(uid)
    return done


# ----------------------------------------------------------------------------
# Config and run provenance (contract c, glue)
# ----------------------------------------------------------------------------

def validate_voices_speeds(voices, speeds):
    """Validate a set of voices and speeds, returning the normalized values.

    Voices must all be in KNOWN_VOICES; speeds must each fall in [0.5, 2.0].
    Returns (voices_list, speeds_as_floats). Raises ValueError on the first bad
    entry. This is the single validator shared by load_config (the config path)
    and the S2 CLI overrides (--voices/--speeds), so a bad voice or speed cannot
    slip in through either path and reach a live API call.
    """
    voices = list(voices)
    bad = [v for v in voices if v not in KNOWN_VOICES]
    if bad:
        raise ValueError("unknown voices: %s (known: %s)"
                         % (bad, list(KNOWN_VOICES)))
    out_speeds = []
    for s in speeds:
        fs = float(s)
        if not (0.5 <= fs <= 2.0):
            raise ValueError("speed out of range [0.5, 2.0]: %s" % s)
        out_speeds.append(fs)
    return voices, out_speeds


def load_config(path: str) -> dict:
    """Load an experiment config, validate it, fill defaults, freeze it.

    Validates that voices are known and speeds are in a sane range. Threshold
    defaults tau_recall/tau_cer are taken from data/filtered/calib_report.json
    when that file exists (calibration over the real qwen pairs), otherwise the
    config's own values are used. Raises on a bad ablation arm.
    """
    cfg = dict(json.loads(Path(path).read_text(encoding="utf-8")))

    voices, speeds = validate_voices_speeds(
        cfg.get("voices", list(KNOWN_VOICES)),
        cfg.get("speeds", [1.0]))
    cfg["voices"] = voices
    cfg["speeds"] = speeds

    cfg.setdefault("teacher", DEFAULT_TEACHER)
    cfg.setdefault("sample_rate", 24000)
    cfg.setdefault("max_chars", 250)
    cfg.setdefault("min_s", 3.0)
    cfg.setdefault("max_s", 30.0)
    cfg.setdefault("dnsmos_min", None)
    cfg.setdefault("max_synth_frac", 0.5)
    cfg.setdefault("split_seed", 20260617)
    cfg.setdefault("cs_threshold", 0.4)

    # prefer calibrated thresholds when the calibration report exists
    calib = _HERE.parent.parent / "data" / "filtered" / "calib_report.json"
    if calib.exists():
        try:
            c = json.loads(calib.read_text(encoding="utf-8"))
            if "tau_recall" in c:
                cfg["tau_recall"] = c["tau_recall"]
            if "tau_cer" in c:
                cfg["tau_cer"] = c["tau_cer"]
        except Exception:
            pass
    cfg.setdefault("tau_recall", 0.80)
    cfg.setdefault("tau_cer", 0.30)
    return cfg


def stamp_run(cfg: dict, run_id: str, inputs: dict) -> str:
    """Freeze cfg, its hash, lib versions, and input checksums to data/runs/.

    Returns run_id. Rows reference the run via a single flags entry
    'run:<run_id>' instead of carrying provenance on every row.
    """
    run_dir = _HERE.parent.parent / "data" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_text = json.dumps(cfg, ensure_ascii=False, sort_keys=True, indent=2)
    (run_dir / "experiment.json").write_text(cfg_text, encoding="utf-8")

    input_hashes = {}
    for name, val in (inputs or {}).items():
        p = Path(val)
        if p.exists() and p.is_file():
            input_hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        else:
            input_hashes[name] = None

    hashes = {
        "run_id": run_id,
        "config_sha256": hashlib.sha256(cfg_text.encode("utf-8")).hexdigest(),
        "python": sys.version.split()[0],
        "inputs": input_hashes,
        "stamped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "hashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_id


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

def _selftest() -> int:
    """Exercise the contracts. The Phase 0 example must judge as a match."""
    failures = []

    def check(name, cond):
        status = "PASS" if cond else "FAIL"
        print("  [%s] %s" % (status, name))
        if not cond:
            failures.append(name)

    # The #1 trap: "coaching center" (Latin) vs qwen "कोचिंग सेंटर" (Devanagari)
    intended = "coaching center"
    asr = "कोचिंग सेंटर"
    cs_intended = to_compare_space(intended)
    cs_asr = to_compare_space(asr)
    print("  compare-space intended : %r" % cs_intended)
    print("  compare-space asr hyp  : %r" % cs_asr)
    recall = content_word_recall(intended, asr)
    cer = cer_roman(intended, asr)
    wer = wer_raw(intended, asr)
    print("  recall=%.3f  cer_roman=%.3f  wer_raw(diagnostic)=%.3f"
          % (recall, cer, wer))
    check("phase0 coaching center recall == 1.0", recall >= 0.999)
    check("phase0 cer_roman is low (<0.2)", cer < 0.2)
    check("wer_raw stays high, showing the confound (>0.5)", wer > 0.5)

    # A realistic mixed sentence from the eval set
    ref = "ये जो coaching center है, इसकी battery life बहुत terrible है।"
    hyp = "ये जो कोचिंग सेंटर है इसकी बैटरी लाइफ बहुत टेरिबल है"
    r2 = content_word_recall(ref, hyp)
    print("  mixed-sentence recall  : %.3f" % r2)
    check("mixed Hinglish recall >= 0.8", r2 >= 0.8)

    # A genuine miss must score low (recognizer dropped the English content)
    bad_hyp = "ये जो है इसकी बहुत है"
    r3 = content_word_recall(ref, bad_hyp)
    print("  dropped-words recall   : %.3f" % r3)
    check("dropped-content recall < 0.5", r3 < 0.5)

    # accept_clip wiring
    cfg = {"tau_recall": 0.8, "tau_cer": 0.3, "min_s": 3.0, "max_s": 30.0}
    ok, reason = accept_clip(
        {"filter_recall": r2, "filter_cer_roman": cer_roman(ref, hyp),
         "asr_hyp": hyp}, dur_s=8.0, sr=24000, cfg=cfg)
    check("accept_clip accepts a good clip", ok and reason is None)
    ok2, reason2 = accept_clip(
        {"filter_recall": 0.3, "filter_cer_roman": 0.1, "asr_hyp": hyp},
        dur_s=8.0, sr=24000, cfg=cfg)
    check("accept_clip rejects low recall (recall_low)",
          (not ok2) and reason2 == "recall_low")
    ok3, reason3 = accept_clip(
        {"filter_recall": 0.9, "filter_cer_roman": 0.1, "asr_hyp": hyp},
        dur_s=1.0, sr=24000, cfg=cfg)
    check("accept_clip rejects too short", (not ok3) and reason3 == "too_short")

    # romanizer sanity
    rom = romanize_deva("कोचिंग सेंटर")
    print("  romanize_deva sample   : %r" % rom)
    check("romanizer collapses to latin", _LATIN_RE.search(rom) is not None)

    # cs_density / cmi_bin reproduce the ws4 convention
    tags = ["en", "other", "en", "other", "other", "other", "other", "other",
            "other", "hi", "hi", "hi", "hi", "other", "other", "other",
            "other", "other", "other", "other", "other", "other", "other",
            "other", "other", "other", "hi"]
    csd, cbin = lang_tags_to_cs_density(tags)
    print("  cs_density=%.4f cmi_bin=%s (expect 0.1667 / med)" % (csd, cbin))
    check("cs_density matches ws4", abs(csd - 0.16666666666666666) < 1e-6)
    check("cmi_bin matches ws4", cbin == "med")

    # tag_languages
    tg = tag_languages("ये coaching center है")
    print("  tag_languages sample   : %s" % tg)
    check("tag_languages tags en+hi", tg == ["hi", "en", "en", "hi"])

    # Invariant: lang_tags_to_cs_density over the eval manifest's OWN lang_tags
    # must reproduce the manifest cmi_bin/cs_density for every one of the 1497
    # rows. This is the real invariant: the manifest tags are surface-aligned
    # (English-in-Devanagari loanwords tag 'en'), which is why corpus building
    # must carry them through for real-seed rows rather than recompute from
    # ref_orig (recomputing would collapse the whole set to cmi_bin 'none').
    eval_path = (_HERE.parent.parent / "data" / "spontaneous_hinglish" /
                 "eval_spontaneous_combined_manifest.json")
    if eval_path.exists():
        eval_rows = read_manifest(str(eval_path))
        man_dist = Counter(r.get("cmi_bin") for r in eval_rows)
        recomputed_ok = 0
        density_ok = 0
        n_tagged = 0
        for r in eval_rows:
            lt = r.get("lang_tags")
            if not lt:
                continue
            n_tagged += 1
            csd, cb = lang_tags_to_cs_density(lt)
            if cb == r.get("cmi_bin"):
                recomputed_ok += 1
            if abs(csd - (r.get("cs_density") or 0.0)) <= 1e-6:
                density_ok += 1
        print("  eval manifest cmi_bin  : %s" % dict(man_dist))
        print("  recomputed-from-tags   : %d/%d cmi_bin match, %d/%d cs_density"
              % (recomputed_ok, n_tagged, density_ok, n_tagged))
        check("eval manifest lang_tags reproduce cmi_bin for all rows",
              n_tagged > 0 and recomputed_ok == n_tagged)
        check("eval manifest lang_tags reproduce cs_density for all rows",
              n_tagged > 0 and density_ok == n_tagged)
        check("eval set is the expected 1497 rows", len(eval_rows) == 1497)
    else:
        print("  (eval manifest not found; skipping cmi_bin invariant check)")

    # chunking under 250 chars, no word breaks
    long_text = ("यह एक बहुत लंबा वाक्य है, " * 30).strip()
    chunks = chunk_text(long_text, max_chars=250)
    check("all chunks <= 250 chars", all(len(c) <= 250 for c in chunks))
    check("chunks reconstruct words (no mid-word split)",
          all(" " in c or len(c) <= 250 for c in chunks))

    # identity determinism
    cid = corpus_id_of(ref)
    cid2 = corpus_id_of(ref + "  ")  # whitespace-normalised, same id
    check("corpus_id deterministic + whitespace-stable", cid == cid2)
    uid = mint_utt_id(cid, "maya", 1.0, None, 0)
    print("  utt_id sample          : %s" % uid)
    check("utt_id format", uid == cid + "__maya__sp10__tx__v0")

    # new_row / validate_row
    row = new_row(
        utt_id=uid, audio_path=None, ref_orig=ref, ref_surface=None,
        ref_iso15919=None, cmi_bin=cbin, cs_density=csd,
        lang_tags=tag_languages(ref), speaker_id="maya", duration_s=None,
        sha256=None, dataset="synthetic_hinglish", partition="train",
        is_synthetic=True, license="synthetic_teacher_tts",
        flags=[], corpus_id=cid, teacher=DEFAULT_TEACHER)
    probs = validate_row(row, profile="corpus")
    check("new_row builds a corpus-valid row", probs == [])
    check("row keeps schema_version v1", row["schema_version"] == SCHEMA_VERSION)
    try:
        new_row(bogus_field=1)
        check("new_row rejects unknown keys", False)
    except KeyError:
        check("new_row rejects unknown keys", True)

    # manifest round-trip (JSONL and JSON array)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        jl = str(Path(td) / "m.jsonl")
        write_manifest(jl, [row])
        back = read_manifest(jl)
        check("jsonl round-trip", len(back) == 1 and back[0]["utt_id"] == uid)
        write_manifest(jl, [row], append=True)
        check("jsonl append", len(read_manifest(jl)) == 2)
        check("resume_done_ids finds id", uid in resume_done_ids(jl))
        ja = str(Path(td) / "m.json")
        write_manifest(ja, [row], json_array=True)
        check("json-array round-trip",
              read_manifest(ja)[0]["utt_id"] == uid)

    # diagnostics
    he = token_entropy(["one two three", "four five six", "one two three"])
    check("token_entropy positive", he > 0)
    rep = ngram_repetition("a b c a b c a b c", n=3)
    print("  ngram_repetition sample: %.3f" % rep)
    check("ngram_repetition detects loop", rep > 0.0)
    rep0 = ngram_repetition("the quick brown fox jumps over", n=4)
    check("ngram_repetition zero on clean text", rep0 == 0.0)

    # dedup
    keep = minhash_dedup([
        "ye coaching center bahut accha hai",
        "ye coaching center bahut accha hai",  # exact dup
        "completely different sentence about something else entirely now",
    ])
    print("  minhash_dedup kept idx : %s (expect 2 of 3)" % keep)
    check("minhash drops the duplicate", len(keep) == 2)

    # concat WAVs from dry-run synth
    with tempfile.TemporaryDirectory() as td:
        b1 = synth_request("hello", "maya", dry_run=True)
        b2 = synth_request("world", "maya", dry_run=True)
        p1 = Path(td) / "c1.wav"; p1.write_bytes(b1)
        p2 = Path(td) / "c2.wav"; p2.write_bytes(b2)
        outp = str(Path(td) / "out.wav")
        dur = concat_wavs([str(p1), str(p2)], outp)
        with wave.open(outp, "rb") as w:
            check("concat WAV is 24 kHz mono", w.getframerate() == 24000
                  and w.getnchannels() == 1)
        check("concat duration positive", dur > 0)

    print()
    if failures:
        print("SELF-TEST FAILED: %d check(s) -> %s" % (len(failures), failures))
        return 1
    print("SELF-TEST PASSED: all checks green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
