# Romanized-Hinglish → Devanagari normalizer (client-side)

The 90M model was trained on **Devanagari Hindi + Roman English** (the synthetic teacher data is pure
Devanagari for Hindi). So it pronounces best when Hindi is in Devanagari and English stays Roman. But
real users type everything **romanized** ("yaar kal ka match dekha"). Romanized Hindi is out of
distribution → mispronunciation, and Indian names in Roman are especially wrong.

This normalizer converts, under the hood, per word: Hindi → Devanagari, English → kept Roman, names →
Devanagari. Pure JS, zero-server, runs in the browser before tokenization.

## Why this design (vs the SOTA model)

The SOTA Roman→Devanagari transliterator is **AI4Bharat IndicXlit** (11M fairseq seq2seq). It has **no
ONNX export** and needs fairseq + external beam search + a dictionary rescorer — a real porting project,
and it still doesn't solve **LID** (keeping English Roman). So the shipped solution is the research's
recommended pragmatic tier: a **dictionary + context-LID** pipeline. IndicXlit-ONNX is a documented
Tier-2 upgrade for OOV coverage.

## Pipeline (`webgpu/app/js/normalize.js`)

Per word, in order:
1. already Devanagari → keep
2. single Latin char (a, I, initials) → keep Roman
3. `forceHindi[w]` → Devanagari — curated Hindi-dominant words (no common-English collision)
4. `names[w]` → Devanagari — Indian-name gazetteer (incl. the 4 voices)
5. `ambig[w]` → **deferred** — genuinely ambiguous tokens (is, to, me, do, us, par, in, bat, hi)
6. English wordlist has `w` → keep Roman — handles loanwords (match, office, last, over, school…)
7. `lexicon[w]` → Devanagari — Dakshina roman→Devanagari (49k words)
8. Capitalized mid-sentence unknown → transliterate (likely an Indian name)
9. OOV → rule-based informal Roman→Devanagari (assume Hindi) unless it "looks English"

**Ambiguous tokens (step 5) are resolved by neighbour context** (window ±2, distance-weighted; Hindi
prior on tie). So "is" → इस in "is baat ko samjho" but stays English in "this is a great system".
"do" → दो in "kaam kar do" but stays English in "what do you want to do". This two-pass resolution is
what makes pure-English input safe while still fixing Hindi-context function words.

## Data assets (`webgpu/app/assets/normalize.json`, ~2.6 MB)

| key | source | size |
|---|---|---|
| `lexicon` roman→Devanagari | **Google Dakshina** hi romanization lexicon (CC BY-SA 4.0), inverted to most-attested target | 49,090 |
| `english` | hermitdave **FrequencyWords** en_50k (top ~38k, len≥2) | 37,829 |
| `forceHindi` | curated Hindi function/content words | 258 |
| `ambig` | curated both-language tokens, context-resolved | 9 |
| `names` | curated Indian names + cities (incl. aadya/arjun/kaustubh/maya) | 49 |
| `hindiFreq` | Dakshina attestation counts (tie-breaks) | 30,000 |

Build: `scripts/hinglish/normalize/build_assets.py` (curated maps inline; Dakshina TSVs + en_50k in
`scripts/hinglish/normalize/data/`).

## Validation (`webgpu/nodecheck/_norm_edge.mjs` + in-browser)

- romanized "yaar kal ka match dekha? last over me jo hua wo totally insane tha" →
  "यार कल का match देखा? last over में जो हुआ वो totally insane था" (English kept Roman).
- Generating from a romanized prompt (auto-convert ON) yields the **exact same audio codes** as
  generating from the hand-written Devanagari+English form — proven in-browser (e.g. the Kaustubh
  sentence: 60 codes, `sameAsDevanagari=true`). Recall the raw-romanized baseline was 94 vs the
  correct 90 codes; normalization closes that gap.
- Pure English ("this is a great system can you believe it") passes through **unchanged**.
- Names: Kaustubh→कौस्तुभ, Arjun→अर्जुन, Rahul→राहुल, Priya→प्रिया, Delhi→दिल्ली, Mumbai→मुंबई.

## UX (transparency)

`index.html` shows a **live green preview** of the conversion as you type, an **auto-convert romanized**
toggle (default on), and the spoken form in the status after generating. The model never silently
mishears — the user always sees what it will say.

## Known limits / follow-ups

- The OOV rule transliterator is best-effort (e.g. "samjho"→सम्झो vs ideal समझो); dictionaries cover
  the common case, so it only fires on the long tail. Tier-2 IndicXlit-ONNX would fix OOV/novel spellings.
- Ambiguity is context-resolved but context-free at sentence boundaries (a lone "do" defaults Hindi).
- Coverage is the Dakshina 49k + curated overrides; rarer Hindi words fall to the rule transliterator.
