// Romanized-Hinglish -> (Devanagari Hindi + Roman English) normalizer.
// Pure JS, zero-server. The model pronounces best when Hindi is in Devanagari and English stays Roman;
// users type everything romanized. This converts under the hood, per word.
//
// Per-token decision (order matters):
//   already Devanagari      -> keep
//   forceHindi[w]           -> Devanagari   (curated Hindi-dominant words incl. ambiguous to/me/is/par)
//   names[w] / Capitalized   -> Devanagari   (Indian names — the explicit pain point)
//   english has w           -> keep Roman    (English loanwords: match, office, last, over, ...)
//   lexicon[w]              -> Devanagari   (Dakshina roman->Devanagari, 49k words)
//   OOV                     -> rule-transliterate to Devanagari (assume Hindi) unless looks-English
//
// Data: assets/normalize.json {forceHindi, names, lexicon, english[], hindiFreq}.

const DEVA = /[ऀ-ॿ]/;
const WORD = /[A-Za-zऀ-ॿ]+(?:['’\-][A-Za-zऀ-ॿ]+)*/g;

// ---- rule-based informal Roman -> Devanagari (OOV fallback only) ----
// Longest-match consonant/vowel syllabification. Dictionaries cover the common case; this handles the
// long tail so a missed Hindi word is still Devanagari-ized rather than left mispronounced in Roman.
const CONS = {
  chh: "छ", chr: "च्र", sh: "श", ch: "च", th: "थ", ph: "फ", kh: "ख", gh: "घ", bh: "भ", dh: "ध",
  jh: "झ", tt: "ट", dd: "ड", nn: "न", ng: "ंग", ny: "न्य", gy: "ग्य", ksh: "क्ष", tr: "त्र", shr: "श्र",
  k: "क", q: "क", g: "ग", j: "ज", z: "ज़", t: "त", d: "द", n: "न", p: "प", b: "ब", m: "म",
  y: "य", r: "र", l: "ल", v: "व", w: "व", s: "स", h: "ह", f: "फ़", c: "क", x: "क्स",
};
const VOW_IND = { aa: "आ", a: "अ", ee: "ई", ii: "ई", i: "इ", oo: "ऊ", uu: "ऊ", u: "उ",
  ai: "ऐ", au: "औ", e: "ए", o: "ओ" };
const VOW_MAT = { aa: "ा", a: "", ee: "ी", ii: "ी", i: "ि", oo: "ू", uu: "ू", u: "ु",
  ai: "ै", au: "ौ", e: "े", o: "ो" };
const VKEYS = ["aa", "ee", "ii", "oo", "uu", "ai", "au", "a", "i", "u", "e", "o"];
const CKEYS = ["chh", "ksh", "shr", "chr", "sh", "ch", "th", "ph", "kh", "gh", "bh", "dh", "jh",
  "tt", "dd", "nn", "ng", "ny", "gy", "tr", "k", "q", "g", "j", "z", "t", "d", "n", "p", "b",
  "m", "y", "r", "l", "v", "w", "s", "h", "f", "c", "x"];

function ruleTranslit(w) {
  w = w.toLowerCase();
  let out = "", i = 0, atStart = true;
  const n = w.length;
  while (i < n) {
    let cons = null, ck = "";
    for (const k of CKEYS) { if (w.startsWith(k, i)) { ck = k; cons = CONS[k]; break; } }
    if (cons) {
      i += ck.length;
      let vk = "";
      for (const v of VKEYS) { if (w.startsWith(v, i)) { vk = v; break; } }
      if (vk) { out += cons + VOW_MAT[vk]; i += vk.length; }
      else {
        // bare consonant: inherent schwa mid-word, halant if followed by another consonant or word-end
        const nextIsCons = i < n && CKEYS.some((k) => w.startsWith(k, i));
        out += cons + ((i >= n || nextIsCons) ? "्" : "");
      }
      atStart = false;
    } else {
      let vk = "";
      for (const v of VKEYS) { if (w.startsWith(v, i)) { vk = v; break; } }
      if (vk) { out += atStart ? VOW_IND[vk] : VOW_IND[vk]; i += vk.length; atStart = false; }
      else { out += w[i]; i += 1; }
    }
  }
  // drop a dangling word-final halant (schwa-deletion already handled by inherent schwa)
  return out.replace(/्$/, "");
}

// crude "looks English" guard for OOV: English-typical letter patterns rare in romanized Hindi
function looksEnglish(w) {
  return /(tion|ing|ment|ous|que|wh|ck$|ed$|ly$)/.test(w) || /[qwxf]/.test(w) && w.length > 6;
}

export async function loadNormalizer(url) {
  const d = await (await fetch(url)).json();
  const english = new Set(d.english);
  const { forceHindi, names, lexicon } = d;
  const ambig = d.ambig || {};

  // pass 1: confident label per word (lang null = ambiguous, resolve by context in pass 2)
  function label(w, isFirst) {
    if (DEVA.test(w)) return { src: w, deva: w, lang: "hi", how: "already" };
    if (w.length === 1 && /[A-Za-z]/.test(w)) return { src: w, deva: w, lang: "en", how: "single" }; // a, I, initials
    const lw = w.toLowerCase();
    if (forceHindi[lw]) return { src: w, deva: forceHindi[lw], lang: "hi", how: "force" };
    if (names[lw]) return { src: w, deva: names[lw], lang: "hi", how: "name" };
    if (ambig[lw]) return { src: w, lang: null, how: "ambig", hiDeva: ambig[lw] };  // defer
    if (english.has(lw)) return { src: w, deva: w, lang: "en", how: "english" };
    if (lexicon[lw]) return { src: w, deva: lexicon[lw], lang: "hi", how: "lexicon" };
    if (!isFirst && /^[A-Z]/.test(w)) return { src: w, deva: ruleTranslit(lw), lang: "hi", how: "name?" };
    if (looksEnglish(lw)) return { src: w, deva: w, lang: "en", how: "looks-en" };
    return { src: w, deva: ruleTranslit(lw), lang: "hi", how: "rule" };
  }

  // pass 2: resolve ambiguous tokens by nearest confident neighbours (window ±2); Hindi prior on tie
  function resolveAmbig(words) {
    for (let i = 0; i < words.length; i++) {
      if (words[i].lang !== null) continue;
      let hi = 0, en = 0;
      for (let j = Math.max(0, i - 2); j <= Math.min(words.length - 1, i + 2); j++) {
        if (j === i || words[j].lang === null) continue;
        const dist = Math.abs(j - i);
        const wgt = dist === 1 ? 2 : 1;
        if (words[j].lang === "hi") hi += wgt; else if (words[j].lang === "en") en += wgt;
      }
      if (en > hi) { words[i].lang = "en"; words[i].deva = words[i].src; words[i].how = "ambig→en"; }
      else { words[i].lang = "hi"; words[i].deva = words[i].hiDeva; words[i].how = "ambig→hi"; }
    }
    return words;
  }

  function normalize(text) {
    // split into word tokens + the separator spans between them
    const seps = [], words = [];
    let last = 0, m, isFirst = true;
    WORD.lastIndex = 0;
    while ((m = WORD.exec(text)) !== null) {
      seps.push(text.slice(last, m.index));
      words.push(label(m[0], isFirst));
      last = m.index + m[0].length;
      isFirst = false;
    }
    seps.push(text.slice(last));
    resolveAmbig(words);
    let out = "";
    for (let i = 0; i < words.length; i++) out += seps[i] + words[i].deva;
    out += seps[words.length];
    return { out, spans: words };
  }

  return { normalize, label, ruleTranslit, _data: d };
}

export { ruleTranslit };
