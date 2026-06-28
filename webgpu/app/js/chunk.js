// Intelligent, length-independent chunking for the autoregressive GPT-TTS.
//
// WHY: the AR mel decoder was trained on short clips. On long inputs its output
// duration saturates (~15s / ~335 mel tokens) and it silently rushes/drops the
// tail content — a 6-sentence paragraph that needs ~22s renders as ~15s (≈30%
// of the content lost). Measured onset is ~4 sentences / ~120 text ids.
//
// FIX: split the (already-normalized) text into sentence-aligned chunks, each
// within a safe text-token budget, synthesize each at full fidelity, and
// concatenate. A single-chunk input is returned verbatim so short-input
// behavior stays bit-identical to the pre-chunking pipeline.
//
// Pure functions only (no model/ORT deps) so they run identically in the browser
// app and in the node regression harness.

// Default text-token budget per chunk. At ~60 ids the one-pass decoder still
// renders content faithfully (coverage ≈ 1.0); degradation only appears well
// beyond this. Override via config.chunk.maxTextIds.
export const DEFAULT_MAX_TEXT_IDS = 60;

// Split text into sentences, keeping terminal punctuation attached. Handles the
// Devanagari danda (।), ?, !, newlines, and sentence-final ASCII '.' (period
// followed by whitespace/end — the app contract is digit-free so no decimals).
export function splitSentences(text) {
  const chars = [...text];
  const out = [];
  let buf = '';
  for (let i = 0; i < chars.length; i++) {
    const c = chars[i];
    buf += c;
    const next = chars[i + 1];
    const isEnd =
      c === '।' || c === '!' || c === '?' || c === '\n' ||
      (c === '.' && (next === undefined || /\s/.test(next)));
    if (isEnd) {
      // pull trailing closing quotes/brackets and spaces into this sentence
      while (i + 1 < chars.length && /["'”’)\]\s]/.test(chars[i + 1]) && chars[i + 1] !== '\n') buf += chars[++i];
      if (buf.trim()) out.push(buf.trim());
      buf = '';
    }
  }
  if (buf.trim()) out.push(buf.trim());
  if (out.length) return out;
  return text.trim() ? [text.trim()] : [];
}

// Split a too-long sentence on clause separators (comma, semicolon, colon,
// Devanagari comma ‘،’/'،' and the danda already consumed), keeping delimiters.
function splitClauses(s) {
  return s.split(/(?<=[,;:،])\s+/).map((x) => x.trim()).filter(Boolean);
}

// Last-resort split of an over-budget clause on word boundaries so a single
// chunk never exceeds maxIds. Greedy pack of words.
function splitWords(s, countIds, maxIds) {
  const words = s.split(/\s+/).filter(Boolean);
  const pieces = [];
  let cur = '';
  for (const w of words) {
    const cand = cur ? cur + ' ' + w : w;
    if (!cur || countIds(cand) <= maxIds) cur = cand;
    else { pieces.push(cur); cur = w; }
  }
  if (cur) pieces.push(cur);
  return pieces;
}

// Break input into atomic units each <= maxIds (sentence -> clauses -> words).
function toUnits(text, countIds, maxIds) {
  const units = [];
  for (const s of splitSentences(text)) {
    if (countIds(s) <= maxIds) { units.push(s); continue; }
    let cur = '';
    for (const cl of splitClauses(s)) {
      const cand = cur ? cur + ' ' + cl : cl;
      if (countIds(cand) <= maxIds) { cur = cand; continue; }
      if (cur) { units.push(cur); cur = ''; }
      if (countIds(cl) <= maxIds) cur = cl;
      else units.push(...splitWords(cl, countIds, maxIds));
    }
    if (cur) units.push(cur);
  }
  return units;
}

// Public: turn text into a list of chunk strings, each within maxIds text tokens.
// Greedily packs consecutive units up to the budget to minimize concatenation
// seams while staying in the decoder's faithful-rendering zone.
export function chunkText(text, countIds, maxIds = DEFAULT_MAX_TEXT_IDS) {
  const units = toUnits(text, countIds, maxIds);
  if (units.length <= 1) return units.length ? units : [];
  const chunks = [];
  let cur = '';
  for (const u of units) {
    const cand = cur ? cur + ' ' + u : u;
    if (!cur || countIds(cand) <= maxIds) cur = cand;
    else { chunks.push(cur); cur = u; }
  }
  if (cur) chunks.push(cur);
  return chunks;
}

// ---- waveform concatenation ----

// Trim leading/trailing near-silence from a mono Float32 PCM clip, leaving a
// small pad. ampThresh is linear amplitude (≈ -40 dBFS default).
export function trimSilence(wav, sr, { ampThresh = 0.01, padMs = 30 } = {}) {
  let a = 0, b = wav.length;
  while (a < b && Math.abs(wav[a]) < ampThresh) a++;
  while (b > a && Math.abs(wav[b - 1]) < ampThresh) b--;
  if (a >= b) return wav.slice(0, Math.min(wav.length, Math.round(sr * 0.02))); // all-silence guard
  const pad = Math.round((padMs / 1000) * sr);
  a = Math.max(0, a - pad);
  b = Math.min(wav.length, b + pad);
  return wav.slice(a, b);
}

// Concatenate chunk PCM clips into one natural-sounding clip: trim each clip's
// edge silence, insert a short inter-chunk pause, and apply a tiny equal-power
// crossfade at each join to remove boundary clicks.
export function concatWavs(wavs, sr, { gapMs = 90, crossfadeMs = 4, trim = true, ampThresh = 0.01, padMs = 30 } = {}) {
  const clips = wavs.filter((w) => w && w.length);
  if (clips.length === 0) return new Float32Array(0);
  const prepped = clips.map((w) => (trim ? trimSilence(w, sr, { ampThresh, padMs }) : w));
  if (prepped.length === 1) return prepped[0];

  const gap = Math.max(0, Math.round((gapMs / 1000) * sr));
  const xf = Math.max(0, Math.round((crossfadeMs / 1000) * sr));
  // total length: sum of clips + gaps, minus crossfade overlaps
  let total = prepped.reduce((s, w) => s + w.length, 0) + gap * (prepped.length - 1) - xf * (prepped.length - 1);
  total = Math.max(0, total);
  const out = new Float32Array(total);
  let pos = 0;
  for (let i = 0; i < prepped.length; i++) {
    const w = prepped[i];
    if (i === 0) { out.set(w, 0); pos = w.length; continue; }
    // crossfade the previous tail with this clip's head over xf samples
    const start = pos - xf + gap;
    for (let k = 0; k < xf && k < w.length; k++) {
      const idx = start + k;
      const t = (k + 1) / (xf + 1);
      const a = Math.cos((Math.PI / 2) * t);   // equal-power fade out (prev)
      const bgain = Math.sin((Math.PI / 2) * t); // equal-power fade in (curr)
      if (idx >= 0 && idx < out.length) out[idx] = (out[idx] || 0) * a + w[k] * bgain;
    }
    const tail = w.subarray(Math.min(xf, w.length));
    out.set(tail, start + xf);
    pos = start + xf + tail.length;
  }
  return out.subarray(0, Math.min(pos, out.length));
}
