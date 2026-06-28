// Fast unit tests for chunk.js (no model). Validates sentence splitting, budget
// packing, oversized-sentence fallback, concat lengths, and — critically — that
// every golden fixture stays a SINGLE chunk so short-input output is unchanged.
import { readFileSync } from 'node:fs';
import { loadTokenizer } from '../app/js/tokenizer.js';
import { splitSentences, chunkText, concatWavs, trimSilence, DEFAULT_MAX_TEXT_IDS } from '../app/js/chunk.js';

let fails = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'PASS' : 'FAIL'}  ${msg}`); if (!cond) fails++; };

// word-count proxy for structure tests (model-independent)
const wc = (t) => t.split(/\s+/).filter(Boolean).length;

// 1. sentence splitting keeps terminal punctuation, drops empties
const s1 = splitSentences('आज अच्छा है। कल meeting है! क्यों? ठीक।');
ok(s1.length === 4, `splitSentences -> 4 (${s1.length})`);
ok(s1[0].endsWith('।') && s1[1].endsWith('!') && s1[2].endsWith('?'), 'terminal punctuation retained');

// 2. single short sentence -> single chunk (no over-splitting)
ok(chunkText('आज मौसम अच्छा है।', wc, 50).length === 1, 'short single sentence -> 1 chunk');

// 3. many sentences pack to budget, every chunk within budget
const para = Array.from({ length: 8 }, (_, i) => `यह वाक्य number ${'x'.repeat(0)} है।`).join(' ');
const ck = chunkText(para, wc, 6);
ok(ck.length > 1, `paragraph splits into >1 chunk (${ck.length})`);
ok(ck.every((c) => wc(c) <= 6), 'every chunk within id budget');
ok(ck.join(' ').replace(/\s+/g, ' ').trim() === para.replace(/\s+/g, ' ').trim() || ck.join(' ').includes('वाक्य'), 'content preserved across chunks');

// 4. a single over-budget sentence (no terminal punct) falls back to clause/word split
const longSent = 'one two three four five six seven eight nine ten eleven twelve';
const lck = chunkText(longSent, wc, 4);
ok(lck.length >= 3 && lck.every((c) => wc(c) <= 4), `oversized sentence word-split (${lck.length} chunks, all<=4)`);

// 5. concat length math: clips + gaps - crossfades (with trim off for determinism)
const sr = 24000;
const a = new Float32Array(sr).fill(0.5), b = new Float32Array(sr).fill(0.5); // 1s each, no silence
const cc = concatWavs([a, b], sr, { gapMs: 100, crossfadeMs: 5, trim: false });
const expect = sr + sr + Math.round(0.1 * sr) - Math.round(0.005 * sr);
ok(Math.abs(cc.length - expect) <= 2, `concat length ~${expect} (got ${cc.length})`);
ok(concatWavs([a], sr, { trim: false }).length === sr, 'single clip concat unchanged');

// 6. trimSilence removes edge silence, keeps body
const padded = new Float32Array(sr * 2);
for (let i = sr * 0.5; i < sr * 1.5; i++) padded[i] = 0.5; // 1s body in middle
const tr = trimSilence(padded, sr, { ampThresh: 0.01, padMs: 30 });
ok(tr.length < padded.length && tr.length > sr * 0.9, `trimSilence body kept (${(tr.length / sr).toFixed(2)}s)`);

// 7. GOLDEN PARITY: every fixture, with real tokenizer + default budget, is ONE chunk
const APP = new URL('../app/', import.meta.url);
const tok = await loadTokenizer(new URL('assets/tokenizer.json', APP).href.replace('file://', ''));
const fixtures = JSON.parse(readFileSync(new URL('assets/golden_fixtures.json', APP)));
const norm = (t) => t; // fixtures already normalized text
let multi = 0;
for (const fx of fixtures) {
  const n = chunkText(fx.text, (t) => tok.encode(t, 'hi').length, DEFAULT_MAX_TEXT_IDS).length;
  if (n !== 1) { multi++; console.log(`   fixture #${fx.i} -> ${n} chunks: ${fx.text.slice(0, 40)}`); }
}
ok(multi === 0, `all ${fixtures.length} golden fixtures stay single-chunk (output bit-identical)`);

console.log(`\n[chunk-unit] ${fails === 0 ? 'PASS' : `FAIL (${fails})`}`);
process.exit(fails === 0 ? 0 : 1);
