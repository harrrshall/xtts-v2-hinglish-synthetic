// Regenerate the EXACT audio the app produces for a given paragraph (normalize -> chunk ->
// per-chunk gen -> concat) and write a 16-bit WAV. Deterministic/greedy, so bit-identical to
// the in-browser output. Usage: node _gen_wav.mjs <voice> <outpath>  (text on stdin)
import { writeFileSync, readFileSync } from 'node:fs';
import { loadEngine } from './_engine.mjs';
import { chunkText, concatWavs } from '../app/js/chunk.js';

const VOICE = process.argv[2] || 'maya';
const OUT = process.argv[3] || 'ui_audio.wav';
const text = readFileSync(0, 'utf8').trim();

const eng = await loadEngine();
const normText = eng.norm.normalize(text).out;
const countIds = (t) => eng.tok.encode(t, 'hi').length;
const chunks = chunkText(normText, countIds, 60);
console.error(`voice=${VOICE} chunks=${chunks.length}: ${chunks.map((c,i)=>`[${i}]${countIds(c)}ids`).join(' ')}`);

const wavs = [];
for (const ch of chunks) { const r = await eng.gen(eng.tok.encode(ch, 'hi'), VOICE); wavs.push(r.wav); }
const pcm = chunks.length > 1 ? concatWavs(wavs, eng.SR, { gapMs: 90, crossfadeMs: 4 }) : wavs[0];

// write 16-bit PCM WAV
const sr = eng.SR, n = pcm.length, buf = Buffer.alloc(44 + n * 2);
buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * 2, 4); buf.write('WAVE', 8); buf.write('fmt ', 12);
buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22); buf.writeUInt32LE(sr, 24);
buf.writeUInt32LE(sr * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34); buf.write('data', 36);
buf.writeUInt32LE(n * 2, 40);
for (let i = 0; i < n; i++) { let s = Math.max(-1, Math.min(1, pcm[i])); buf.writeInt16LE(s < 0 ? s * 0x8000 : s * 0x7fff, 44 + i * 2); }
writeFileSync(OUT, buf);
console.error(`wrote ${OUT} (${(n / sr).toFixed(2)}s, ${buf.length} bytes)`);
