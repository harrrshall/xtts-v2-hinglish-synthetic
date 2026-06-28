// End-to-end proof that chunking removes long-input degradation, through the REAL model.
// Metric: content coverage = audio_dur / expected_dur, where expected_dur is the sum of the
// per-sentence solo durations (the faithful target). One-pass drops content as length grows
// (coverage << 1); chunked restores it (coverage ~ 1). Also checks no truncation / no loops.
import { loadEngine, maxRun, ngramRepeatRate } from './_engine.mjs';
import { chunkText, concatWavs, trimSilence } from '../app/js/chunk.js';

const VOICE = process.env.VOICE || 'maya';
const eng = await loadEngine();
const dur = (w) => w.length / eng.SR;
// speech content = duration after trimming edge silence (isolates words from silence/pauses)
const speech = (w) => trimSilence(w, eng.SR, { ampThresh: 0.01, padMs: 0 }).length / eng.SR;
const countIds = (t) => eng.tok.encode(eng.norm.normalize(t).out, 'hi').length;
const genText = (t) => eng.gen(eng.tok.encode(eng.norm.normalize(t).out, 'hi'), VOICE);

const SENTS = [
  'आज मौसम बहुत अच्छा है और मैं घूमने जाना चाहता हूँ।',
  'मुझे यह project बहुत interesting लग रहा है।',
  'कल office में एक important meeting है इसलिए जल्दी पहुँचना है।',
  'यार ये नया phone का camera बिल्कुल insane है।',
  'Please इस bug का fix deploy कर दो और मुझे link भेज देना।',
  'शाम को बारिश होने की पूरी संभावना है, छाता साथ रखना।',
];
const PARA = SENTS.join(' ');

console.log(`engine ready. voice=${VOICE}\n`);

// 1. expected SPEECH content = sum of per-sentence solo speech (silence-trimmed) — faithful target
let expected = 0;
for (const s of SENTS) { const r = await genText(s); expected += speech(r.wav); }
console.log(`expected speech (silence-trimmed sum of solos) = ${expected.toFixed(2)}s\n`);

// 2. one-pass on the whole paragraph
const op = await genText(PARA);
const opCov = speech(op.wav) / expected;
console.log(`one-pass:  speech=${speech(op.wav).toFixed(2)}s rawdur=${dur(op.wav).toFixed(2)}s codes=${op.codes.length} hitCap=${op.hitCap} ` +
  `maxRun=${maxRun(op.codes)} rep3=${ngramRepeatRate(op.codes).toFixed(3)} coverage=${opCov.toFixed(3)}`);

// 3. chunked via the SAME chunk.js the app uses
const chunks = chunkText(eng.norm.normalize(PARA).out, countIds, 60);
const wavs = []; let chHitCap = false, chMaxRun = 0, chRep = 0, chSpeech = 0;
for (const ch of chunks) {
  const r = await eng.gen(eng.tok.encode(ch, 'hi'), VOICE);
  wavs.push(r.wav); chHitCap = chHitCap || r.hitCap; chSpeech += speech(r.wav);
  chMaxRun = Math.max(chMaxRun, maxRun(r.codes)); chRep = Math.max(chRep, ngramRepeatRate(r.codes));
}
const merged = concatWavs(wavs, eng.SR, { gapMs: 90, crossfadeMs: 4 });
const chCov = chSpeech / expected;            // content faithfulness (silence-independent)
console.log(`chunked:   speech=${chSpeech.toFixed(2)}s mergeddur=${dur(merged).toFixed(2)}s nChunks=${chunks.length} hitCap=${chHitCap} ` +
  `maxRun=${chMaxRun} rep3=${chRep.toFixed(3)} coverage=${chCov.toFixed(3)}`);
console.log(`\nchunks: ${chunks.map((c, i) => `[${i}] ${countIds(c)}ids`).join('  ')}`);

// ---- gates ----
let fails = 0;
const gate = (c, m) => { console.log(`${c ? 'PASS' : 'FAIL'}  ${m}`); if (!c) fails++; };
console.log('');
gate(opCov < 0.90, `one-pass IS degraded (speech coverage ${opCov.toFixed(3)} < 0.90)`);
gate(chCov >= 0.97, `chunked is content-faithful (speech coverage ${chCov.toFixed(3)} >= 0.97)`);
gate(chCov - opCov >= 0.12, `chunked beats one-pass by >=0.12 (Δ=${(chCov - opCov).toFixed(3)})`);
gate(!chHitCap, 'chunked: no truncation (no chunk hit MAXG)');
gate(chMaxRun <= 12, `chunked: no stuck-token loop (maxRun ${chMaxRun} <= 12)`);
gate(chRep <= 0.30, `chunked: low 3-gram repetition (${chRep.toFixed(3)} <= 0.30)`);

console.log(`\n[chunk-eval] ${fails === 0 ? 'PASS — chunking fixes long-input degradation' : `FAIL (${fails})`}`);
process.exit(fails === 0 ? 0 : 1);
