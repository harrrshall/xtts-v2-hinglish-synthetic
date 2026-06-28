// Kernel optimization benchmark for the AR decode loop.
// Measures total generation time + ONNX-only time to isolate JS overhead.
// Run:  node _bench.mjs [WARMUP=2] [RUNS=5] [VOICE=maya] [TEXT=...]
//
// Appends results to bench_results.log for persistent tracking across sessions.
// Reports per-run breakdown and aggregate stats.

import * as ortNS from 'onnxruntime-web';
import { readFileSync, appendFileSync } from 'node:fs';
const ort = ortNS.default || ortNS;
ort.env.wasm.numThreads = 1;

const APP = new URL('../app/', import.meta.url);
const rd  = (p) => readFileSync(new URL(p, APP));
const bufOf = (p) => { const b = rd(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };

// ---- load assets ----
const cfg   = JSON.parse(rd('config.json'));
const C     = cfg.constants;
const { layers: NL, heads: H, start_audio_token: SA, stop_audio_token: EA,
        start_text_token: ST, stop_text_token: ET, max_gen_mel_tokens: MAXG,
        num_audio_tokens: NAUDIO, code_stride_len: STRIDE, sample_rate: SR } = C;
const PEN   = cfg.decode.repetition_penalty;
const D     = 640;

const vmeta = JSON.parse(rd('assets/voices.json'));
const vbuf  = new Float32Array(rd('assets/voices.bin').buffer.slice());
const voices = {};
for (const [v, m] of Object.entries(vmeta)) {
  const cn = m.cond_shape[0] * m.cond_shape[1], sn = m.spk_shape[0] * m.spk_shape[1];
  voices[v] = { cond: vbuf.slice(m.cond_off, m.cond_off + cn), condShape: m.cond_shape,
                spk:  vbuf.slice(m.spk_off, m.spk_off + sn),  spkShape:  m.spk_shape };
}

const emeta = JSON.parse(rd('assets/embeddings.json'));
const ebuf  = new Float32Array(rd('assets/embeddings.bin').buffer.slice());
const E     = {};
for (const [k, m] of Object.entries(emeta)) E[k] = ebuf.subarray(m.off, m.off + m.shape[0] * m.shape[1]);

// ---- load tokenizer ----
import { loadTokenizer } from '../app/js/tokenizer.js';
const tok = await loadTokenizer(new URL('assets/tokenizer.json', APP).href.replace('file://', ''));

// ---- load ONNX sessions ----
console.log('Loading sessions (wasm)...');
const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
const sStep = await ort.InferenceSession.create(bufOf('models/gpt_step.onnx'), opt);
const sVoc  = await ort.InferenceSession.create(bufOf('models/vocoder.onnx'),  opt);
console.log('Sessions ready.\n');

// ---- BASELINE generation (current code, no optimizations) ----
function prefillEmbeds(ids, cond) {
  const ti = [ST, ...ids, ET], n = ti.length, L = 32 + n + 1;
  const emb = new Float32Array(L * D);
  emb.set(cond, 0);
  for (let i = 0; i < n; i++) {
    const dst = (32 + i) * D, te = ti[i] * D, tp = i * D;
    for (let d = 0; d < D; d++) emb[dst + d] = E.text_emb[te + d] + E.text_pos[tp + d];
  }
  const dst = (32 + n) * D, me = SA * D;
  for (let d = 0; d < D; d++) emb[dst + d] = E.mel_emb[me + d] + E.mel_pos[d];
  return { data: emb, L };
}

function decodeEmbedBaseline(t, pos) {
  const e = new Float32Array(D), me = t * D, mp = pos * D;
  for (let d = 0; d < D; d++) e[d] = E.mel_emb[me + d] + E.mel_pos[mp + d];
  return e;
}

function emptyPast() {
  const f = {};
  for (let j = 0; j < NL; j++) {
    f[`past_key_values.${j}.key`]   = new ort.Tensor('float32', new Float32Array(0), [1, H, 0, 64]);
    f[`past_key_values.${j}.value`] = new ort.Tensor('float32', new Float32Array(0), [1, H, 0, 64]);
  }
  return f;
}

const repPen = (l, seq) => { for (const t of seq) { const s = l[t]; l[t] = s < 0 ? s * PEN : s / PEN; } };
const argmax = (a) => { let bi = 0, bv = a[0]; for (let k = 1; k < a.length; k++) if (a[k] > bv) { bv = a[k]; bi = k; } return bi; };

async function genBaseline(ids, voice) {
  const v = voices[voice], seqSet = new Set([1, SA]);
  const { data, L } = prefillEmbeds(ids, v.cond);

  let tOrt = 0;

  let t = performance.now();
  let out = await sStep.run({ inputs_embeds: new ort.Tensor('float32', data, [1, L, D]), ...emptyPast() });
  const tPrefill = performance.now() - t;
  tOrt += tPrefill;

  // BASELINE: Float32Array.from() copies + lats[] array
  let logits = Float32Array.from(out.logits.data);
  const lats = [Float32Array.from(out.latent.data)];
  let past = out;
  const codes = [];

  for (let step = 0; step < MAXG; step++) {
    repPen(logits, seqSet);
    const nxt = argmax(logits);
    if (nxt === EA) { codes.push(nxt); break; }
    codes.push(nxt); seqSet.add(nxt);

    const feed = { inputs_embeds: new ort.Tensor('float32', decodeEmbedBaseline(nxt, codes.length), [1, 1, D]) };
    for (let j = 0; j < NL; j++) {
      feed[`past_key_values.${j}.key`]   = past[`present.${j}.key`];
      feed[`past_key_values.${j}.value`] = past[`present.${j}.value`];
    }
    t = performance.now();
    out = await sStep.run(feed);
    tOrt += performance.now() - t;

    logits = Float32Array.from(out.logits.data);
    lats.push(Float32Array.from(out.latent.data));
    past = out;
  }

  const N = lats.length, lat = new Float32Array(N * D);
  for (let k = 0; k < N; k++) lat.set(lats[k], k * D);
  const g = new ort.Tensor('float32', v.spk, [1, ...v.spkShape]);

  t = performance.now();
  const vout = await sVoc.run({ lat640: new ort.Tensor('float32', lat, [1, N, D]), g });
  const tVocoder = performance.now() - t;
  tOrt += tVocoder;

  const wav = Float32Array.from(vout.wav.data);
  return { codes, wav, tPrefill, tVocoder, tOrt, nSteps: N };
}

// ---- OPTIMIZED generation ----
// O1: pre-allocate stacked latent buffer (no lats[] array, no post-loop copy)
// O2: use logits .data directly (skip Float32Array.from for logits — safe since consumed before next run())
// O3: pre-allocate reusable decode embed buffer
const _decBuf = new Float32Array(D);
function decodeEmbedOpt(t, pos) {
  const me = t * D, mp = pos * D;
  for (let d = 0; d < D; d++) _decBuf[d] = E.mel_emb[me + d] + E.mel_pos[mp + d];
  return _decBuf;
}

async function genOptimized(ids, voice) {
  const v = voices[voice], seqSet = new Set([1, SA]);
  const { data, L } = prefillEmbeds(ids, v.cond);

  let tOrt = 0;

  let t = performance.now();
  let out = await sStep.run({ inputs_embeds: new ort.Tensor('float32', data, [1, L, D]), ...emptyPast() });
  const tPrefill = performance.now() - t;
  tOrt += tPrefill;

  // O1: pre-alloc stacked latent buffer, write directly each step
  const lat = new Float32Array((MAXG + 1) * D);
  lat.set(out.latent.data, 0);
  let N = 1;

  // O2: use .data directly for logits (no copy)
  let logits = out.logits.data;
  let past = out;
  const codes = [];

  for (let step = 0; step < MAXG; step++) {
    repPen(logits, seqSet);
    const nxt = argmax(logits);
    if (nxt === EA) { codes.push(nxt); break; }
    codes.push(nxt); seqSet.add(nxt);

    // O3: reuse decode embed buffer
    const feed = { inputs_embeds: new ort.Tensor('float32', decodeEmbedOpt(nxt, codes.length), [1, 1, D]) };
    for (let j = 0; j < NL; j++) {
      feed[`past_key_values.${j}.key`]   = past[`present.${j}.key`];
      feed[`past_key_values.${j}.value`] = past[`present.${j}.value`];
    }
    t = performance.now();
    out = await sStep.run(feed);
    tOrt += performance.now() - t;

    // O6: dispose previous step KV tensors to prevent WASM heap accumulation (~170 MB GC pressure)
    for (let j = 0; j < NL; j++) {
      past[`present.${j}.key`]?.dispose?.();
      past[`present.${j}.value`]?.dispose?.();
    }

    // O1: direct write to stacked buffer
    lat.set(out.latent.data, N * D);
    N++;
    // O2: no copy for logits
    logits = out.logits.data;
    past = out;
  }
  // Dispose final step KV tensors
  for (let j = 0; j < NL; j++) {
    past[`present.${j}.key`]?.dispose?.();
    past[`present.${j}.value`]?.dispose?.();
  }

  // O1: pass subarray directly (no assembly loop)
  const g = new ort.Tensor('float32', v.spk, [1, ...v.spkShape]);
  t = performance.now();
  const vout = await sVoc.run({ lat640: new ort.Tensor('float32', lat.subarray(0, N * D), [1, N, D]), g });
  const tVocoder = performance.now() - t;
  tOrt += tVocoder;

  const wav = Float32Array.from(vout.wav.data);
  return { codes, wav, tPrefill, tVocoder, tOrt, nSteps: N };
}

// ---- benchmark runner ----
const WARMUP = +(process.env.WARMUP ?? 2);
const RUNS   = +(process.env.RUNS ?? 5);
const VOICE  = process.env.VOICE ?? 'maya';
const TEXT   = process.env.TEXT  ?? 'mera naam Kaustubh hai aur main Delhi se hoon';

const ids = tok.encode(TEXT, 'hi');
console.log(`Input: "${TEXT}"`);
console.log(`Tokens: ${ids.length}, voice: ${VOICE}, warmup: ${WARMUP}, runs: ${RUNS}\n`);

function stats(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const mean = arr.reduce((s, x) => s + x, 0) / arr.length;
  const std  = Math.sqrt(arr.reduce((s, x) => s + (x - mean) ** 2, 0) / arr.length);
  return { mean, std, min: sorted[0], max: sorted[sorted.length - 1], median: sorted[Math.floor(arr.length / 2)] };
}

function fmtStats(s) {
  return `mean=${s.mean.toFixed(0)}ms  std=${s.std.toFixed(0)}ms  min=${s.min.toFixed(0)}  median=${s.median.toFixed(0)}  max=${s.max.toFixed(0)}`;
}

async function runBench(label, genFn) {
  console.log(`=== ${label} ===`);
  const all = [];
  for (let i = 0; i < WARMUP + RUNS; i++) {
    const t0 = performance.now();
    const r = await genFn(ids, VOICE);
    const tTotal = performance.now() - t0;
    const durSec  = r.codes.length * STRIDE / SR;
    const rt      = durSec / (tTotal / 1000);
    const jsMs    = tTotal - r.tOrt;
    if (i < WARMUP) {
      console.log(`  warmup #${i + 1}: ${tTotal.toFixed(0)}ms (${r.nSteps} steps, ${rt.toFixed(2)}x RT)`);
    } else {
      console.log(`  run #${i - WARMUP + 1}: total=${tTotal.toFixed(0)}ms  ort=${r.tOrt.toFixed(0)}ms  js=${jsMs.toFixed(0)}ms  prefill=${r.tPrefill.toFixed(0)}ms  vocoder=${r.tVocoder.toFixed(0)}ms  steps=${r.nSteps}  RT=${rt.toFixed(2)}x`);
      all.push({ tTotal, tOrt: r.tOrt, jsMs, tPrefill: r.tPrefill, tVocoder: r.tVocoder });
    }
  }
  const totals  = all.map(x => x.tTotal);
  const jsTimes = all.map(x => x.jsMs);
  const ortTimes = all.map(x => x.tOrt);
  console.log(`  TOTAL: ${fmtStats(stats(totals))}`);
  console.log(`  ORT:   ${fmtStats(stats(ortTimes))}`);
  console.log(`  JS:    ${fmtStats(stats(jsTimes))}`);
  console.log();
  return { totals, jsTimes, ortTimes };
}

const baseResult = await runBench('BASELINE (current code)',   genBaseline);
const optResult  = await runBench('OPTIMIZED (O1+O2+O3+O6)', genOptimized);

// ---- summary delta ----
console.log('=== SUMMARY DELTA ===');
const bSt = stats(baseResult.totals), oSt = stats(optResult.totals);
const bJs = stats(baseResult.jsTimes), oJs = stats(optResult.jsTimes);
console.log(`Total   median: ${bSt.median.toFixed(0)}ms → ${oSt.median.toFixed(0)}ms  delta=${((bSt.median - oSt.median) / bSt.median * 100).toFixed(1)}%`);
console.log(`Total   std:    ${bSt.std.toFixed(0)}ms → ${oSt.std.toFixed(0)}ms  (variance reduction)`);
console.log(`JS overhead:    ${bJs.median.toFixed(1)}ms → ${oJs.median.toFixed(1)}ms`);

// ---- persistent log ----
const ts = new Date().toISOString().slice(0, 19).replace('T', ' ');
const logLine = [
  `\n--- ${ts} ---`,
  `text="${TEXT}" voice=${VOICE} warmup=${WARMUP} runs=${RUNS}`,
  `BASELINE  median=${bSt.median.toFixed(0)}ms  std=${bSt.std.toFixed(0)}ms  min=${bSt.min.toFixed(0)}ms  max=${bSt.max.toFixed(0)}ms  js=${bJs.median.toFixed(1)}ms`,
  `OPTIMIZED median=${oSt.median.toFixed(0)}ms  std=${oSt.std.toFixed(0)}ms  min=${oSt.min.toFixed(0)}ms  max=${oSt.max.toFixed(0)}ms  js=${oJs.median.toFixed(1)}ms`,
  `DELTA     total=${((bSt.median - oSt.median) / bSt.median * 100).toFixed(1)}%  std_reduction=${((bSt.std - oSt.std) / bSt.std * 100).toFixed(0)}%`,
].join('\n');
const logFile = new URL('bench_results.log', import.meta.url).pathname;
appendFileSync(logFile, logLine + '\n');
console.log(`\nResults appended to bench_results.log`);
