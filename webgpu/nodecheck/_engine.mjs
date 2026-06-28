// Shared node harness around the REAL merged gpt_step graph + vocoder (onnxruntime-web wasm EP).
// Mirrors webgpu/app/js/tts.js generation loop exactly, and exposes the tokenizer + a gen() that
// returns {codes, wav, hitCap}. Used by the long-input / chunking experiments and regression tests.
import * as ortNS from 'onnxruntime-web';
import { readFileSync } from 'node:fs';
import { loadTokenizer } from '../app/js/tokenizer.js';
import { loadNormalizer } from '../app/js/normalize.js';

const ort = ortNS.default || ortNS;
ort.env.wasm.numThreads = 1;

const APP = new URL('../app/', import.meta.url);
const rd = (p) => readFileSync(new URL(p, APP));
const bufOf = (p) => { const b = rd(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };

export async function loadEngine() {
  const cfg = JSON.parse(rd('config.json'));
  const C = cfg.constants;
  const NL = C.layers, H = C.heads, SA = C.start_audio_token, EA = C.stop_audio_token;
  const ST = C.start_text_token, ET = C.stop_text_token, PEN = cfg.decode.repetition_penalty;
  const MAXG = C.max_gen_mel_tokens, D = 640, SR = C.sample_rate;

  const vmeta = JSON.parse(rd('assets/voices.json'));
  const vbuf = new Float32Array(rd('assets/voices.bin').buffer.slice());
  const voices = {};
  for (const [v, m] of Object.entries(vmeta)) {
    const cn = m.cond_shape[0] * m.cond_shape[1], sn = m.spk_shape[0] * m.spk_shape[1];
    voices[v] = { cond: vbuf.slice(m.cond_off, m.cond_off + cn), condShape: m.cond_shape,
                  spk: vbuf.slice(m.spk_off, m.spk_off + sn), spkShape: m.spk_shape };
  }
  const emeta = JSON.parse(rd('assets/embeddings.json'));
  const ebuf = new Float32Array(rd('assets/embeddings.bin').buffer.slice());
  const E = {}; for (const [k, m] of Object.entries(emeta)) E[k] = ebuf.subarray(m.off, m.off + m.shape[0] * m.shape[1]);

  const tok = await loadTokenizer(new URL('assets/tokenizer.json', APP).href.replace('file://', ''));
  // loadNormalizer fetch()es its url; node's fetch can't read local files, so shim it for this call.
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (u) => ({ json: async () => JSON.parse(readFileSync(new URL('assets/normalize.json', APP))) });
  const norm = await loadNormalizer('file:normalize');
  globalThis.fetch = realFetch;

  const opt = { executionProviders: ['wasm'] };
  const sStep = await ort.InferenceSession.create(bufOf('models/gpt_step.onnx'), opt);
  const sVoc = await ort.InferenceSession.create(bufOf('models/vocoder.onnx'), opt);

  function prefillEmbeds(ids, cond) {
    const ti = [ST, ...ids, ET], n = ti.length, L = 32 + n + 1, emb = new Float32Array(L * D);
    emb.set(cond, 0);
    for (let i = 0; i < n; i++) { const dst = (32 + i) * D, te = ti[i] * D, tp = i * D; for (let d = 0; d < D; d++) emb[dst + d] = E.text_emb[te + d] + E.text_pos[tp + d]; }
    const dst = (32 + n) * D, me = SA * D; for (let d = 0; d < D; d++) emb[dst + d] = E.mel_emb[me + d] + E.mel_pos[d];
    return { data: emb, L };
  }
  // O3: reusable decode-embed buffer (ORT copies data on Tensor creation)
  const _decBuf = new Float32Array(D);
  function decodeEmbed(t, pos) { const me = t * D, mp = pos * D; for (let d = 0; d < D; d++) _decBuf[d] = E.mel_emb[me + d] + E.mel_pos[mp + d]; return _decBuf; }
  function emptyPast() { const f = {}; for (let j = 0; j < NL; j++) { f[`past_key_values.${j}.key`] = new ort.Tensor('float32', new Float32Array(0), [1, H, 0, 64]); f[`past_key_values.${j}.value`] = new ort.Tensor('float32', new Float32Array(0), [1, H, 0, 64]); } return f; }
  const repPen = (l, seq) => { for (const t of seq) { const s = l[t]; l[t] = s < 0 ? s * PEN : s / PEN; } };
  const argmax = (a) => { let bi = 0, bv = a[0]; for (let k = 1; k < a.length; k++) if (a[k] > bv) { bv = a[k]; bi = k; } return bi; };

  // O1: pre-allocated stacked-latent buffer (max steps + prefill), reused across calls
  const _latBuf = new Float32Array((MAXG + 1) * D);

  // returns {codes, wav, hitCap} for one chunk of token ids
  async function gen(ids, voice) {
    const v = voices[voice], seq = new Set([1, SA]);
    const { data, L } = prefillEmbeds(ids, v.cond);
    let out = await sStep.run({ inputs_embeds: new ort.Tensor('float32', data, [1, L, D]), ...emptyPast() });

    // O1: write each latent directly into the pre-allocated buffer; no lats[] array
    _latBuf.set(out.latent.data, 0);
    let N = 1;
    // O2: use .data directly — no Float32Array.from; consumed before next run()
    let logits = out.logits.data;
    let past = out; const codes = []; let hitCap = true;

    for (let s = 0; s < MAXG; s++) {
      repPen(logits, seq); const nxt = argmax(logits);
      if (nxt === EA) { codes.push(nxt); hitCap = false; break; }
      codes.push(nxt); seq.add(nxt);
      // O3: decodeEmbed reuses _decBuf
      const feed = { inputs_embeds: new ort.Tensor('float32', decodeEmbed(nxt, codes.length), [1, 1, D]) };
      for (let j = 0; j < NL; j++) { feed[`past_key_values.${j}.key`] = past[`present.${j}.key`]; feed[`past_key_values.${j}.value`] = past[`present.${j}.value`]; }
      out = await sStep.run(feed);
      // O6: dispose previous step's KV tensors — prevents ~170 MB WASM heap accumulation
      for (let j = 0; j < NL; j++) {
        past[`present.${j}.key`]?.dispose?.();
        past[`present.${j}.value`]?.dispose?.();
      }
      // O1: direct write, no allocation
      _latBuf.set(out.latent.data, N * D); N++;
      // O2: no copy
      logits = out.logits.data; past = out;
    }
    // Dispose final step's KV tensors
    for (let j = 0; j < NL; j++) {
      past[`present.${j}.key`]?.dispose?.();
      past[`present.${j}.value`]?.dispose?.();
    }

    // O1: subarray — no extra allocation, no post-loop copy loop
    const g = new ort.Tensor('float32', v.spk, [1, ...v.spkShape]);
    const vout = await sVoc.run({ lat640: new ort.Tensor('float32', _latBuf.subarray(0, N * D), [1, N, D]), g });
    return { codes, wav: Float32Array.from(vout.wav.data), hitCap };
  }

  return { gen, tok, norm, voices, C, SR, MAXG };
}

// ---- code-level degradation metrics (no audio model needed) ----
// max consecutive identical token run
export function maxRun(codes) {
  let best = 1, cur = 1;
  for (let i = 1; i < codes.length; i++) { if (codes[i] === codes[i - 1]) { cur++; best = Math.max(best, cur); } else cur = 1; }
  return codes.length ? best : 0;
}
// fraction of 3-gram positions that are repeats of an earlier 3-gram (loop/stutter proxy)
export function ngramRepeatRate(codes, n = 3) {
  if (codes.length < n + 1) return 0;
  const seen = new Set(); let rep = 0, tot = 0;
  for (let i = 0; i + n <= codes.length; i++) {
    const key = codes.slice(i, i + n).join(',');
    tot++; if (seen.has(key)) rep++; else seen.add(key);
  }
  return rep / tot;
}
