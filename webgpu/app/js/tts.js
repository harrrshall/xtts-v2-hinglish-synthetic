// syntts — in-browser Hinglish TTS. ONE merged GPT graph + host-side embeddings (deduped weights).
// Mirrors the verified Python pipeline (webgpu/export/10_merge_export.py) EXACTLY:
//   inputs_embeds = [cond(32), text_emb+text_pos, start_audio_emb+mel_pos0]   (prefill, empty cache)
//   gpt_step(inputs_embeds, past) -> logits, latent, present
//   greedy + repetition_penalty over seq=[1]*(32+T+2)+[SA]+codes ; decode step feeds one token embed
//   lat640 = stack(latents) ; vocoder(lat640, g=spk) -> wav @24kHz

import { loadTokenizer } from './tokenizer.js';
import { loadNormalizer } from './normalize.js';
import { chunkText, concatWavs, DEFAULT_MAX_TEXT_IDS } from './chunk.js';

const ORT_VER = '1.22.0';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
const HF_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';
const CACHE = 'syntts-v1';
const D = 640;

let ort = null;
async function loadORT() {
  if (ort) return ort;
  ort = await import(`${ORT_CDN}ort.webgpu.min.mjs`);
  ort.env.wasm.wasmPaths = ORT_CDN;
  ort.env.wasm.numThreads = 1;
  return ort;
}
const i64 = (arr) => BigInt64Array.from(arr, (x) => BigInt(x));

export class HinglishTTS {
  constructor(cfg) {
    this.cfg = cfg;
    this.C = cfg.constants;
    this.PEN = cfg.decode.repetition_penalty;
    this.NL = this.C.layers;
    this.H = this.C.heads;
    this.SA = this.C.start_audio_token;
    this.EA = this.C.stop_audio_token;
    this.ST = this.C.start_text_token;
    this.ET = this.C.stop_text_token;
    this.MAXG = this.C.max_gen_mel_tokens;
    this.chunkCfg = cfg.chunk || {};
    this.maxTextIds = this.chunkCfg.maxTextIds || DEFAULT_MAX_TEXT_IDS;
    this.provider = null;
  }

  async _fetchBuf(url, key, label, onFile) {
    const cache = await caches.open(CACHE).catch(() => null);
    if (cache) {
      const hit = await cache.match(url);
      if (hit) { const buf = await hit.arrayBuffer();
        onFile({ key, label, loaded: buf.byteLength, total: buf.byteLength, fromCache: true, done: true }); return buf; }
    }
    const res = await fetch(url);
    const total = +(res.headers.get('content-length') || 0);
    const reader = res.body.getReader(); const chunks = []; let loaded = 0;
    onFile({ key, label, loaded: 0, total, fromCache: false, done: false });
    for (;;) { const { done, value } = await reader.read(); if (done) break;
      chunks.push(value); loaded += value.length; onFile({ key, label, loaded, total, fromCache: false, done: false }); }
    const blob = new Blob(chunks);
    if (cache) await cache.put(url, new Response(blob.slice(), { headers: { 'content-length': String(loaded) } })).catch(() => {});
    onFile({ key, label, loaded, total: loaded, fromCache: false, done: true });
    return await blob.arrayBuffer();
  }

  async init({ onProgress, onFile } = {}) {
    const o = await loadORT();
    const prog = onProgress || (() => {});
    const file = onFile || (() => {});
    const wantGPU = !!navigator.gpu;
    const eps = wantGPU ? ['webgpu', 'wasm'] : ['wasm'];
    if (navigator.storage?.persist) navigator.storage.persist().catch(() => {});

    prog('loading runtime + tokenizer');
    const { PreTrainedTokenizer } = await import(/* @vite-ignore */ HF_CDN);
    this.tok = await loadTokenizer(new URL('assets/tokenizer.json', location.href).href, { PreTrainedTokenizer });
    this.norm = await loadNormalizer(new URL('assets/normalize.json', location.href).href);
    await this._loadVoices();
    await this._loadEmbeddings();

    const opt = { executionProviders: eps, graphOptimizationLevel: 'all' };
    const M = this.cfg.models;
    const url = (p) => this.cfg.modelBase ? new URL(p, this.cfg.modelBase).href : p;
    prog('downloading model');
    const stepBuf = await this._fetchBuf(url(M.step), 'step', 'Acoustic model', file);
    this.sStep = await o.InferenceSession.create(new Uint8Array(stepBuf), opt);
    const vocBuf = await this._fetchBuf(url(M.vocoder), 'vocoder', 'Vocoder', file);
    this.sVoc = await o.InferenceSession.create(new Uint8Array(vocBuf), opt);
    this.provider = wantGPU ? 'webgpu' : 'wasm';
    prog('warming up');
    await this._warmup().catch(() => {});
    prog('ready');
    return this;
  }

  async _loadVoices() {
    const meta = await (await fetch('assets/voices.json')).json();
    const buf = new Float32Array(await (await fetch('assets/voices.bin')).arrayBuffer());
    this.voices = {};
    for (const [v, m] of Object.entries(meta)) {
      const cn = m.cond_shape[0] * m.cond_shape[1], sn = m.spk_shape[0] * m.spk_shape[1];
      this.voices[v] = { cond: buf.slice(m.cond_off, m.cond_off + cn), condShape: m.cond_shape,
                         spk: buf.slice(m.spk_off, m.spk_off + sn), spkShape: m.spk_shape };
    }
  }

  async _loadEmbeddings() {
    const meta = await (await fetch('assets/embeddings.json')).json();
    const buf = new Float32Array(await (await fetch('assets/embeddings.bin')).arrayBuffer());
    this.emb = {};
    for (const [k, m] of Object.entries(meta)) this.emb[k] = buf.subarray(m.off, m.off + m.shape[0] * m.shape[1]);
  }

  // build prefill inputs_embeds: cond + (text_emb+text_pos) + (start_audio_emb+mel_pos[0])
  _prefillEmbeds(ids, cond) {
    const ti = [this.ST, ...ids, this.ET], n = ti.length, L = 32 + n + 1;
    const emb = new Float32Array(L * D);
    emb.set(cond, 0);                                   // 32 cond rows
    const { text_emb: TE, text_pos: TP, mel_emb: ME, mel_pos: MP } = this.emb;
    for (let i = 0; i < n; i++) {
      const dst = (32 + i) * D, te = ti[i] * D, tp = i * D;
      for (let d = 0; d < D; d++) emb[dst + d] = TE[te + d] + TP[tp + d];
    }
    const dst = (32 + n) * D, me = this.SA * D;          // start_audio at mel pos 0
    for (let d = 0; d < D; d++) emb[dst + d] = ME[me + d] + MP[d];
    return { data: emb, L };
  }

  _decodeEmbed(token, pos) {
    const { mel_emb: ME, mel_pos: MP } = this.emb, e = new Float32Array(D), me = token * D, mp = pos * D;
    for (let d = 0; d < D; d++) e[d] = ME[me + d] + MP[mp + d];
    return e;
  }

  _emptyPast(o) {
    const f = {};
    for (let j = 0; j < this.NL; j++) {
      f[`past_key_values.${j}.key`] = new o.Tensor('float32', new Float32Array(0), [1, this.H, 0, 64]);
      f[`past_key_values.${j}.value`] = new o.Tensor('float32', new Float32Array(0), [1, this.H, 0, 64]);
    }
    return f;
  }

  async _warmup() {
    const o = ort, v = this.voices[this.cfg.voices[0]];
    const { data, L } = this._prefillEmbeds([10, 11], v.cond);
    const out = await this.sStep.run({ inputs_embeds: new o.Tensor('float32', data, [1, L, 640]), ...this._emptyPast(o) });
    const feed = { inputs_embeds: new o.Tensor('float32', this._decodeEmbed(5, 1), [1, 1, 640]) };
    for (let j = 0; j < this.NL; j++) { feed[`past_key_values.${j}.key`] = out[`present.${j}.key`]; feed[`past_key_values.${j}.value`] = out[`present.${j}.value`]; }
    await this.sStep.run(feed);
    const g = new o.Tensor('float32', v.spk, [1, ...v.spkShape]);
    await this.sVoc.run({ lat640: new o.Tensor('float32', new Float32Array(640), [1, 1, 640]), g });
  }

  _repPenalty(logits, seqSet) { const p = this.PEN; for (const t of seqSet) { const s = logits[t]; logits[t] = s < 0 ? s * p : s / p; } }
  _argmax(a) { let bi = 0, bv = a[0]; for (let k = 1; k < a.length; k++) if (a[k] > bv) { bv = a[k]; bi = k; } return bi; }
  normalizeText(text) { return this.norm ? this.norm.normalize(text) : { out: text, spans: [] }; }

  // Synthesize ONE chunk of already-normalized text. Bit-identical to the
  // original single-pass generate() loop. Returns { wav, codes, latents... }.
  async _genChunk(normText, v, { onToken, tokenBase = 0 } = {}) {
    const o = ort;
    const ids = this.tok.encode(normText, 'hi');
    const seqSet = new Set([1, this.SA]);

    const { data: pe, L } = this._prefillEmbeds(ids, v.cond);
    let out = await this.sStep.run({ inputs_embeds: new o.Tensor('float32', pe, [1, L, 640]), ...this._emptyPast(o) });
    let logits = Float32Array.from(out.logits.data);
    const lats = [Float32Array.from(out.latent.data)];
    let past = out;
    const codes = [];
    for (let step = 0; step < this.MAXG; step++) {
      this._repPenalty(logits, seqSet);
      const nxt = this._argmax(logits);
      if (nxt === this.EA) { codes.push(nxt); break; }
      codes.push(nxt); seqSet.add(nxt);
      if (onToken && (step % 8 === 0)) onToken(tokenBase + codes.length);
      const feed = { inputs_embeds: new o.Tensor('float32', this._decodeEmbed(nxt, codes.length), [1, 1, 640]) };
      for (let j = 0; j < this.NL; j++) { feed[`past_key_values.${j}.key`] = past[`present.${j}.key`]; feed[`past_key_values.${j}.value`] = past[`present.${j}.value`]; }
      out = await this.sStep.run(feed);
      logits = Float32Array.from(out.logits.data);
      lats.push(Float32Array.from(out.latent.data));
      past = out;
    }
    const N = lats.length, lat = new Float32Array(N * D);
    for (let k = 0; k < N; k++) lat.set(lats[k], k * D);
    const g = new o.Tensor('float32', v.spk, [1, ...v.spkShape]);
    const vout = await this.sVoc.run({ lat640: new o.Tensor('float32', lat, [1, N, 640]), g });
    return { wav: Float32Array.from(vout.wav.data), codes };
  }

  async generate(text, voice, { onToken, autoNormalize = true } = {}) {
    const v = this.voices[voice] || this.voices[this.cfg.voices[0]];
    const normText = autoNormalize ? this.normalizeText(text).out : text;
    const t0 = performance.now();

    // Length-independent path: split long input into sentence-aligned chunks,
    // synthesize each at full fidelity, concatenate. Single chunk => unchanged.
    const chunks = chunkText(normText, (t) => this.tok.encode(t, 'hi').length, this.maxTextIds);
    if (chunks.length > 1) {
      const wavs = []; const allCodes = []; let tokenBase = 0;
      for (const ch of chunks) {
        const r = await this._genChunk(ch, v, { onToken, tokenBase });
        wavs.push(r.wav); allCodes.push(...r.codes); tokenBase += r.codes.length;
      }
      const cc = this.chunkCfg;
      const wav = concatWavs(wavs, this.C.sample_rate, {
        gapMs: cc.gapMs ?? 90, crossfadeMs: cc.crossfadeMs ?? 4,
        ampThresh: cc.ampThresh ?? 0.01, padMs: cc.padMs ?? 30,
      });
      const tTotal = performance.now() - t0;
      return { wav, sampleRate: this.C.sample_rate, codes: allCodes, nCodes: allCodes.length,
               tGen: tTotal, tTotal, nChunks: chunks.length, chunks,
               durSec: wav.length / this.C.sample_rate, provider: this.provider, normText, inputText: text };
    }

    const { wav, codes } = await this._genChunk(normText, v, { onToken });
    const tTotal = performance.now() - t0;
    return { wav, sampleRate: this.C.sample_rate, codes, nCodes: codes.length, tGen: tTotal, tTotal,
             nChunks: 1, durSec: wav.length / this.C.sample_rate, provider: this.provider, normText, inputText: text };
  }
}

// ---- audio helpers ----
export async function playPCM(pcm, sampleRate) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
  if (ctx.state === 'suspended') await ctx.resume();
  const buf = ctx.createBuffer(1, pcm.length, sampleRate);
  buf.copyToChannel(pcm, 0);
  const src = ctx.createBufferSource(); src.buffer = buf; src.connect(ctx.destination); src.start();
  return ctx;
}

export function pcmToWav(pcm, sampleRate) {
  const n = pcm.length, buf = new ArrayBuffer(44 + n * 2), dv = new DataView(buf);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, 'RIFF'); dv.setUint32(4, 36 + n * 2, true); ws(8, 'WAVE'); ws(12, 'fmt ');
  dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, sampleRate, true); dv.setUint32(28, sampleRate * 2, true);
  dv.setUint16(32, 2, true); dv.setUint16(34, 16, true); ws(36, 'data'); dv.setUint32(40, n * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++) { let s = Math.max(-1, Math.min(1, pcm[i])); dv.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true); o += 2; }
  return new Blob([buf], { type: 'audio/wav' });
}
