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
const HF_CDN  = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';
const CACHE   = 'syntts-v1';
const D       = 640;

let ort = null;
async function loadORT() {
  if (ort) return ort;
  ort = await import(`${ORT_CDN}ort.webgpu.min.mjs`);
  ort.env.wasm.wasmPaths = ORT_CDN;
  // Use 4 threads when crossOriginIsolated (COOP+COEP headers set); fall back to 1 otherwise.
  ort.env.wasm.numThreads = (typeof self !== 'undefined' && self.crossOriginIsolated) ? 4 : 1;
  return ort;
}

export class HinglishTTS {
  constructor(cfg) {
    this.cfg     = cfg;
    this.C       = cfg.constants;
    this.PEN     = cfg.decode.repetition_penalty;
    this.NL      = this.C.layers;
    this.H       = this.C.heads;
    this.SA      = this.C.start_audio_token;
    this.EA      = this.C.stop_audio_token;
    this.ST      = this.C.start_text_token;
    this.ET      = this.C.stop_text_token;
    this.MAXG    = this.C.max_gen_mel_tokens;
    this.chunkCfg  = cfg.chunk || {};
    this.maxTextIds = this.chunkCfg.maxTextIds || DEFAULT_MAX_TEXT_IDS;
    this.provider  = null;
    // O3: single reusable decode-embed buffer (ORT copies data into WASM/GPU on Tensor creation)
    this._decBuf = new Float32Array(D);
    // O1: one reusable stacked-latent buffer (max decode steps + prefill)
    this._latBuf = new Float32Array((this.MAXG + 1) * D);
    // O7: pre-compute KV key strings once; avoids template-literal concat per decode step
    this._pastKeys = []; this._pastVals = []; this._presKeys = []; this._presVals = [];
    for (let j = 0; j < this.NL; j++) {
      this._pastKeys.push(`past_key_values.${j}.key`);
      this._pastVals.push(`past_key_values.${j}.value`);
      this._presKeys.push(`present.${j}.key`);
      this._presVals.push(`present.${j}.value`);
    }
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
    const file  = onFile    || (() => {});
    const wantGPU = !!navigator.gpu && !!(await navigator.gpu.requestAdapter().catch(() => null));
    const eps = wantGPU ? ['webgpu', 'wasm'] : ['wasm'];
    if (navigator.storage?.persist) navigator.storage.persist().catch(() => {});

    // O4: request high-performance GPU when available
    if (wantGPU && o.env.webgpu) o.env.webgpu.powerPreference = 'high-performance';

    prog('loading runtime + tokenizer');
    const { PreTrainedTokenizer } = await import(/* @vite-ignore */ HF_CDN);
    this.tok  = await loadTokenizer(new URL('assets/tokenizer.json', location.href).href, { PreTrainedTokenizer });
    this.norm = await loadNormalizer(new URL('assets/normalize.json', location.href).href);
    await this._loadVoices();
    await this._loadEmbeddings();

    // O5 (WebGPU path): keep KV-cache output tensors on the GPU across decode steps.
    // This eliminates 2 × NL GPU→CPU→GPU transfers per decode step. On the WASM path
    // this option is ignored. Output names that need to stay on CPU (logits, latent)
    // are left at their default 'cpu' location.
    const kvGpuLoc = {};
    if (wantGPU) {
      for (let j = 0; j < this.NL; j++) {
        kvGpuLoc[`present.${j}.key`]   = 'gpu-buffer';
        kvGpuLoc[`present.${j}.value`] = 'gpu-buffer';
      }
    }

    const stepOpt = {
      executionProviders: eps,
      graphOptimizationLevel: 'all',
      ...(wantGPU && Object.keys(kvGpuLoc).length ? { preferredOutputLocation: kvGpuLoc } : {}),
    };
    const vocOpt = { executionProviders: eps, graphOptimizationLevel: 'all' };

    const M = this.cfg.models;
    const url = (p) => this.cfg.modelBase ? new URL(p, this.cfg.modelBase).href : p;

    // On WebGPU, prefer fp16 models (~2x faster: WebGPU EP supports fp16 natively,
    // unlike WASM EP which requires fp16 inputs and has no auto-cast).
    let useFp16 = false;
    if (wantGPU && M.step_fp16) {
      try {
        const probe = await fetch(url(M.step_fp16), { method: 'HEAD' });
        if (probe.ok) useFp16 = true;
      } catch {}
    }
    const stepPath  = useFp16 ? M.step_fp16  : M.step;
    const vocPath   = useFp16 ? M.vocoder_fp16 : M.vocoder;
    const stepLabel = useFp16 ? 'Acoustic model (fp16)' : 'Acoustic model';
    const vocLabel  = useFp16 ? 'Vocoder (fp16)'        : 'Vocoder';

    prog('downloading model');
    const stepBuf = await this._fetchBuf(url(stepPath), 'step', stepLabel, file);
    this.sStep = await o.InferenceSession.create(new Uint8Array(stepBuf), stepOpt);
    const vocBuf = await this._fetchBuf(url(vocPath), 'vocoder', vocLabel, file);
    this.sVoc = await o.InferenceSession.create(new Uint8Array(vocBuf), vocOpt);
    this.provider = wantGPU ? (useFp16 ? 'webgpu-fp16' : 'webgpu') : 'wasm';

    // Pre-warm voice conditioning KV: run sStep once per voice on its 32 cond tokens.
    // Each gen() call then only processes the shorter text+SA sequence as the prefill.
    // Math: identical to full-prefill by KV-cache / causal-mask property (proven by parity gate).
    prog('precomputing voice embeddings');
    for (const [, v] of Object.entries(this.voices)) {
      const condOut = await this.sStep.run({
        inputs_embeds: new o.Tensor('float32', v.cond, [1, v.condShape[0], D]),
        ...this._emptyPast(o),
      });
      v.condKv = {};
      for (let j = 0; j < this.NL; j++) {
        v.condKv[this._pastKeys[j]] = condOut[this._presKeys[j]];
        v.condKv[this._pastVals[j]] = condOut[this._presVals[j]];
      }
    }

    prog('warming up');
    await this._warmup().catch(() => {});
    prog('ready');
    return this;
  }

  async _loadVoices() {
    const meta = await (await fetch('assets/voices.json')).json();
    const buf  = new Float32Array(await (await fetch('assets/voices.bin')).arrayBuffer());
    this.voices = {};
    for (const [v, m] of Object.entries(meta)) {
      const cn = m.cond_shape[0] * m.cond_shape[1], sn = m.spk_shape[0] * m.spk_shape[1];
      this.voices[v] = { cond: buf.slice(m.cond_off, m.cond_off + cn), condShape: m.cond_shape,
                         spk:  buf.slice(m.spk_off,  m.spk_off  + sn), spkShape:  m.spk_shape };
    }
  }

  async _loadEmbeddings() {
    const meta = await (await fetch('assets/embeddings.json')).json();
    const buf  = new Float32Array(await (await fetch('assets/embeddings.bin')).arrayBuffer());
    this.emb = {};
    for (const [k, m] of Object.entries(meta)) this.emb[k] = buf.subarray(m.off, m.off + m.shape[0] * m.shape[1]);
  }

  // Build prefill inputs_embeds: (text_emb+text_pos) + (start_audio_emb+mel_pos[0]).
  // Cond tokens are excluded — they are pre-processed per-voice via condKv in init().
  _prefillEmbeds(ids) {
    const ti = [this.ST, ...ids, this.ET], n = ti.length, L = n + 1;
    const emb = new Float32Array(L * D);
    const { text_emb: TE, text_pos: TP, mel_emb: ME, mel_pos: MP } = this.emb;
    for (let i = 0; i < n; i++) {
      const dst = i * D, te = ti[i] * D, tp = i * D;
      for (let d = 0; d < D; d++) emb[dst + d] = TE[te + d] + TP[tp + d];
    }
    const dst = n * D, me = this.SA * D;
    for (let d = 0; d < D; d++) emb[dst + d] = ME[me + d] + MP[d];
    return { data: emb, L };
  }

  // O3: write into reusable buffer; ORT copies it during Tensor creation so the buffer
  // is safe to overwrite on the next call (after the preceding run() has returned).
  _decodeEmbed(token, pos) {
    const { mel_emb: ME, mel_pos: MP } = this.emb, me = token * D, mp = pos * D;
    for (let d = 0; d < D; d++) this._decBuf[d] = ME[me + d] + MP[mp + d];
    return this._decBuf;
  }

  _emptyPast(o) {
    const f = {};
    for (let j = 0; j < this.NL; j++) {
      f[this._pastKeys[j]] = new o.Tensor('float32', new Float32Array(0), [1, this.H, 0, 64]);
      f[this._pastVals[j]] = new o.Tensor('float32', new Float32Array(0), [1, this.H, 0, 64]);
    }
    return f;
  }

  async _warmup() {
    const o = ort, v = this.voices[this.cfg.voices[0]];
    const { data, L } = this._prefillEmbeds([10, 11]);
    const out = await this.sStep.run({ inputs_embeds: new o.Tensor('float32', data, [1, L, 640]), ...v.condKv });
    const feed = { inputs_embeds: new o.Tensor('float32', this._decodeEmbed(5, 1), [1, 1, 640]) };
    for (let j = 0; j < this.NL; j++) { feed[this._pastKeys[j]] = out[this._presKeys[j]]; feed[this._pastVals[j]] = out[this._presVals[j]]; }
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

    const { data: pe, L } = this._prefillEmbeds(ids);
    const tPrefillStart = performance.now();
    let out = await this.sStep.run({ inputs_embeds: new o.Tensor('float32', pe, [1, L, 640]), ...v.condKv });
    const tPrefill = performance.now() - tPrefillStart;

    // O1: pre-allocated stacked latent buffer — write each step directly, no lats[] array,
    //     no per-step allocation, no post-loop assembly. this._latBuf is reused across calls.
    const lat = this._latBuf;
    lat.set(out.latent.data, 0);
    let N = 1;

    // O2: use logits .data directly — no Float32Array.from() copy.
    // Safe: we consume logits fully (repPenalty + argmax) before the next run() call,
    // and ORT allocates new tensor buffers on every run(), so the ref won't alias future output.
    let logits = out.logits.data;
    let past = out;
    const codes = [];
    const tDecodeStart = performance.now();

    for (let step = 0; step < this.MAXG; step++) {
      this._repPenalty(logits, seqSet);
      const nxt = this._argmax(logits);
      if (nxt === this.EA) { codes.push(nxt); break; }
      codes.push(nxt); seqSet.add(nxt);
      if (onToken && (step % 8 === 0)) onToken(tokenBase + codes.length);

      // O3: _decodeEmbed reuses this._decBuf — ORT copies on Tensor creation
      const feed = { inputs_embeds: new o.Tensor('float32', this._decodeEmbed(nxt, codes.length), [1, 1, 640]) };
      // O7: use pre-cached key arrays instead of template-literal concat per step
      for (let j = 0; j < this.NL; j++) {
        feed[this._pastKeys[j]] = past[this._presKeys[j]];
        feed[this._pastVals[j]] = past[this._presVals[j]];
      }
      out = await this.sStep.run(feed);

      // O6: dispose previous step's KV tensors immediately after run() consumes them.
      // Without this, WASM heap / GPU VRAM accumulates ~170 MB of KV garbage across all
      // decode steps, which triggers a massive GC pause (we measured a 32s outlier).
      for (let j = 0; j < this.NL; j++) {
        past[this._presKeys[j]]?.dispose?.();
        past[this._presVals[j]]?.dispose?.();
      }

      // O1: direct write into pre-allocated buffer
      lat.set(out.latent.data, N * D);
      N++;
      // O2: no copy
      logits = out.logits.data;
      past = out;
    }
    const tDecode = performance.now() - tDecodeStart;

    // Dispose final step's KV tensors (loop ended without another step consuming them)
    for (let j = 0; j < this.NL; j++) {
      past[this._presKeys[j]]?.dispose?.();
      past[this._presVals[j]]?.dispose?.();
    }

    // O1: subarray avoids a separate N*D allocation; vocoder reads the live view
    const g = new o.Tensor('float32', v.spk, [1, ...v.spkShape]);
    const tVocStart = performance.now();
    const vout = await this.sVoc.run({ lat640: new o.Tensor('float32', lat.subarray(0, N * D), [1, N, 640]), g });
    const tVocoder = performance.now() - tVocStart;

    return {
      wav: Float32Array.from(vout.wav.data),
      codes,
      tPrefill,
      tDecode,
      tVocoder,
      nSteps: N,
    };
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
      let tPrefill = 0, tDecode = 0, tVocoder = 0, nSteps = 0;
      for (const ch of chunks) {
        const r = await this._genChunk(ch, v, { onToken, tokenBase });
        wavs.push(r.wav); allCodes.push(...r.codes); tokenBase += r.codes.length;
        tPrefill += r.tPrefill; tDecode += r.tDecode; tVocoder += r.tVocoder; nSteps += r.nSteps;
      }
      const cc = this.chunkCfg;
      const wav = concatWavs(wavs, this.C.sample_rate, {
        gapMs: cc.gapMs ?? 90, crossfadeMs: cc.crossfadeMs ?? 4,
        ampThresh: cc.ampThresh ?? 0.01, padMs: cc.padMs ?? 30,
      });
      const tTotal = performance.now() - t0;
      return { wav, sampleRate: this.C.sample_rate, codes: allCodes, nCodes: allCodes.length,
               tGen: tTotal, tTotal, nChunks: chunks.length, chunks,
               tPrefill, tDecode, tVocoder, nSteps,
               durSec: wav.length / this.C.sample_rate, provider: this.provider, normText, inputText: text };
    }

    const { wav, codes, tPrefill, tDecode, tVocoder, nSteps } = await this._genChunk(normText, v, { onToken });
    const tTotal = performance.now() - t0;
    return { wav, sampleRate: this.C.sample_rate, codes, nCodes: codes.length,
             tGen: tTotal, tTotal, nChunks: 1,
             tPrefill, tDecode, tVocoder, nSteps,
             durSec: wav.length / this.C.sample_rate, provider: this.provider, normText, inputText: text };
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
