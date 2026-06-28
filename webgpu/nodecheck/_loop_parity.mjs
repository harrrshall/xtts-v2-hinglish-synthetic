// Validate the JS generation loop (KV-cache feed/fetch, int64 tensors, rep-penalty, latent
// stacking, vocoder) against golden codes — using the REAL onnxruntime-web API under node (wasm EP),
// before touching WebGPU. Mirrors webgpu/app/js/tts.js exactly.
import * as ortNS from 'onnxruntime-web';
import { readFileSync } from 'node:fs';
const ort = ortNS.default || ortNS;
ort.env.wasm.numThreads = 1;

const APP = new URL('../app/', import.meta.url);
const rd = (p) => readFileSync(new URL(p, APP));
const cfg = JSON.parse(rd('config.json'));
const C = cfg.constants, PEN = cfg.decode.repetition_penalty, NL = C.layers;
const SA = C.start_audio_token, EA = C.stop_audio_token, MAXG = C.max_gen_mel_tokens;
const fixtures = JSON.parse(rd('assets/golden_fixtures.json'));
const vmeta = JSON.parse(rd('assets/voices.json'));
const vbuf = new Float32Array(rd('assets/voices.bin').buffer);
const voices = {};
for (const [v, m] of Object.entries(vmeta)) {
  const cn = m.cond_shape[0] * m.cond_shape[1], sn = m.spk_shape[0] * m.spk_shape[1];
  voices[v] = { cond: vbuf.slice(m.cond_off, m.cond_off + cn), condShape: m.cond_shape,
                spk: vbuf.slice(m.spk_off, m.spk_off + sn), spkShape: m.spk_shape };
}
const i64 = (a) => BigInt64Array.from(a, (x) => BigInt(x));
const buf = (p) => { const b = rd(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };

console.log('loading sessions (wasm)…');
const opt = { executionProviders: ['wasm'] };
const sPre = await ort.InferenceSession.create(buf('models/gpt_prefill.onnx'), opt);
const sDec = await ort.InferenceSession.create(buf('models/gpt_decode.onnx'), opt);
const sVoc = await ort.InferenceSession.create(buf('models/vocoder.onnx'), opt);
console.log('sessions ready\n');

function repPen(logits, seqSet) { for (const t of seqSet) { const s = logits[t]; logits[t] = s < 0 ? s * PEN : s / PEN; } }
function argmax(a) { let bi = 0, bv = a[0]; for (let k = 1; k < a.length; k++) if (a[k] > bv) { bv = a[k]; bi = k; } return bi; }

async function gen(ids, voice) {
  const v = voices[voice], T = ids.length;
  const cond = new ort.Tensor('float32', v.cond, [1, ...v.condShape]);
  const seqSet = new Set([1, SA]);
  let out = await sPre.run({ text_ids: new ort.Tensor('int64', i64(ids), [1, T]), cond });
  let logits = Float32Array.from(out.logits.data);
  const lats = [Float32Array.from(out.latent.data)];
  let past = out; const codes = [];
  for (let step = 0; step < MAXG; step++) {
    repPen(logits, seqSet);
    const nxt = argmax(logits);
    if (nxt === EA) { codes.push(nxt); break; }
    codes.push(nxt); seqSet.add(nxt);
    const feed = { input_id: new ort.Tensor('int64', i64([nxt]), [1, 1]),
                   pos: new ort.Tensor('int64', i64([codes.length]), [1]) };
    for (let j = 0; j < NL; j++) { feed[`past_key_values.${j}.key`] = past[`present.${j}.key`];
                                   feed[`past_key_values.${j}.value`] = past[`present.${j}.value`]; }
    out = await sDec.run(feed); logits = Float32Array.from(out.logits.data);
    lats.push(Float32Array.from(out.latent.data)); past = out;
  }
  const N = lats.length, lat = new Float32Array(N * 640);
  for (let k = 0; k < N; k++) lat.set(lats[k], k * 640);
  const g = new ort.Tensor('float32', v.spk, [1, ...v.spkShape]);
  const vout = await sVoc.run({ lat640: new ort.Tensor('float32', lat, [1, N, 640]), g });
  return { codes, wav: Float32Array.from(vout.wav.data) };
}

const N = process.env.NFIX ? +process.env.NFIX : fixtures.length;
let ok = true;
for (const fx of fixtures.slice(0, N)) {
  const t0 = Date.now();
  const { codes, wav } = await gen(fx.ids, fx.voice);
  const exact = codes.length === fx.codes.length && codes.every((c, k) => c === fx.codes[k]);
  const nm = codes.filter((c, k) => c === fx.codes[k]).length;
  console.log(`#${fx.i} ${fx.voice.padEnd(9)} golden=${fx.codes.length} got=${codes.length} ` +
              `exact=${exact ? 'YES' : `NO(${nm}/${fx.codes.length})`} wav=${(wav.length/24000).toFixed(2)}s ${((Date.now()-t0)/1000).toFixed(1)}s`);
  ok = ok && exact;
}
console.log(`\n[C9-node] ${ok ? 'PASS — JS loop (ort-web wasm) reproduces golden codes exactly' : 'FAIL'}`);
process.exit(ok ? 0 : 1);
