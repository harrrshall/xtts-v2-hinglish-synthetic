// Verify the merged single-graph JS pipeline (host embeddings + gpt_step) reproduces golden codes,
// using the real onnxruntime-web API under node. Mirrors webgpu/app/js/tts.js exactly.
import * as ortNS from 'onnxruntime-web';
import { readFileSync } from 'node:fs';
const ort = ortNS.default || ortNS; ort.env.wasm.numThreads = 1;
const APP = new URL('../app/', import.meta.url);
const rd = (p) => readFileSync(new URL(p, APP));
const cfg = JSON.parse(rd('config.json')); const C = cfg.constants;
const NL = C.layers, H = C.heads, SA = C.start_audio_token, EA = C.stop_audio_token;
const ST = C.start_text_token, ET = C.stop_text_token, PEN = cfg.decode.repetition_penalty, MAXG = C.max_gen_mel_tokens, D = 640;
const fx = JSON.parse(rd('assets/golden_fixtures.json'));
const vmeta = JSON.parse(rd('assets/voices.json'));
const vbuf = new Float32Array(rd('assets/voices.bin').buffer.slice());
const voices = {};
for (const [v, m] of Object.entries(vmeta)) voices[v] = { cond: vbuf.slice(m.cond_off, m.cond_off + m.cond_shape[0]*m.cond_shape[1]) };
const emeta = JSON.parse(rd('assets/embeddings.json'));
const ebuf = new Float32Array(rd('assets/embeddings.bin').buffer.slice());
const E = {}; for (const [k, m] of Object.entries(emeta)) E[k] = ebuf.subarray(m.off, m.off + m.shape[0]*m.shape[1]);
const buf = (p) => { const b = rd(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };

const sStep = await ort.InferenceSession.create(buf('models/gpt_step.onnx'), { executionProviders: ['wasm'] });
console.log('session ready\n');

function prefillEmbeds(ids, cond) {
  const ti = [ST, ...ids, ET], n = ti.length, L = 32 + n + 1, emb = new Float32Array(L * D);
  emb.set(cond, 0);
  for (let i = 0; i < n; i++) { const dst=(32+i)*D, te=ti[i]*D, tp=i*D; for (let d=0;d<D;d++) emb[dst+d]=E.text_emb[te+d]+E.text_pos[tp+d]; }
  const dst=(32+n)*D, me=SA*D; for (let d=0;d<D;d++) emb[dst+d]=E.mel_emb[me+d]+E.mel_pos[d];
  return { data: emb, L };
}
function decodeEmbed(tok, pos){ const e=new Float32Array(D), me=tok*D, mp=pos*D; for(let d=0;d<D;d++) e[d]=E.mel_emb[me+d]+E.mel_pos[mp+d]; return e; }
function emptyPast(){ const f={}; for(let j=0;j<NL;j++){ f[`past_key_values.${j}.key`]=new ort.Tensor('float32',new Float32Array(0),[1,H,0,64]); f[`past_key_values.${j}.value`]=new ort.Tensor('float32',new Float32Array(0),[1,H,0,64]); } return f; }
function repPen(l,seq){ for(const t of seq){ const s=l[t]; l[t]=s<0?s*PEN:s/PEN; } }
function argmax(a){ let bi=0,bv=a[0]; for(let k=1;k<a.length;k++) if(a[k]>bv){bv=a[k];bi=k;} return bi; }

async function gen(ids, voice) {
  const cond = voices[voice].cond, seq = new Set([1, SA]);
  const { data, L } = prefillEmbeds(ids, cond);
  let out = await sStep.run({ inputs_embeds: new ort.Tensor('float32', data, [1, L, D]), ...emptyPast() });
  let logits = Float32Array.from(out.logits.data); let past = out; const codes = [];
  for (let s = 0; s < MAXG; s++) {
    repPen(logits, seq); const nxt = argmax(logits);
    if (nxt === EA) { codes.push(nxt); break; }
    codes.push(nxt); seq.add(nxt);
    const feed = { inputs_embeds: new ort.Tensor('float32', decodeEmbed(nxt, codes.length), [1,1,D]) };
    for (let j=0;j<NL;j++){ feed[`past_key_values.${j}.key`]=past[`present.${j}.key`]; feed[`past_key_values.${j}.value`]=past[`present.${j}.value`]; }
    out = await sStep.run(feed); logits = Float32Array.from(out.logits.data); past = out;
  }
  return codes;
}

let ok = true;
for (const f of fx) {
  const t0 = Date.now(); const codes = await gen(f.ids, f.voice);
  const exact = codes.length === f.codes.length && codes.every((c,k)=>c===f.codes[k]);
  console.log(`#${f.i} ${f.voice.padEnd(9)} golden=${f.codes.length} got=${codes.length} ${exact?'EXACT':'DIFF'} ${((Date.now()-t0)/1000).toFixed(1)}s`);
  ok = ok && exact;
}
console.log(`\n[merge-parity] ${ok ? 'PASS — merged JS pipeline reproduces golden codes exactly' : 'FAIL'}`);
process.exit(ok ? 0 : 1);
