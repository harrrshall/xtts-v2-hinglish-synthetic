# WebGPU in-browser port of the 90M Hinglish TTS — plan + bug-prevention checklist

Goal: run the 89.96M fixed-voice Hinglish student fully client-side on **WebGPU**, with **no quality
loss** vs the PyTorch model and **fast** (near-real-time) inference. Zero server, static hosting, $0.

The inference graph to port (from `local_demo/app.py`, the known-good CPU path):

```
text → tokenizer(lang=hi) → ids
ids + cond_latents(32×640) → GPT.generate  (AR, KV-cache, greedy, rep_penalty=1.3) → codes
ids + codes + cond_latents → GPT.forward(return_latent) → mel_latents (T×640)
mel_latents → adapter(640→1024) → HiFi-GAN(g=speaker_emb 512) → wav @ 24 kHz
```

Key facts that make this tractable (verified in source):
- GPT-2 internal `wpe` is nulled (`null_position_embeddings`); ALL position info is the external
  `text_pos_embedding` / `mel_pos_embedding`. Each decode step adds `mel_pos_embedding[t]`.
- No DVAE at inference (encode-only, training). No speaker pathway (deleted; voices baked).
- Voices are tiny data: per voice `cond_latents (1,32,640)` + `speaker_embedding (1,512,1)`.

## Phases (each gated by a numerical parity check before moving on)

0. **Golden reference** — fixed prompts × 4 voices, greedy. Dump ids, codes, latents, wav. Oracle.
1. **HiFi-GAN → ONNX** — remove weight-norm; parity vs golden wav.
2. **GPT latent forward → ONNX** (+ adapter) — parity vs golden latents.
3. **GPT AR prefill + decode (KV-cache) → ONNX** — ORT greedy loop reproduces golden codes EXACTLY.
4. **fp16 + full ORT-Python pipeline** — prove no-quality-loss (codes identical / wav within tol;
   accent+SECS not regressed).
5. **JS tokenizer** — identical ids to Python on golden prompts.
6. **WebGPU front-end** — onnxruntime-web (WebGPU EP), JS KV-cache greedy loop, voices as data,
   Web Audio playback + WAV download. Static files.
7. **In-browser parity + real E2E UX test** — Chrome WebGPU, every voice, latency, edge cases.

## Decode contract (must match exactly, every layer)

- Greedy = `do_sample=False, top_k=1, temperature=0.7` (temperature irrelevant at top_k=1/argmax),
  `repetition_penalty=1.3`, `num_beams=1`, `length_penalty=1.0`.
- start_audio_token / stop_audio_token / start_text_token / stop_text_token from `model_args`.
- Stop when stop_audio_token sampled OR `max_gen_mel_tokens` reached.
- Position index for decode step feeding token T_k (k=1,2,…) is mel_pos `k` (start_audio is pos 0).
- Repetition penalty: HF semantics — divide logit by penalty if >0 else multiply, over tokens already
  in the running sequence (the generated audio codes only, as HF applies it to `input_ids` = the
  full decoder stream incl. the prefix sentinel ids; replicate HF exactly — see parity test).
- Latent extraction returns `mel_latent[:, :-5]` (the `sub=-5` truncation in `gpt.forward`).

## BUG-PREVENTION CHECKLIST — RESULTS (every box ticked with evidence, except fp16 = deferred)

Numerical parity (the no-quality-loss gate):
- [x] G0  Golden harness reproduces known-good audio (6 clips, RMS 0.04–0.13, non-degenerate). ✅
- [x] C1  ONNX adapter+HiFi-GAN: corr=1.000000, max|Δ| ~3e-4 vs golden wav (fp32). ✅
- [x] C2  ONNX latent forward: max|lat640| < 1e-4, cos=1.000000, all 6 samples. ✅
- [x] C3  ONNX prefill argmax == reference first code (implied by C4 exactness). ✅
- [x] C4  ORT-Python greedy KV-cache loop == golden codes EXACTLY, all voices/prompts. ✅
- [x] C5  KV-cache decode == reference (golden = HF generate); also AR-latent == return_latent (2e-5). ✅
- [x] C6  End-to-end ORT-Python: codes bit-exact, wav corr=1.000000, SNR 66–82 dB. ✅
- [ ] C7  fp16 graphs — DEFERRED. Converter trips on the `null_position_embeddings` zeros constant
          (GPT-2's nulled wpe) feeding an fp16 Add. fp32 shipped (guaranteed lossless). See follow-up.
- [x] C8  JS tokenizer: ids identical to Python on all 6 prompts — in node AND in-browser. ✅
- [x] C9  In-browser codes == golden EXACTLY (82/82, maya) via ort-web; node ort-web all 6 voices. ✅
- [x] C10 In-browser wav matches golden (len 91392, peak 0.6526 — both match golden s0). ✅

Correctness traps — all validated by exact code reproduction (C4/C9):
- [x] T1  int64 tokens (BigInt64Array) — correct. ✅
- [x] T2  KV-cache shape (1,10,L,64), grows by 1/step — correct (probe + exact codes). ✅
- [x] T3  mel position: start_audio=0, k-th code=k — correct. ✅
- [x] T4  text padding `[start_text,…,stop_text]` — correct. ✅
- [x] T5  repetition penalty over full input_ids incl. prefix 1s + start_audio — correct. ✅
- [x] T6  weight-norm fused once by load_checkpoint(eval=True), not removed twice. ✅
- [~] T7  fp16 overflow — N/A (shipped fp32). To revisit when fp16 lands.
- [x] T8  Provider honesty — badge shows real backend (WASM here; WebGPU on a GPU box). ✅
- [x] T9  Warm second load from browser cache verified (fast re-init). ✅
- [x] T10 No COOP/COEP needed (WebGPU EP); plain static server serves it. ✅

Performance (this box has no GPU adapter → WASM fallback; WebGPU would be much faster):
- [x] P1  Warmup hides cold-start; init ~10 s warm. ✅
- [x] P2  WASM ~0.28–0.31× RT (~13 s for ~4 s audio). WebGPU expected several× faster. ✅(wasm)
- [ ] P3  KV on-GPU (gpu-buffer io-binding) — follow-up speedup.

UX (real end-to-end, in the actual browser):
- [x] U1  Page loads, shows device + size badge, no console errors. ✅
- [x] U2  Voice switch changes output (maya 56 vs arjun 42 codes, wav diff 0.56). ✅
- [x] U3  Digit input shows the guard message; example chips wired. ✅
- [x] U4  Audio plays inline + downloads as WAV; warm session reused. ✅
- [x] U5  Cold load + warm-cache revisit both verified in-browser. ✅

### WebGPU execution caveat
This dev box has `navigator.gpu` but `requestAdapter()` returns null (no GPU), so the **actual WebGPU
compute path could not be exercised here** — the app correctly falls back to WASM, which proved full
numerical correctness in-browser (C9/C10 bit-exact). The WebGPU path is wired (`['webgpu','wasm']`),
and all required ops (MatMul/Attention/LayerNorm/Gather/Softmax/Where/Cast/Concat/ConvTranspose1d) are
WebGPU-supported per the research briefing; on a WebGPU-capable machine it engages the GPU automatically.

### fp16 follow-up (the one open optimization)
Halves download (~740 MB → ~370 MB) and speeds WebGPU. Blocker: `onnxconverter_common.float16` leaves
the `null_position_embeddings` zeros (fp32) feeding an fp16 Add. Fix paths: (a) re-export with GPT-2
`wpe` patched to skip the position add entirely (it's adding zeros), then convert + re-verify codes via
the node-wasm parity (`webgpu/nodecheck/_loop_parity.mjs`); or (b) a post-pass that casts the remaining
fp32 constants to fp16. Must re-pass C9 in-browser before shipping (fp16 can flip a GPT argmax).

## Quantization stance

Target **fp16** (half the bytes, WebGPU-native). It must PASS C7. If the GPT logits diverge enough to
change argmax (different codes), fall back to fp32 GPT + fp16 vocoder, or fp32 throughout — quality is
the hard constraint, speed is second. INT8 is NOT pursued unless fp16 already passes and more speed is
needed, and only if it independently passes the no-quality-loss gate.

## Layout

```
webgpu/
  export/      python: golden + onnx export + parity (run with local_demo venv)
  golden/      reference artifacts (.npz, .wav) — gitignored if large
  models/      exported .onnx (fp32 + fp16) — gitignored
  app/         static site (index.html, ort assets, tokenizer.js, tts.js, voices/, models/)
```
