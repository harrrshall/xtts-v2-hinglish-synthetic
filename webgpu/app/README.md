# Hinglish TTS (89.96M) — fully in-browser, WebGPU

A static web app that runs the sub-100M fixed-voice Hindi-English code-switch TTS model **entirely
client-side** on **WebGPU** (with a WASM CPU fallback). No server, no backend. The visitor's device
does all the inference.

Verified: in-browser output is **bit-exact** to the PyTorch reference (audio codes identical, waveform
length + peak match the golden), so there is **no quality loss** vs the original model.

## UI

Premium minimal light theme (white + blue), **Geist** for Latin and **Noto Sans Devanagari** for Hindi.
- **First visit:** a download panel shows overall percent, MB downloaded of total, and a per-file bar for
  each model (prefill / decode / vocoder), streamed and cached via the Cache API so revisits are instant.
- **Editable conversion:** the "Model will speak" field shows the romanized-to-Devanagari conversion live
  and is editable, so the user can tweak exactly what gets spoken (an `auto` / `edited` pill + reset).
- **Generate:** an embedded gradient button that shows a spinner and a live timer while generating, with a
  token progress bar; the result has an inline player and a Download .wav link.
- Decoration is the speech waveform the model produces (top-right) and a flowing voice signal (bottom-left).
  The whole app fits one screen without scrolling.

## Run locally

```bash
cd webgpu/app
python3 -m http.server 8011        # any static server works
# open http://127.0.0.1:8011/index.html  in a WebGPU browser (Chrome/Edge 121+, Safari 26+)
```

First load downloads ~740 MB of ONNX (cached by the browser afterwards). On a machine with a real
GPU it uses WebGPU and runs near/above real-time; with no GPU adapter it falls back to WASM (slower,
still correct). The badge shows which backend is active.

## What it is

- 4 fixed voices: **aadya, arjun, kaustubh, maya**. No zero-shot cloning (that trade is what makes
  <100M possible).
- **Type romanized Hinglish** ("yaar kal ka match dekha") OR Devanagari+English. A client-side
  normalizer (`js/normalize.js`) auto-converts Hindi words to **Devanagari** and keeps English in
  **Roman** — the form the model pronounces best — and fixes Indian-name pronunciation (Kaustubh →
  कौस्तुभ). A live preview shows the conversion; toggle it off to type the exact form yourself.
  See `docs/HINGLISH_NORMALIZER.md`.
- **Numbers spelled as words** (digits are rejected — they crash the Hindi tokenizer). The UI guards.
- Greedy decode, `repetition_penalty=1.3` — identical contract to the reference model.

## How it works (the pipeline, all in JS + ONNX)

```
text ─tokenizer.js (Transformers.js BPE + hi cleaners)→ ids
ids + cond ─gpt_prefill.onnx→ logits, latent₀, KV          (onnxruntime-web, WebGPU EP)
loop: greedy+rep-penalty → code ; gpt_decode.onnx(code,pos,KV) → logits, latentₖ, KV
lat = stack(latents) ─vocoder.onnx(lat, g=speaker_emb)→ wav @24kHz   (adapter folded in)
Web Audio playback + WAV download
```

Three ONNX graphs (no separate latent graph — the AR loop emits the vocoder latents directly):
- `models/gpt_prefill.onnx` — text+cond → first logits + latent + KV cache.
- `models/gpt_decode.onnx` — one token + position + KV → logits + latent + new KV.
- `models/vocoder.onnx` — 640-d latents + speaker embedding → waveform (640→1024 adapter + HiFi-GAN).

Assets: `assets/tokenizer.json` (HF BPE), `assets/voices.bin`+`.json` (4 baked voices), `config.json`.

## Files

| file | what |
|---|---|
| `index.html` | UI + glue |
| `js/tts.js` | ORT-web sessions, greedy KV-cache generation loop, audio + WAV |
| `js/tokenizer.js` | XTTS tokenizer port (Transformers.js BPE + `hi` cleaners); ids bit-exact to Python |
| `js/tokenizer.test.mjs` | node parity test for the tokenizer |
| `models/*.onnx` | the three fp32 graphs |
| `assets/*` | tokenizer, voices, parity fixtures |

## Deploy for free ($0, static)

Any static host works since WebGPU needs **no COOP/COEP headers**:
- **GitHub Pages / Cloudflare Pages / HF Static Space**: push the `app/` dir. Serve `.onnx` with
  `Cache-Control: immutable` so revisits load from cache. Large files are fine (90M model, ~740 MB
  fp32 total split across 3 files; well under any limit).

## Notes / follow-ups

- **fp16 build** would halve the download (~370 MB) and speed up WebGPU. The naive float16 converter
  trips on the `null_position_embeddings` zeros constant (GPT-2's nulled `wpe`) feeding an fp16 Add;
  fix = re-export with `wpe` patched to skip the add (or a Cast-insertion post-pass), then convert and
  re-verify codes in-browser. Tracked in `docs/WEBGPU_PLAN.md`.
- KV cache currently round-trips CPU↔GPU each step; pinning `present.*` to `gpu-buffer`
  (`preferredOutputLocation`) is a further WebGPU speedup.
- Build provenance + numerical parity gates: `docs/WEBGPU_PLAN.md`, `webgpu/export/`.
