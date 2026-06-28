# Deploying the Hinglish TTS model (free hosting options)

How to put the sub-100M Hinglish model (or the 265M/443M) in front of users for **$0**. Recorded for next
time. Model live at [`harrrshall/xtts-hinglish-90m`](https://huggingface.co/harrrshall/xtts-hinglish-90m)
(fp16, 180 MB + base XTTS vocoder auto-downloaded).

## TL;DR

- **Want a live demo in hours:** free **Hugging Face Space** (CPU, or ZeroGPU if available). Recommended first step.
- **Want zero-server, runs-on-each-device, free forever:** **WebGPU in the browser** (ONNX + onnxruntime-web).
  Proven possible, but a multi-day porting project. The ONNX export of the autoregressive GPT is the gate.

---

## Option 1 — Hugging Face Spaces (free, recommended first)

A free Space hosts a Gradio web app with a public URL, pulling the model straight from the Hub.

- **Free CPU** (2 vCPU, 16 GB): always works, persistent, $0. XTTS is autoregressive, so ~15-40s per clip on
  CPU. Fine for a demo, slow for heavy traffic.
- **ZeroGPU** (free shared A100, quota-limited): ~1-2s per clip. Enable if the account qualifies.
- Disk is a non-issue: model 180 MB, base XTTS vocoder ~2 GB into the Space's ephemeral 50 GB.

What to ship in the Space (`app.py` + `requirements.txt`):
- `requirements.txt`: `coqui-tts`, `gradio`, `soundfile`, `torch`.
- `app.py`: load `student640.py` + `student640b_rft.pt` once; download base XTTS for the frozen HiFi-GAN +
  tokenizer (`COQUI_TOS_AGREED=1`); Gradio UI = voice dropdown (aadya/arjun/kaustubh/maya) + Hinglish textbox
  → audio. The HF repo's `inference.py` already does the load + generate; wrap it in a Gradio callback.
- Caveats to bake into the UI: write Hindi in Devanagari + English in Latin, spell numbers as words, chunk
  text over ~150 chars, greedy + `repetition_penalty≈1.3`.

## Option 2 — Client-side WebGPU (the zero-cost dream)

Run the whole model in the visitor's browser. No backend, infinitely free, scales to anyone; ship as **static
files** on GitHub Pages or a free HF Static Space.

**It is feasible.** Autoregressive TTS/ASR in-browser is proven (Whisper runs via Transformers.js + WebGPU, and
Whisper is an AR decoder like our GPT). A 90M GPT + a conv vocoder is well within WebGPU.

**The stack:** `ONNX export → onnxruntime-web (WebGPU execution provider) + JS tokenizer + JS generation loop`.

**Key simplification — no DVAE at inference.** The DVAE only encodes audio→codes during training. The browser
graph is just: `text → GPT (autoregressive) → 640→1024 adapter → HiFi-GAN → waveform`. The 4 baked voices are
small tensors shipped as data (`ck["voices"][v]["cond_latents"]` 32×640 + `speaker_embedding` 512).

**The work, by risk:**
1. ONNX-export the **autoregressive GPT** with KV-cache for step-wise generation (the fiddly part; Coqui wraps
   it in `GPT2InferenceModel`). Export as a single-step model, run the loop in JS.
2. ONNX-export the **HiFi-GAN** (fuse/strip weight-norm first; known-doable).
3. Reimplement the **tokenizer + cleaners** in JS (multilingual BPE + Devanagari; numbers spelled as words).
4. JS **sampling loop** (greedy/top-k, repetition penalty, stop token) calling onnxruntime-web per step.
5. Wire WebGPU EP, load voices.

**Effort:** a focused multi-day project; step 1 is the main risk. (Kokoro got an easy web port because it is
*non*-autoregressive, a single forward; XTTS is token-by-token, hence harder.)

**The gate to commit:** export the **HiFi-GAN and the GPT to ONNX and verify they run in `onnxruntime` (Python)
first**. If both export cleanly, the browser port is on solid ground. Do this before any JS work.

**On-device cost:** ~100-200 MB download (int8/fp16 ONNX); near-real-time on a decent GPU, a few seconds on
weaker devices; per-step JS↔WASM↔WebGPU overhead is the likely bottleneck, mitigated by KV-cache.

## Option 3 — Other free-ish routes

- **Colab / Kaggle:** free GPU, but ephemeral. A `gradio share=True` link dies when the notebook stops; not
  real hosting.
- **Modal:** ~$30/mo free credits, fast serverless GPU, free *within* credits.
- **Render / Railway / Fly.io free tiers:** CPU-only, shrinking, slow for AR TTS; not great fits.
- **Local on your own machine:** free but only up while the computer is on.

## Recommendation

1. **HF Space now** (CPU, flip to ZeroGPU if available) for an immediate shareable demo.
2. If the zero-server browser version is wanted, **start with the ONNX export of HiFi-GAN + GPT** (the gate),
   then build the onnxruntime-web + JS port and host the static files for $0 forever.
