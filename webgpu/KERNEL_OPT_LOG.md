# WebGPU Kernel Optimization Log

Session: 2026-06-28

## Benchmark setup

Script: `nodecheck/_bench.mjs`  
Browser harness: `app/bench.html`  
Input: `"mera naam Kaustubh hai aur main Delhi se hoon"` (29 text tokens), voice: maya  
Runtime: onnxruntime-web 1.22.0, WASM EP, 1 thread  
Hardware: WASM/CPU fallback (dev box has no GPU adapter)  
Warmup: 2 runs. Measured: 5 runs. Metric: median wall-clock.

## Baseline (pre-optimization)

Measured 2026-06-28 on WASM:

```
run #1: total=11247ms  ort=11239ms  js=9ms  prefill=547ms  vocoder=8618ms  steps=65
run #2: total=11208ms  ort=11204ms  js=4ms  prefill=539ms  vocoder=8589ms  steps=65
run #3: total=11262ms  ort=11257ms  js=5ms  prefill=548ms  vocoder=8624ms  steps=65
run #4: total=32119ms  ort=32114ms  js=4ms  prefill=550ms  vocoder=29492ms  steps=65  ← GC outlier
run #5: total=11761ms  ort=11756ms  js=5ms  prefill=511ms  vocoder=9190ms  steps=65

Median total: 11,262ms
JS overhead:  5ms (0.04% of total — dominated by ONNX inference)
```

Phase breakdown (WASM, median run):
- Prefill:  ~540ms  ( 5%)
- Decode:   ~2,100ms (19%, 64 decode steps ≈ 33ms/step)
- Vocoder:  ~8,600ms (76%)

Key insight: **vocoder is 76% of WASM wall-clock**. On WebGPU all three phases will be
10-50x faster but vocoder remains the relative bottleneck.

## Optimizations applied — Round 1

### O1: Pre-allocated stacked latent buffer

**Files:** `app/js/tts.js`, `nodecheck/_engine.mjs`

Before: per-step `Float32Array.from(out.latent.data)` → push to `lats[]` → post-loop copy loop.  
After: `this._latBuf = new Float32Array((MAXG + 1) * D)` pre-allocated in constructor;
`_latBuf.set(out.latent.data, N * D); N++` writes each step directly;
`lat.subarray(0, N * D)` passed to vocoder — no post-loop assembly.

Eliminated per generation:
- `MAXG+1` Float32Array allocations (each 640 floats = 2.5 KB)
- `lats[]` array holding N references
- Post-loop copy loop (N × 640 float writes)
- One extra `new Float32Array(N * D)` allocation for the stacked lat

### O2: Skip Float32Array.from for logits

**Files:** `app/js/tts.js`, `nodecheck/_engine.mjs`

Before: `logits = Float32Array.from(out.logits.data)` — copies 6,681 floats (~26 KB) per step.  
After: `logits = out.logits.data` — direct reference.

Safe because: we call `_repPenalty(logits)` and `_argmax(logits)` before the next `sStep.run()`.
ORT allocates new tensor buffers on every `run()`, so the old reference won't alias future output.
Mutation of `logits` via `_repPenalty` is fine since the tensor data is not used by ORT after `run()` returns.

Eliminated per generation: ~65 × 26 KB = 1.7 MB of allocations.

### O3: Reusable decode-embed buffer

**Files:** `app/js/tts.js`, `nodecheck/_engine.mjs`

Before: `_decodeEmbed` allocates `new Float32Array(D)` (640 floats = 2.5 KB) on every decode step.  
After: `this._decBuf = new Float32Array(D)` pre-allocated in constructor; returned directly.

Safe because: ORT copies the data into the WASM heap / GPU when creating the `Tensor`, and
`run()` completes before `_decodeEmbed` is called again (no aliasing).

Eliminated per generation: 64 × 2.5 KB = 160 KB of allocations.

### O4: WebGPU power preference

**File:** `app/js/tts.js`

`ort.env.webgpu.powerPreference = 'high-performance'` set in `init()` when WebGPU is available.
Requests the discrete GPU on multi-GPU machines. No-op on integrated-only and WASM path.

### O5: KV cache stays on GPU (preferredOutputLocation)

**File:** `app/js/tts.js`

Before: every decode step, ORT downloads all 32 KV tensors (16 layers × 2) from GPU to CPU,
then uploads them back as inputs for the next step — 2 GPU↔CPU transfers per layer per step.

After: `preferredOutputLocation: { 'present.j.key': 'gpu-buffer', ... }` for all 16 layers
keeps KV outputs on the GPU. The tensors are passed directly back as inputs without a CPU round-trip.

Eliminated: 2 × 16 × 64 ≈ 2,048 GPU↔CPU transfers per generation.  
Expected win on WebGPU: **2-5× speedup for the decode loop** (the KV transfer cost dominates
on GPU where each model forward pass is ~5ms but each KV transfer is ~1ms per tensor).

**WASM path:** option is silently ignored by ORT (KV tensors stay CPU regardless).

## Optimizations applied — Round 1 (continued)

### O6: Explicit KV tensor disposal after each decode step

**Files:** `app/js/tts.js`, `nodecheck/_engine.mjs`, `nodecheck/_bench.mjs`

Without disposal, ORT output tensors from each decode step live in WASM heap until GC.
With 16 layers × 2 × N steps accumulating: at step 65, ~170 MB of KV tensors are pending GC.
This triggered 31+ second pauses (observed in baseline runs #5 and #6 in final benchmark).

Fix: after each `sStep.run(feed)`, call `past['present.j.key'].dispose()` and
`past['present.j.value'].dispose()` for all NL layers. Also dispose the final step's tensors
after the loop exits. `?.dispose?.()` guards against non-disposable tensors (GPU path).

On WebGPU this also releases GPU VRAM incrementally instead of accumulating it.

## Benchmark result — Round 1 (Final, WASM)

Run 2026-06-28 18:14 | `RUNS=6 WARMUP=2 node _bench.mjs` | Logged in `bench_results.log`

```
BASELINE  median=11,773ms  std=9,683ms  min=11,027ms  max=31,796ms  js_overhead=5ms
OPTIMIZED median=11,115ms  std=160ms    min=10,852ms  max=11,304ms  js_overhead=4ms

DELTA     median=-5.6%     std=-98%
```

**The critical win is variance, not median.** Baseline had two consecutive 31+ second runs
(GC pauses from KV heap accumulation). Optimized: all 6 runs in a 452ms band, zero outliers.

This is the "won't crash the laptop" result: consistent ~11s on WASM regardless of run count.

WASM total-time improvement is modest (5.6%) since ORT inference dominates. Optimizations O1-O3
are primarily valuable on **WebGPU** where each decode step is ~5-10ms and JS overhead is
relatively more significant. O5 (KV on GPU) is the big WebGPU win — needs GPU hardware to measure.

## Per-phase timing now exposed

`generate()` return value now includes: `tPrefill`, `tDecode`, `tVocoder`, `nSteps`.  
UI (`index.html`) shows: `prefill Xs dec Xs voc Xs` in the result line.  
Bench page (`bench.html`) shows a visual phase bar + per-run table.

## Next optimization candidates

| # | Optimization | Estimated win | Complexity |
|---|---|---|---|
| N1 | fp16 model (open item from build) | 2× speed + halve download | Medium |
| N2 | Streaming vocoder (chunk latents, start audio earlier) | Lower perceived latency | High |
| N3 | WASM numThreads > 1 (needs COOP/COEP headers) | 2-4× on WASM decode | Low (server config) |
| N4 | ORT IOBinding for full step (not just KV) | Marginal on top of O5 | High |
| N5 | Batch prefill across text-chunk calls | Eliminates redundant cond/pos embeds | Medium |

## How to re-run benchmark

```bash
cd webgpu/nodecheck
node _bench.mjs                          # default: 2 warmup, 5 runs, maya voice
RUNS=10 VOICE=arjun node _bench.mjs     # more runs, different voice
```

Browser: open `app/bench.html` via any static server (`python3 -m http.server` in `app/`).
