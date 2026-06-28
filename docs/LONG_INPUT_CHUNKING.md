# Length-independent synthesis (intelligent chunking)

## The problem

The acoustic model is an autoregressive GPT decoder trained on short clips. On long
inputs its mel-token output **saturates** and it silently rushes/drops the tail
content instead of rendering it. This is not the hard `max_gen_mel_tokens` cap (602)
firing — it saturates well below that, around ~335 tokens / ~15s, regardless of how
much text follows.

Measured on a 6-sentence Hinglish paragraph (voice `maya`, wasm EP, greedy decode):

| input | text ids | mel codes | audio |
|---|---|---|---|
| 1 sentence | 30 | 82 | 3.81s |
| 4 sentences | 121 | 316 | 14.67s |
| 5 sentences | 158 | 322 | 14.94s |
| 6 sentences | 190 | 335 | **15.55s** |

The 6-sentence paragraph needs ~18.6s of speech (sum of the silence-trimmed solo
sentences) but one-pass renders only 14.5s — **content coverage 0.78, i.e. ~22% of
the words dropped**. Degradation onset is ~4 sentences / ~120 text ids.

## The fix

Split the (already-normalized) text into sentence-aligned chunks, each within a safe
text-token budget, synthesize each at full fidelity with the **same** voice
conditioning, and concatenate with edge-silence trimming + a short inter-chunk pause
+ a tiny equal-power crossfade (removes boundary clicks).

- `webgpu/app/js/chunk.js` — pure, dependency-free logic: `splitSentences`,
  `chunkText`, `trimSilence`, `concatWavs`. Shared verbatim by the app and the node
  harness so they cannot drift.
- `webgpu/app/js/tts.js` — `generate()` now chunks; each chunk runs through
  `_genChunk()` (the original single-pass loop, unchanged). **A single-chunk input
  takes the original path and is bit-identical to before** — short-input quality and
  golden parity are untouched.
- `webgpu/app/config.json` — `chunk: { maxTextIds: 60, gapMs: 90, crossfadeMs: 4,
  ampThresh: 0.01, padMs: 30 }`. `maxTextIds=60` keeps every chunk inside the
  decoder's faithful-rendering zone while still packing short sentences to minimize
  seams.

Splitting precedence: sentence (danda / `?` / `!` / sentence-final `.` / newline) →
clause (`,` `;` `:`) → word, so no chunk ever exceeds the budget even for a single
run-on sentence.

## Verification

Run from `webgpu/nodecheck/` against the **real** ONNX graphs (onnxruntime-web, wasm):

```
npm run test:chunk    # fast, no model: split/pack/concat units + golden-parity guard
npm run eval:chunk    # end-to-end through the model: content-coverage gates
```

`test:chunk` also asserts **all 6 golden fixtures stay single-chunk** (so their output
is unchanged).

`eval:chunk` gates (all passing):

| metric | one-pass | chunked | gate |
|---|---|---|---|
| speech coverage | 0.779 | **1.001** | one-pass < 0.90; chunked ≥ 0.97 |
| improvement Δ | — | +0.222 | ≥ 0.12 |
| truncation (hit MAXG) | — | none | none |
| stuck-token max run | — | 2 | ≤ 12 |
| 3-gram repeat rate | — | 0.000 | ≤ 0.30 |

Coverage is measured on **silence-trimmed speech content** so it reflects dropped
words, not pause length. Chunked synthesis reproduces 100% of the content; one-pass
loses 22%.
