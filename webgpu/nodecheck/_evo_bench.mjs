// Evo-compatible benchmark for WebGPU kernel optimization.
// Metric: mean-of-medians generation latency (ms), lower is better (--metric min).
//
// Writes EVO_RESULT_PATH with {"score": ms, "tasks": {per_case_ms}}.
// Runs from the evo worktree — imports _engine.mjs from the same directory.

import { loadEngine } from './_engine.mjs';
import { writeFileSync } from 'node:fs';

const WARMUP = 1;
const RUNS   = 2;

const TESTS = [
  { id: 'maya_short',   text: 'mera naam Kaustubh hai aur main Delhi se hoon', voice: 'maya' },
  { id: 'arjun_medium', text: 'yaar kal ka match dekha kya scene tha bhai',    voice: 'arjun' },
];

console.log('Loading engine...');
const { gen, tok, C } = await loadEngine();
const STRIDE = C.code_stride_len, SR = C.sample_rate;
console.log('Engine ready.\n');

const taskScores = {};
let totalMs = 0;

for (const t of TESTS) {
  const ids = tok.encode(t.text, 'hi');
  const times = [];

  for (let i = 0; i < WARMUP + RUNS; i++) {
    const t0 = performance.now();
    const r = await gen(ids, t.voice);
    const ms = performance.now() - t0;
    const durSec = r.codes.length * STRIDE / SR;
    const rt = durSec / (ms / 1000);
    const label = i < WARMUP ? 'warmup' : `run ${i - WARMUP + 1}`;
    console.log(`  [${t.id}] ${label}: ${ms.toFixed(0)}ms  codes=${r.codes.length}  RT=${rt.toFixed(2)}x`);
    if (i >= WARMUP) times.push(ms);
  }

  const sorted = [...times].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  taskScores[t.id] = median;
  totalMs += median;
  console.log(`  [${t.id}] median=${median.toFixed(0)}ms\n`);
}

const overallScore = totalMs / TESTS.length;
console.log(`Overall score (mean of medians): ${overallScore.toFixed(0)}ms`);

const resultPath = process.env.EVO_RESULT_PATH;
if (resultPath) {
  writeFileSync(resultPath, JSON.stringify({ score: overallScore, tasks: taskScores }));
  console.log(`Result written to ${resultPath}`);
} else {
  console.log('(no EVO_RESULT_PATH set — standalone mode)');
}
