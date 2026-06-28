// Gate: verify _engine.mjs gen() produces bit-exact codes against golden fixtures.
// Used as the evo correctness gate. Exit 0 = pass, exit 1 = fail.
// Run:  NFIX=2 node _engine_parity.mjs

import { loadEngine } from './_engine.mjs';
import { readFileSync } from 'node:fs';

const APP      = new URL('../app/', import.meta.url);
const fixtures = JSON.parse(readFileSync(new URL('assets/golden_fixtures.json', APP)));
const NFIX     = +(process.env.NFIX ?? 2);

console.log(`Loading engine (parity gate, ${NFIX} fixtures)...`);
const { gen } = await loadEngine();
console.log('Engine ready.\n');

let ok = true;
for (const fx of fixtures.slice(0, NFIX)) {
  const r = await gen(fx.ids, fx.voice);
  const exact = r.codes.length === fx.codes.length && r.codes.every((c, k) => c === fx.codes[k]);
  const nm    = r.codes.filter((c, k) => c === fx.codes[k]).length;
  console.log(`#${fx.i} ${fx.voice.padEnd(9)} golden=${fx.codes.length} got=${r.codes.length} exact=${exact ? 'YES' : `NO(${nm}/${fx.codes.length})`}`);
  ok = ok && exact;
}

console.log(`\n[engine-parity] ${ok ? 'PASS' : 'FAIL'}`);
process.exit(ok ? 0 : 1);
