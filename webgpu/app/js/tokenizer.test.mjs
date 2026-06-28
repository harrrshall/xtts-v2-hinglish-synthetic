// Parity test: encode each golden fixture's `text` with lang="hi" and assert
// the produced ids match the Python tokenizer's `ids` EXACTLY.
//
// Run: node js/tokenizer.test.mjs   (from the app dir)
// Exits non-zero on any mismatch.

import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { readFile } from "node:fs/promises";
import { loadTokenizer } from "./tokenizer.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_DIR = resolve(__dirname, "..");
const TOKENIZER_PATH = resolve(APP_DIR, "assets/tokenizer.json");
const FIXTURES_PATH = resolve(APP_DIR, "assets/golden_fixtures.json");

function eq(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function firstDiff(a, b) {
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i++) {
    if (a[i] !== b[i]) return i;
  }
  return -1;
}

const tok = await loadTokenizer(TOKENIZER_PATH);
const fixtures = JSON.parse(await readFile(FIXTURES_PATH, "utf8"));

let pass = 0;
let fail = 0;

for (const fx of fixtures) {
  const got = tok.encode(fx.text, fx.lang ?? "hi");
  const exp = fx.ids;
  if (eq(got, exp)) {
    pass++;
    console.log(`PASS  #${fx.i}  (${exp.length} ids)  voice=${fx.voice}`);
  } else {
    fail++;
    const d = firstDiff(got, exp);
    console.log(`FAIL  #${fx.i}  voice=${fx.voice}`);
    console.log(`  text: ${JSON.stringify(fx.text)}`);
    console.log(`  exp (${exp.length}): ${JSON.stringify(exp)}`);
    console.log(`  got (${got.length}): ${JSON.stringify(got)}`);
    console.log(`  first diff at index ${d}: exp=${exp[d]} got=${got[d]}`);
  }
}

console.log("");
console.log(`Summary: ${pass}/${fixtures.length} passed, ${fail} failed.`);

if (fail > 0) process.exit(1);
console.log("ALL FIXTURES MATCH EXACTLY.");
