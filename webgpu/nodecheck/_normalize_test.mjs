// Test the romanized-Hinglish normalizer against hand-written correct-mixed targets.
import { readFileSync } from 'node:fs';
const APP = new URL('../app/', import.meta.url);
// shim fetch so normalize.js can "fetch" the local JSON
globalThis.fetch = async (u) => {
  const p = u.startsWith('http') ? new URL(u).pathname : u;
  const path = new URL(p.replace(/^.*assets/, 'assets/'), APP);
  return { json: async () => JSON.parse(readFileSync(path, 'utf8')) };
};
const { loadNormalizer } = await import(new URL('js/normalize.js', APP));
const norm = await loadNormalizer('assets/normalize.json');

const CASES = [
  ["yaar kal ka match dekha? last over me jo hua wo totally insane tha",
   "यार कल का match देखा? last over में जो हुआ वो totally insane था"],
  ["mera naam Kaustubh hai aur main Delhi se hoon",
   "मेरा नाम कौस्तुभ है और मैं Delhi से हूँ"],
  ["main aaj office nahi ja raha, can you believe it?",
   "मैं आज office नहीं जा रहा, can you believe it?"],
  ["bhai mujhe ye wala product chahiye, please order kar do",
   "(no gold)"],
  ["Arjun ne kaha ki movie bahut acchi thi",
   "(no gold)"],
];

for (const [src, gold] of CASES) {
  const { out, spans } = norm.normalize(src);
  console.log("IN  :", src);
  console.log("OUT :", out);
  if (gold && !gold.startsWith("(no")) console.log("GOLD:", gold, out === gold ? "  ✅ EXACT" : "  (differs)");
  console.log("tok :", spans.map(s => `${s.src}→${s.deva}[${s.how}]`).join("  "));
  console.log();
}
