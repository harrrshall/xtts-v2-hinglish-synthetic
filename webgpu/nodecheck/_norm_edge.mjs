import { readFileSync } from 'node:fs';
const APP = new URL('../app/', import.meta.url);
globalThis.fetch = async (u) => { const path=new URL('assets/normalize.json',APP); return { json: async()=>JSON.parse(readFileSync(path,'utf8')) }; };
const { loadNormalizer } = await import(new URL('js/normalize.js', APP));
const n = await loadNormalizer('assets/normalize.json');
const t = s => n.normalize(s).out;
const cases = [
  ["romanized", "yaar kal ka match dekha? last over me jo hua wo totally insane tha"],
  ["pureEnglish", "this is a great system can you believe it"],
  ["mixedPhrase", "bro this movie is amazing yaar must watch"],
  ["nameHeavy", "Rahul aur Priya Mumbai ja rahe hain"],
  ["isHindi", "is baat ko samjho yaar"],
  ["toHindi", "agar tum aaoge to main bhi aaunga"],
  ["doHindi", "ye kaam kar do please"],
  ["doEnglish", "what do you want to do today"],
  ["alreadyMixed", "मुझे यह project बहुत interesting लग रहा है, can you believe it?"],
];
for (const [k,s] of cases) console.log(k.padEnd(13), "::", t(s));
