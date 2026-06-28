// Reproduce long-input degradation. Synthesize the same content as (a) one long pass and
// (b) sentence-by-sentence, and report the objective code-level failure signals:
//   hitCap (truncation at MAXG), maxRun (stuck-token), ngramRepeatRate (loop/stutter), duration.
import { loadEngine, maxRun, ngramRepeatRate } from './_engine.mjs';

const VOICE = process.env.VOICE || 'maya';
const eng = await loadEngine();
console.log(`engine ready (MAXG=${eng.MAXG}, SR=${eng.SR})\n`);

// Natural Hinglish sentences (Devanagari + English code-switch), each fine on its own.
const SENTS = [
  'आज मौसम बहुत अच्छा है और मैं घूमने जाना चाहता हूँ।',
  'मुझे यह project बहुत interesting लग रहा है।',
  'कल office में एक important meeting है इसलिए जल्दी पहुँचना है।',
  'यार ये नया phone का camera बिल्कुल insane है।',
  'Please इस bug का fix deploy कर दो और मुझे link भेज देना।',
  'शाम को बारिश होने की पूरी संभावना है, छाता साथ रखना।',
];

function report(tag, codes, wav, hitCap) {
  console.log(`  ${tag.padEnd(22)} codes=${String(codes.length).padStart(3)} ` +
    `hitCap=${hitCap ? 'YES' : 'no '} maxRun=${String(maxRun(codes)).padStart(3)} ` +
    `rep3=${ngramRepeatRate(codes).toFixed(3)} dur=${(wav.length / eng.SR).toFixed(2)}s`);
}

// Sweep: 1..N sentences joined into ONE pass.
for (let k = 1; k <= SENTS.length; k++) {
  const text = SENTS.slice(0, k).join(' ');
  const ids = eng.tok.encode(eng.norm.normalize(text).out, 'hi');
  const { codes, wav, hitCap } = await eng.gen(ids, VOICE);
  report(`onepass[${k} sent, ${ids.length} ids]`, codes, wav, hitCap);
}

console.log('');
// Per-sentence baseline (what chunking would feed).
let totDur = 0, anyBad = false;
for (let i = 0; i < SENTS.length; i++) {
  const ids = eng.tok.encode(eng.norm.normalize(SENTS[i]).out, 'hi');
  const { codes, wav, hitCap } = await eng.gen(ids, VOICE);
  totDur += wav.length / eng.SR;
  anyBad = anyBad || hitCap || maxRun(codes) > 12 || ngramRepeatRate(codes) > 0.30;
  report(`sent[${i}]`, codes, wav, hitCap);
}
console.log(`\nper-sentence total audio = ${totDur.toFixed(2)}s, any-bad=${anyBad}`);
