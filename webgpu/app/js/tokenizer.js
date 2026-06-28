// JS port of Coqui XTTS-v2 VoiceBpeTokenizer.encode(text, lang="hi").
//
// Produces token ids IDENTICAL to the Python implementation in
// TTS/tts/layers/xtts/tokenizer.py for lang="hi".
//
// The preprocessing (multilingual_cleaners + "[hi]" prefix + space->"[SPACE]")
// is reimplemented here exactly; the BPE itself is delegated to
// @huggingface/transformers (Transformers.js), which loads the bare
// tokenizer.json (HF `tokenizers` v1.0 format) faithfully: added-tokens trie,
// null normalizer, Whitespace pre_tokenizer, BPE merges/vocab.
//
// Works in node (import from the installed package) and in the browser
// (import from a CDN, see loadTokenizer note below).

// --- Preprocessing constants ported verbatim from tokenizer.py (lang="hi") ---

// expand_symbols_multilingual: _symbols_multilingual["hi"] entries.
// Each pattern is matched case-insensitively (re.IGNORECASE in Python; the
// symbols contain no letters so the flag is a no-op, but we keep it for parity).
const HI_SYMBOLS = [
  ["&", " और "],            // " और "
  ["@", " ऐट दी रेट "], // " ऐट दी रेट "
  ["%", " प्रतिशत "],   // " प्रतिशत "
  ["#", " हैश "],      // " हैश "
  ["$", " डॉलर "],// " डॉलर "
  ["£", " पाउंड "], // £ -> " पाउंड "
  ["°", " डिग्री "], // ° -> " डिग्री "
];

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// _abbreviations["hi"] is empty -> no-op. Kept for clarity/parity.
function expandAbbreviationsHi(text) {
  return text;
}

// expand_symbols_multilingual(text, "hi")
function expandSymbolsHi(text) {
  for (const [sym, repl] of HI_SYMBOLS) {
    const re = new RegExp(escapeRegExp(sym), "gi");
    text = text.replace(re, repl);
    // Python: text = text.replace("  ", " ")  (all non-overlapping double spaces)
    text = text.replaceAll("  ", " ");
  }
  return text.trim(); // Python .strip()
}

// expand_numbers_multilingual(text, "hi").
// For digit-free input (the app guards against ASCII digits) every regex in the
// Python path requires an ASCII [0-9] to match, so the function is a no-op.
// We replicate that exactly: if there are no ASCII digits, return unchanged.
// num2words is not available in JS, and valid app input is always digit-free.
function expandNumbersHi(text) {
  if (!/[0-9]/.test(text)) return text;
  // Digits are out of contract for this app; num2words(lang="hi") cannot be
  // reproduced in-browser. Leave digits untouched rather than diverge silently.
  return text;
}

// collapse_whitespace: re.sub(r"\s+", " ", text).strip()
function collapseWhitespace(text) {
  return text.replace(/\s+/g, " ").trim();
}

// multilingual_cleaners(text, "hi")
function multilingualCleanersHi(text) {
  text = text.replaceAll('"', ""); // text.replace('"', "")  (str.replace replaces all)
  text = text.toLowerCase();        // lowercase(text)
  text = expandNumbersHi(text);     // expand_numbers_multilingual
  text = expandAbbreviationsHi(text); // expand_abbreviations_multilingual (no-op for hi)
  text = expandSymbolsHi(text);     // expand_symbols_multilingual
  text = collapseWhitespace(text);  // collapse_whitespace
  return text;
}

// preprocess_text(txt, "hi") -> multilingual_cleaners (hi has no extra transliteration)
function preprocessTextHi(text) {
  return multilingualCleanersHi(text);
}

/**
 * Load the XTTS-v2 Hinglish tokenizer.
 *
 * @param {string} tokenizerJsonUrl - URL/path to assets/tokenizer.json.
 *   In node, pass a file path or file:// URL; in the browser pass an http(s) URL.
 * @param {object} [opts]
 * @param {Function} [opts.PreTrainedTokenizer] - the Transformers.js class. If
 *   omitted, it is dynamically imported from "@huggingface/transformers".
 *   In the browser, import it from a CDN and pass it here, e.g.:
 *     import { PreTrainedTokenizer } from
 *       "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
 *     const tok = await loadTokenizer(url, { PreTrainedTokenizer });
 * @returns {Promise<{encode: (text: string, lang?: string) => number[], _hf: any}>}
 */
export async function loadTokenizer(tokenizerJsonUrl, opts = {}) {
  let PreTrainedTokenizer = opts.PreTrainedTokenizer;
  if (!PreTrainedTokenizer) {
    ({ PreTrainedTokenizer } = await import("@huggingface/transformers"));
  }

  // Fetch + parse the bare tokenizer.json. http(s)/blob URLs go through fetch
  // (browser); bare paths and file:// URLs are read from disk (node).
  let tokenizerJSON;
  const isHttp = /^(https?|blob):/.test(tokenizerJsonUrl);
  if (isHttp && typeof fetch === "function") {
    const res = await fetch(tokenizerJsonUrl);
    if (!res.ok) throw new Error(`Failed to fetch tokenizer.json: ${res.status}`);
    tokenizerJSON = await res.json();
  } else {
    // node: local file path or file:// URL
    const { readFile } = await import("node:fs/promises");
    const { fileURLToPath } = await import("node:url");
    const p = tokenizerJsonUrl.startsWith("file:")
      ? fileURLToPath(tokenizerJsonUrl)
      : tokenizerJsonUrl;
    tokenizerJSON = JSON.parse(await readFile(p, "utf8"));
  }

  // Construct a PreTrainedTokenizer directly from the parsed JSON. The second
  // arg is tokenizer_config; we pass an empty config. The post_processor in
  // tokenizer.json is null, so no BOS/EOS is ever added regardless, but we also
  // call encode with add_special_tokens=false to match Python exactly.
  const hf = new PreTrainedTokenizer(tokenizerJSON, {});

  function encode(text, lang = "hi") {
    lang = String(lang).split("-")[0]; // remove region -> "hi"
    if (lang !== "hi") {
      throw new Error(`This JS port only implements lang="hi" (got "${lang}").`);
    }
    let txt = preprocessTextHi(text);
    txt = `[${lang}]${txt}`;          // prefix "[hi]"
    txt = txt.replaceAll(" ", "[SPACE]");
    return hf.encode(txt, { add_special_tokens: false });
  }

  return { encode, _hf: hf };
}

// Exported for testing the preprocessing in isolation.
export const _internals = {
  multilingualCleanersHi,
  expandSymbolsHi,
  collapseWhitespace,
  preprocessTextHi,
};

export default loadTokenizer;
