#!/usr/bin/env python3
"""Build the browser normalizer assets:
  - lexicon: roman -> Devanagari (inverted Dakshina hi lexicon, most-attested target per romanization)
  - english: frequency-thresholded English wordset for LID (en_50k)
  - forceHindi: curated Hindi-dominant words (no common-English collision) -> Devanagari
  - names: Indian-name gazetteer -> Devanagari
  - hindiFreq: Devanagari word -> attestation (for tie-breaks)

Emitted to webgpu/app/assets/normalize.json (+ a copy under this dir for inspection).
"""
import json, re
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
DAK = HERE / "data" / "dakshina_dataset_v1.0" / "hi" / "lexicons"
EN = HERE / "data" / "en_50k.txt"
APP_ASSETS = HERE.parents[2] / "webgpu" / "app" / "assets"

# --- curated force-Hindi: words that are essentially ALWAYS Hindi in Hinglish (no common English
#     collision). Truly-ambiguous tokens (to, me, is, in, so, do, us, on, the, he, it...) are LEFT OUT
#     and resolved by neighbour context at runtime. ---
FORCE_HINDI = {
    # pronouns
    "main": "मैं", "mai": "मैं", "mein_p": "मैं", "hum": "हम", "tum": "तुम", "tu": "तू", "aap": "आप",
    "ye": "ये", "yeh": "यह", "wo": "वो", "woh": "वह", "vo": "वो", "voh": "वह",
    "mera": "मेरा", "meri": "मेरी", "mere": "मेरे", "tera": "तेरा", "teri": "तेरी", "tere": "तेरे",
    "hamara": "हमारा", "hamari": "हमारी", "tumhara": "तुम्हारा", "tumhari": "तुम्हारी",
    "uska": "उसका", "uski": "उसकी", "unka": "उनका", "iska": "इसका", "iski": "इसकी",
    "koi": "कोई", "kuch": "कुछ", "kuchh": "कुछ", "sab": "सब", "sabhi": "सभी", "apna": "अपना", "apni": "अपनी",
    # postpositions / particles
    "ka": "का", "ki": "की", "ke": "के", "ko": "को", "se": "से", "mein": "में", "ne": "ने",
    "tak": "तक", "bhi": "भी", "na": "ना", "nahi": "नहीं", "nahin": "नहीं", "mat": "मत",
    # verbs / auxiliaries
    "hai": "है", "hain": "हैं", "hai": "है", "ho": "हो", "hu": "हूँ", "hoon": "हूँ", "hun": "हूँ",
    "tha": "था", "thi": "थी", "kar": "कर", "karo": "करो", "karu": "करूँ", "kiya": "किया",
    "karna": "करना", "karne": "करने", "karta": "करता", "karti": "करती", "karte": "करते",
    "raha": "रहा", "rahi": "रही", "rahe": "रहे", "gaya": "गया", "gayi": "गई", "gaye": "गए",
    "jaa": "जा", "jaana": "जाना", "jana": "जाना", "jaata": "जाता", "aa": "आ", "aaya": "आया", "aayi": "आई",
    "hoga": "होगा", "hogi": "होगी", "honge": "होंगे", "chahiye": "चाहिए", "chahiye_": "चाहिए",
    "sakta": "सकता", "sakti": "सकती", "sakte": "सकते", "milega": "मिलेगा", "lagta": "लगता", "lagi": "लगी",
    # question / adverb / time
    "kya": "क्या", "kyun": "क्यों", "kyu": "क्यों", "kyon": "क्यों", "kaise": "कैसे", "kaisa": "कैसा",
    "kahan": "कहाँ", "kaha": "कहाँ", "kab": "कब", "kaun": "कौन", "kitna": "कितना", "kitni": "कितनी",
    "jab": "जब", "tab": "तब", "ab": "अब", "abhi": "अभी", "aaj": "आज", "kal": "कल", "parso": "परसों",
    "phir": "फिर", "fir": "फिर", "bahut": "बहुत", "thoda": "थोड़ा", "thodi": "थोड़ी", "zyada": "ज़्यादा",
    "accha": "अच्छा", "acha": "अच्छा", "achha": "अच्छा", "theek": "ठीक", "thik": "ठीक", "sahi": "सही",
    "jaldi": "जल्दी", "dhyan": "ध्यान", "matlab": "मतलब", "waqt": "वक़्त", "samay": "समय",
    # conjunctions / discourse
    "aur": "और", "ya": "या", "lekin": "लेकिन", "magar": "मगर", "kyunki": "क्योंकि", "kyuki": "क्योंकि",
    "agar": "अगर", "warna": "वरना", "isliye": "इसलिए", "matlab_": "मतलब", "yaar": "यार", "arre": "अरे",
    "haan": "हाँ", "han": "हाँ", "ji": "जी", "bilkul": "बिल्कुल", "shayad": "शायद", "zaroor": "ज़रूर",
    # very common content
    "naam": "नाम", "kaam": "काम", "baat": "बात", "din": "दिन", "raat": "रात", "ghar": "घर", "log": "लोग",
    "dost": "दोस्त", "khush": "खुश", "pyaar": "प्यार", "dil": "दिल", "duniya": "दुनिया", "zindagi": "ज़िंदगी",
    "khana": "खाना", "paani": "पानी", "pani": "पानी", "shukriya": "शुक्रिया", "dhanyavaad": "धन्यवाद",
    "dekha": "देखा", "dekho": "देखो", "suno": "सुनो", "bolo": "बोलो", "chalo": "चलो", "khelo": "खेलो",
    "samajh": "समझ", "pata": "पता", "malum": "मालूम", "yaad": "याद", "intezaar": "इंतज़ार",
    # --- ambiguous-but-Hindi-dominant (collide with short English; in romanized Hindi these are
    #     overwhelmingly Hindi). Curated to FORCE Hindi over the English dict. ---
    "jo": "जो", "hua": "हुआ",
    "hui": "हुई", "hue": "हुए", "ban": "बन", "bana": "बना", "bani": "बनी", "baat": "बात",
    "kaha": "कहा", "kaho": "कहो", "wala": "वाला", "wali": "वाली", "wale": "वाले",
    "har": "हर", "kam": "कम", "liye": "लिए", "saath": "साथ", "sath": "साथ", "paas": "पास", "pass": "पास",
    "baad": "बाद", "pehle": "पहले", "pahle": "पहले", "niche": "नीचे", "upar": "ऊपर", "uper": "ऊपर",
    "andar": "अंदर", "bahar": "बाहर", "idhar": "इधर", "udhar": "उधर", "yahan": "यहाँ", "wahan": "वहाँ",
    "aag": "आग", "rok": "रोक", "chal": "चल", "chalo": "चलो", "ruko": "रुको", "suno": "सुनो",
    "desh": "देश", "sirf": "सिर्फ़", "re": "रे",
    # common Hindi words shadowed by rare en_50k entries / abbreviations -> force Hindi
    "ja": "जा", "bhai": "भाई", "bhaiya": "भैया", "behen": "बहन", "behan": "बहन", "didi": "दीदी",
    "beta": "बेटा", "beti": "बेटी", "de": "दे", "le": "ले", "lo": "लो", "dena": "देना",
    "lena": "लेना", "dega": "देगा", "lega": "लेगा", "mil": "मिल", "mila": "मिला", "mili": "मिली",
    "jata": "जाता", "jati": "जाती", "jate": "जाते", "aata": "आता", "aati": "आती", "aate": "आते",
    "chahta": "चाहता", "chahti": "चाहती", "chahte": "चाहते", "hona": "होना", "mann": "मन",
    "pyar": "प्यार", "saara": "सारा", "sara": "सारा", "wahi": "वही", "yahi": "यही", "aise": "ऐसे",
    "waise": "वैसे", "jaise": "जैसे", "itna": "इतना", "utna": "उतना", "jitna": "जितना",
    "khud": "खुद", "waqai": "वाक़ई", "sach": "सच", "jhuth": "झूठ", "galat": "ग़लत", "badi": "बड़ी",
    "bada": "बड़ा", "chota": "छोटा", "choti": "छोटी", "naya": "नया", "nayi": "नई", "purana": "पुराना",
}
# strip the disambiguation suffixes (_p, _h, _) used to allow duplicate Devanagari values above
FORCE_HINDI = {re.sub(r"_[a-z]?$", "", k): v for k, v in FORCE_HINDI.items()}

# --- genuinely ambiguous tokens (common in BOTH languages): resolved by neighbour context at
#     runtime (Hindi if surrounded by Hindi, English if surrounded by English). NOT blanket-forced. ---
AMBIG = {
    "is": "इस", "to": "तो", "me": "में", "do": "दो", "us": "उस", "par": "पर", "in": "इन",
    "bat": "बात", "hi": "ही",
}

# --- Indian names gazetteer (incl. the 4 model voices) ---
NAMES = {
    "aadya": "आद्या", "arjun": "अर्जुन", "kaustubh": "कौस्तुभ", "maya": "माया",
    "rahul": "राहुल", "rohan": "रोहन", "rohit": "रोहित", "amit": "अमित", "ankit": "अंकित",
    "raj": "राज", "ravi": "रवि", "vikram": "विक्रम", "vijay": "विजय", "sanjay": "संजय",
    "suresh": "सुरेश", "rajesh": "राजेश", "mahesh": "महेश", "ramesh": "रमेश", "dinesh": "दिनेश",
    "priya": "प्रिया", "pooja": "पूजा", "puja": "पूजा", "neha": "नेहा", "kavya": "काव्या",
    "ananya": "अनन्या", "aishwarya": "ऐश्वर्या", "deepika": "दीपिका", "sneha": "स्नेहा",
    "harshal": "हर्षल", "harsh": "हर्ष", "aryan": "आर्यन", "ishaan": "ईशान", "kabir": "कबीर",
    "krishna": "कृष्ण", "shiva": "शिव", "ganesh": "गणेश", "lakshmi": "लक्ष्मी", "sita": "सीता",
    "delhi": "दिल्ली", "mumbai": "मुंबई", "bangalore": "बैंगलोर", "kolkata": "कोलकाता",
    "chennai": "चेन्नई", "pune": "पुणे", "jaipur": "जयपुर", "lucknow": "लखनऊ", "bharat": "भारत",
    "india": "इंडिया", "hindustan": "हिंदुस्तान",
}


def build_lexicon():
    """Invert Dakshina: romanization -> Devanagari with the highest total attestation."""
    roman_to = defaultdict(lambda: defaultdict(int))   # roman -> {deva: count}
    hindi_freq = defaultdict(int)
    for split in ("train", "dev", "test"):
        f = DAK / f"hi.translit.sampled.{split}.tsv"
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            deva, roman, cnt = parts[0].strip(), parts[1].strip().lower(), parts[2].strip()
            if not deva or not roman:
                continue
            try:
                c = int(cnt)
            except ValueError:
                c = 1
            roman_to[roman][deva] += c
            hindi_freq[deva] += c
    lexicon = {}
    for roman, devas in roman_to.items():
        lexicon[roman] = max(devas.items(), key=lambda kv: kv[1])[0]
    return lexicon, dict(hindi_freq)


def build_english():
    words = set()
    for i, line in enumerate(EN.read_text(encoding="utf-8").splitlines()):
        w = line.split(" ")[0].strip().lower()
        if w.isalpha() and len(w) >= 2:
            words.add(w)
        if i >= 40000:
            break
    return words


def main():
    lexicon, hindi_freq = build_lexicon()
    english = build_english()
    # force-Hindi and names take precedence; drop their keys from the English set so they don't
    # masquerade as confidently-English.
    for k in list(FORCE_HINDI) + list(NAMES) + list(AMBIG):
        english.discard(k)
    # keep only Devanagari mappings the model can actually use; prune lexicon entries that are pure ASCII
    # (Dakshina has loanword rows like uncle->अंकल; harmless, English dict catches them first)
    out = {
        "forceHindi": FORCE_HINDI,
        "ambig": AMBIG,
        "names": NAMES,
        "lexicon": lexicon,
        "english": sorted(english),
        # ship only reasonably-attested hindi freq to keep size down
        "hindiFreq": {k: v for k, v in hindi_freq.items() if v >= 2},
    }
    APP_ASSETS.mkdir(parents=True, exist_ok=True)
    p = APP_ASSETS / "normalize.json"
    p.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"[assets] forceHindi={len(FORCE_HINDI)} names={len(NAMES)} "
          f"lexicon={len(lexicon)} english={len(english)} hindiFreq={len(out['hindiFreq'])}")
    print(f"[assets] -> {p}  ({p.stat().st_size/1e6:.2f} MB)")
    # coverage check on test words
    for w in ["dekha", "match", "kal", "yaar", "office", "insane", "totally", "naam", "raha",
              "believe", "last", "over", "jo", "hua"]:
        tag = ("forceHindi" if w in FORCE_HINDI else "name" if w in NAMES
               else "lexicon:" + lexicon.get(w, "") if w in lexicon
               else "english" if w in english else "OOV")
        print(f"   {w:10s} -> {tag}")


if __name__ == "__main__":
    main()
