#!/usr/bin/env python3
"""Author NEW held-out code-switch Hinglish prompts for the powered (n>=150) certification.
All hand-written, English embedded in a Hindi matrix. Verified at build time to NOT overlap the
training corpus or the existing held-out 89 (no leakage). Voice-balanced manifest out."""
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import english_words  # noqa: E402

PROMPTS = [
    # gaming
    "यार इस ranked match में हमारी पूरी team ने अच्छा push किया but last round में clutch नहीं हो पाया",
    "मेरा headshot aim इतना off था कि enemy ने आराम से मुझे spawn पर ही knock कर दिया",
    "नई update के बाद से game का frame rate बहुत drop हो रहा है lag भी आ रहा है",
    "तू basement में camp मत कर बाहर निकल और साथ में rotate कर वरना हम पूरा zone खो देंगे",
    "उस streamer का reaction time इतना fast है कि वो हर ambush को pre fire कर देता है",
    "मैंने नया controller खरीदा but उसकी trigger sensitivity मुझे थोड़ी weird लग रही है",
    "हमने पूरी lobby को carry किया फिर भी मेरा teammate बार बार revive मांग रहा था",
    "इस season का battle pass इतना grindy है कि सारे rewards unlock करना मुश्किल है",
    # work / office
    "कल का client call subah जल्दी है इसलिए मुझे presentation आज रात ही finalize करनी पड़ेगी",
    "मेरा manager चाहता है कि हम इस sprint में सारे pending tickets close कर दें",
    "office की नई policy के हिसाब से अब हमें हर हफ्ते एक report submit करनी होगी",
    "उसने meeting के बीच में ही अपना camera off कर दिया और बोला network issue है",
    "मुझे लगता है इस deadline को थोड़ा extend करना पड़ेगा क्योंकि scope काफी बढ़ गया है",
    "नए intern ने पहले ही दिन पूरा deployment pipeline समझ लिया वो काफी sharp है",
    "हमारी team का morale थोड़ा low है इसलिए मैंने एक casual lunch plan किया है",
    "इस quarter का revenue target hit करने के लिए हमें marketing पर ज्यादा focus करना होगा",
    "बॉस ने feedback दिया कि मेरी slides पर बहुत ज्यादा text है उन्हें simple रखो",
    "remote work की वजह से अब commute का time बच जाता है but थोड़ा isolation feel होता है",
    # tech / gadgets
    "मेरे laptop की battery इतनी जल्दी drain हो रही है शायद कोई background app चल रहा है",
    "नया phone लेने से पहले मैं उसके camera और display reviews ध्यान से पढ़ रहा हूँ",
    "router को restart करने के बाद भी internet speed बहुत slow आ रही है",
    "उसने अपने पूरे setup में RGB lights लगा दी हैं अब desk एकदम gaming station लगता है",
    "इस app का latest version install करने के बाद से notifications आना ही बंद हो गई",
    "मेरी smartwatch का heart rate sensor कभी कभी गलत reading दिखाता है",
    "cloud पर backup लेना safe रहता है warna अगर drive crash हो जाए तो data चला जाता है",
    "इस keyboard के mechanical switches की typing sound मुझे बहुत satisfying लगती है",
    # daily life / home
    "ceiling fan की speed धीरे हो गई है शायद उसका capacitor weak हो गया होगा",
    "मेरे घर का geyser सुबह सुबह बहुत time लेता है पानी गरम करने में",
    "कल plumber आया था but वो leak पूरी तरह से fix नहीं कर पाया",
    "हमने drawing room का पूरा furniture rearrange किया अब space ज्यादा खुला लगता है",
    "बारिश की वजह से balcony के सारे plants में extra water जमा हो गया था",
    "मेरी maid आज छुट्टी पर है इसलिए सारे dishes मुझे खुद ही wash करने पड़ेंगे",
    "नई washing machine का spin cycle इतना quiet है कि पता ही नहीं चलता",
    "हमारे society की lift पिछले तीन दिन से under maintenance है सीढ़ियां चढ़नी पड़ रही हैं",
    # food / cooking
    "मैंने आज पहली बार homemade pasta try किया but sauce थोड़ा bland रह गया",
    "इस restaurant की service slow है but उनका butter chicken एकदम authentic है",
    "तू recipe में थोड़ा कम oil डाल वरना डिश बहुत heavy हो जाएगी",
    "weekend पर हम एक नए cafe गए थे जहां का cold coffee सच में amazing था",
    "मेरी diet में अब मैं ज्यादा protein और कम processed food add करने की कोशिश कर रहा हूँ",
    "उसने birthday के लिए एक custom cake order किया जिस पर chocolate ganache था",
    "रात के leftovers को microwave करने से पहले उन्हें एक proper container में रखना",
    # travel / commute
    "मेरी flight का gate change हो गया और मुझे last minute में पूरे terminal भागना पड़ा",
    "traffic इतना heavy था कि एक छोटे से signal को cross करने में बीस मिनट लग गए",
    "हमने road trip के लिए एक SUV rent की और पूरे रास्ते अच्छी playlist चलाई",
    "metro की नई line खुलने के बाद से मेरा office तक का commute काफी easy हो गया है",
    "उसने hotel booking confirm करने से पहले सारे cancellation terms ध्यान से पढ़े",
    "airport पर security check में इतनी लंबी queue थी कि मैं boarding लगभग miss कर देता",
    "बारिश की वजह से कई trains delay हो गईं और platform पर भारी crowd जमा हो गया",
    # health / fitness
    "मेरा gym trainer बोलता है कि form सही रखना weight बढ़ाने से ज्यादा important है",
    "देर रात तक screen देखने की वजह से मेरा sleep cycle पूरी तरह से बिगड़ गया है",
    "doctor ने कहा है कि मुझे रोज कम से कम दस हजार steps walk करने चाहिए",
    "उसने एक नया fitness app download किया जो हर workout का progress track करता है",
    "ज्यादा junk खाने के बाद अब मैं एक proper meal plan follow करने की सोच रहा हूँ",
    "yoga की morning session के बाद पूरे दिन energy और focus काफी better रहता है",
    # shopping / finance
    "इस sale में discount तो अच्छा है but delivery charges काफी ज्यादा लग रहे हैं",
    "मैंने अपनी monthly budget में एक अलग category investment के लिए add की है",
    "उस online store ने मेरा refund process करने में पूरा एक हफ्ता लगा दिया",
    "credit card का bill समय पर pay ना करो तो interest बहुत तेजी से बढ़ता है",
    "नई policy लेने से पहले उसके सारे hidden charges को carefully compare करना",
    "मैंने एक budget phone लिया जिसकी value for money सच में बहुत बढ़िया है",
    # social media / entertainment
    "उसकी latest reel इतनी viral हो गई कि एक रात में हजारों followers बढ़ गए",
    "नई web series का ending इतना unexpected था कि पूरा group chat उसी पर discuss करता रहा",
    "मैंने उस podcast का पूरा episode सुना जिसमें startup founders ने अपनी journey share की",
    "उसने अपने vlog के लिए एक नया mic लिया ताकि audio quality professional लगे",
    "इस movie का background score इतना powerful है कि हर scene और intense लगती है",
    "मेरा screen time इस हफ्ते इतना ज्यादा था कि app ने खुद एक warning भेज दी",
    # education / study
    "exam से पहले उसने पूरे syllabus का एक quick revision schedule बना लिया",
    "online lecture के बीच में मेरा connection बार बार freeze हो रहा था",
    "professor ने assignment की submission deadline को एक हफ्ते आगे बढ़ा दिया",
    "मैंने एक नया course join किया जो practical projects पर ज्यादा focus करता है",
    "library में इतनी शांति रहती है कि वहां concentration अपने आप better हो जाता है",
    "उसने अपने notes को digital format में convert कर लिया ताकि search करना आसान हो",
    # relationships / social
    "उसने आखिरी मिनट पर plan cancel कर दिया और कोई proper reason भी नहीं बताया",
    "हमने एक surprise party organize की but उसे पहले ही पूरी बात पता चल गई",
    "मेरे cousin की wedding में पूरा family एक लंबे arse बाद एक साथ इकट्ठा हुआ",
    "तू उससे directly बात कर ले warna ये misunderstanding और बढ़ती चली जाएगी",
    "उसका positive attitude पूरे group की energy को हमेशा उठा देता है",
    "weekend पर हम सब मिलकर एक board game night करने का plan बना रहे हैं",
    # misc daily
    "मेरे area में आज सुबह से power cut है और inverter की battery भी low हो रही है",
    "उसने अपनी car का insurance renew करना भूल गया और अब penalty भरनी पड़ेगी",
    "इस महीने का electricity bill पिछले से लगभग double आया है कुछ तो गड़बड़ है",
    "नया pet लाने से पहले हमने उसके vaccination और diet की पूरी planning कर ली",
    "मेरे phone का storage full हो गया है इसलिए मुझे कुछ पुरानी photos delete करनी होंगी",
    "उसने अपने garden में एक small fountain लगाया जिससे पूरा backyard काफी peaceful लगता है",
    "बच्चों की online classes की वजह से घर का internet अक्सर overloaded रहता है",
    "मैंने अपनी पुरानी cycle की servicing करवाई अब उसकी braking बहुत smooth हो गई है",
    "उस shop का customer service इतना अच्छा है कि वो हर complaint को seriously लेते हैं",
    "मेरी morning routine में अब मैं एक छोटा meditation session भी add करने लगा हूँ",
    "उसने अपनी resume को update किया और कई नई skills को highlight भी किया",
    "गर्मी इतनी ज्यादा है कि AC के बिना एक मिनट भी बैठना मुश्किल हो रहा है",
    "नई coffee machine आने के बाद से सुबह की शुरुआत काफी refreshing हो गई है",
    "उसने अपने project का demo इतने confidence के साथ दिया कि सब impressed रह गए",
    "मेरे neighbour का dog रात भर bark करता रहता है जिससे नींद पूरी तरह खराब हो जाती है",
    "इस app का dark mode आंखों के लिए काफी comfortable है खासकर रात में",
    "हमने पुराने सारे cartons को recycle के लिए अलग कर दिया ताकि घर थोड़ा declutter हो जाए",
    "उसकी handwriting इतनी neat है कि उसके notes हर कोई borrow करना चाहता है",
    "मैंने weekend के लिए एक छोटा trek plan किया है मौसम भी काफी pleasant रहने वाला है",
    "नई printer की setup में थोड़ा time लगा but अब wireless printing एकदम smooth है",
]

VOICES = ["kaustubh", "arjun", "maya", "aadya"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train_texts = set()
    for r in csv.reader(open("data/xtts/metadata_train.csv"), delimiter="|"):
        if len(r) >= 2:
            train_texts.add(r[1].strip())
    held = set(json.loads(l)["ref_text"].strip() for l in open("data/eval_heldout89.jsonl") if l.strip())

    rows, leak = [], 0
    for i, t in enumerate(PROMPTS):
        t = t.strip()
        if t in train_texts or t in held:
            leak += 1
            print("LEAK (skipped):", t[:50]); continue
        ew = len(english_words(t))
        cs = "high" if ew >= 4 else ("med" if ew >= 2 else "none")
        rows.append({"utt_id": f"ep{len(rows):03d}__{VOICES[len(rows) % 4]}",
                     "ref_text": t, "voice": VOICES[len(rows) % 4], "cs_mode": cs})
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    from collections import Counter
    scor = sum(1 for r in rows if len(english_words(r["ref_text"])) >= 1)
    print(f"[powered_prompts] wrote {len(rows)} prompts ({leak} leaks dropped) -> {args.out}")
    print(f"  scorable-for-accent (>=1 EN word): {scor}  cs={Counter(r['cs_mode'] for r in rows)}  voices={Counter(r['voice'] for r in rows)}")


if __name__ == "__main__":
    main()
