#!/usr/bin/env python3
"""Batch 2 of NEW held-out code-switch prompts for the powered cert (target accent n~300 for a clean PASS).
Representative difficulty: varied English density (2-6 words) and length, NOT biased easy like batch 1.
Verified at build time against training + held89 + batch-1 (no leakage/dupes)."""
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import english_words  # noqa: E402

PROMPTS = [
    # denser / technical code-switch (where accent discrimination is highest)
    "उस deployment के बाद production server पर एक critical memory leak detect हुआ जिसे rollback करके ही ठीक किया गया",
    "हमारी backend team ने database queries को optimize किया तो overall latency लगभग आधी हो गई",
    "नई authentication layer में token expiry का logic इतना strict है कि users बार बार logout हो रहे हैं",
    "उसने machine learning model को fine tune किया but validation accuracy अभी भी expected baseline से नीचे है",
    "इस microservice का API response time spike कर रहा है शायद किसी downstream dependency की वजह से",
    "हमने पूरे codebase में legacy functions को refactor किया ताकि future maintenance आसान हो जाए",
    "encryption key rotate करने के बाद कुछ पुराने sessions invalid हो गए और support tickets बढ़ गए",
    "उसका pull request merge होने से पहले दो reviewers ने उसमें कई edge cases highlight किए",
    "हमारी monitoring dashboard ने एक unusual traffic pattern flag किया जो potential security breach हो सकता है",
    "नए caching layer ने read performance तो improve कर दी but cache invalidation थोड़ा tricky हो गया",
    # business / startup
    "हमारी startup ने अपनी latest funding round close की और अब hiring पर aggressively focus कर रहे हैं",
    "investor ने pitch deck देखने के बाद हमारी unit economics और customer retention पर सवाल पूछे",
    "नई pricing strategy launch करने के बाद हमारा monthly churn rate थोड़ा बढ़ गया है",
    "हमें इस product का roadmap फिर से prioritize करना होगा क्योंकि market demand shift हो रही है",
    "हमारी sales team ने इस quarter में पिछले से कहीं ज्यादा qualified leads convert किए",
    "उस competitor की aggressive marketing campaign के बाद हमारा market share थोड़ा pressure में है",
    "हमने customer feedback के आधार पर onboarding flow को पूरी तरह redesign कर दिया",
    "इस partnership से हमें distribution channels तक बेहतर access मिलेगा और reach भी बढ़ेगी",
    # daily life mid-density
    "मेरी car का service due है but workshop वाले अगले हफ्ते से पहले कोई slot नहीं दे रहे",
    "उसने अपने पूरे wardrobe को declutter किया और बहुत सारे कपड़े donate भी कर दिए",
    "इस महीने का grocery budget लगभग खत्म हो गया है अभी भी पंद्रह दिन बाकी हैं",
    "नए apartment में shift करने के बाद address proof update करना अभी बाकी है",
    "मेरे building की parking इतनी tight है कि हर बार car निकालना एक challenge बन जाता है",
    "उसने weekend पर पूरे घर की deep cleaning की और सारे cabinets भी organize कर दिए",
    "मेरा neighbour अपने terrace पर एक small organic garden maintain करता है काफी impressive है",
    "बिजली का बिल इस बार unexpectedly high आया तो मैंने पूरे appliances का usage check किया",
    # health / well-being denser
    "doctor ने मेरी blood report देखकर कहा कि vitamin D की deficiency है और supplements शुरू करने को कहा",
    "उसने अपनी sedentary lifestyle बदलने के लिए रोज एक structured workout routine follow करना शुरू किया",
    "लगातार screen exposure की वजह से मेरी eyes में strain रहता है इसलिए मैंने blue light filter लगाया",
    "stress manage करने के लिए उसने एक breathing exercise और journaling की practice अपनाई है",
    "मेरे physiotherapist ने कहा कि posture correct रखना long term back pain से बचने के लिए जरूरी है",
    "नई diet में मैंने refined sugar लगभग बंद कर दी और natural alternatives पर switch कर लिया",
    # finance denser
    "उसने अपने portfolio को diversify करने के लिए कुछ funds mutual funds में और कुछ fixed deposits में रखे",
    "tax filing से पहले मुझे सारे investment proofs और rent receipts एक जगह organize करने होंगे",
    "उस loan की interest rate floating है इसलिए हर rate revision पर मेरी EMI बदल जाती है",
    "मैंने एक emergency fund बनाया है जो लगभग छह महीने के expenses cover कर सकता है",
    "credit score improve करने के लिए उसने अपने सारे bills को auto pay पर set कर दिया",
    "इस insurance plan का premium थोड़ा high है but coverage और claim settlement ratio दोनों बेहतर हैं",
    # travel / commute denser
    "उसकी connecting flight का layover इतना short था कि बीच के airport में लगभग दौड़ना पड़ा",
    "हमने पूरे trip का itinerary पहले से plan किया ताकि last minute की booking में extra ना देना पड़े",
    "monsoon की वजह से highway पर visibility बहुत कम थी इसलिए driver ने speed काफी reduce कर दी",
    "नए travel app ने मुझे एक cheaper alternate route suggest किया जिससे काफी time बच गया",
    "international trip से पहले उसने अपना forex card load किया और sim का roaming pack भी activate किया",
    "train की waiting list confirm नहीं हुई तो आखिर में हमें एक last minute flight book करनी पड़ी",
    # food denser
    "उस new fusion restaurant का menu interesting है but कुछ dishes का portion size काफी small लगा",
    "मैंने एक healthy meal prep routine शुरू की है ताकि हफ्ते भर का खाना पहले से ready रहे",
    "उसने अपनी coffee की addiction कम करने के लिए धीरे धीरे green tea पर switch करना शुरू किया",
    "इस bakery का sourdough bread इतना fresh है कि लगभग हर सुबह जल्दी ही sold out हो जाता है",
    "restaurant ने हमारा online order गलत pack कर दिया तो उन्होंने तुरंत replacement भेज दिया",
    # education / career denser
    "उसने अपनी certification complete करने के बाद resume update किया और कई companies में apply किया",
    "online course का final assessment काफी comprehensive था जिसमें practical assignments भी शामिल थे",
    "mentor ने उसे suggest किया कि networking events attend करना career growth के लिए बहुत valuable है",
    "नई job के first week में उसे पूरे internal tools और workflows समझने में थोड़ा time लगा",
    "उसने अपनी public speaking improve करने के लिए एक local toastmasters club join किया",
    "interview की preparation के लिए मैंने कई mock sessions किए और अपने weak areas पर काम किया",
    # entertainment / social denser
    "उस documentary ने climate change के impact को इतने detail में cover किया कि सोचने पर मजबूर कर दिया",
    "नई gaming console का pre order इतनी जल्दी sold out हो गया कि बहुत से लोग disappointed रह गए",
    "उसने अपने podcast के लिए एक guest invite किया जो इस industry का काफी experienced expert है",
    "इस concert के tickets मिनटों में बिक गए और resale platforms पर price लगभग double हो गई",
    "हमने weekend पर एक movie marathon किया और साथ में काफी सारा junk food भी order किया",
    "उसकी photography का aesthetic इतना unique है कि उसका instagram engagement लगातार बढ़ रहा है",
    # home / gadgets denser
    "मेरे smart home setup में कभी कभी automation routines randomly trigger हो जाते हैं",
    "नई vacuum cleaner की suction power तो अच्छी है but उसकी battery backup थोड़ी limited है",
    "उसने पूरे घर में energy efficient LED bulbs लगाए जिससे electricity consumption काफी कम हुआ",
    "मेरे laptop का thermal throttling heavy tasks के दौरान performance को काफी degrade कर देता है",
    "नए noise cancelling headphones ने मेरे daily commute के experience को पूरी तरह बदल दिया",
    "उसका mechanical keyboard का custom keycap set finally एक लंबे shipping delay के बाद आ गया",
    # sports / fitness denser
    "उसने marathon की training के दौरान अपनी running pace और recovery time दोनों carefully track किए",
    "कल के match में हमारी team ने आखिरी over में एक brilliant comeback करके game जीत लिया",
    "नए fitness tracker ने मेरी sleep quality और resting heart rate के बारे में काफी insights दिए",
    "coach ने कहा कि strength training के साथ proper nutrition भी equally important है",
    "उसने अपना पहला cycling tournament जीता और अब एक professional team के साथ train कर रहा है",
    # misc representative
    "उसने अपने side project को finally launch किया और पहले ही हफ्ते में अच्छा traction मिला",
    "मेरे area में नई metro construction की वजह से traffic diversion ने commute और लंबा कर दिया",
    "हमने एक community event organize किया जिसमें local artists ने अपना work showcase किया",
    "उसका new freelance contract remote है इसलिए अब वो किसी भी timezone के clients के साथ काम कर सकता है",
    "मैंने अपने पुराने gadgets को exchange offer में देकर एक नया tablet काफी कम price में लिया",
    "उस webinar में speaker ने productivity hacks और time management पर कुछ practical tips share किए",
    "नई operating system update के बाद कुछ apps का compatibility issue आ रहा है",
    "उसने अपने blog पर एक detailed tutorial publish किया जो beginners के बीच काफी popular हो गया",
    "हमारी society ने एक rainwater harvesting system install किया ताकि water wastage कम हो सके",
    "मेरे phone का face unlock कम light में कभी कभी fail हो जाता है इसलिए मैं fingerprint use करता हूँ",
    "उसने अपने पूरे financial goals को एक spreadsheet में track करना शुरू किया है काफी disciplined approach है",
    "नए electric scooter की range एक single charge में लगभग पूरे शहर का commute cover कर लेती है",
    "उस startup का culture इतना transparent है कि हर decision के पीछे का reasoning पूरी team के साथ share होता है",
    "मैंने weekend पर एक online workshop attend किया जो creative writing के fundamentals पर था",
    "उसकी team ने एक tight deadline के बावजूद पूरा feature ship किया without compromising on quality",
    "नई subscription model launch करने के बाद हमारी recurring revenue काफी stable हो गई है",
    "उसने अपने garden में एक drip irrigation system लगाया जिससे पानी की काफी बचत होती है",
    "मेरे का area में internet provider ने finally fiber connection रोल आउट किया और speed कई गुना बढ़ गई",
    "उस museum की नई exhibition में ancient artifacts के साथ interactive digital displays भी थे",
    "हमने एक budget friendly weekend getaway plan किया जो nature के काफी close था",
    "उसकी presentation में data visualization इतनी clear थी कि complex numbers भी आसानी से समझ आ गए",
    "नए हुए regulation की वजह से कई companies को अपने data privacy policies update करनी पड़ीं",
    "मेरे laptop में एक SSD upgrade करने के बाद boot time लगभग कुछ ही seconds का रह गया",
    "उसने अपने fitness journey को document करने के लिए एक dedicated youtube channel शुरू किया",
    "इस app का offline mode काफी reliable है जिससे बिना internet के भी काम चलता रहता है",
    "हमारी team ने एक hackathon में हिस्सा लिया और एक prototype सिर्फ चौबीस घंटे में build किया",
    "उसने अपने investment decisions को emotion से अलग रखने के लिए एक systematic rule based approach अपनाया",
    "नए traffic management system ने शहर के कई busy intersections पर congestion काफी कम कर दिया",
    "मेरे का बच्चा अब coding सीख रहा है और उसने अपना पहला छोटा game भी खुद design किया",
    "उस conference में कई industry leaders ने emerging technologies के future पर अपने views share किए",
    "हमने एक old warehouse को renovate करके एक modern co working space में बदल दिया",
    "उसकी नई book का pre order इतना strong रहा कि publisher ने first edition की print run बढ़ा दी",
    "मेरे का सबसे पुराना दोस्त अब एक successful entrepreneur है और कई youngsters को mentor भी करता है",
    "नए solar panels install करने के बाद हमारी monthly electricity की लागत काफी हद तक कम हो गई",
    "उसने अपनी पूरी thesis को एक structured manner में organize किया with proper citations और references",
    "इस software की learning curve थोड़ी steep है but एक बार समझ आने के बाद productivity काफी बढ़ जाती है",
    "हमने अपने पुराने office space को छोड़कर एक बड़े और better located building में shift कर लिया",
    "उसका startup अब profitable है और वो अपने पहले set of employees को equity भी offer कर रहा है",
    "नई camera की low light performance इतनी अच्छी है कि रात की photos भी काफी sharp आती हैं",
    "मेरे का mentor हमेशा कहते हैं कि consistency किसी भी skill में mastery की असली key है",
    "उसने अपने daily commute के time को productive बनाने के लिए audiobooks सुनना शुरू कर दिया",
    "इस region में tourism बढ़ने के बाद local economy और small businesses दोनों को काफी benefit हुआ",
    "हमारी team ने customer support के लिए एक AI powered chatbot deploy किया जिससे response time घट गया",
    "उसने अपने पुराने furniture को upcycle करके एक creative और budget friendly home makeover किया",
    "नई policy के तहत employees अब अपने working hours को flexible तरीके से manage कर सकते हैं",
    "मेरे phone की storage लगभग full है इसलिए मैंने सारी पुरानी files को cloud पर migrate कर दिया",
    "उसकी research paper एक reputed journal में accept हुई जो उसके लिए एक बड़ी achievement है",
    "हमने एक local NGO के साथ collaborate करके एक education drive organize की weekend पर",
    "उस electric car की charging एक overnight session में पूरी हो जाती है जो काफी convenient है",
    "मेरे का छोटा भाई अब एक professional gamer बनना चाहता है और daily practice भी करता है",
    "नए recipe app ने मेरी cooking को काफी आसान बना दिया with step by step video instructions",
    "उसने अपनी savings से एक small rental property खरीदी जो अब एक steady passive income देती है",
    "इस गर्मी की heatwave इतनी intense है कि कई शहरों में power demand record level तक पहुंच गई",
    "हमारी team ने एक legacy system को पूरी तरह से एक modern cloud based architecture पर migrate किया",
    "उसने अपने interview के बाद एक thoughtful follow up email भेजा जिसने अच्छा impression बनाया",
]

VOICES = ["kaustubh", "arjun", "maya", "aadya"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seen = set()
    for r in csv.reader(open("data/xtts/metadata_train.csv"), delimiter="|"):
        if len(r) >= 2:
            seen.add(r[1].strip())
    for l in open("data/eval_heldout89.jsonl"):
        if l.strip():
            seen.add(json.loads(l)["ref_text"].strip())
    p1 = Path("data/eval_pow/new_prompts.jsonl")
    if p1.exists():
        for l in open(p1):
            if l.strip():
                seen.add(json.loads(l)["ref_text"].strip())

    rows, leak = [], 0
    for t in PROMPTS:
        t = t.strip()
        if t in seen:
            leak += 1; print("DUP/LEAK skip:", t[:45]); continue
        seen.add(t)
        ew = len(english_words(t))
        cs = "high" if ew >= 4 else ("med" if ew >= 2 else "none")
        v = VOICES[len(rows) % 4]
        rows.append({"utt_id": f"eq{len(rows):03d}__{v}", "ref_text": t, "voice": v, "cs_mode": cs})
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    from collections import Counter
    scor = sum(1 for r in rows if len(english_words(r["ref_text"])) >= 1)
    print(f"[powered_prompts2] wrote {len(rows)} ({leak} leak/dup dropped) scorable={scor} "
          f"cs={Counter(r['cs_mode'] for r in rows)} voices={Counter(r['voice'] for r in rows)}")


if __name__ == "__main__":
    main()
