# Code-switch boundary A/B spot-check — 16L vs round-1 RFT

The one check no automatic scorer can make: does the **Hindi→English boundary prosody** stay natural
after RFT? For each utterance below, listen to **A (16L, pre-RL)** then **B (round-1 RFT)**. Focus on
the **English words** (bold) — in B they should be *more clearly English* without sounding slowed,
over-articulated, or robotic, and the Hindi around them should keep its natural rhythm.

Files: `<utt>__A_16L.wav` (before) and `<utt>__B_round1.wav` (after).

## Accent-gain showcase (B should sound more natively English on the bold words)

| utt | 16L→R1 recall | text (English words bolded) |
|---|---|---|
| ev053__kaustubh | 0.33 → 0.83 (+0.50) | Ceiling fan की **speed** slow है, मुझे लगता है **winding burn** हो गई होगी अंदर से |
| ev046__arjun | 0.50 → 0.83 (+0.33) | गर्मी में मैं अपने **pet** को **hydrated** रखने के लिए **fresh water bowl** दिन में तीन… |
| ev079__kaustubh | 0.71 → 1.00 (+0.29) | अरे **loot drop** उस **rooftop** पर है, फटाफट **rotate** कर वरना **enemy team grab** कर लेगा |
| ev071__kaustubh | 0.71 → 1.00 (+0.29) | मुझे **doctor** का **appointment book** करना है **but slot** सब **evening** के already full |
| ev041__aadya | 0.80 → 1.00 (+0.20) | कल रात मैंने **ranked match** खेला था लेकिन **teammate** ने पूरी **game throw** कर दी |
| ev038__kaustubh | 0.80 → 1.00 (+0.20) | (same text, different voice) |

## Regression check (recall unchanged — confirm voice + prosody did NOT degrade)

| utt | recall | text |
|---|---|---|
| ev064__aadya | 0.33 → 0.33 | मेरा **phone** का **recharge** खत्म हो गया **data** बिल्कुल नहीं बचा |
| ev063__maya | 0.33 → 0.33 | (same text, different voice) |

## What "pass" looks like
- B's English words are clearer / more native than A's, **and**
- B does not sound slower, monotone, or robotic (the reward floors were designed to prevent this), **and**
- on the regression-check pair, B's voice + delivery are indistinguishable in quality from A.

If all three hold, the boundary-prosody gate passes and round-1 is shippable (pending the n≥150 powered cert).
