# Hinglish TTS via Synthetic Data (teacher TTS teacher) — Concrete Plan

**Goal:** Train a Hinglish (Hindi-English code-switched) TTS model for a few fixed voices,
using the teacher TTS as the synthetic-data teacher, without prosodic collapse.

**Context:** No real Hinglish data on hand. Few fixed target voices. Research project
(licensing not a blocker per user). This puts us on the **TDSC** branch of the
Stability-Expressivity paper (#3) plus the **NileTTS** template (#15).

---

## Why teacher TTS changes the plan

The earlier blocker was: with no real data, you cannot fine-tune a teacher on code-switching,
so you depend entirely on an off-the-shelf teacher's *native* Hinglish ability. Most teachers
fail this. **the teacher TTS explicitly supports Hindi+English mid-utterance switching** via its
Indic Group (shared phoneme space). If that claim holds in practice, the project is feasible.

**Single-teacher risk (real):** Paper #70 used 3 teachers to diversify style; the review warns
"a single teacher's style bias dominates the student." Mitigate by using multiple teacher TTS
voices + speed variation, and ideally blending a second teacher (CosyVoice2 / XTTS-v2) for a
fraction of the corpus.

---

## Teacher API reference (confirmed)

- Endpoint: `POST https://api.teacher TTS/waves/v1/teacher-tts/get_speech`
- Auth: `Authorization: Bearer $TEACHER_TTS_API_KEY`
- Python SDK: `pip install teacherai` -> `from teacherai.waves import WavesClient`
- Audio: native 44.1kHz; request `sample_rate: 24000`, `output_format: "wav"`
- Chunk limit: ~250 chars/request (split long text on sentence/clause boundaries)
- Text rules: Hindi words in **Devanagari**, English words in **Latin**, no transliteration
- Voices: stock Indic voices + clone from 5-15s reference clip -> voice_id
- Cost: ~$0.01/min => ~$0.60/h of audio (a 60h corpus ~ $36)

```python
from teacherai.waves import WavesClient
client = WavesClient(api_key="...")
client.synthesize(
    text="मुझे एक coffee चाहिए, can you make it strong?",
    voice_id="<indic_voice_id>",
    sample_rate=24000,
    speed=1.0,
    save_as="out.wav",
)
```

---

## Phase 0 — Teacher go/no-go gate (1 afternoon, do this first)

Do not build the corpus until this passes.

1. Write ~50 real Hinglish sentences with dense intra-sentence switches
   (e.g. "कल मैंने नया phone खरीदा but the battery life is terrible").
2. Synthesize each with 3-4 candidate teacher TTS Indic voices, at speeds 0.9 / 1.0 / 1.1.
3. Listen for the failure mode that kills the project: does it pronounce English words
   English-style and Hindi words Hindi-style *within one sentence*, or smear one phonology
   across both ("uniform thick accent" failure)?
4. Run your chosen ASR over the outputs; spot-check WER on the switch points specifically.

**Decision:** if >=2 voices code-switch cleanly, proceed. If none do, stop and reconsider
(no recipe tuning fixes a teacher that cannot switch).

---

## Phase 0.5 — Real anchor data (1-2h, the DGSA upgrade)

Purpose: define the fixed target voice AND inject real human prosody so synthetic data
cannot collapse the model into flatness. Quality bar is higher here than for the synthetic
corpus. Rules: one speaker, clean, consistent, **expressive** delivery (not monotone).
You need ~1-2h *after* cleaning, so source more than that raw.

### Path A — Record yourself (preferred if the voice is available)
- Mic: decent USB mic or good phone, not laptop built-in.
- Room: small, soft furnishings, no echo; kill AC/fan/traffic; phone on airplane mode.
- Settings: 48kHz (or 44.1kHz), **mono**, WAV/lossless, 16/24-bit. Downsample to 24kHz later.
- Script: read the Phase 1 Hinglish transcripts. Cover switch points, questions, numbers.
  ~700-1000 sentences ~ 1-2h.
- Consistency: same mic / distance (a fist away) / room; 1-2 sittings; take breaks.
- After: slice into 3-30s clips, each paired with its transcript.

### Path B — YouTube (only if target is a specific public voice you can't record)
Emilia-Pipe recipe (#34/#77):
1. `yt-dlp` audio (solo podcasts/vlogs/lectures, not music videos).
2. Source-separate (Demucs / UVR) to strip music/background.
3. Diarize (pyannote), keep only the target speaker.
4. VAD-segment into 3-30s clips.
5. ASR pseudo-transcribe (IndicWhisper / Sarvam).
6. Filter: DNSMOS > 3.0, lang-conf > 0.8, drop residual-music/overlap clips.
Download ~3-4h raw to net ~1-2h clean. Keep recording setup consistent across videos.
Note: YouTube audio is lossy (~128kbps) and acceptable as anchor, not studio-grade.

### Decision
Path A if you can record the voice directly (clean + real, zero cleanup). Path B only for
a specific public voice. Ethics: cloning an identifiable real person carries weight beyond
licensing; keep to a voice you have a reasonable basis to use.

---

## Existing asset: `data/spontaneous_hinglish/` (copied from VoiceSangam)

A real spontaneous Hinglish dataset is already in the repo: 1,497 YouTube clips, ~2.51 h,
7 speakers, 16 kHz, with Devanagari + romanized transcripts, code-mix density labels, and
multi-ASR hypotheses scored against consensus gold. See its `README.md`. Honest roles:
- **Phase 1 text:** real code-switch transcripts to use as read-scripts / distribution.
- **Phase 3 ASR filter:** `HYP_*`/`SCORE_*` already benchmark which Hinglish ASR wins.
- **Phase 6 eval:** the 1,497-clip set is your hard out-of-domain test.
- **Phase 0.5 anchor:** only a weak bonus (153 single-speaker clips, 0.22 h, 16 kHz).
  Re-extract at 24 kHz from raw if used. Still record fresh 1-2 h for the real anchor.

---

## Phase 1 — Text corpus (LLM-generated, your strongest card)

1. Prompt an LLM (Claude) to generate Hinglish transcripts across your target domains.
   Explicitly require: dense code-switch points, numbers/dates, questions, exclamations,
   varied sentence length. Hindi in Devanagari, English in Latin.
2. Deduplicate (fuzzy/MinHash) to avoid mode collapse (the #B "low lexical diversity" pitfall).
3. Target volume: enough text for ~60-100h of audio after filtering. With ~12s/utterance
   average, 100h ~ 30k utterances; over-generate ~1.4x to survive the WER filter.
4. Track diversity: unique-word ratio, switch-point frequency per utterance, domain balance.

---

## Phase 2 — Synthesis (teacher TTS teacher)

1. For each transcript, synthesize across **2-4 fixed voices** (these become your student's
   output voices) plus optional cloned voices for your real targets.
2. Vary speed in [0.9, 1.1] for prosodic variety (cheap diversity lever).
3. Chunk text to <=250 chars on clause boundaries; concatenate audio per utterance.
4. Save 24kHz WAV + the exact intended transcript (this pairing is your dataset).
5. (Optional, recommended) generate ~15-20% of utterances with a second teacher
   (CosyVoice2 / XTTS-v2) to break single-teacher style bias.

---

## Phase 3 — Filter HARD (highest-leverage step)

1. ASR every synthesized utterance. **Use a code-switch-capable ASR**: Whisper-large-v3,
   or IndicWhisper / Sarvam / AI4Bharat for better Hindi+English. This choice matters; a weak
   ASR filter is the biggest practical risk.
2. Keep only utterances with **WER < 0.20** vs the intended transcript. Regenerate failures
   up to ~10x (different voice/speed) before discarding.
3. Add quality gates: DNSMOS > 3.0, duration in [3s, 30s], language-confidence check.
4. Log how much you dropped per gate (no silent truncation).

---

## Phase 4 — Train the student (SFT)

- **Base model:** CosyVoice2 (preferred — prosody-timbre disentanglement gives you the
  expressivity-recovery levers in Phase 5) or XTTS-v2 (NileTTS-proven, simpler).
- Fine-tune on filtered real-transcript + synthetic-audio pairs.
- Volume reference: NileTTS reached a working dialectal model with ~38h / 2 speakers.
  Budget 40-100h for your fixed voices.
- Monitor during training: **token entropy** and **4-gram repetition rate** — these are your
  Synthetic-Erosion alarms from #3 (repetition climbing toward ~9% = prosody collapsing).

---

## Phase 5 — Recover expressivity (TDSC, since no real data)

Pure-synthetic SFT will be intelligible but flat. Recover prosody with the no-real-data method
from paper #3:

1. Generate candidates at low / mid / high sampling temperature.
2. Strict ASR filter: WER < tau_w, repetition < tau_r, length in [gamma_min|x|, gamma_max|x|].
3. Iterative SFT on accepted samples.
4. **DPO** on filtered preference pairs with a temperature curriculum.
   (Reference: Lao 100% synthetic via TDSC reached WER 29.8%, NMOS 3.94, beating Azure/Gemini.)

---

## Phase 6 — Evaluation

- WER (code-switch ASR), UTMOS/NMOS, speaker SIM.
- A held-out **hard** set with dense code-switching (the real test).
- Human A/B vs the teacher TTS teacher itself (can the student match its teacher?).

---

## Honest expectations

- **Achievable:** a good, production-usable fixed-voice Hinglish TTS (target NMOS ~3.8-4.0).
- **Unlikely without any real data:** studio-indistinguishable quality. No published result
  shows pure-synthetic *code-switched* TTS matching real-data quality.
- **Highest-ROI upgrade:** record even **1-2 hours** of your actual target voices reading
  Hinglish. That moves you from TDSC to **DGSA** (#3), the gap between NMOS 3.94 and 4.42.
  For fixed voices that is a single recording session.

---

## Risk register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Teacher can't truly code-switch | Fatal | Phase 0 gate before any spend |
| Single-teacher style bias | High | Multi-voice + speed var + 2nd teacher blend |
| Weak Hinglish ASR -> bad filter | High | Use IndicWhisper/Sarvam; spot-check switch points |
| Synthetic Erosion (prosody collapse) | High | Cap reliance, monitor entropy/repetition, TDSC+DPO |
| Transliteration errors in text | Medium | Enforce Devanagari/Latin per word; normalize numbers |
| Mode collapse in LLM text | Medium | Fuzzy dedup, diversity metrics |
