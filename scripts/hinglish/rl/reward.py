#!/usr/bin/env python3
"""RL reward + expressivity monitors, computed on the DECODED 24 kHz waveform (never on DVAE tokens).

Design (see docs/RL_EXPRESSIVITY_PLAN.md):
  ONE trusted maximizer  -> R_accent = harmonic_mean(whisper-en recall, exp(-NLL/3))
  CAPPED one-sided floors -> UTMOS, SECS, duration, F0-std, energy dynamic-range.
Each floor penalizes regression BELOW the frozen-16L base on the same prompt; ZERO credit for exceeding.
The UTMOS cap is what removes the prosody-flattening incentive (only clean because 16L starts at teacher UTMOS).

Used two ways:
  - offline RFT winner-selection: a candidate is an eligible WINNER only if it passes ALL floors; among the
    eligible, pick max R_accent. You literally cannot train toward a flatter/slower/voice-drifted sample.
  - scalar reward (GRPO escalation): total = w_acc*R_accent + sum(w_i * floor_penalty_i), reward-scaling OFF.

Heavy models (whisper-large-v3, utmos22_strong, resemblyzer) load once; reuse the scorer across a run.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
import numpy as np
import librosa
import torch

# ---- accent helpers (mirrors scripts/hinglish/10_accent_eval.py; that file's numeric name is unimportable) ----
EN_TOK = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")
STOP = set("a an the to of in on at is are am be been being and or but so if it i you we he she they "
           "my your his her our their this that these those for with as it's i'm".split())


def english_words(text: str):
    return [w.lower().strip("'-") for w in text.split() if EN_TOK.match(w) and w.lower() not in STOP]


def fuzzy_in(word: str, hypwords) -> bool:
    for h in hypwords:
        if h == word:
            return True
        if abs(len(h) - len(word)) <= 2:
            dp = list(range(len(h) + 1))
            for i, a in enumerate(word, 1):
                prev, dp[0] = dp[0], i
                for j, b in enumerate(h, 1):
                    cur = dp[j]; dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a != b)); prev = cur
            if dp[-1] <= max(1, len(word) // 4):
                return True
    return False


def harmonic_mean(a: float, b: float) -> float:
    a = max(a, 1e-6); b = max(b, 1e-6)
    return 2 * a * b / (a + b)


# ---- prosody / degenerate features (cheap, CPU, no pyworld) ----
def f0_std_semitones(wav: np.ndarray, sr: int) -> float:
    """SD of pitch in semitones over voiced frames; the primary flattening detector."""
    try:
        f0, vflag, _ = librosa.pyin(wav, fmin=65, fmax=400, sr=sr, frame_length=1024)
    except Exception:
        return 0.0
    f0 = f0[np.isfinite(f0)]
    f0 = f0[f0 > 0]
    if f0.size < 5:
        return 0.0
    med = np.median(f0)
    semis = 12.0 * np.log2(f0 / med)
    return float(np.std(semis))


def energy_dr_db(wav: np.ndarray, sr: int) -> float:
    """Dynamic range (p95-p5) of frame RMS in dB over voiced (above-floor) frames; gain-invariant via peak-norm."""
    peak = np.max(np.abs(wav)) + 1e-9
    w = wav / peak
    rms = librosa.feature.rms(y=w, frame_length=1024, hop_length=256)[0]
    db = 20 * np.log10(rms + 1e-7)
    voiced = db[db > (db.max() - 40)]  # ignore the silent floor
    if voiced.size < 5:
        return 0.0
    return float(np.percentile(voiced, 95) - np.percentile(voiced, 5))


def silence_ratio(wav: np.ndarray, sr: int) -> float:
    """Fraction of frames below -40 dB of peak (energy-based VAD; webrtcvad not installed)."""
    peak = np.max(np.abs(wav)) + 1e-9
    rms = librosa.feature.rms(y=wav / peak, frame_length=1024, hop_length=256)[0]
    db = 20 * np.log10(rms + 1e-7)
    return float(np.mean(db < -40))


def ngram_rep(words, n: int = 3) -> float:
    """Max repeated n-gram fraction in the transcript (loop/babble detector)."""
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    from collections import Counter
    top = Counter(grams).most_common(1)[0][1]
    return top / len(grams)


@dataclass
class Floors:
    """Per-prompt frozen-16L baseline + acceptance thresholds."""
    utmos: float            # base UTMOS (e.g. 3.141 global, or per-prompt)
    secs: float             # base SECS
    f0_std: float           # base pitch-SD (semitones)
    energy_dr: float        # base energy dynamic range (dB)
    dur: float              # reference (teacher/base) duration (s) for pacing
    secs_margin: float = 0.03
    dur_tol: float = 0.15
    sil_cap: float = 0.45
    rep_cap: float = 0.30


@dataclass
class Weights:
    accent: float = 1.0
    utmos: float = 0.3
    secs: float = 0.5
    duration: float = 0.1
    f0_std: float = 0.2
    energy_dr: float = 0.1


class RewardScorer:
    def __init__(self, device: str = "cuda", whisper_model: str = "openai/whisper-large-v3"):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        from resemblyzer import VoiceEncoder
        self.dev = device if torch.cuda.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(whisper_model)
        self.asr = AutoModelForSpeechSeq2Seq.from_pretrained(
            whisper_model, torch_dtype=torch.float16).to(self.dev).eval()
        self.utmos = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True).to(self.dev).eval()
        self.venc = VoiceEncoder(device=self.dev)
        self._ref_emb = {}

    def register_voice(self, voice: str, ref_wav: str):
        from resemblyzer import preprocess_wav
        self._ref_emb[voice] = self.venc.embed_utterance(preprocess_wav(ref_wav))

    # ---- individual scorers ----
    def _whisper_en(self, wav16: np.ndarray):
        feat = self.proc(wav16, sampling_rate=16000, return_tensors="pt").input_features.to(self.dev, torch.float16)
        with torch.no_grad():
            out = self.asr.generate(feat, language="en", task="transcribe", max_new_tokens=128,
                                    return_dict_in_generate=True, output_scores=True)
        ids = out.sequences
        hyp = self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        # mean NLL of generated tokens (confidence densifier)
        try:
            ts = self.asr.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            lp = ts[0].float().cpu().numpy()
            lp = lp[np.isfinite(lp)]
            nll = float(-np.mean(lp)) if lp.size else 5.0
        except Exception:
            nll = 5.0
        return hyp, nll

    def _utmos(self, wav16: np.ndarray) -> float:
        with torch.no_grad():
            return float(self.utmos(torch.from_numpy(wav16).unsqueeze(0).to(self.dev), 16000))

    def _secs(self, wav_path_or_arr, voice: str):
        from resemblyzer import preprocess_wav
        if voice not in self._ref_emb:
            return None
        emb = self.venc.embed_utterance(preprocess_wav(wav_path_or_arr))
        r = self._ref_emb[voice]
        return float(np.dot(emb, r) / (np.linalg.norm(emb) * np.linalg.norm(r)))

    def components(self, wav: np.ndarray, sr: int, ref_text: str, voice: str, wav_path: str = None) -> dict:
        """Raw per-candidate measurements on the decoded waveform."""
        wav = np.asarray(wav, dtype=np.float32)
        wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav
        hyp, nll = self._whisper_en(wav16)
        hypw = [w.lower().strip(".,!?'\"-") for w in hyp.split()]
        intended = english_words(ref_text)
        en_recall = (sum(1 for w in intended if fuzzy_in(w, hypw)) / len(intended)) if intended else None
        return {
            "hyp": hyp, "nll": nll, "n_en": len(intended),
            "en_recall": en_recall,
            "utmos": self._utmos(wav16),
            "secs": self._secs(wav_path if wav_path else wav16.astype(np.float32), voice),
            "f0_std": f0_std_semitones(wav, sr),
            "energy_dr": energy_dr_db(wav, sr),
            "dur": len(wav) / sr,
            "silence_ratio": silence_ratio(wav, sr),
            "ngram_rep": ngram_rep(hypw),
        }

    def score(self, comp: dict, floors: Floors, weights: Weights = None) -> dict:
        """Turn raw components into R_accent, per-floor penalties, a scalar total, and the eligibility flag."""
        w = weights or Weights()
        rec = comp["en_recall"]
        # R_accent: only the prompts that carry English words drive accent; pure-Hindi prompts get a neutral 1.0
        if rec is None:
            r_accent = 1.0
        else:
            conf = float(np.exp(-comp["nll"] / 3.0))
            r_accent = harmonic_mean(rec, conf)

        p_utmos = -max(0.0, floors.utmos - comp["utmos"])
        p_secs = -max(0.0, (floors.secs - floors.secs_margin) - (comp["secs"] if comp["secs"] is not None else 1.0))
        p_dur = -abs(comp["dur"] - floors.dur) / max(floors.dur, 1e-6)
        p_f0 = -max(0.0, floors.f0_std - comp["f0_std"])
        p_edr = -max(0.0, floors.energy_dr - comp["energy_dr"])

        degenerate = (comp["silence_ratio"] > floors.sil_cap) or (comp["ngram_rep"] > floors.rep_cap)
        # eligibility for RFT winner = all floors held + not degenerate
        eligible = (
            (not degenerate)
            and comp["utmos"] >= floors.utmos
            and (comp["secs"] is None or comp["secs"] >= floors.secs - floors.secs_margin)
            and comp["f0_std"] >= floors.f0_std
            and comp["energy_dr"] >= floors.energy_dr
            and abs(comp["dur"] - floors.dur) <= floors.dur_tol * floors.dur
        )
        total = (w.accent * r_accent + w.utmos * p_utmos + w.secs * p_secs
                 + w.duration * p_dur + w.f0_std * p_f0 + w.energy_dr * p_edr)
        if degenerate:
            total -= 5.0  # hard sink so degenerate rollouts never win a group
        return {
            "r_accent": round(r_accent, 4),
            "p_utmos": round(p_utmos, 4), "p_secs": round(p_secs, 4), "p_dur": round(p_dur, 4),
            "p_f0": round(p_f0, 4), "p_edr": round(p_edr, 4),
            "degenerate": bool(degenerate), "eligible": bool(eligible),
            "total": round(float(total), 4),
        }
