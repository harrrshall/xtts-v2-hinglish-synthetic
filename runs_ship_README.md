---
license: other
license_name: coqui-public-model-license
license_link: https://coqui.ai/cpml
language:
- hi
- en
tags:
- text-to-speech
- tts
- xtts
- hinglish
- code-switching
- knowledge-distillation
- dpo
- rlhf
pipeline_tag: text-to-speech
base_model: coqui/XTTS-v2
library_name: coqui-tts
---

# XTTS-v2 Hinglish — 265M (distilled + RL-tuned)

A **compressed, accent-recovered Hinglish (Hindi-English code-switch) TTS** model. It is the autoregressive
GPT of Coqui **XTTS-v2**, fine-tuned for Hinglish, then **distilled from ~443M → 265M parameters (1.67×
smaller)** and repaired with reinforcement learning — all under a strict rule: **prove no quality loss
before keeping any change.**

4 built-in voices: `kaustubh`, `arjun`, `maya`, `aadya`.

## What makes this model

The goal was a smaller, faster Hinglish model that sounds identical to the 443M original. The work:

1. **Depth distillation** (30 → 16 transformer layers) with logit + hidden-state KD. This held naturalness
   and voice identity, but the code-switch **accent** regressed.
2. **Reinforcement learning to recover accent without flattening prosody.** Offline RFT (best-of-N
   rejection-sampling) + length-normalized DPO, with a reward that has **one trusted maximizer**
   (English-word ASR recall) and **capped one-sided floors** on naturalness, voice similarity, pitch
   variance, energy and duration — so accent could only improve *without* degrading anything else.

## Certified results (vs the 443M teacher, n≈314 held-out, TOST / BCa bootstrap equivalence gate)

| axis | Δ vs teacher | margin | verdict |
|---|---|---|---|
| **naturalness** (UTMOS) | −0.03 | 0.10 | ✅ **no quality loss** |
| **voice similarity** (SECS) | −0.007 | 0.03 | ✅ **no quality loss** |
| **code-switch accent** (whisper-en recall) | −0.019 | 0.03 | near-parity (within 0.019 of teacher) |
| **expressivity** (pitch variance) | preserved (+1.3%) | — | ✅ no flattening |

The model is **certified to have no quality loss in naturalness and voice** relative to the 443M teacher,
with **code-switch accent recovered to near-parity** (within 0.019; just shy of a strict formal pass at the
confidence bound) and **expressivity preserved** — the RL improved accent *without* the prosody-flattening
that naive reward optimization causes.

## Usage

```python
# pip install coqui-tts soundfile librosa
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf, numpy as np
from pathlib import Path

repo = "."  # local clone of this repo
cfg = XttsConfig(); cfg.load_json(f"{repo}/config.json")
cfg.model_args.gpt_layers = 16
cfg.model_args.dvae_checkpoint = f"{repo}/dvae.pth"
cfg.model_args.mel_norm_file  = f"{repo}/mel_stats.pth"

model = Xtts.init_from_config(cfg)
model.load_checkpoint(cfg, checkpoint_path=f"{repo}/model.pth",
                      vocab_path=f"{repo}/vocab.json", use_deepspeed=False)
model.cuda().eval()

gpt_cond, speaker = model.get_conditioning_latents(audio_path=[f"{repo}/refs/kaustubh.wav"])
out = model.inference(
    "मैंने आज ek नया project start किया है and यह बहुत interesting लग रहा है",
    "hi", gpt_cond, speaker, temperature=0.7, enable_text_splitting=False)
sf.write("out.wav", np.asarray(out["wav"], dtype=np.float32), 24000)
```

> Note: on recent torch/torchaudio, route XTTS audio I/O through `soundfile` (see the `xtts_patch.py`
> approach) if `torchaudio.load` errors.

## Files

- `model.pth` — the 265M (16-layer) GPT + inherited DVAE/HiFi-GAN weights (fp32).
- `config.json` — XTTS config with `gpt_layers=16`.
- `vocab.json`, `dvae.pth`, `mel_stats.pth`, `speakers_xtts.pth` — frozen/inherited assets.
- `refs/*.wav` — the 4 speaker reference clips.
- `samples/*.wav` — demo outputs.

## Limitations

- **Hinglish-focused.** Tuned for Hindi-English code-switch (`language="hi"`); not a general multilingual model.
- **Accent is near-parity, not a certified strict pass** — within 0.019 of the 443M teacher; on the hardest
  code-switch cases the larger model is marginally crisper.
- **4 fixed reference voices** are provided; zero-shot cloning from arbitrary audio is inherited from XTTS-v2
  but not the focus here.

## License & attribution

Derivative of **Coqui XTTS-v2**, distributed under the **Coqui Public Model License (CPML)** —
**non-commercial use only**. See https://coqui.ai/cpml. By using this model you agree to the CPML terms.
Base model: [`coqui/XTTS-v2`](https://huggingface.co/coqui/XTTS-v2).

## Method details

Full methodology, benchmark design, and the sub-100M roadmap are documented in the project repo
(distillation trainer, capped-floor reward, RFT/DPO pipeline, and the TOST/BCa equivalence harness).
