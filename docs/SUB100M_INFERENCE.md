# Sub-100M Hinglish TTS — Inference & Usage

How to load and run the **89.96M** fixed-voice Hinglish student (`student640b_rft.pt`). See
[`SUB100M_RESULTS.md`](SUB100M_RESULTS.md) for how it was built and certified.

## What the checkpoint contains

`student640b_rft.pt` is a dict:
- `gpt` — the d=640 GPT state-dict (16 layers, 10 heads×64, FFN 2560, no speaker pathway).
- `adapter` — the `Linear(640→1024)` weights mapping GPT latents to the frozen vocoder's input dim.
- `voices` — the 4 baked voices, each `{cond_latents: (1,32,640), speaker_embedding: (1,512,1)}`.
- `model_args`, `d_student`, `heads`, `keep_idx`, `head_keep`, `ffn_keep` — architecture + provenance.

It does **not** contain the DVAE or HiFi-GAN — those are the frozen base XTTS assets, loaded separately
(the vocoder + tokenizer come from any 16-layer XTTS checkpoint; the round1 265M ckpt is used by default).

## Minimal inference

The loader and decode path are in `scripts/hinglish/rl/gen_student640.py`. Minimal use:

```python
import sys, torch, soundfile as sf, numpy as np
sys.path.insert(0, "scripts/hinglish")
from rl.gen_student640 import load_student

BASE = ".tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2"
student, vocoder_model, tok, ck = load_student("runs/rl/sub100m/student640b_rft.pt", BASE)
# vocoder_model supplies the FROZEN hifigan_decoder + tokenizer (a 16-layer XTTS shell)

voice = "maya"                       # one of: aadya, arjun, kaustubh, maya
text  = "आज office में एक important meeting है तो मैं थोड़ा busy रहूँगा"   # Devanagari + English, spell digits as words
cond  = ck["voices"][voice]["cond_latents"].cuda()
spk   = ck["voices"][voice]["speaker_embedding"].cuda()
tt    = torch.IntTensor(tok.encode(text, lang="hi")).unsqueeze(0).cuda()

with torch.no_grad():
    codes = student.gpt.generate(cond_latents=cond, text_inputs=tt, do_sample=False, top_k=1,
                                 temperature=0.7, repetition_penalty=1.3, num_return_sequences=1,
                                 num_beams=1, length_penalty=1.0)
    exp  = torch.tensor([codes.shape[-1] * student.gpt.code_stride_len], device="cuda")
    tlen = torch.tensor([tt.shape[-1]], device="cuda")
    lat640  = student.gpt(tt, tlen, codes, exp, cond_latents=cond, return_latent=True)  # (1,T,640)
    lat1024 = student.vocoder_latents(lat640)                                           # (1,T,1024) via adapter
    wav = vocoder_model.hifigan_decoder(lat1024, g=spk).cpu().squeeze().numpy()         # frozen vocoder
sf.write("out.wav", np.asarray(wav, np.float32), 24000)
```

Or just use the CLI wrappers:
```bash
# one clip per prompt (greedy, most faithful)
python scripts/hinglish/rl/gen_student640.py --student runs/rl/sub100m/student640b_rft.pt \
    --prompts scripts/hinglish/rl/sample_prompts.jsonl --out-dir out/ --n 1 --greedy --rep-penalty 1.3
# objective scoring (accent / UTMOS / SECS / tail)
python scripts/hinglish/rl/eval_student640.py --student runs/rl/sub100m/student640b_rft.pt \
    --prompts <prompts.jsonl> --out-dir out/ --greedy --rep-penalty 1.3
```

## Operational notes

- **Fixed voices only.** The speaker pathway is deleted; the model speaks the 4 baked voices (aadya, arjun,
  kaustubh, maya). No zero-shot cloning — that is the trade that makes <100M possible.
- **Language tag is `"hi"`** for both pure-Hindi and Hinglish; write Hindi in Devanagari, English in Latin.
- **Spell numbers as words.** Digits crash `num2words` for `hi` (e.g. write "बीस" not "20"); the helper scripts
  skip digit-bearing prompts rather than crash.
- **Long text:** chunk inputs over ~150 characters (XTTS `hi` truncation limit), same as the 265M model.
- **Decoding:** greedy with `repetition_penalty≈1.3` is the most faithful. ~14.6% of generations may still run
  long; resample or apply a length guard if needed.
- **Runtime:** ~0.5–1.1 s/clip on the GPU box (codes + latents + vocoder), faster than the 265M model.
