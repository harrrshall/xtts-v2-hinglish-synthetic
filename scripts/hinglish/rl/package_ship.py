#!/usr/bin/env python3
"""Package the round-1 RFT checkpoint into a self-contained, portable HF release dir, then VERIFY it
loads + generates before any upload."""
import json, shutil, sys, glob
from pathlib import Path
import torch, numpy as np, soundfile as sf
sys.path.insert(0, "scripts/hinglish/fp16")
import xtts_patch  # noqa
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

SHIP = Path("runs/rl/ship_265m")
BASE = Path(".tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2")
REL = Path("runs/xtts_hinglish/RELEASE")
SHIP.mkdir(parents=True, exist_ok=True)

# 1. strip round-1 checkpoint -> model.pth (fp32 model dict only, no optimizer/trainer state)
ck = sorted(glob.glob("runs/rl/round1_rft/*/best_model*.pth"))[-1]
full = torch.load(ck, map_location="cpu", weights_only=False)
model_sd = full.get("model", full)
torch.save({"model": model_sd}, SHIP / "model.pth")
print(f"[pkg] model.pth: {len(model_sd)} tensors, {(SHIP/'model.pth').stat().st_size/1e9:.2f} GB (from {ck})")

# 2. config.json with gpt_layers=16 + portable bare-filename asset paths
cfg = json.load(open(REL / "config.json"))
cfg["model_args"]["gpt_layers"] = 16
for k, fn in [("dvae_checkpoint", "dvae.pth"), ("mel_norm_file", "mel_stats.pth"),
              ("xtts_checkpoint", "model.pth"), ("tokenizer_file", "vocab.json")]:
    if k in cfg["model_args"]:
        cfg["model_args"][k] = fn
json.dump(cfg, open(SHIP / "config.json", "w"), indent=2)
print(f"[pkg] config.json gpt_layers={cfg['model_args']['gpt_layers']}")

# 3. copy frozen/inherited assets + speaker refs
for src, dst in [(BASE / "vocab.json", "vocab.json"), (BASE / "dvae.pth", "dvae.pth"),
                 (BASE / "mel_stats.pth", "mel_stats.pth"), (REL / "speakers_xtts.pth", "speakers_xtts.pth")]:
    if Path(src).exists():
        shutil.copy(src, SHIP / dst)
(SHIP / "refs").mkdir(exist_ok=True)
for v in ["kaustubh", "arjun", "maya", "aadya"]:
    rp = REL / "refs" / f"{v}.wav"
    if rp.exists():
        shutil.copy(rp, SHIP / "refs" / f"{v}.wav")
print(f"[pkg] assets: {sorted(p.name for p in SHIP.iterdir())}")

# 4. VERIFY: load from the ship dir + generate one clip
cfg_v = XttsConfig(); cfg_v.load_json(str(SHIP / "config.json"))
cfg_v.model_args.gpt_layers = 16
for k, fn in [("dvae_checkpoint", "dvae.pth"), ("mel_norm_file", "mel_stats.pth")]:
    setattr(cfg_v.model_args, k, str(SHIP / fn))
m = Xtts.init_from_config(cfg_v)
m.load_checkpoint(cfg_v, checkpoint_path=str(SHIP / "model.pth"), vocab_path=str(SHIP / "vocab.json"), use_deepspeed=False)
m.cuda().eval()
gpt_cond, spk = m.get_conditioning_latents(audio_path=[str(SHIP / "refs" / "kaustubh.wav")])
text = "मैंने आज ek नया project start किया है and यह बहुत interesting लग रहा है"
o = m.inference(text, "hi", gpt_cond, spk, temperature=0.7, enable_text_splitting=False)
wav = np.asarray(o["wav"], dtype=np.float32)
(SHIP / "samples").mkdir(exist_ok=True)
sf.write(str(SHIP / "samples" / "demo_kaustubh.wav"), wav, 24000)
assert np.isfinite(wav).all() and len(wav) > 24000, "verification gen failed"
print(f"[pkg] VERIFY OK: generated {len(wav)/24000:.1f}s -> samples/demo_kaustubh.wav")
print("SHIP_PACKAGE_OK")
