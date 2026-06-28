#!/usr/bin/env python3
"""Package everything the static web app needs: onnx models, baked voices, tokenizer, config,
and parity fixtures (golden token ids + golden codes for the in-browser checks C8/C9)."""
import os, sys, json, shutil
os.environ.setdefault("COQUI_TOS_AGREED", "1")
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; MODELS = REPO.parent / "syntts_models"
DEMO = MODELS / "local_demo"
OUT = REPO / "webgpu" / "models"; GOLD = REPO / "webgpu" / "golden"
APP = REPO / "webgpu" / "app"; ASSETS = APP / "assets"; ASSETS.mkdir(parents=True, exist_ok=True)
(APP / "models").mkdir(exist_ok=True)

meta = json.loads((GOLD / "meta.json").read_text()); C = meta["constants"]

# --- onnx models (fp32; symlink-copy) ---
for n in ("gpt_prefill", "gpt_decode", "vocoder"):
    shutil.copy(OUT / f"{n}.onnx", APP / "models" / f"{n}.onnx")
    print(f"[model] {n}.onnx -> app/models ({(APP/'models'/f'{n}.onnx').stat().st_size/1e6:.0f} MB)")

# --- voices: concat Float32 blob + json offsets ---
vz = np.load(GOLD / "voices.npz")
blob = bytearray(); voices = {}
for v in meta["voices"]:
    cond = vz[f"{v}_cond"].astype(np.float32)      # (32,640)
    spk = vz[f"{v}_spk"].astype(np.float32)        # (512,1)
    co = len(blob) // 4; blob += cond.tobytes()
    so = len(blob) // 4; blob += spk.tobytes()
    voices[v] = {"cond_off": co, "cond_shape": list(cond.shape),
                 "spk_off": so, "spk_shape": list(spk.shape)}
(ASSETS / "voices.bin").write_bytes(bytes(blob))
(ASSETS / "voices.json").write_text(json.dumps(voices))
print(f"[voices] voices.bin {len(blob)/1e3:.0f} KB  ({list(voices)})")

# --- tokenizer.json (the base XTTS vocab.json IS the HF tokenizers BPE file) ---
from TTS.utils.manage import ModelManager
base = Path(ModelManager().download_model("tts_models/multilingual/multi-dataset/xtts_v2")[0])
tjson = base / "vocab.json"
shutil.copy(tjson, ASSETS / "tokenizer.json")
print(f"[tok] tokenizer.json {tjson.stat().st_size/1e3:.0f} KB")

# --- config for the app ---
cfg = {
    "constants": C,
    "decode": {"repetition_penalty": float(meta["decode"]["repetition_penalty"])},
    "voices": meta["voices"],
    "models": {"prefill": "models/gpt_prefill.onnx", "decode": "models/gpt_decode.onnx",
               "vocoder": "models/vocoder.onnx"},
}
(APP / "config.json").write_text(json.dumps(cfg, indent=2))

# --- parity fixtures: golden token ids + codes for the first prompts ---
z = np.load(GOLD / "golden.npz")
fix = []
for i, smp in enumerate(meta["samples"]):
    fix.append({"i": i, "text": smp["text"], "voice": smp["voice"], "lang": "hi",
                "ids": z[f"s{i}_ids"].astype(int).tolist(),
                "codes": z[f"s{i}_codes"].astype(int).tolist(),
                "n_codes": int(z[f"s{i}_codes"].shape[0])})
(ASSETS / "golden_fixtures.json").write_text(json.dumps(fix))
print(f"[fixtures] {len(fix)} prompts (ids+codes) -> assets/golden_fixtures.json")
print("[done] web assets packaged ->", APP)
