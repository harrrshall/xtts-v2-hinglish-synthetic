#!/usr/bin/env python3
"""Export adapter(640->1024) + frozen HiFi-GAN as one graph (C1).  Input is the GPT latent640.

Target:  lat640 (1,T,640) + g (1,512,1)  ->  wav (1,1,T*1024)
Folds the student's Linear(640->1024) adapter in front of HifiDecoder.forward so the browser
feeds raw GPT latents straight to one vocoder session (no separate adapter/latent graph).
Weight-norm already fused by load_checkpoint(eval=True) — not removed again (T6).
"""
import os, sys, json
os.environ.setdefault("COQUI_TOS_AGREED", "1")
from pathlib import Path
import numpy as np, torch, torch.nn as nn
REPO = Path(__file__).resolve().parents[2]; MODELS = REPO.parent / "syntts_models"
DEMO = MODELS / "local_demo"; CKPT = MODELS / "sub100M_d640_final" / "student640b_rft.pt"
OUT = REPO / "webgpu" / "models"; OUT.mkdir(parents=True, exist_ok=True)
GOLD = REPO / "webgpu" / "golden"; ONNX_PATH = OUT / "vocoder.onnx"
sys.path.insert(0, str(DEMO))
import student640 as S
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager

base = Path(ModelManager().download_model("tts_models/multilingual/multi-dataset/xtts_v2")[0])
cfg = XttsConfig(); cfg.load_json(str(base / "config.json"))
VOC = Xtts.init_from_config(cfg)
VOC.load_checkpoint(cfg, checkpoint_dir=str(base), use_deepspeed=False, eval=True); VOC.eval()
ck = torch.load(str(CKPT), map_location="cpu")
adapter = nn.Linear(ck["d_student"], 1024); adapter.load_state_dict(ck["adapter"]); adapter.eval()


class AdapterVocoder(nn.Module):
    def __init__(self, adapter, hifi): super().__init__(); self.adapter = adapter; self.hifi = hifi
    def forward(self, lat640, g):
        return self.hifi(self.adapter(lat640), g=g)


net = AdapterVocoder(adapter, VOC.hifigan_decoder).eval()

z = np.load(GOLD / "golden.npz"); meta = json.loads((GOLD / "meta.json").read_text())
vz = np.load(GOLD / "voices.npz")
lat = torch.from_numpy(z["s0_lat640"]).unsqueeze(0)               # (1,T,640)
spk = torch.from_numpy(vz[f"{meta['samples'][0]['voice']}_spk"]).unsqueeze(0)
with torch.no_grad():
    wav_torch = net(lat, spk).squeeze().numpy()

torch.onnx.export(net, (lat, spk), str(ONNX_PATH),
                  input_names=["lat640", "g"], output_names=["wav"],
                  dynamic_axes={"lat640": {1: "T"}, "wav": {2: "S"}},
                  opset_version=17, do_constant_folding=True, dynamo=False)
print(f"[export] {ONNX_PATH}  ({ONNX_PATH.stat().st_size/1e6:.1f} MB)")

import onnx, onnxruntime as ort
onnx.checker.check_model(onnx.load(str(ONNX_PATH)))
sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
ok = True
for i in (0, 2, 5):
    lat_i = z[f"s{i}_lat640"][None]; spk_i = vz[f"{meta['samples'][i]['voice']}_spk"][None]
    w = sess.run(["wav"], {"lat640": lat_i, "g": spk_i})[0].squeeze()
    g = z[f"s{i}_wav"]; n = min(len(w), len(g))
    d = float(np.abs(w[:n] - g[:n]).max()); corr = float(np.corrcoef(w[:n], g[:n])[0, 1])
    print(f"[C1 s{i}] max|Δ|={d:.2e} corr={corr:.6f}")
    ok &= corr > 0.9999
print(f"[C1] {'PASS (adapter+vocoder, corr>0.9999)' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
