#!/usr/bin/env python3
"""Export the GPT latent-extraction forward (return_latent) + folded 640->1024 adapter (C2).

Mirrors gpt.forward(..., return_latent=True) for batch=1, no padding:
  text_inp  = [start_text, ids..., stop_text]                       (T_text+2)
  audio_inp = [start_audio, codes..., stop_audio x4]                (n_codes+5)
  emb = cat[cond_latents(32), txt_emb+txt_pos, mel_emb+mel_pos]
  enc = final_norm( gpt2(emb, causal)[:, 32:] )
  lat640 = enc[:, -(n_codes+5):][:, :n_codes]    (== the [:-5] truncation)
  lat1024 = adapter(lat640)

Graph:  text_ids (1,T) + codes (1,N) + cond (1,32,640) -> lat640 (1,N,640), lat1024 (1,N,1024)

Run: <venv>/bin/python webgpu/export/02_export_gpt_latent.py
"""
import os, sys, json
os.environ.setdefault("COQUI_TOS_AGREED", "1")
from pathlib import Path
import numpy as np, torch, torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO.parent / "syntts_models"
DEMO = MODELS / "local_demo"
CKPT = MODELS / "sub100M_d640_final" / "student640b_rft.pt"
OUT = REPO / "webgpu" / "models"; OUT.mkdir(parents=True, exist_ok=True)
GOLD = REPO / "webgpu" / "golden"
ONNX_PATH = OUT / "gpt_latent.onnx"
sys.path.insert(0, str(DEMO))
import student640 as S

ck = torch.load(str(CKPT), map_location="cpu")
gpt = S.build_student_gpt(ck["model_args"], d_student=ck["d_student"], heads=ck["heads"])
gpt.load_state_dict(ck["gpt"])
student = S.Student640(gpt, d_student=ck["d_student"]); student.adapter.load_state_dict(ck["adapter"])
student.eval()
C = json.loads((GOLD / "meta.json").read_text())["constants"]

from export_common import prep_gpt2_for_export
prep_gpt2_for_export(gpt.gpt)   # eager attn + neutralize vmap causal-mask path


class LatentNet(nn.Module):
    """Clean, export-friendly reproduction of the return_latent path (batch=1)."""
    def __init__(self, gpt, adapter, C):
        super().__init__()
        self.g = gpt; self.adapter = adapter
        self.st, self.et = C["start_text_token"], C["stop_text_token"]
        self.sa, self.ea = C["start_audio_token"], C["stop_audio_token"]

    def forward(self, text_ids, codes, cond):
        b = text_ids.shape[0]
        dev = text_ids.device
        st = torch.full((b, 1), self.st, dtype=torch.long, device=dev)
        et = torch.full((b, 1), self.et, dtype=torch.long, device=dev)
        sa = torch.full((b, 1), self.sa, dtype=torch.long, device=dev)
        ea = torch.full((b, 4), self.ea, dtype=torch.long, device=dev)
        text_inp = torch.cat([st, text_ids.long(), et], dim=1)          # (1, T+2)
        audio_inp = torch.cat([sa, codes.long(), ea], dim=1)            # (1, N+5)
        text_emb = self.g.text_embedding(text_inp) + self.g.text_pos_embedding(text_inp)
        mel_emb = self.g.mel_embedding(audio_inp) + self.g.mel_pos_embedding(audio_inp)
        emb = torch.cat([cond, text_emb, mel_emb], dim=1)               # (1, 32+T+2+N+5, d)
        out = self.g.gpt(inputs_embeds=emb, use_cache=False, return_dict=True).last_hidden_state
        enc = self.g.final_norm(out[:, cond.shape[1]:])                 # drop cond rows
        n = codes.shape[1]
        mel_lat = enc[:, -(n + 5):]                                     # (1, N+5, d)
        lat640 = mel_lat[:, :n]                                         # (1, N, d)  == [:-5]
        lat1024 = self.adapter(lat640)
        return lat640, lat1024


net = LatentNet(gpt, student.adapter, C).eval()

z = np.load(GOLD / "golden.npz")
meta = json.loads((GOLD / "meta.json").read_text())
vz = np.load(GOLD / "voices.npz")
# sample 0
ids = torch.from_numpy(z["s0_ids"]).unsqueeze(0)
codes = torch.from_numpy(z["s0_codes"]).unsqueeze(0)
cond = torch.from_numpy(vz[f"{meta['samples'][0]['voice']}_cond"]).unsqueeze(0)  # (1,32,640)

with torch.no_grad():
    lat640_t, lat1024_t = net(ids, codes, cond)
d640 = np.abs(lat640_t.numpy()[0] - z["s0_lat640"]).max()
print(f"[wrapper vs golden] max|lat640| = {d640:.2e}  (sanity that wrapper == real forward)")

torch.onnx.export(
    net, (ids, codes, cond), str(ONNX_PATH),
    input_names=["text_ids", "codes", "cond"], output_names=["lat640", "lat1024"],
    dynamic_axes={"text_ids": {1: "T"}, "codes": {1: "N"},
                  "lat640": {1: "N"}, "lat1024": {1: "N"}},
    opset_version=17, do_constant_folding=True, dynamo=False,
)
print(f"[export] {ONNX_PATH}  ({ONNX_PATH.stat().st_size/1e6:.1f} MB)")

import onnx, onnxruntime as ort
onnx.checker.check_model(onnx.load(str(ONNX_PATH)))
sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])


def check(i):
    ids = z[f"s{i}_ids"][None].astype(np.int64)
    codes = z[f"s{i}_codes"][None].astype(np.int64)
    cond = vz[f"{meta['samples'][i]['voice']}_cond"][None]
    l640, l1024 = sess.run(["lat640", "lat1024"], {"text_ids": ids, "codes": codes, "cond": cond})
    d6 = np.abs(l640[0] - z[f"s{i}_lat640"]).max()
    d10 = np.abs(l1024[0] - z[f"s{i}_lat1024"]).max()
    cos = float((l640[0] * z[f"s{i}_lat640"]).sum() /
                (np.linalg.norm(l640[0]) * np.linalg.norm(z[f"s{i}_lat640"])))
    print(f"[C2 s{i}] max|lat640|={d6:.2e}  max|lat1024|={d10:.2e}  cos={cos:.6f}")
    return d6 < 1e-3 and cos > 0.9999


ok = all(check(i) for i in range(len(meta["samples"])))
print(f"\n[C2] {'PASS' if ok else 'FAIL'}  (gate max|lat640|<1e-3, cos>0.9999, all samples)")
sys.exit(0 if ok else 1)
