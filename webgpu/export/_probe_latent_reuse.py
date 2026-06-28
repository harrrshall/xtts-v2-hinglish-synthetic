#!/usr/bin/env python3
"""Verify: the final_norm hidden states produced DURING AR generation (prefill@start_audio +
each decode@fed-code) equal the return_latent lat640. If so, we drop the separate latent graph."""
import os, sys, json
os.environ.setdefault("COQUI_TOS_AGREED", "1")
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[2]; MODELS = REPO.parent / "syntts_models"
DEMO = MODELS / "local_demo"; CKPT = MODELS / "sub100M_d640_final" / "student640b_rft.pt"
GOLD = REPO / "webgpu" / "golden"
sys.path.insert(0, str(DEMO)); sys.path.insert(0, str(REPO / "webgpu" / "export"))
import student640 as S
from export_common import prep_gpt2_for_export
from transformers.cache_utils import DynamicCache

ck = torch.load(str(CKPT), map_location="cpu")
gpt = S.build_student_gpt(ck["model_args"], d_student=ck["d_student"], heads=ck["heads"])
gpt.load_state_dict(ck["gpt"]); gpt.eval(); prep_gpt2_for_export(gpt.gpt)
C = json.loads((GOLD / "meta.json").read_text())["constants"]
ST, ET, SA = C["start_text_token"], C["stop_text_token"], C["start_audio_token"]
z = np.load(GOLD / "golden.npz"); meta = json.loads((GOLD / "meta.json").read_text())
vz = np.load(GOLD / "voices.npz")

i = 0
ids = torch.from_numpy(z[f"s{i}_ids"]).unsqueeze(0)
codes = torch.from_numpy(z[f"s{i}_codes"]).unsqueeze(0)   # includes trailing 1025
cond = torch.from_numpy(vz[f"{meta['samples'][i]['voice']}_cond"]).unsqueeze(0)
gold_lat = z[f"s{i}_lat640"]                              # (N,640)
N = codes.shape[1]
g = gpt

with torch.no_grad():
    # prefill: [cond, text, start_audio]
    st = torch.full((1, 1), ST); et = torch.full((1, 1), ET); sa = torch.full((1, 1), SA)
    text_inp = torch.cat([st, ids.long(), et], 1)
    text_emb = g.text_embedding(text_inp) + g.text_pos_embedding(text_inp)
    sa_emb = g.mel_embedding(sa) + g.mel_pos_embedding(sa)
    emb = torch.cat([cond, text_emb, sa_emb], 1)
    out = g.gpt(inputs_embeds=emb, use_cache=True, return_dict=True)
    lat = [g.final_norm(out.last_hidden_state[:, -1:])[:, 0, 0:0]]  # placeholder
    lat = [g.final_norm(out.last_hidden_state[:, -1])]              # (1,640) latent@start_audio
    past = out.past_key_values
    # decode feeding c1..c_{N-1} (NOT the trailing stop)
    for k in range(1, N):                                          # feed codes[k-1]? careful
        pass
    # feed each code at index 0..N-2 (c1..c_{N-1}); codes[N-1] is the stop (1025) -> not fed
    lat = [g.final_norm(out.last_hidden_state[:, -1])]
    past = out.past_key_values
    for j in range(0, N - 1):                                      # codes[0..N-2]
        tok = codes[:, j:j + 1].long()
        pos = torch.tensor([j + 1])                                # mel pos = j+1
        e = g.mel_embedding(tok) + g.mel_pos_embedding.emb(pos).unsqueeze(0)
        o = g.gpt(inputs_embeds=e, past_key_values=past, use_cache=True, return_dict=True)
        lat.append(g.final_norm(o.last_hidden_state[:, -1]))
        past = o.past_key_values
    ar_lat = torch.cat(lat, 0).numpy()                             # (N,640)

print(f"gold_lat {gold_lat.shape}  ar_lat {ar_lat.shape}")
d = np.abs(ar_lat - gold_lat).max()
cos = float((ar_lat * gold_lat).sum() / (np.linalg.norm(ar_lat) * np.linalg.norm(gold_lat)))
print(f"max|AR_latent - return_latent| = {d:.2e}  cos={cos:.6f}  "
      f"{'EQUIVALENT — drop the latent graph' if d < 1e-3 else 'NOT equal'}")
