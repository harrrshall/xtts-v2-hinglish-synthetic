#!/usr/bin/env python3
"""Probe how GPT2Model handles KV cache in this transformers version, before the real export.
Checks: (1) prefill present type, (2) legacy round-trip, (3) manual 2-step == single full forward."""
import os, sys, json
os.environ.setdefault("COQUI_TOS_AGREED", "1")
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[2]; MODELS = REPO.parent / "syntts_models"
DEMO = MODELS / "local_demo"; CKPT = MODELS / "sub100M_d640_final" / "student640b_rft.pt"
sys.path.insert(0, str(DEMO)); sys.path.insert(0, str(REPO / "webgpu" / "export"))
import student640 as S
from export_common import prep_gpt2_for_export

ck = torch.load(str(CKPT), map_location="cpu")
gpt = S.build_student_gpt(ck["model_args"], d_student=ck["d_student"], heads=ck["heads"])
gpt.load_state_dict(ck["gpt"]); gpt.eval()
g2 = prep_gpt2_for_export(gpt.gpt)
import transformers; print("transformers", transformers.__version__)
from transformers.cache_utils import DynamicCache

d = ck["d_student"]
torch.manual_seed(0)
emb_full = torch.randn(1, 6, d)   # pretend 6-token prefix

with torch.no_grad():
    # --- single full forward (reference) ---
    out_full = g2(inputs_embeds=emb_full, use_cache=True, return_dict=True)
    h_full = out_full.last_hidden_state
    pkv = out_full.past_key_values
    print("present type:", type(pkv).__name__)
    legacy = pkv.to_legacy_cache() if hasattr(pkv, "to_legacy_cache") else pkv
    print("n_layers in legacy:", len(legacy), "layer0 k shape:", tuple(legacy[0][0].shape))

    # --- manual: prefix of 5, then 1 step with cache ---
    out1 = g2(inputs_embeds=emb_full[:, :5], use_cache=True, return_dict=True)
    past = out1.past_key_values
    out2 = g2(inputs_embeds=emb_full[:, 5:6], past_key_values=past, use_cache=True, return_dict=True)
    h_step = out2.last_hidden_state   # (1,1,d) hidden for the 6th token
    diff = (h_step[:, -1] - h_full[:, -1]).abs().max().item()
    print(f"[cache] hidden(step) vs hidden(full) last token max|Δ| = {diff:.2e}  {'OK' if diff<1e-4 else 'BAD'}")

    # --- can we rebuild a cache from flat tensors and feed it? (export needs this) ---
    flat = []
    for k, v in legacy:
        flat.append(k[:, :, :5]); flat.append(v[:, :, :5])
    rebuilt = DynamicCache.from_legacy_cache(tuple((flat[2*i], flat[2*i+1]) for i in range(len(legacy))))
    out3 = g2(inputs_embeds=emb_full[:, 5:6], past_key_values=rebuilt, use_cache=True, return_dict=True)
    diff2 = (out3.last_hidden_state[:, -1] - h_full[:, -1]).abs().max().item()
    print(f"[rebuilt-from-flat] max|Δ| = {diff2:.2e}  {'OK' if diff2<1e-4 else 'BAD'}")
    print("DynamicCache has from_legacy_cache:", hasattr(DynamicCache, "from_legacy_cache"))
