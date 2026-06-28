#!/usr/bin/env python3
"""Export the AR GPT as prefill + single-step decode (KV-cache) and prove the ORT greedy loop
reproduces the golden audio codes EXACTLY (C4). Cross-checks vs golden = the HF-generate reference.

Graphs:
  gpt_prefill.onnx :  text_ids(1,T) + cond(1,32,640)
                      -> logits(1,1026) + present.{i}.key|value (1,10,P+1,64)   P=32+T+2
  gpt_decode.onnx  :  input_id(1,1) + pos(1) + past.{i}.key|value
                      -> logits(1,1026) + present.{i}.key|value (past+1)

Decode contract (must match HF generate, see WEBGPU_PLAN.md):
  greedy argmax, repetition_penalty=1.3 applied over the FULL input_ids =
  [1]*(32+T+2) + [start_audio] + generated_codes  (the prefix sentinel 1s and start_audio count).
  mel position of the k-th generated code = k (start_audio is pos 0).
"""
import os, sys, json
os.environ.setdefault("COQUI_TOS_AGREED", "1")
from pathlib import Path
import numpy as np, torch, torch.nn as nn
REPO = Path(__file__).resolve().parents[2]; MODELS = REPO.parent / "syntts_models"
DEMO = MODELS / "local_demo"; CKPT = MODELS / "sub100M_d640_final" / "student640b_rft.pt"
OUT = REPO / "webgpu" / "models"; GOLD = REPO / "webgpu" / "golden"
sys.path.insert(0, str(DEMO)); sys.path.insert(0, str(REPO / "webgpu" / "export"))
import student640 as S
from export_common import prep_gpt2_for_export
from transformers.cache_utils import DynamicCache

ck = torch.load(str(CKPT), map_location="cpu")
gpt = S.build_student_gpt(ck["model_args"], d_student=ck["d_student"], heads=ck["heads"])
gpt.load_state_dict(ck["gpt"]); gpt.eval()
prep_gpt2_for_export(gpt.gpt)
C = json.loads((GOLD / "meta.json").read_text())["constants"]
NL = C["layers"]; SA, EA = C["start_audio_token"], C["stop_audio_token"]
ST, ET = C["start_text_token"], C["stop_text_token"]


class PrefillNet(nn.Module):
    """Emits logits (to sample) AND latent640 (= the vocoder latent at the start_audio position)."""
    def __init__(self, gpt): super().__init__(); self.g = gpt
    def forward(self, text_ids, cond):
        b, dev = text_ids.shape[0], text_ids.device
        st = torch.full((b, 1), ST, dtype=torch.long, device=dev)
        et = torch.full((b, 1), ET, dtype=torch.long, device=dev)
        sa = torch.full((b, 1), SA, dtype=torch.long, device=dev)
        text_inp = torch.cat([st, text_ids.long(), et], dim=1)
        text_emb = self.g.text_embedding(text_inp) + self.g.text_pos_embedding(text_inp)
        sa_emb = self.g.mel_embedding(sa) + self.g.mel_pos_embedding(sa)   # mel pos 0
        emb = torch.cat([cond, text_emb, sa_emb], dim=1)
        out = self.g.gpt(inputs_embeds=emb, use_cache=True, return_dict=True)
        latent = self.g.final_norm(out.last_hidden_state[:, -1])           # (1,640) == lat640[0]
        logits = self.g.mel_head(latent)                                   # (1,1026)
        present = out.past_key_values.to_legacy_cache()
        flat = []
        for k, v in present: flat += [k, v]
        return (logits, latent, *flat)


class DecodeNet(nn.Module):
    """Emits logits AND latent640 (= the vocoder latent at the fed-code position)."""
    def __init__(self, gpt): super().__init__(); self.g = gpt
    def forward(self, input_id, pos, past):
        mel_emb = self.g.mel_embedding(input_id.long())                   # (1,1,d)
        pos_emb = self.g.mel_pos_embedding.emb(pos.long()).unsqueeze(0)    # (1,1,d)
        emb = mel_emb + pos_emb
        legacy = tuple((past[2 * i], past[2 * i + 1]) for i in range(NL))
        cache = DynamicCache.from_legacy_cache(legacy)
        out = self.g.gpt(inputs_embeds=emb, past_key_values=cache, use_cache=True, return_dict=True)
        latent = self.g.final_norm(out.last_hidden_state[:, -1])
        logits = self.g.mel_head(latent)
        present = out.past_key_values.to_legacy_cache()
        flat = []
        for k, v in present: flat += [k, v]
        return (logits, latent, *flat)


# ---- sample inputs ----
z = np.load(GOLD / "golden.npz"); meta = json.loads((GOLD / "meta.json").read_text())
vz = np.load(GOLD / "voices.npz")
ids0 = torch.from_numpy(z["s0_ids"]).unsqueeze(0)
cond0 = torch.from_numpy(vz[f"{meta['samples'][0]['voice']}_cond"]).unsqueeze(0)

SKIP = os.environ.get("SKIP_EXPORT") == "1" and (OUT / "gpt_prefill.onnx").exists()
pre = PrefillNet(gpt).eval(); dec = DecodeNet(gpt).eval()
with torch.no_grad():
    pre_out = pre(ids0, cond0)
logits0, latent0, past0 = pre_out[0], pre_out[1], list(pre_out[2:])
print(f"[prefill] logits {tuple(logits0.shape)} latent {tuple(latent0.shape)} "
      f"past0 k0 {tuple(past0[0].shape)}  (P+1={past0[0].shape[2]})")

# ---- export names ----
past_names, present_names = [], []
for i in range(NL):
    past_names += [f"past_key_values.{i}.key", f"past_key_values.{i}.value"]
    present_names += [f"present.{i}.key", f"present.{i}.value"]
dyn_pre = {"text_ids": {1: "T"}}
for n in present_names: dyn_pre[n] = {2: "P1"}
dyn_dec = {}
for n in past_names: dyn_dec[n] = {2: "past"}
for n in present_names: dyn_dec[n] = {2: "past1"}

if not SKIP:
    torch.onnx.export(pre, (ids0, cond0), str(OUT / "gpt_prefill.onnx"),
                      input_names=["text_ids", "cond"], output_names=["logits", "latent", *present_names],
                      dynamic_axes=dyn_pre, opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"[export] gpt_prefill.onnx ({(OUT/'gpt_prefill.onnx').stat().st_size/1e6:.1f} MB)")
    input_id = torch.tensor([[int(z['s0_codes'][0])]], dtype=torch.long)
    pos = torch.tensor([1], dtype=torch.long)
    with torch.no_grad():
        _ = dec(input_id, pos, past0)
    torch.onnx.export(dec, (input_id, pos, tuple(past0)), str(OUT / "gpt_decode.onnx"),
                      input_names=["input_id", "pos", *past_names], output_names=["logits", "latent", *present_names],
                      dynamic_axes=dyn_dec, opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"[export] gpt_decode.onnx ({(OUT/'gpt_decode.onnx').stat().st_size/1e6:.1f} MB)")
else:
    print("[skip] reusing existing gpt_prefill/gpt_decode onnx")

# ================= C4: ORT greedy loop must reproduce golden codes =================
import onnxruntime as ort
sp = ort.InferenceSession(str(OUT / "gpt_prefill.onnx"), providers=["CPUExecutionProvider"])
sd = ort.InferenceSession(str(OUT / "gpt_decode.onnx"), providers=["CPUExecutionProvider"])
PEN = float(meta["decode"]["repetition_penalty"])
MAXG = C["max_gen_mel_tokens"]


def rep_penalty(logits, seq_ids, pen):
    idx = np.asarray(seq_ids, dtype=np.int64)
    s = logits[idx]
    s = np.where(s < 0, s * pen, s / pen)
    out = logits.copy(); out[idx] = s
    return out


def greedy(text_ids_np, cond_np):
    T = text_ids_np.shape[1]
    seq = [1] * (32 + T + 2) + [SA]          # HF input_ids exactly
    outs = sp.run(None, {"text_ids": text_ids_np, "cond": cond_np})
    logits = outs[0][0]; past = outs[2:]
    codes = []
    for step in range(MAXG):
        nxt = int(rep_penalty(logits, seq, PEN).argmax())
        if nxt == EA:
            codes.append(nxt)          # HF returns the trailing stop token in the sequence
            break
        codes.append(nxt); seq.append(nxt)
        feed = {"input_id": np.array([[nxt]], np.int64), "pos": np.array([len(codes)], np.int64)}
        for j in range(NL):
            feed[f"past_key_values.{j}.key"] = past[2 * j]
            feed[f"past_key_values.{j}.value"] = past[2 * j + 1]
        outs = sd.run(None, feed)
        logits = outs[0][0]; past = outs[2:]
    return np.array(codes, np.int64)


ok_all = True
for i, smp in enumerate(meta["samples"]):
    tids = z[f"s{i}_ids"][None].astype(np.int64)
    cond = vz[f"{smp['voice']}_cond"][None]
    got = greedy(tids, cond)
    gold = z[f"s{i}_codes"]
    same = got.shape == gold.shape and bool((got == gold).all())
    n_match = int((got[:min(len(got), len(gold))] == gold[:min(len(got), len(gold))]).sum())
    print(f"[C4 s{i}] {smp['voice']:9s} golden={len(gold)} got={len(got)} "
          f"exact={'YES' if same else f'NO ({n_match}/{len(gold)} match)'}")
    ok_all &= same

print(f"\n[C4] {'PASS — ORT greedy == golden codes for all samples' if ok_all else 'FAIL'}")
sys.exit(0 if ok_all else 1)
