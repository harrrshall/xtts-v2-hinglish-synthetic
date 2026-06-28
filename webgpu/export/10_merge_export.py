#!/usr/bin/env python3
"""Merge prefill+decode into ONE GPT graph with embeddings lifted to host (dedup the duplicate
transformer weights). Single gpt_step.onnx: inputs_embeds + past_kv -> logits, latent, present_kv.
Prefill = empty past (0-length); decode = growing past. Embedding tables exported as data for JS.

Stage 1: torch probe (greedy loop using the merged net + host-built embeds) == golden codes.
Stage 2: export gpt_step.onnx + embeddings, ORT greedy loop == golden codes EXACTLY.
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
gpt.load_state_dict(ck["gpt"]); gpt.eval(); prep_gpt2_for_export(gpt.gpt)
C = json.loads((GOLD / "meta.json").read_text())["constants"]
NL, SA, EA = C["layers"], C["start_audio_token"], C["stop_audio_token"]
ST, ET = C["start_text_token"], C["stop_text_token"]
PEN = float(json.loads((GOLD / "meta.json").read_text())["decode"]["repetition_penalty"]); MAXG = C["max_gen_mel_tokens"]
D = C["d_student"]

# ---- lifted embedding tables (these go to JS) ----
TEXT_EMB = gpt.text_embedding.weight.detach()          # (6681,640)
MEL_EMB = gpt.mel_embedding.weight.detach()             # (1026,640)
TEXT_POS = gpt.text_pos_embedding.emb.weight.detach()   # (maxT,640)
MEL_POS = gpt.mel_pos_embedding.emb.weight.detach()     # (maxM,640)


def prefill_embeds(text_ids, cond):
    """Build inputs_embeds exactly as the old prefill graph did, but with host-side gather+add."""
    ti = torch.cat([torch.tensor([ST]), torch.tensor(text_ids, dtype=torch.long), torch.tensor([ET])])
    n = ti.shape[0]
    te = TEXT_EMB[ti] + TEXT_POS[:n]
    sa = MEL_EMB[SA] + MEL_POS[0]
    emb = torch.cat([cond[0], te, sa[None]], dim=0)      # (32+n+1, 640)
    return emb[None]                                     # (1,L,640)


def decode_embed(token, pos):
    return (MEL_EMB[token] + MEL_POS[pos])[None, None]   # (1,1,640)


class StepNet(nn.Module):
    """One graph for prefill AND decode: inputs_embeds + flat past -> logits, latent, flat present."""
    def __init__(self, gpt): super().__init__(); self.g = gpt
    def forward(self, inputs_embeds, past):
        legacy = tuple((past[2 * i], past[2 * i + 1]) for i in range(NL))
        cache = DynamicCache.from_legacy_cache(legacy)
        out = self.g.gpt(inputs_embeds=inputs_embeds, past_key_values=cache, use_cache=True, return_dict=True)
        latent = self.g.final_norm(out.last_hidden_state[:, -1])
        logits = self.g.mel_head(latent)
        present = out.past_key_values.to_legacy_cache()
        flat = []
        for k, v in present: flat += [k, v]
        return (logits, latent, *flat)


net = StepNet(gpt).eval()
z = np.load(GOLD / "golden.npz"); meta = json.loads((GOLD / "meta.json").read_text())
vz = np.load(GOLD / "voices.npz")


def rep_penalty_np(logits, seq, pen):
    idx = np.asarray(seq, np.int64); s = logits[idx]
    s = np.where(s < 0, s * pen, s / pen); o = logits.copy(); o[idx] = s; return o


def empty_past():
    return [torch.zeros(1, NL_HEADS, 0, 64) for _ in range(2 * NL)]


NL_HEADS = C["heads"]


@torch.no_grad()
def greedy_torch(text_ids, cond):
    T = len(text_ids); seq = [1] * (32 + T + 2) + [SA]
    emb = prefill_embeds(text_ids, cond)
    out = net(emb, empty_past())
    logits = out[0][0].numpy(); past = list(out[2:])
    codes = []
    for _ in range(MAXG):
        nxt = int(rep_penalty_np(logits, seq, PEN).argmax())
        if nxt == EA: codes.append(nxt); break
        codes.append(nxt); seq.append(nxt)
        out = net(decode_embed(nxt, len(codes)), past)
        logits = out[0][0].numpy(); past = list(out[2:])
    return np.array(codes, np.int64)


print("=== Stage 1: torch merged-net greedy vs golden ===")
ok = True
for i, smp in enumerate(meta["samples"]):
    cond = torch.from_numpy(vz[f"{smp['voice']}_cond"]).unsqueeze(0)
    got = greedy_torch(z[f"s{i}_ids"].tolist(), cond)
    gold = z[f"s{i}_codes"]
    same = got.shape == gold.shape and bool((got == gold).all())
    print(f"  s{i} {smp['voice']:9s} golden={len(gold)} got={len(got)} {'EXACT' if same else 'DIFF'}")
    ok &= same
print(f"[stage1] {'PASS' if ok else 'FAIL'}")
assert ok, "stage1 failed"

# ================= Stage 2: export gpt_step.onnx + embeddings, ORT parity =================
ONNX_PATH = OUT / "gpt_step.onnx"
EMB_DIR = REPO / "webgpu" / "app" / "assets"; EMB_DIR.mkdir(parents=True, exist_ok=True)

past_names, present_names = [], []
for i in range(NL):
    past_names += [f"past_key_values.{i}.key", f"past_key_values.{i}.value"]
    present_names += [f"present.{i}.key", f"present.{i}.value"]
dyn = {"inputs_embeds": {1: "L"}}
for n in past_names: dyn[n] = {2: "past"}
for n in present_names: dyn[n] = {2: "past1"}

# trace with a representative decode step (non-empty past) so the past axis is a normal dynamic dim
emb0 = prefill_embeds(z["s0_ids"].tolist(), torch.from_numpy(vz["maya_cond"]).unsqueeze(0))
with torch.no_grad():
    o0 = net(emb0, empty_past()); past0 = list(o0[2:])
de = decode_embed(int(z["s0_codes"][0]), 1)
with torch.no_grad():
    _ = net(de, past0)
torch.onnx.export(net, (de, tuple(past0)), str(ONNX_PATH),
                  input_names=["inputs_embeds", *past_names], output_names=["logits", "latent", *present_names],
                  dynamic_axes=dyn, opset_version=17, do_constant_folding=True, dynamo=False)
print(f"[export] gpt_step.onnx ({ONNX_PATH.stat().st_size/1e6:.1f} MB)  (was prefill 342 + decode 324)")

# embedding tables for JS (fp16 to keep it tiny)
np.savez(EMB_DIR / "embeddings.npz",
         text_emb=TEXT_EMB.numpy().astype(np.float16), mel_emb=MEL_EMB.numpy().astype(np.float16),
         text_pos=TEXT_POS.numpy().astype(np.float16), mel_pos=MEL_POS.numpy().astype(np.float16))
print(f"[export] embeddings.npz ({(EMB_DIR/'embeddings.npz').stat().st_size/1e6:.1f} MB)")

import onnxruntime as ort
sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
TE, ME, TP, MP = TEXT_EMB.numpy(), MEL_EMB.numpy(), TEXT_POS.numpy(), MEL_POS.numpy()


def prefill_embeds_np(text_ids, cond):
    ti = [ST] + list(text_ids) + [ET]; n = len(ti)
    te = TE[ti] + TP[:n]; sa = ME[SA] + MP[0]
    return np.concatenate([cond[0], te, sa[None]], 0)[None].astype(np.float32)


def decode_embed_np(token, pos):
    return (ME[token] + MP[pos])[None, None].astype(np.float32)


EP = [np.zeros((1, NL_HEADS, 0, 64), np.float32) for _ in range(2 * NL)]


def greedy_ort(text_ids, cond):
    T = len(text_ids); seq = [1] * (32 + T + 2) + [SA]
    feed = {"inputs_embeds": prefill_embeds_np(text_ids, cond)}
    for j in range(NL):
        feed[f"past_key_values.{j}.key"] = EP[2 * j]; feed[f"past_key_values.{j}.value"] = EP[2 * j + 1]
    out = sess.run(None, feed); logits = out[0][0]; past = out[2:]
    codes = []
    for _ in range(MAXG):
        nxt = int(rep_penalty_np(logits, seq, PEN).argmax())
        if nxt == EA: codes.append(nxt); break
        codes.append(nxt); seq.append(nxt)
        feed = {"inputs_embeds": decode_embed_np(nxt, len(codes))}
        for j in range(NL):
            feed[f"past_key_values.{j}.key"] = past[2 * j]; feed[f"past_key_values.{j}.value"] = past[2 * j + 1]
        out = sess.run(None, feed); logits = out[0][0]; past = out[2:]
    return np.array(codes, np.int64)


print("\n=== Stage 2: ORT merged-graph greedy vs golden ===")
ok2 = True
for i, smp in enumerate(meta["samples"]):
    cond = vz[f"{smp['voice']}_cond"][None]
    got = greedy_ort(z[f"s{i}_ids"].tolist(), cond)
    gold = z[f"s{i}_codes"]
    same = got.shape == gold.shape and bool((got == gold).all())
    print(f"  s{i} {smp['voice']:9s} golden={len(gold)} got={len(got)} {'EXACT' if same else 'DIFF'}")
    ok2 &= same
print(f"[stage2] {'PASS - single graph reproduces golden codes exactly' if ok2 else 'FAIL'}")
sys.exit(0 if ok2 else 1)
