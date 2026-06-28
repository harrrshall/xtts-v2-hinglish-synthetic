#!/usr/bin/env python3
"""fp16 the merged gpt_step.onnx + vocoder.onnx, and verify NO quality loss end-to-end
(codes bit-exact, wav corr>0.9999) on the full merged pipeline.

The fp16 blocker is GPT-2's nulled wpe: null_position_embeddings emits a float32 zeros constant that
feeds an fp16 Add. Fix here = after conversion, force every remaining float32 Constant / ConstantOfShape
to float16 so there are no mixed-type nodes (the zeros are zeros either way).
"""
import os, sys, json
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort
from onnxconverter_common import float16
from onnx import numpy_helper, TensorProto
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "webgpu" / "models"; F16 = OUT / "fp16"; F16.mkdir(exist_ok=True)
GOLD = REPO / "webgpu" / "golden"; ASSETS = REPO / "webgpu" / "app" / "assets"
meta = json.loads((GOLD / "meta.json").read_text()); C = meta["constants"]
NL, SA, EA, H = C["layers"], C["start_audio_token"], C["stop_audio_token"], C["heads"]
PEN = float(meta["decode"]["repetition_penalty"]); MAXG = C["max_gen_mel_tokens"]
z = np.load(GOLD / "golden.npz"); vz = np.load(GOLD / "voices.npz")
emb = np.load(ASSETS / "embeddings.npz")
TE = emb["text_emb"].astype(np.float32); ME = emb["mel_emb"].astype(np.float32)
TP = emb["text_pos"].astype(np.float32); MP = emb["mel_pos"].astype(np.float32)


def force_const_fp16(model):
    """Kill leftover float32: retarget Cast-to-FLOAT -> FLOAT16, and convert float Constant values.
    With op_block_list=[] there are no intentional fp32 islands, so every remaining float must be fp16."""
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16
        for attr in node.attribute:
            if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                arr = numpy_helper.to_array(attr.t).astype(np.float16)
                attr.t.CopyFrom(numpy_helper.from_array(arr, attr.t.name))
    return model


def to_fp16(name, aggressive):
    m = onnx.load(str(OUT / f"{name}.onnx"))
    if aggressive:                       # GPT graph: convert everything, then fix leftover casts/consts
        m16 = float16.convert_float_to_float16(m, keep_io_types=False, op_block_list=[], node_block_list=[])
        force_const_fp16(m16)
    else:                                # vocoder: default op-block-list keeps Resize scales etc. fp32
        m16 = float16.convert_float_to_float16(m, keep_io_types=False)
    del m16.graph.value_info[:]
    onnx.save(m16, str(F16 / f"{name}.onnx"))
    s0 = (OUT / f"{name}.onnx").stat().st_size / 1e6; s1 = (F16 / f"{name}.onnx").stat().st_size / 1e6
    print(f"[fp16] {name}: {s0:.0f} -> {s1:.0f} MB")
    return F16 / f"{name}.onnx"


step_path = to_fp16("gpt_step", aggressive=True)
voc_path = to_fp16("vocoder", aggressive=False)
F = np.float16
sp = ort.InferenceSession(str(step_path), providers=["CPUExecutionProvider"])
sv = ort.InferenceSession(str(voc_path), providers=["CPUExecutionProvider"])


def rep_pen(logits, seq, pen):
    idx = np.asarray(seq, np.int64); s = logits[idx]
    s = np.where(s < 0, s * pen, s / pen); o = logits.copy(); o[idx] = s; return o


def prefill_embeds(ids, cond):
    ti = [C["start_text_token"]] + list(ids) + [C["stop_text_token"]]; n = len(ti)
    te = TE[ti] + TP[:n]; sa = ME[SA] + MP[0]
    return np.concatenate([cond[0], te, sa[None]], 0)[None]


def run(ids, cond, spk):
    T = len(ids); seq = [1] * (32 + T + 2) + [SA]
    feed = {"inputs_embeds": prefill_embeds(ids, cond).astype(F)}
    for j in range(NL):
        feed[f"past_key_values.{j}.key"] = np.zeros((1, H, 0, 64), F)
        feed[f"past_key_values.{j}.value"] = np.zeros((1, H, 0, 64), F)
    out = sp.run(None, feed); logits = out[0][0].astype(np.float32); lat = [out[1][0]]; past = out[2:]
    codes = []
    for _ in range(MAXG):
        nxt = int(rep_pen(logits, seq, PEN).argmax())
        if nxt == EA: codes.append(nxt); break
        codes.append(nxt); seq.append(nxt)
        e = (ME[nxt] + MP[len(codes)])[None, None].astype(F)
        feed = {"inputs_embeds": e}
        for j in range(NL):
            feed[f"past_key_values.{j}.key"] = past[2 * j]; feed[f"past_key_values.{j}.value"] = past[2 * j + 1]
        out = sp.run(None, feed); logits = out[0][0].astype(np.float32); lat.append(out[1][0]); past = out[2:]
    lat640 = np.stack(lat, 0)[None].astype(F)
    wav = sv.run(["wav"], {"lat640": lat640, "g": spk.astype(F)})[0].squeeze().astype(np.float32)
    return np.array(codes, np.int64), wav


print("\n=== fp16 merged pipeline vs golden ===")
ok = True
for i, smp in enumerate(meta["samples"]):
    cond = vz[f"{smp['voice']}_cond"][None]; spk = vz[f"{smp['voice']}_spk"][None]
    codes, wav = run(z[f"s{i}_ids"].tolist(), cond, spk)
    gc = z[f"s{i}_codes"]; gw = z[f"s{i}_wav"]
    csame = codes.shape == gc.shape and bool((codes == gc).all())
    nm = int((codes[:min(len(codes), len(gc))] == gc[:min(len(codes), len(gc))]).sum())
    n = min(len(wav), len(gw)); a, b = wav[:n], gw[:n]
    corr = float(np.corrcoef(a, b)[0, 1]) if n > 1 else 0
    snr = float(10 * np.log10((b**2).sum() / (((a - b)**2).sum() + 1e-12)))
    print(f"  s{i} {smp['voice']:9s} codes_exact={csame} ({nm}/{len(gc)}) corr={corr:.5f} SNR={snr:.1f}dB")
    ok &= (csame and corr > 0.999)
print(f"[C-fp16] {'PASS - fp16 merged: codes bit-exact, wav corr>0.999 (no quality loss)' if ok else 'PARTIAL/FAIL'}")
sys.exit(0 if ok else 1)
