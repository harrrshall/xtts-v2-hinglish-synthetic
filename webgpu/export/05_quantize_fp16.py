#!/usr/bin/env python3
"""Convert the 3 ONNX graphs to fp16 (half the bytes, WebGPU-native) and verify no quality loss (C7).

Mixed precision: keep_io_types=True (graph I/O + KV cache stay fp32 for stable feedback), internal
compute fp16. Then run the FULL pipeline on the fp16 graphs and require codes still bit-exact vs
golden (the strict no-quality-loss bar). NOTE: ORT-CPU upcasts fp16 internally, so this is a floor;
the true fp16-WebGPU verdict is the in-browser check (C9). fp32 graphs remain the lossless fallback.
"""
import os, sys, json, time
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort
from onnxconverter_common import float16
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "webgpu" / "models"; GOLD = REPO / "webgpu" / "golden"
F16 = OUT / "fp16"; F16.mkdir(exist_ok=True)
meta = json.loads((GOLD / "meta.json").read_text()); C = meta["constants"]
NL, SA, EA = C["layers"], C["start_audio_token"], C["stop_audio_token"]
PEN = float(meta["decode"]["repetition_penalty"]); MAXG = C["max_gen_mel_tokens"]
z = np.load(GOLD / "golden.npz"); vz = np.load(GOLD / "voices.npz")

for name in ("gpt_prefill", "gpt_decode", "vocoder"):
    m = onnx.load(str(OUT / f"{name}.onnx"))
    # keep_io_types=False -> all float I/O fp16; op_block_list=[] -> convert EVERY op (no fp32
    # islands, so no mixed-type Add/Cast clashes). Uniform fp16 matches native WebGPU fp16.
    m16 = float16.convert_float_to_float16(m, keep_io_types=False, op_block_list=[],
                                           node_block_list=[])
    del m16.graph.value_info[:]                 # drop stale intermediate type annotations; ORT re-infers
    onnx.save(m16, str(F16 / f"{name}.onnx"))
    sz0 = (OUT / f"{name}.onnx").stat().st_size / 1e6; sz1 = (F16 / f"{name}.onnx").stat().st_size / 1e6
    print(f"[fp16] {name}: {sz0:.0f} MB -> {sz1:.0f} MB")

F16T = np.float16

prov = ["CPUExecutionProvider"]
sp = ort.InferenceSession(str(F16 / "gpt_prefill.onnx"), providers=prov)
sd = ort.InferenceSession(str(F16 / "gpt_decode.onnx"), providers=prov)
sv = ort.InferenceSession(str(F16 / "vocoder.onnx"), providers=prov)


def rep_penalty(logits, seq, pen):
    idx = np.asarray(seq, np.int64); s = logits[idx]
    s = np.where(s < 0, s * pen, s / pen); o = logits.copy(); o[idx] = s; return o


def generate(text_ids, cond, spk):
    T = text_ids.shape[1]; seq = [1] * (32 + T + 2) + [SA]
    outs = sp.run(None, {"text_ids": text_ids, "cond": cond.astype(F16T)})
    logits = outs[0][0].astype(np.float32); lat = [outs[1][0]]; past = outs[2:]
    codes = []
    for _ in range(MAXG):
        nxt = int(rep_penalty(logits, seq, PEN).argmax())
        if nxt == EA: codes.append(nxt); break
        codes.append(nxt); seq.append(nxt)
        feed = {"input_id": np.array([[nxt]], np.int64), "pos": np.array([len(codes)], np.int64)}
        for j in range(NL):
            feed[f"past_key_values.{j}.key"] = past[2*j]; feed[f"past_key_values.{j}.value"] = past[2*j+1]
        outs = sd.run(None, feed); logits = outs[0][0].astype(np.float32); lat.append(outs[1][0]); past = outs[2:]
    lat640 = np.stack(lat, 0)[None].astype(F16T)
    wav = sv.run(["wav"], {"lat640": lat640, "g": spk.astype(F16T)})[0].squeeze().astype(np.float32)
    return np.array(codes, np.int64), wav


ok = True
for i, smp in enumerate(meta["samples"]):
    tids = z[f"s{i}_ids"][None].astype(np.int64)
    cond = vz[f"{smp['voice']}_cond"][None]; spk = vz[f"{smp['voice']}_spk"][None]
    codes, wav = generate(tids, cond, spk)
    gold = z[f"s{i}_wav"]; gc = z[f"s{i}_codes"]
    csame = codes.shape == gc.shape and bool((codes == gc).all())
    nmatch = int((codes[:min(len(codes), len(gc))] == gc[:min(len(codes), len(gc))]).sum())
    n = min(len(wav), len(gold)); a, b = wav[:n], gold[:n]
    corr = float(np.corrcoef(a, b)[0, 1]) if n > 1 else 0.0
    print(f"[C7 s{i}] {smp['voice']:9s} codes_exact={csame} ({nmatch}/{len(gc)}) corr={corr:.5f}")
    ok &= csame
print(f"\n[C7] {'PASS — fp16 codes bit-exact (no quality loss on CPU floor)' if ok else 'PARTIAL — see per-sample; rely on fp32 or in-browser C9'}")
