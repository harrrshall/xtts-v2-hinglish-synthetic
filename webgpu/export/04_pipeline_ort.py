#!/usr/bin/env python3
"""Full end-to-end pipeline in onnxruntime-Python — the EXACT graph the browser runs (C6).
3 ONNX sessions, no separate latent graph (latents are collected from the AR loop):

  prefill(text_ids,cond) -> logits, latent[0], kv
  loop: sample code (greedy+rep_penalty); decode(code,pos,kv) -> logits, latent[k], kv
  lat640 = stack(latents)            # one per fed token (start_audio + each non-stop code)
  vocoder(lat640, g=spk) -> wav      # adapter folded in

Proves: codes bit-exact vs golden, wav perceptually identical (corr>0.9999, SNR>45dB).
Run: <venv>/bin/python webgpu/export/04_pipeline_ort.py
"""
import os, sys, json, time
from pathlib import Path
import numpy as np, soundfile as sf, onnxruntime as ort
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "webgpu" / "models"; GOLD = REPO / "webgpu" / "golden"
RES = GOLD / "ort_out"; RES.mkdir(exist_ok=True)
meta = json.loads((GOLD / "meta.json").read_text()); C = meta["constants"]
NL, SA, EA = C["layers"], C["start_audio_token"], C["stop_audio_token"]
PEN = float(meta["decode"]["repetition_penalty"]); MAXG = C["max_gen_mel_tokens"]
z = np.load(GOLD / "golden.npz"); vz = np.load(GOLD / "voices.npz")
prov = ["CPUExecutionProvider"]
sp = ort.InferenceSession(str(OUT / "gpt_prefill.onnx"), providers=prov)
sd = ort.InferenceSession(str(OUT / "gpt_decode.onnx"), providers=prov)
sv = ort.InferenceSession(str(OUT / "vocoder.onnx"), providers=prov)


def rep_penalty(logits, seq, pen):
    idx = np.asarray(seq, np.int64); s = logits[idx]
    s = np.where(s < 0, s * pen, s / pen); o = logits.copy(); o[idx] = s; return o


def generate(text_ids, cond, spk):
    T = text_ids.shape[1]; seq = [1] * (32 + T + 2) + [SA]
    outs = sp.run(None, {"text_ids": text_ids, "cond": cond})
    logits = outs[0][0]; lat = [outs[1][0]]; past = outs[2:]
    codes = []
    for _ in range(MAXG):
        nxt = int(rep_penalty(logits, seq, PEN).argmax())
        if nxt == EA: codes.append(nxt); break
        codes.append(nxt); seq.append(nxt)
        feed = {"input_id": np.array([[nxt]], np.int64), "pos": np.array([len(codes)], np.int64)}
        for j in range(NL):
            feed[f"past_key_values.{j}.key"] = past[2 * j]; feed[f"past_key_values.{j}.value"] = past[2 * j + 1]
        outs = sd.run(None, feed); logits = outs[0][0]; lat.append(outs[1][0]); past = outs[2:]
    lat640 = np.stack(lat, 0)[None].astype(np.float32)      # (1,N,640)
    wav = sv.run(["wav"], {"lat640": lat640, "g": spk})[0].squeeze()
    return np.array(codes, np.int64), wav


ok = True
for i, smp in enumerate(meta["samples"]):
    tids = z[f"s{i}_ids"][None].astype(np.int64)
    cond = vz[f"{smp['voice']}_cond"][None]; spk = vz[f"{smp['voice']}_spk"][None]
    t0 = time.perf_counter(); codes, wav = generate(tids, cond, spk); dt = time.perf_counter() - t0
    gold = z[f"s{i}_wav"]; n = min(len(wav), len(gold)); a, b = wav[:n], gold[:n]
    csame = codes.shape == z[f"s{i}_codes"].shape and bool((codes == z[f"s{i}_codes"]).all())
    corr = float(np.corrcoef(a, b)[0, 1]); snr = float(10 * np.log10((b**2).sum() / (((a-b)**2).sum()+1e-12)))
    sf.write(RES / f"ort_{i}_{smp['voice']}.wav", wav.astype(np.float32), 24000)
    good = csame and corr > 0.9999 and snr > 45
    print(f"[C6 s{i}] {smp['voice']:9s} codes_exact={csame} corr={corr:.6f} SNR={snr:.1f}dB "
          f"lat={codes.shape[0]} {dt:.1f}s {'OK' if good else 'CHECK'}")
    ok &= good
print(f"\n[C6] {'PASS — codes bit-exact + wav corr>0.9999 (no latent graph, adapter folded)' if ok else 'FAIL'}")
print(f"     wavs -> {RES}")
sys.exit(0 if ok else 1)
