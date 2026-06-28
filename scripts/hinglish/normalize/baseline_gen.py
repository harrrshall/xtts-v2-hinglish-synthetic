#!/usr/bin/env python3
"""Baseline: generate audio for fully-romanized Hinglish vs the correct Devanagari-Hindi+Roman-English
form, to quantify the problem and give the text-normalizer a validation target.

Output wavs + codes/duration -> scripts/hinglish/normalize/out/
"""
import os, sys, json, time
os.environ.setdefault("COQUI_TOS_AGREED", "1")
from pathlib import Path
import numpy as np, soundfile as sf, torch
REPO = Path(__file__).resolve().parents[3]
MODELS = REPO.parent / "syntts_models"; DEMO = MODELS / "local_demo"
CKPT = MODELS / "sub100M_d640_final" / "student640b_rft.pt"
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(DEMO))
import student640 as S
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager

base = Path(ModelManager().download_model("tts_models/multilingual/multi-dataset/xtts_v2")[0])
cfg = XttsConfig(); cfg.load_json(str(base / "config.json"))
VOC = Xtts.init_from_config(cfg); VOC.load_checkpoint(cfg, checkpoint_dir=str(base), use_deepspeed=False, eval=True); VOC.eval()
ck = torch.load(str(CKPT), map_location="cpu")
gpt = S.build_student_gpt(ck["model_args"], d_student=ck["d_student"], heads=ck["heads"]); gpt.load_state_dict(ck["gpt"])
student = S.Student640(gpt, d_student=ck["d_student"]); student.adapter.load_state_dict(ck["adapter"]); student.eval()
student.gpt.init_gpt_for_inference(kv_cache=True)
VD = ck["voices"]
DEC = dict(do_sample=False, top_k=1, top_p=1.0, temperature=0.7, repetition_penalty=1.3,
           num_return_sequences=1, num_beams=1, length_penalty=1.0)


@torch.no_grad()
def gen(text, voice):
    cond = VD[voice]["cond_latents"].float(); spk = VD[voice]["speaker_embedding"].float()
    ids = torch.IntTensor(VOC.tokenizer.encode(text, lang="hi")).unsqueeze(0)
    codes = student.gpt.generate(cond_latents=cond, text_inputs=ids, **DEC)
    exp = torch.tensor([codes.shape[-1] * student.gpt.code_stride_len])
    lat = student.gpt(ids, torch.tensor([ids.shape[-1]]), codes, exp, cond_latents=cond, return_latent=True)
    wav = VOC.hifigan_decoder(student.vocoder_latents(lat), g=spk).squeeze().numpy()
    return codes.shape[-1], wav


PAIRS = [
    ("p1_romanized", "yaar kal ka match dekha? last over me jo hua wo totally insane tha", "arjun"),
    ("p1_correct",   "यार कल का match देखा? last over में जो हुआ वो totally insane था", "arjun"),
    ("p2_romanized", "mera naam Kaustubh hai aur main Delhi se hoon", "kaustubh"),
    ("p2_correct",   "मेरा नाम कौस्तुभ है और मैं Delhi से हूँ", "kaustubh"),
    ("p3_romanized", "main aaj office nahi ja raha, can you believe it?", "maya"),
    ("p3_correct",   "मैं आज office नहीं जा रहा, can you believe it?", "maya"),
]
for label, text, voice in PAIRS:
    t0 = time.perf_counter(); n, wav = gen(text, voice); dt = time.perf_counter() - t0
    sf.write(OUT / f"{label}.wav", wav.astype(np.float32), 24000)
    print(f"{label:14s} {voice:9s} codes={n:3d} dur={len(wav)/24000:.2f}s ({dt:.1f}s) | {text[:48]}")
print(f"\nwavs -> {OUT}  (listen: romanized vs correct for each pair)")
