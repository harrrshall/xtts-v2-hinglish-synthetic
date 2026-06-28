#!/usr/bin/env python3
"""Golden reference harness — the numerical oracle for the WebGPU port.

Loads the 89.96M student on CPU (exactly like local_demo/app.py), runs a fixed set of
(voice, text) prompts with greedy decode, and dumps every intermediate so each ONNX/JS
stage can be checked against it:

  token ids -> audio codes -> 640-d mel latents -> 1024-d adapter out -> 24 kHz wav

Outputs (webgpu/golden/):
  golden.npz            per-sample arrays (ids, codes, lat640, lat1024, wav) + meta
  meta.json             prompts, voices list, decode kwargs, model constants
  voices.npz            the 4 baked voices (cond_latents, speaker_embedding)
  sample_<i>_<voice>.wav  audio to listen to

Run:  <local_demo venv>/bin/python webgpu/export/00_golden.py
"""
import os, sys, json, time
os.environ.setdefault("COQUI_TOS_AGREED", "1")
from pathlib import Path
import numpy as np, soundfile as sf, torch

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO.parent / "syntts_models"
DEMO = MODELS / "local_demo"
CKPT = MODELS / "sub100M_d640_final" / "student640b_rft.pt"
OUT = REPO / "webgpu" / "golden"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DEMO))            # student640.py lives here
import student640 as S
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager

DEV = "cpu"
torch.manual_seed(0)

# --- fixed golden prompts: Devanagari, mixed code-switch, romanized ---
PROMPTS = [
    "आज मौसम बहुत अच्छा है और मैं घूमने जाना चाहता हूँ।",
    "मुझे यह project बहुत interesting लग रहा है, can you believe it?",
    "यार कल का match देखा? last over में जो हुआ वो totally insane था!",
    "main aaj bahut khush hoon kyunki mera result aa gaya.",
]
# voices: first two prompts on maya+arjun (main parity), cover all 4 voices at least once
SAMPLES = [
    (0, "maya"), (1, "maya"), (2, "arjun"), (3, "arjun"),
    (0, "aadya"), (0, "kaustubh"),
]

print(f"[load] base XTTS (tokenizer + frozen HiFi-GAN) ...", flush=True)
base = Path(ModelManager().download_model("tts_models/multilingual/multi-dataset/xtts_v2")[0])
cfg = XttsConfig(); cfg.load_json(str(base / "config.json"))
VOC = Xtts.init_from_config(cfg)
VOC.load_checkpoint(cfg, checkpoint_dir=str(base), use_deepspeed=False, eval=True); VOC.eval()

print(f"[load] 90M student ...", flush=True)
ck = torch.load(str(CKPT), map_location="cpu")
gpt = S.build_student_gpt(ck["model_args"], d_student=ck["d_student"], heads=ck["heads"])
gpt.load_state_dict(ck["gpt"])
student = S.Student640(gpt, d_student=ck["d_student"])
student.adapter.load_state_dict(ck["adapter"]); student.eval()
student.gpt.init_gpt_for_inference(kv_cache=True)
VOICE_DATA = ck["voices"]
ma = ck["model_args"]
print(f"[load] READY  params={ck.get('n_params',0)/1e6:.2f}M  voices={list(VOICE_DATA)}", flush=True)

DECODE = dict(do_sample=False, top_k=1, top_p=1.0, temperature=0.7,
              repetition_penalty=1.3, num_return_sequences=1, num_beams=1, length_penalty=1.0)


@torch.no_grad()
def run(text, voice):
    cond = VOICE_DATA[voice]["cond_latents"].float()       # (1,32,640)
    spk  = VOICE_DATA[voice]["speaker_embedding"].float()  # (1,512,1)
    ids  = torch.IntTensor(VOC.tokenizer.encode(text, lang="hi")).unsqueeze(0)
    codes = student.gpt.generate(cond_latents=cond, text_inputs=ids, **DECODE)
    exp  = torch.tensor([codes.shape[-1] * student.gpt.code_stride_len])
    lat640 = student.gpt(ids, torch.tensor([ids.shape[-1]]), codes, exp,
                         cond_latents=cond, return_latent=True)     # (1,T,640)
    lat1024 = student.vocoder_latents(lat640)                       # (1,T,1024)
    wav = VOC.hifigan_decoder(lat1024, g=spk).squeeze().numpy().astype(np.float32)
    return dict(ids=ids.squeeze(0).numpy().astype(np.int64),
                codes=codes.squeeze(0).numpy().astype(np.int64),
                lat640=lat640.squeeze(0).numpy().astype(np.float32),
                lat1024=lat1024.squeeze(0).numpy().astype(np.float32),
                wav=wav)


store = {}
meta_samples = []
for i, (pi, voice) in enumerate(SAMPLES):
    text = PROMPTS[pi]
    t0 = time.perf_counter()
    r = run(text, voice)
    dt = time.perf_counter() - t0
    dur = len(r["wav"]) / 24000
    print(f"[gen {i}] {voice:9s} codes={r['codes'].shape[0]:4d} lat={r['lat640'].shape} "
          f"wav={dur:.2f}s in {dt:.1f}s  | {text[:42]}", flush=True)
    for k, v in r.items():
        store[f"s{i}_{k}"] = v
    sf.write(OUT / f"sample_{i}_{voice}.wav", r["wav"], 24000)
    meta_samples.append(dict(i=i, prompt_idx=pi, voice=voice, text=text,
                             n_ids=int(r["ids"].shape[0]), n_codes=int(r["codes"].shape[0]),
                             gen_s=round(dt, 2), dur_s=round(dur, 2)))

np.savez(OUT / "golden.npz", **store)

# voices as data for the browser
vstore = {}
for v, d in VOICE_DATA.items():
    vstore[f"{v}_cond"] = d["cond_latents"].float().squeeze(0).numpy().astype(np.float32)   # (32,640)
    vstore[f"{v}_spk"]  = d["speaker_embedding"].float().squeeze(0).numpy().astype(np.float32)  # (512,1)
np.savez(OUT / "voices.npz", **vstore)

meta = dict(
    samples=meta_samples,
    voices=list(VOICE_DATA),
    decode=DECODE,
    constants=dict(
        d_student=int(ck["d_student"]), d_vocoder=1024, heads=int(ck["heads"]),
        layers=int(ma["gpt_layers"]), head_dim=64, code_stride_len=int(gpt.code_stride_len),
        start_text_token=int(ma["gpt_start_text_token"]), stop_text_token=int(ma["gpt_stop_text_token"]),
        start_audio_token=int(ma["gpt_start_audio_token"]), stop_audio_token=int(ma["gpt_stop_audio_token"]),
        num_audio_tokens=int(ma["gpt_num_audio_tokens"]), number_text_tokens=int(ma["gpt_number_text_tokens"]),
        max_gen_mel_tokens=int(gpt.max_gen_mel_tokens), sample_rate=24000,
    ),
)
(OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print(f"\n[done] golden -> {OUT}")
print(json.dumps(meta["constants"], indent=2))
