#!/usr/bin/env python3
"""Diagnose the silent smoke output: is it the adapter's channel-zeroing or degenerate student latents?"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import student640 as S
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

BASE = ".tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2"
CKPT = "runs/rl/round1_rft/xtts_hinglish-June-25-2026_06+39PM-0000000/best_model.pth"
ST = "runs/rl/sub100m/student640_init.pt"
TEXT = "मुझे यह project interesting लग रहा है, can you believe it?"
VREF = "runs/xtts_hinglish/RELEASE/refs/maya.wav"


def stats(name, t):
    t = t.detach().float().cpu()
    print(f"  {name:28s} shape={tuple(t.shape)} mean={t.mean():+.4f} std={t.std():.4f} "
          f"min={t.min():+.3f} max={t.max():+.3f} nan={torch.isnan(t).any().item()}")


cfg = XttsConfig(); cfg.load_json(BASE + "/config.json"); cfg.model_args.gpt_layers = 16
teacher = Xtts.init_from_config(cfg)
teacher.load_checkpoint(cfg, checkpoint_path=CKPT, vocab_path=BASE + "/vocab.json", use_deepspeed=False)
teacher.eval(); teacher.cuda()
gpt_cond, spk = teacher.get_conditioning_latents(audio_path=[VREF])

print("=== TEACHER real path ===")
with torch.no_grad():
    o = teacher.inference(TEXT, "hi", gpt_cond, spk, temperature=0.7, enable_text_splitting=False)
w = np.asarray(o["wav"], np.float32)
print(f"  teacher wav dur={len(w)/24000:.2f}s rms={np.sqrt((w**2).mean()):.4f} peak={np.abs(w).max():.3f}")
tl = torch.tensor(o["gpt_latents"]).cuda()           # (1,T,1024) teacher latents
stats("teacher latents(1024)", tl)
with torch.no_grad():
    w2 = teacher.hifigan_decoder(tl, g=spk.cuda()).cpu().squeeze().numpy()
print(f"  re-vocoded teacher latents rms={np.sqrt((w2**2).mean()):.4f}")

# zero the dropped channels of the TEACHER latents -> does the frozen vocoder survive?
ck = torch.load(ST, map_location="cpu"); keep = ck["keep_idx"].long().cuda()
tl_kept = torch.zeros_like(tl); tl_kept[..., keep] = tl[..., keep]
with torch.no_grad():
    w3 = teacher.hifigan_decoder(tl_kept, g=spk.cuda()).cpu().squeeze().numpy()
print(f"  teacher latents w/ 384 dropped ch ZEROED rms={np.sqrt((w3**2).mean()):.4f} peak={np.abs(w3).max():.3f}")

print("=== STUDENT latents (teacher-forced on teacher codes) ===")
# build student, load init
gpt = S.build_student_gpt(ck["model_args"], d_student=ck["d_student"], heads=ck["heads"])
gpt.load_state_dict(ck["gpt"]); student = S.Student640(gpt, d_student=ck["d_student"])
student.adapter.load_state_dict(ck["adapter"]); student.cuda().eval()
cond640 = ck["voices"]["maya"]["cond_latents"].cuda()
# teacher-force the student on the TEACHER's codes to remove generation degeneracy from the test
tt = torch.IntTensor(teacher.tokenizer.encode(TEXT, lang="hi")).unsqueeze(0).cuda()
codes = torch.tensor(o["gpt_latents"]).new_zeros(0)  # placeholder
import torch.nn.functional as F
with torch.no_grad():
    # regenerate codes from teacher to feed student (same codes the teacher used aren't returned; re-derive)
    gpt_codes = teacher.gpt.generate(cond_latents=gpt_cond.cuda(), text_inputs=tt, do_sample=False,
                                     top_k=1, top_p=1.0, temperature=0.7, num_return_sequences=1,
                                     num_beams=1, length_penalty=1.0, repetition_penalty=1.0)
    exp = torch.tensor([gpt_codes.shape[-1] * student.gpt.code_stride_len], device="cuda")
    tlen = torch.tensor([tt.shape[-1]], device="cuda")
    sl = student.gpt(tt, tlen, gpt_codes, exp, cond_latents=cond640, return_latent=True)
    stats("student latents(640)", sl)
    a1024 = student.vocoder_latents(sl)
    stats("adapter out(1024)", a1024)
    stats("teacher latents(1024) ref", tl)
    ws = teacher.hifigan_decoder(a1024, g=spk.cuda()).cpu().squeeze().numpy()
print(f"  student->adapter->vocoder rms={np.sqrt((ws**2).mean()):.4f} peak={np.abs(ws).max():.3f}")
print(f"  teacher code count={gpt_codes.shape[-1]}")
