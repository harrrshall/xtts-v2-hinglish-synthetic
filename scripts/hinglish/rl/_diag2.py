#!/usr/bin/env python3
"""Pinpoint the 19x latent-magnitude shrink: where does the teacher's latent energy live, and did
   activation-importance keep those channels? Also compare student vs teacher latents channel-wise."""
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

cfg = XttsConfig(); cfg.load_json(BASE + "/config.json"); cfg.model_args.gpt_layers = 16
teacher = Xtts.init_from_config(cfg)
teacher.load_checkpoint(cfg, checkpoint_path=CKPT, vocab_path=BASE + "/vocab.json", use_deepspeed=False)
teacher.eval(); teacher.cuda()
gpt_cond, spk = teacher.get_conditioning_latents(audio_path=[VREF])
tt = torch.IntTensor(teacher.tokenizer.encode(TEXT, lang="hi")).unsqueeze(0).cuda()

ck = torch.load(ST, map_location="cpu"); keep = ck["keep_idx"].long()
with torch.no_grad():
    codes = teacher.gpt.generate(cond_latents=gpt_cond.cuda(), text_inputs=tt, do_sample=False, top_k=1,
                                 top_p=1.0, temperature=0.7, num_return_sequences=1, num_beams=1,
                                 length_penalty=1.0, repetition_penalty=1.0)
    exp = torch.tensor([codes.shape[-1] * teacher.gpt.code_stride_len], device="cuda")
    tlen = torch.tensor([tt.shape[-1]], device="cuda")
    t_lat = teacher.gpt(tt, tlen, codes, exp, cond_latents=gpt_cond.cuda(), return_latent=True)[0]  # (T,1024)

# per-channel teacher latent energy
ch_energy = t_lat.float().abs().mean(0).cpu()                    # (1024,)
order = torch.argsort(ch_energy, descending=True)
keptset = set(keep.tolist())
top50 = order[:50].tolist()
in_kept = sum(1 for c in top50 if c in keptset)
print(f"teacher latent per-channel |.| : top-50 energy channels, {in_kept}/50 are in our kept-640")
print(f"  mean energy kept={ch_energy[keep].mean():.3f}  dropped={ch_energy[[c for c in range(1024) if c not in keptset]].mean():.3f}")

# final_norm weight magnitude on kept vs dropped
fnw = teacher.gpt.final_norm.weight.detach().abs().cpu()
dropped = torch.tensor([c for c in range(1024) if c not in keptset])
print(f"  final_norm |weight| kept={fnw[keep].mean():.3f}  dropped={fnw[dropped].mean():.3f}  "
      f"max_kept={fnw[keep].max():.2f} max_dropped={fnw[dropped].max():.2f}")

# student latents teacher-forced, channel-aligned compare
gpt = S.build_student_gpt(ck["model_args"], d_student=ck["d_student"], heads=ck["heads"])
gpt.load_state_dict(ck["gpt"]); student = S.Student640(gpt, d_student=ck["d_student"]); student.cuda().eval()
cond640 = ck["voices"]["maya"]["cond_latents"].cuda()
with torch.no_grad():
    s_lat = student.gpt(tt, tlen, codes, exp, cond_latents=cond640, return_latent=True)[0]   # (T,640)
T = min(s_lat.shape[0], t_lat.shape[0])
s = s_lat[:T].float().cpu(); t_on_kept = t_lat[:T][:, keep].float().cpu()
# correlation per kept channel
sc = s - s.mean(0); tc = t_on_kept - t_on_kept.mean(0)
corr = (sc * tc).sum(0) / (sc.norm(dim=0) * tc.norm(dim=0) + 1e-8)
print(f"student vs teacher(kept) latents: std_student={s.std():.3f} std_teacher_kept={t_on_kept.std():.3f} "
      f"mean_corr={corr.mean():+.3f} frac_corr>0.5={float((corr>0.5).float().mean()):.2f}")
print(f"  ratio std teacher_kept/student = {t_on_kept.std()/s.std():.1f}")
