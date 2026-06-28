#!/usr/bin/env python3
"""Identity test: run the slicing code with keep_idx=arange(1024), 16 heads, ffn=4096.
If slice_teacher_into_student is correct, the 'student' == teacher and latents must match (corr~1.0).
This separates a real slicing bug from the expected LayerNorm-coupling effect at d=640."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import student640 as S
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

BASE = ".tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2"
CKPT = "runs/rl/round1_rft/xtts_hinglish-June-25-2026_06+39PM-0000000/best_model.pth"
TEXT = "मुझे यह project interesting लग रहा है, can you believe it?"
VREF = "runs/xtts_hinglish/RELEASE/refs/maya.wav"

cfg = XttsConfig(); cfg.load_json(BASE + "/config.json"); cfg.model_args.gpt_layers = 16
teacher = Xtts.init_from_config(cfg)
teacher.load_checkpoint(cfg, checkpoint_path=CKPT, vocab_path=BASE + "/vocab.json", use_deepspeed=False)
teacher.eval(); teacher.cuda()
gpt_cond, spk = teacher.get_conditioning_latents(audio_path=[VREF])
tt = torch.IntTensor(teacher.tokenizer.encode(TEXT, lang="hi")).unsqueeze(0).cuda()

full_sd = torch.load(CKPT, map_location="cpu"); full_sd = full_sd.get("model", full_sd)
tsd = {k[len(S.GPT_PREFIX):]: v for k, v in full_sd.items() if k.startswith(S.GPT_PREFIX)}
ma = {k: getattr(cfg.model_args, k) for k in (
    "gpt_layers", "gpt_start_text_token", "gpt_stop_text_token", "gpt_max_text_tokens",
    "gpt_max_audio_tokens", "gpt_max_prompt_tokens", "gpt_number_text_tokens", "gpt_num_audio_tokens",
    "gpt_start_audio_token", "gpt_stop_audio_token", "gpt_use_perceiver_resampler", "gpt_code_stride_len")}

# IDENTITY: keep all 1024 channels, 16 heads. ffn=4096. The head/ffn importance topk will just reorder;
# to be a true identity we must keep heads/ffn in ORIGINAL order, so monkeypatch selection to identity.
import student640
student640.head_importance = lambda ca, cp: torch.arange(student640.TEACHER_HEADS).float()   # topk -> last k; we want all 16 so fine, but order matters
# force identity selections via sorted topk of an increasing score = keeps [0..k-1] in order
keep_idx = torch.arange(1024)
gpt = S.build_student_gpt(ma, d_student=1024, heads=16)
rec = S.slice_teacher_into_student(gpt, tsd, keep_idx, student_ffn=4096)
print("heads kept L0:", rec["heads"][0].tolist()[:6], "... n=", rec["heads"][0].numel())
print("ffn kept L0 first6:", rec["ffn"][0].tolist()[:6], "n=", rec["ffn"][0].numel())
student = S.Student640(gpt, d_student=1024); student.cuda().eval()

with torch.no_grad():
    codes = teacher.gpt.generate(cond_latents=gpt_cond.cuda(), text_inputs=tt, do_sample=False, top_k=1,
                                 top_p=1.0, temperature=0.7, num_return_sequences=1, num_beams=1,
                                 length_penalty=1.0, repetition_penalty=1.0)
    exp = torch.tensor([codes.shape[-1] * teacher.gpt.code_stride_len], device="cuda")
    tlen = torch.tensor([tt.shape[-1]], device="cuda")
    t_lat = teacher.gpt(tt, tlen, codes, exp, cond_latents=gpt_cond.cuda(), return_latent=True)[0]
    s_lat = student.gpt(tt, tlen, codes, exp, cond_latents=gpt_cond.cuda(), return_latent=True)[0]

T = min(t_lat.shape[0], s_lat.shape[0])
t = t_lat[:T].float().cpu(); s = s_lat[:T].float().cpu()
diff = (t - s).abs().max().item()
sc = s - s.mean(0); tc = t - t.mean(0)
corr = ((sc * tc).sum(0) / (sc.norm(dim=0) * tc.norm(dim=0) + 1e-8)).mean().item()
print(f"IDENTITY check: std_t={t.std():.3f} std_s={s.std():.3f} max|t-s|={diff:.4f} mean_corr={corr:+.4f}")
print("PASS (slicing correct)" if diff < 1e-2 else "FAIL (slicing bug -- not just LN coupling)")
