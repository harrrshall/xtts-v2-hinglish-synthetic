#!/usr/bin/env python3
"""Reference floors: teacher-forced CE + next-token top-1 accuracy on the eval set, for
teacher(d=1024) vs student-init vs student-distilled. Tells us the true content-capacity gap and
whether the student even learned the text->code mapping (top-1 acc), separate from generation drift."""
import sys
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import student640 as S
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

BASE = ".tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2"
CKPT = "runs/rl/round1_rft/xtts_hinglish-June-25-2026_06+39PM-0000000/best_model.pth"
REFS = "runs/xtts_hinglish/RELEASE/refs"
EVAL = "runs/rl/sub100m/distill_eval.pt"

cfg = XttsConfig(); cfg.load_json(BASE + "/config.json"); cfg.model_args.gpt_layers = 16
teacher = Xtts.init_from_config(cfg)
teacher.load_checkpoint(cfg, checkpoint_path=CKPT, vocab_path=BASE + "/vocab.json", use_deepspeed=False)
teacher.eval(); teacher.cuda()
cond_t = {}
for v in ["aadya", "arjun", "kaustubh", "maya"]:
    gc, _ = teacher.get_conditioning_latents(audio_path=[f"{REFS}/{v}.wav"])
    cond_t[v] = gc.cpu()

ev = torch.load(EVAL, map_location="cpu")
dev = "cuda"


def collate(items):
    b = len(items); nt = max(it["n_text"] for it in items); nc = max(it["n_codes"] for it in items)
    text = torch.zeros(b, nt, dtype=torch.long); codes = torch.zeros(b, nc, dtype=torch.long)
    tlen = torch.zeros(b, dtype=torch.long); wavlen = torch.zeros(b, dtype=torch.long); voices = []
    for i, it in enumerate(items):
        t = it["tokens"].long(); c = it["codes"].long()
        text[i, :t.numel()] = t; codes[i, :c.numel()] = c
        tlen[i] = t.numel(); wavlen[i] = it["n_codes"] * 1024; voices.append(it["voice"])
    return text.to(dev), tlen.to(dev), codes.to(dev), wavlen.to(dev), voices


@torch.no_grad()
def score(gpt, cond_map, adapter=None, name=""):
    ce_t = 0.0; corr = 0; ntok = 0
    for i in range(0, len(ev), 16):
        items = ev[i:i + 16]
        text, tlen, codes, wavlen, voices = collate(items)
        cond = torch.cat([cond_map[v].to(dev) for v in voices], dim=0)
        ce, logits, lat, mask = S.forward_both(gpt, text, tlen, codes, wavlen, cond)
        ce_t += float(ce) * len(items)
        pred = logits.argmax(1)                       # (b, L)
        tgt = logits.new_zeros(0)
        # rebuild targets the same way forward_both does is complex; use mask + shift-free check:
        # approximate top-1 acc over valid positions vs the GT codes aligned by forward's targets
        corr += int(((pred == _targets(gpt, codes, wavlen)) & mask).sum())
        ntok += int(mask.sum())
    print(f"  {name:22s} CE={ce_t/len(ev):.3f}  top1_acc={corr/max(ntok,1):.3f}")


def _targets(gpt, audio_codes, wav_lengths):
    code_stride_len = gpt.code_stride_len
    code_lengths = torch.ceil(wav_lengths / code_stride_len).long() + 3
    max_mel_len = code_lengths.max()
    if max_mel_len > audio_codes.shape[-1]:
        audio_codes = F.pad(audio_codes, (0, max_mel_len - audio_codes.shape[-1]))
    audio_codes = F.pad(audio_codes[:, :max_mel_len], (0, 1), value=gpt.stop_audio_token)
    audio_codes = gpt.set_mel_padding(audio_codes, code_lengths - 3)
    _, mel_targets = gpt.set_inputs_and_targets(audio_codes, gpt.start_audio_token, gpt.stop_audio_token)
    for idx, l in enumerate(code_lengths):
        mel_targets[idx, l + 1:] = -1
    return mel_targets.long()


print("=== teacher-forced CE + next-code top-1 acc on eval (n=%d) ===" % len(ev))
score(teacher.gpt, cond_t, name="teacher d=1024")

for tag, path in [("student-init", "runs/rl/sub100m/student640_init.pt"),
                  ("student-distilled", "runs/rl/sub100m/student640_distilled.pt")]:
    ck = torch.load(path, map_location="cpu")
    gpt = S.build_student_gpt(ck["model_args"], d_student=ck["d_student"], heads=ck["heads"])
    gpt.load_state_dict(ck["gpt"]); gpt.cuda().eval()
    cond_s = {v: ck["voices"][v]["cond_latents"] for v in ck["voices"]}
    score(gpt, cond_s, name=tag)
