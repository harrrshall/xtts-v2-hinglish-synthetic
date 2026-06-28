#!/usr/bin/env python3
"""Stage-2 of the staged prune: build a d=640 student INITIALIZED FROM A RECOVERED d=768 STUDENT
(not the raw teacher). Gradual cuts (1024->768->640) preserve content capacity better than one-shot.

The d=768 student's channels are teacher channels re-indexed; we keep the 640 of its 768 channels that
correspond to the ORIGINAL teacher top-640 (consistent final channel set, d=768-recovered weights).
The d=640 adapter inits from the d=768 adapter's kept columns (a recovered 640->1024 map, not scatter).
Conditioning is re-sliced from the d=768 baked cond; the frozen vocoder + speaker embeddings are unchanged.
"""
import argparse, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
import student640 as S
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-student", default="runs/rl/sub100m/student768_distilled.pt", help="recovered d=768 ckpt")
    ap.add_argument("--orig-chan", default="runs/rl/sub100m/channel_importance.pt", help="for the final top-640 set")
    ap.add_argument("--base", default=".tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    ap.add_argument("--vocoder-ckpt", default="runs/rl/round1_rft/xtts_hinglish-June-25-2026_06+39PM-0000000/best_model.pth")
    ap.add_argument("--d-student", type=int, default=640)
    ap.add_argument("--heads", type=int, default=10)
    ap.add_argument("--out", default="runs/rl/sub100m/student640b_init.pt")
    ap.add_argument("--smoke-dir", default="runs/rl/sub100m/smoke640b")
    args = ap.parse_args()
    assert torch.cuda.is_available()
    dev = "cuda"; base = Path(args.base)
    student_ffn = 4 * args.d_student

    # -------- load the recovered d=768 student (the teacher for this stage) --------
    print("[1] loading recovered d=768 student ...", flush=True)
    tck = torch.load(args.teacher_student, map_location="cpu")
    T768 = tck["keep_idx"].long()                       # teacher(1024) channels the d=768 kept (sorted)
    t_gpt = S.build_student_gpt(tck["model_args"], d_student=tck["d_student"], heads=tck["heads"])
    t_gpt.load_state_dict(tck["gpt"])
    tsd = t_gpt.state_dict()                              # no GPT_PREFIX (already a bare GPT)
    t_adapter_w = tck["adapter"]["weight"]               # (1024, 768)
    t_adapter_b = tck["adapter"]["bias"]                 # (1024,)

    # -------- which 640 of the d=768's positions == original teacher top-640 --------
    T640 = set(torch.load(args.orig_chan, map_location="cpu")["keep_idx"].long().tolist())
    stage2_pos = torch.tensor([j for j, t in enumerate(T768.tolist()) if t in T640], dtype=torch.long)
    assert stage2_pos.numel() == args.d_student, f"got {stage2_pos.numel()} positions != {args.d_student}"
    print(f"    stage2 keep: {stage2_pos.numel()} of {T768.numel()} d=768 channels (== teacher top-640)")

    # -------- build + slice d=640 from the d=768 student --------
    print("[2] building + slicing d=640 from d=768 ...", flush=True)
    gpt = S.build_student_gpt(tck["model_args"], d_student=args.d_student, heads=args.heads)
    record = S.slice_teacher_into_student(gpt, tsd, stage2_pos, student_ffn)
    student = S.Student640(gpt, d_student=args.d_student)
    # adapter init: recovered d=768 adapter restricted to kept input columns -> 640->1024
    with torch.no_grad():
        student.adapter.weight.copy_(t_adapter_w[:, stage2_pos])
        student.adapter.bias.copy_(t_adapter_b)
    gpt_p = sum(p.numel() for p in student.gpt.parameters())
    print(f"    student params: gpt={gpt_p/1e6:.2f}M adapter={sum(p.numel() for p in student.adapter.parameters())/1e6:.3f}M "
          f"TOTAL={student.n_params()/1e6:.2f}M")
    student.cuda().eval()

    # -------- re-slice baked conditioning (d=768 -> d=640) --------
    voices = {}
    for v, d in tck["voices"].items():
        cond768 = d["cond_latents"]                       # (1,32,768)
        voices[v] = {"cond_latents": cond768[..., stage2_pos].contiguous(),
                     "speaker_embedding": d["speaker_embedding"]}
    print(f"[3] re-sliced cond for voices={list(voices)} -> (1,32,{args.d_student})")

    # final keep_idx in TEACHER(1024) index space (for any further stage / analysis)
    final_keep_1024 = T768[stage2_pos]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"gpt": student.gpt.state_dict(), "adapter": student.adapter.state_dict(),
                "keep_idx": final_keep_1024, "head_keep": record["heads"], "ffn_keep": record["ffn"],
                "model_args": tck["model_args"], "d_student": args.d_student, "heads": args.heads,
                "student_ffn": student_ffn, "voices": voices, "n_params": student.n_params(),
                "source": args.teacher_student, "staged": "1024->768->640"}, args.out)
    print(f"[4] saved -> {args.out} ({student.n_params()/1e6:.2f}M)")

    # -------- smoke decode via frozen vocoder --------
    print("[5] smoke decode ...", flush=True)
    cfg = XttsConfig(); cfg.load_json(str(base / "config.json")); cfg.model_args.gpt_layers = 16
    voc = Xtts.init_from_config(cfg)
    voc.load_checkpoint(cfg, checkpoint_path=args.vocoder_ckpt, vocab_path=str(base / "vocab.json"), use_deepspeed=False)
    voc.eval(); voc.cuda()
    student.gpt.init_gpt_for_inference(kv_cache=True); student.gpt.cuda().eval()
    Path(args.smoke_dir).mkdir(parents=True, exist_ok=True)
    text = "मुझे यह project बहुत interesting लग रहा है, can you believe it?"
    for vname, v in voices.items():
        cond = v["cond_latents"].cuda(); spk = v["speaker_embedding"].cuda()
        tt = torch.IntTensor(voc.tokenizer.encode(text, lang="hi")).unsqueeze(0).cuda()
        with torch.no_grad():
            torch.manual_seed(20260626)
            codes = student.gpt.generate(cond_latents=cond, text_inputs=tt, do_sample=True, top_p=0.9,
                                         top_k=50, temperature=0.85, num_return_sequences=1, num_beams=1,
                                         length_penalty=1.0, repetition_penalty=2.0)
            exp = torch.tensor([codes.shape[-1] * student.gpt.code_stride_len], device=dev)
            tlen = torch.tensor([tt.shape[-1]], device=dev)
            lat = student.gpt(tt, tlen, codes, exp, cond_latents=cond, return_latent=True)
            wav = voc.hifigan_decoder(student.vocoder_latents(lat), g=spk).cpu().squeeze().numpy()
        sf.write(str(Path(args.smoke_dir) / f"{vname}.wav"), np.asarray(wav, np.float32), 24000)
        print(f"    {vname}: codes={codes.shape[-1]} wav={len(wav)/24000:.2f}s")
    print("BUILD_640B_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
