#!/usr/bin/env python3
"""Task #12 — build + structured-init the d=640 fixed-voice student, bake voices, smoke-test audio.

  teacher (265M RL'd) --slice top-640 ch / top-10 heads / top-2560 ffn--> student GPT (d=640)
                       + 640->1024 adapter (scatter init)
                       + 4 baked voices (cond_latents sliced to 640, speaker_embedding 512)
  smoke: student.generate -> latents640 -> adapter -> 1024 -> FROZEN hifigan -> wav

Saves runs/rl/sub100m/student640_init.pt and one wav per voice under runs/rl/sub100m/smoke/.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))            # scripts/hinglish
import student640 as S
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    ap.add_argument("--ckpt", default="runs/rl/round1_rft/xtts_hinglish-June-25-2026_06+39PM-0000000/best_model.pth")
    ap.add_argument("--chan", default="runs/rl/sub100m/channel_importance.pt")
    ap.add_argument("--refs-dir", default="runs/xtts_hinglish/RELEASE/refs")
    ap.add_argument("--gpt-layers", type=int, default=16)
    ap.add_argument("--d-student", type=int, default=640)
    ap.add_argument("--heads", type=int, default=10)
    ap.add_argument("--out", default="runs/rl/sub100m/student640_init.pt")
    ap.add_argument("--smoke-dir", default="runs/rl/sub100m/smoke")
    ap.add_argument("--smoke-text", default="मुझे यह project बहुत interesting लग रहा है, can you believe it?")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    dev = "cuda"
    base = Path(args.base)
    student_ffn = 4 * args.d_student

    # -------- load teacher (full d=1024) for GPT weights, baking, frozen vocoder --------
    print("[1] loading teacher ...", flush=True)
    cfg = XttsConfig(); cfg.load_json(str(base / "config.json"))
    cfg.model_args.gpt_layers = args.gpt_layers
    teacher = Xtts.init_from_config(cfg)
    teacher.load_checkpoint(cfg, checkpoint_path=args.ckpt, vocab_path=str(base / "vocab.json"), use_deepspeed=False)
    teacher.eval(); teacher.cuda()
    a = cfg.model_args.__dict__ if hasattr(cfg.model_args, "__dict__") else dict(cfg.model_args)
    ma = {k: getattr(cfg.model_args, k) for k in (
        "gpt_layers", "gpt_start_text_token", "gpt_stop_text_token", "gpt_max_text_tokens",
        "gpt_max_audio_tokens", "gpt_max_prompt_tokens", "gpt_number_text_tokens", "gpt_num_audio_tokens",
        "gpt_start_audio_token", "gpt_stop_audio_token", "gpt_use_perceiver_resampler", "gpt_code_stride_len")}

    # teacher GPT state-dict, prefix stripped
    full_sd = torch.load(args.ckpt, map_location="cpu")
    full_sd = full_sd["model"] if "model" in full_sd else full_sd
    tsd = {k[len(S.GPT_PREFIX):]: v for k, v in full_sd.items() if k.startswith(S.GPT_PREFIX)}
    print(f"    teacher GPT tensors: {len(tsd)}")

    # -------- build + init student --------
    print("[2] building + slicing student ...", flush=True)
    chan = torch.load(args.chan, map_location="cpu")
    if chan["keep_idx"].numel() == args.d_student:
        keep_idx = chan["keep_idx"].long()
    else:  # derive top-d_student channels by activation importance (works for any width, e.g. 768)
        imp = chan["importance"]
        keep_idx = torch.sort(torch.topk(imp, args.d_student).indices).values.long()
        cap = float((imp[keep_idx].sum() / imp.sum()))
        print(f"    derived keep_idx: top-{args.d_student}/{imp.numel()} channels capture {cap:.1%} activation mass")
    gpt = S.build_student_gpt(ma, d_student=args.d_student, heads=args.heads)
    record = S.slice_teacher_into_student(gpt, tsd, keep_idx, student_ffn)
    student = S.Student640(gpt, d_student=args.d_student)
    S.init_adapter(student.adapter, keep_idx)

    gpt_p = sum(p.numel() for p in student.gpt.parameters())
    ad_p = sum(p.numel() for p in student.adapter.parameters())
    print(f"    student params: gpt={gpt_p/1e6:.2f}M  adapter={ad_p/1e6:.3f}M  TOTAL={student.n_params()/1e6:.2f}M")

    student.cuda().eval()

    # -------- bake the fixed voices --------
    print("[3] baking voices ...", flush=True)
    refs = sorted(Path(args.refs_dir).glob("*.wav"))
    assert refs, f"no refs in {args.refs_dir}"
    voices = {}
    for r in refs:
        gpt_cond, spk = teacher.get_conditioning_latents(audio_path=[str(r)])   # (1,32,1024),(1,512)
        cond640 = gpt_cond[..., keep_idx].contiguous().cpu()                    # slice the d axis
        voices[r.stem] = {"cond_latents": cond640, "speaker_embedding": spk.cpu(),
                          "cond_shape": tuple(cond640.shape)}
        print(f"    {r.stem}: cond {tuple(cond640.shape)} spk {tuple(spk.shape)}")

    # -------- save before smoke (so a decode crash still leaves the checkpoint) --------
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "gpt": student.gpt.state_dict(), "adapter": student.adapter.state_dict(),
        "keep_idx": keep_idx, "head_keep": record["heads"], "ffn_keep": record["ffn"],
        "model_args": ma, "d_student": args.d_student, "heads": args.heads, "student_ffn": student_ffn,
        "voices": voices, "n_params": student.n_params(),
        "source_ckpt": args.ckpt, "channel_importance": args.chan,
    }, args.out)
    print(f"[4] saved -> {args.out}  ({student.n_params()/1e6:.2f}M params)")

    # -------- smoke test: produce audio per voice --------
    print("[5] smoke decode ...", flush=True)
    Path(args.smoke_dir).mkdir(parents=True, exist_ok=True)
    student.gpt.init_gpt_for_inference(kv_cache=True)
    student.gpt.cuda().eval()
    tok = teacher.tokenizer
    text = args.smoke_text
    for vname, v in voices.items():
        cond = v["cond_latents"].cuda()
        spk = v["speaker_embedding"].cuda()
        text_tokens = torch.IntTensor(tok.encode(text, lang="hi")).unsqueeze(0).cuda()
        with torch.no_grad():
            torch.manual_seed(20260626)
            codes = student.gpt.generate(cond_latents=cond, text_inputs=text_tokens,
                                         do_sample=True, top_p=0.9, top_k=50, temperature=0.85,
                                         num_return_sequences=1, num_beams=1, length_penalty=1.0,
                                         repetition_penalty=2.0)
            expected_len = torch.tensor([codes.shape[-1] * student.gpt.code_stride_len], device=dev)
            text_len = torch.tensor([text_tokens.shape[-1]], device=dev)
            latents640 = student.gpt(text_tokens, text_len, codes, expected_len,
                                     cond_latents=cond, return_latent=True)        # (1,T,640)
            latents1024 = student.vocoder_latents(latents640)                      # (1,T,1024)
            wav = teacher.hifigan_decoder(latents1024, g=spk).cpu().squeeze().numpy()
        outw = Path(args.smoke_dir) / f"{vname}.wav"
        sf.write(str(outw), np.asarray(wav, dtype=np.float32), 24000)
        print(f"    {vname}: codes={codes.shape[-1]} latents={tuple(latents640.shape)} "
              f"wav={len(wav)/24000:.2f}s -> {outw}")

    print("BUILD_STUDENT640_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
