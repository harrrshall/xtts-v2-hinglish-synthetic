#!/usr/bin/env python3
"""Generate audio from a d=640 fixed-voice student checkpoint (init or distilled).

Path: student.gpt.generate(baked cond640) -> codes -> forward(return_latent) -> latents640
      -> adapter -> 1024 -> FROZEN HiFi-GAN(g=baked speaker_emb) -> wav.
Reused by #13 (sanity, n=1) and #14 (RFT rollouts, n=N). Writes <out>/<utt>__c<j>.wav + candidates.jsonl.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import student640 as S
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def stable_seed(seed_base: int, utt_id: str) -> int:
    h = 1469598103934665603
    for b in utt_id.encode("utf-8"):
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return (seed_base + (h & 0x7FFFFFFF)) & 0x7FFFFFFF


TEACHER_CKPT = "runs/rl/round1_rft/xtts_hinglish-June-25-2026_06+39PM-0000000/best_model.pth"


def load_student(ckpt_path, base, dev="cuda", teacher_ckpt=TEACHER_CKPT):
    """Returns (student, teacher_for_vocoder, tokenizer, ck). The teacher here only supplies the
    FROZEN hifigan_decoder + tokenizer; load the 16-layer round1 ckpt (its vocoder is identical)."""
    base = Path(base)
    cfg = XttsConfig(); cfg.load_json(str(base / "config.json")); cfg.model_args.gpt_layers = 16
    teacher = Xtts.init_from_config(cfg)
    teacher.load_checkpoint(cfg, checkpoint_path=teacher_ckpt, vocab_path=str(base / "vocab.json"), use_deepspeed=False)
    teacher.eval(); teacher.cuda()
    ck = torch.load(ckpt_path, map_location="cpu")
    gpt = S.build_student_gpt(ck["model_args"], d_student=ck["d_student"], heads=ck["heads"])
    gpt.load_state_dict(ck["gpt"])
    student = S.Student640(gpt, d_student=ck["d_student"]); student.adapter.load_state_dict(ck["adapter"])
    student.cuda().eval(); student.gpt.init_gpt_for_inference(kv_cache=True); student.gpt.cuda().eval()
    return student, teacher, teacher.tokenizer, ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--base", default=".tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep-penalty", type=float, default=2.0)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed-base", type=int, default=20260626)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    dev = "cuda"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    if args.max:
        rows = rows[:args.max]

    student, teacher, tok, ck = load_student(args.student, args.base, dev)
    voices = ck["voices"]
    print(f"[gen_student640] {args.student}  voices={list(voices)}  prompts={len(rows)} n={args.n}")
    n = 1 if args.greedy else args.n
    man, t0 = [], time.perf_counter()
    for r in rows:
        uid, v = r["utt_id"], r["voice"]
        text = r.get("ref_text") or r.get("text")
        cond = voices[v]["cond_latents"].cuda(); spk = voices[v]["speaker_embedding"].cuda()
        try:  # digits crash num2words for 'hi' (spell numbers as words); skip un-tokenizable prompts
            tt = torch.IntTensor(tok.encode(text, lang="hi")).unsqueeze(0).cuda()
        except Exception as e:
            print(f"  SKIP-PROMPT {uid}: {type(e).__name__}"); continue
        for j in range(n):
            torch.manual_seed(stable_seed(args.seed_base + j * 7919, uid))
            kw = dict(do_sample=not args.greedy, num_return_sequences=1, num_beams=1, length_penalty=1.0,
                      repetition_penalty=args.rep_penalty)
            kw.update(top_k=1, top_p=1.0, temperature=0.7) if args.greedy else \
                kw.update(top_k=args.top_k, top_p=args.top_p, temperature=args.temp)
            try:
                with torch.no_grad():
                    codes = student.gpt.generate(cond_latents=cond, text_inputs=tt, **kw)
                    exp = torch.tensor([codes.shape[-1] * student.gpt.code_stride_len], device=dev)
                    tlen = torch.tensor([tt.shape[-1]], device=dev)
                    lat640 = student.gpt(tt, tlen, codes, exp, cond_latents=cond, return_latent=True)
                    lat1024 = student.vocoder_latents(lat640)
                    wav = teacher.hifigan_decoder(lat1024, g=spk).cpu().squeeze().numpy()
            except Exception as e:
                print(f"  SKIP {uid} c{j}: {type(e).__name__} {str(e)[:60]}"); continue
            wav = np.asarray(wav, dtype=np.float32)
            wp = out / f"{uid}__c{j}.wav"; sf.write(str(wp), wav, 24000)
            man.append({"utt_id": uid, "cand_id": j, "wav": str(wp), "ref_text": text, "voice": v,
                        "cs_mode": r.get("cs_mode", "?"), "audio_s": round(len(wav) / 24000, 3),
                        "codes": int(codes.shape[-1])})
    (out / "candidates.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in man))
    dt = time.perf_counter() - t0
    print(f"[gen_student640] {len(man)} clips -> {out}  {dt:.0f}s ({dt/max(len(man),1):.2f}s/clip)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
