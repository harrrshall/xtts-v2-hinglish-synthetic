#!/usr/bin/env python3
"""Decode a d=640 student on a prompt set and score it: accent recall, UTMOS, SECS, audio RMS,
code length + degeneracy. Confirms the distilled student is audible/intelligible and quantifies the
gap to teacher (run with --student on both the distilled student and, for ref, feed teacher wavs)."""
import argparse, json, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.gen_student640 import load_student, stable_seed
from rl.reward import RewardScorer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--base", default=".tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--refs-dir", default="runs/xtts_hinglish/RELEASE/refs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--rep-penalty", type=float, default=2.0)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    dev = "cuda"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    if args.max:
        rows = rows[:args.max]

    student, teacher, tok, ck = load_student(args.student, args.base, dev)
    scorer = RewardScorer(device=dev)
    for v in ck["voices"]:
        scorer.register_voice(v, str(Path(args.refs_dir) / f"{v}.wav"))

    agg = {"accent": [], "utmos": [], "secs": [], "rms": [], "codes": [], "dur": [], "sil": [], "rep": []}
    for r in rows:
        uid, v = r["utt_id"], r["voice"]; text = r.get("ref_text") or r.get("text")
        cond = ck["voices"][v]["cond_latents"].cuda(); spk = ck["voices"][v]["speaker_embedding"].cuda()
        try:
            tt = torch.IntTensor(tok.encode(text, lang="hi")).unsqueeze(0).cuda()
        except Exception:
            continue
        torch.manual_seed(stable_seed(20260626, uid))
        gk = dict(do_sample=not args.greedy, num_return_sequences=1, num_beams=1, length_penalty=1.0,
                  repetition_penalty=args.rep_penalty)
        gk.update(top_k=1, top_p=1.0, temperature=0.7) if args.greedy else \
            gk.update(top_k=args.top_k, top_p=0.9, temperature=args.temp)
        with torch.no_grad():
            codes = student.gpt.generate(cond_latents=cond, text_inputs=tt, **gk)
            exp = torch.tensor([codes.shape[-1] * student.gpt.code_stride_len], device=dev)
            tlen = torch.tensor([tt.shape[-1]], device=dev)
            lat640 = student.gpt(tt, tlen, codes, exp, cond_latents=cond, return_latent=True)
            wav = teacher.hifigan_decoder(student.vocoder_latents(lat640), g=spk).cpu().squeeze().numpy()
        wav = np.asarray(wav, dtype=np.float32)
        wp = out / f"{uid}.wav"; sf.write(str(wp), wav, 24000)
        comp = scorer.components(wav, 24000, text, v, wav_path=str(wp))
        rms = float(np.sqrt((wav ** 2).mean()))
        if comp["en_recall"] is not None:
            agg["accent"].append(comp["en_recall"])
        agg["utmos"].append(comp["utmos"]); agg["secs"].append(comp["secs"] if comp["secs"] else 0.0)
        agg["rms"].append(rms); agg["codes"].append(int(codes.shape[-1])); agg["dur"].append(comp["dur"])
        agg["sil"].append(comp["silence_ratio"]); agg["rep"].append(comp["ngram_rep"])
        print(f"  {uid:18s} {v:9s} acc={comp['en_recall']} utmos={comp['utmos']:.2f} "
              f"secs={comp['secs']:.3f} rms={rms:.3f} codes={codes.shape[-1]} sil={comp['silence_ratio']:.2f} "
              f"hyp='{comp['hyp'][:42]}'", flush=True)

    def m(x): return float(np.mean(x)) if x else 0.0
    print("\n=== SUMMARY (distilled d=640 student) ===")
    print(f"  accent_recall={m(agg['accent']):.3f}  UTMOS={m(agg['utmos']):.3f}  SECS={m(agg['secs']):.3f}")
    print(f"  audio_rms={m(agg['rms']):.4f}  codes_mean={m(agg['codes']):.0f}  dur_mean={m(agg['dur']):.2f}s "
          f"silence={m(agg['sil']):.2f}  rep={m(agg['rep']):.2f}")
    json.dump({k: m(v) for k, v in agg.items()}, open(out / "summary.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
