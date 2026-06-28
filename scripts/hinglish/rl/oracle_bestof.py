#!/usr/bin/env python3
"""RFT-ceiling diagnostic: generate N candidates/prompt, score each, report mean accent vs
per-prompt MAX accent (the best-of-N oracle). RFT can only reinforce samples the model already
produces, so oracle accent is the realistic ceiling RFT could reach. Decides whether #14 is worth scaling."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.gen_student640 import load_student, stable_seed
from rl.reward import RewardScorer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--base", default=".tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--refs-dir", default="runs/xtts_hinglish/RELEASE/refs")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--max", type=int, default=24)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    dev = "cuda"
    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()][:args.max]
    student, teacher, tok, ck = load_student(args.student, args.base, dev)
    scorer = RewardScorer(device=dev)
    for v in ck["voices"]:
        scorer.register_voice(v, str(Path(args.refs_dir) / f"{v}.wav"))

    means, oracles = [], []
    for r in rows:
        uid, v = r["utt_id"], r["voice"]; text = r.get("ref_text") or r.get("text")
        cond = ck["voices"][v]["cond_latents"].cuda(); spk = ck["voices"][v]["speaker_embedding"].cuda()
        tt = torch.IntTensor(tok.encode(text, lang="hi")).unsqueeze(0).cuda()
        accs = []
        for j in range(args.n):
            torch.manual_seed(stable_seed(20260626 + j * 7919, uid))
            with torch.no_grad():
                codes = student.gpt.generate(cond_latents=cond, text_inputs=tt, do_sample=True, top_k=50,
                                             top_p=0.9, temperature=args.temp, num_return_sequences=1,
                                             num_beams=1, length_penalty=1.0, repetition_penalty=1.5)
                exp = torch.tensor([codes.shape[-1] * student.gpt.code_stride_len], device=dev)
                tlen = torch.tensor([tt.shape[-1]], device=dev)
                lat = student.gpt(tt, tlen, codes, exp, cond_latents=cond, return_latent=True)
                wav = teacher.hifigan_decoder(student.vocoder_latents(lat), g=spk).cpu().squeeze().numpy()
            comp = scorer.components(np.asarray(wav, np.float32), 24000, text, v)
            if comp["en_recall"] is not None:
                accs.append(comp["en_recall"])
        if accs:
            means.append(float(np.mean(accs))); oracles.append(float(np.max(accs)))
            print(f"  {uid:18s} mean={np.mean(accs):.2f} best-of-{args.n}={np.max(accs):.2f} "
                  f"(n_en candidates={len(accs)})", flush=True)
    print(f"\n=== RFT CEILING (n={len(means)} prompts, N={args.n}) ===")
    print(f"  mean accent       = {np.mean(means):.3f}")
    print(f"  oracle best-of-{args.n} = {np.mean(oracles):.3f}   <- realistic RFT ceiling")
    print(f"  teacher reference = 0.830")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
