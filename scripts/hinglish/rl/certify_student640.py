#!/usr/bin/env python3
"""Task #14 certification: is the <100M d=640 student EQUIVALENT to the 265M teacher (round1)?

Generates BOTH models on the SAME held-out eval texts (excluded from distillation), scores each with
the same RewardScorer, then paired bootstrap 95% CI + TOST non-inferiority per axis:
  accent (English-as-English recall), UTMOS (naturalness), SECS (voice fidelity).
Non-inferiority PASS = lower bound of (student - teacher) mean-delta CI > -margin.
Margins (pre-registered, same as prior tiers): accent .03, UTMOS .10, SECS .03.
Also reports the runaway-tail rate (codes>=600 or silence>.6) for both models."""
import argparse, csv, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.gen_student640 import load_student, stable_seed
from rl.reward import RewardScorer
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

MARGINS = {"accent": 0.03, "utmos": 0.10, "secs": 0.03}


def boot_delta_ci(s, t, n=5000, seed=0):
    """Paired bootstrap of mean(student - teacher). Returns (mean, lo, hi)."""
    s = np.asarray(s); t = np.asarray(t); d = s - t
    rng = np.random.default_rng(seed); idx = np.arange(len(d))
    means = [d[rng.choice(idx, len(d), replace=True)].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="runs/rl/sub100m/student640b_rft.pt")
    ap.add_argument("--teacher-ckpt", default="runs/rl/round1_rft/xtts_hinglish-June-25-2026_06+39PM-0000000/best_model.pth")
    ap.add_argument("--base", default=".tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    ap.add_argument("--eval-meta", default="data/xtts/metadata_eval.csv")
    ap.add_argument("--eval-jsonl", default=None, help="held-out prompts jsonl (utt_id, ref_text, voice); overrides --eval-meta")
    ap.add_argument("--refs-dir", default="runs/xtts_hinglish/RELEASE/refs")
    ap.add_argument("--out", default="runs/rl/sub100m/cert_report.json")
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()
    assert torch.cuda.is_available(); dev = "cuda"
    base = Path(args.base)

    # held-out prompts (text, voice), skip digit-bearing (num2words 'hi' crash)
    rows = []
    if args.eval_jsonl:
        for l in open(args.eval_jsonl, encoding="utf-8"):
            if not l.strip():
                continue
            d = json.loads(l); txt = (d.get("ref_text") or d.get("text") or "").strip()
            if txt and not any(c.isdigit() for c in txt):
                rows.append({"utt_id": d.get("utt_id", f"cert{len(rows):04d}__{d['voice']}"),
                             "text": txt, "voice": d["voice"]})
    else:
        for r in csv.reader(open(args.eval_meta, encoding="utf-8"), delimiter="|"):
            if len(r) >= 3 and r[1].strip() and not any(c.isdigit() for c in r[1]):
                rows.append({"utt_id": f"cert{len(rows):04d}__{r[2]}", "text": r[1], "voice": r[2]})
    if args.max:
        rows = rows[:args.max]

    student, voc, tok, ck = load_student(args.student, args.base, dev, teacher_ckpt=args.teacher_ckpt)
    # teacher (265M round1) for its own generation
    cfg = XttsConfig(); cfg.load_json(str(base / "config.json")); cfg.model_args.gpt_layers = 16
    teacher = Xtts.init_from_config(cfg)
    teacher.load_checkpoint(cfg, checkpoint_path=args.teacher_ckpt, vocab_path=str(base / "vocab.json"), use_deepspeed=False)
    teacher.eval(); teacher.cuda()
    tcond = {v: teacher.get_conditioning_latents(audio_path=[str(Path(args.refs_dir) / f"{v}.wav")]) for v in ck["voices"]}

    sc = RewardScorer(device=dev)
    for v in ck["voices"]:
        sc.register_voice(v, str(Path(args.refs_dir) / f"{v}.wav"))

    M = {"accent": {"s": [], "t": []}, "utmos": {"s": [], "t": []}, "secs": {"s": [], "t": []}}
    tail = {"s": 0, "t": 0, "n": 0}
    for r in rows:
        uid, v, text = r["utt_id"], r["voice"], r["text"]
        try:
            tt = torch.IntTensor(tok.encode(text, lang="hi")).unsqueeze(0).cuda()
        except Exception:
            continue
        # student (greedy, faithful)
        scond = ck["voices"][v]["cond_latents"].cuda(); sspk = ck["voices"][v]["speaker_embedding"].cuda()
        torch.manual_seed(stable_seed(7, uid))
        with torch.no_grad():
            sc_codes = student.gpt.generate(cond_latents=scond, text_inputs=tt, do_sample=False, top_k=1,
                                            top_p=1.0, temperature=0.7, num_return_sequences=1, num_beams=1,
                                            length_penalty=1.0, repetition_penalty=1.3)
            exp = torch.tensor([sc_codes.shape[-1] * student.gpt.code_stride_len], device=dev)
            lat = student.gpt(tt, torch.tensor([tt.shape[-1]], device=dev), sc_codes, exp, cond_latents=scond, return_latent=True)
            swav = voc.hifigan_decoder(student.vocoder_latents(lat), g=sspk).cpu().squeeze().numpy()
        # teacher
        gptc, tspk = tcond[v]
        torch.manual_seed(stable_seed(7, uid))
        two = teacher.inference(text, "hi", gptc, tspk, temperature=0.7, enable_text_splitting=False,
                                repetition_penalty=1.3, do_sample=False, top_k=1, top_p=1.0)
        twav = np.asarray(two["wav"], np.float32)

        cs = sc.components(np.asarray(swav, np.float32), 24000, text, v)
        ct = sc.components(twav, 24000, text, v)
        if cs["en_recall"] is not None and ct["en_recall"] is not None:
            M["accent"]["s"].append(cs["en_recall"]); M["accent"]["t"].append(ct["en_recall"])
        M["utmos"]["s"].append(cs["utmos"]); M["utmos"]["t"].append(ct["utmos"])
        M["secs"]["s"].append(cs["secs"] or 0.0); M["secs"]["t"].append(ct["secs"] or 0.0)
        tail["n"] += 1
        tail["s"] += int(sc_codes.shape[-1] >= 600 or cs["silence_ratio"] > 0.6)
        tail["t"] += int(len(twav) / 24000 > 25 or ct["silence_ratio"] > 0.6)
        if tail["n"] % 15 == 0:
            print(f"  {tail['n']} done", flush=True)

    rep = {"n": tail["n"], "tail_rate_student": tail["s"] / max(tail["n"], 1),
           "tail_rate_teacher": tail["t"] / max(tail["n"], 1), "axes": {}}
    print(f"\n=== CERTIFICATION: d=640 staged student ({ck['n_params']/1e6:.1f}M) vs 265M teacher (n={tail['n']}) ===")
    all_pass = True
    for ax in ("accent", "utmos", "secs"):
        s, t = M[ax]["s"], M[ax]["t"]
        mean, lo, hi = boot_delta_ci(s, t)
        ok = lo > -MARGINS[ax]
        all_pass &= ok
        rep["axes"][ax] = {"student": float(np.mean(s)), "teacher": float(np.mean(t)), "delta": mean,
                           "ci": [lo, hi], "margin": -MARGINS[ax], "pass": bool(ok), "n": len(s)}
        print(f"  {ax:7s} student={np.mean(s):.3f} teacher={np.mean(t):.3f} delta={mean:+.3f} "
              f"CI=[{lo:+.3f},{hi:+.3f}] margin={-MARGINS[ax]:+.2f}  {'PASS' if ok else 'FAIL'} (n={len(s)})")
    rep["overall_pass"] = bool(all_pass)
    print(f"  tail-rate student={rep['tail_rate_student']:.1%} teacher={rep['tail_rate_teacher']:.1%}")
    print(f"  OVERALL: {'PASS (equivalent within margins)' if all_pass else 'FAIL'}")
    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
