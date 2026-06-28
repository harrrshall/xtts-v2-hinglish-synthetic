#!/usr/bin/env python3
"""Task #13 — multi-signal distillation of the d=640 fixed-voice student from the 265M RL'd teacher.

Per step, teacher (frozen, d=1024) and student (d=640, +adapter) run the SAME shared forward
(student640.forward_both) on the same GT codes + baked per-voice conditioning. Loss:

    L = w_ce * CE(GT codes)                                   # ground-truth anchor
      + w_kl * KL(student||teacher mel-logits, T) * T^2       # logit distillation (Minitron)
      + w_lat * [ MSE + (1 - cos) ]( adapter(student_lat), teacher_lat )   # TIMBRE / vocoder protector

The latent term is load-bearing: it teaches the 640->1024 adapter to feed the FROZEN HiFi-GAN, and
recovers the LayerNorm-coupling decorrelation that width-pruning introduces (see _diag2). Distill on
the teacher corpus (teacher is the quality target); aggressive early-stop on held-out latent+CE.
"""
import argparse, sys, time, math
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import student640 as S
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def collate(items, device):
    b = len(items)
    nt = max(it["n_text"] for it in items); nc = max(it["n_codes"] for it in items)
    text = torch.zeros(b, nt, dtype=torch.long); codes = torch.zeros(b, nc, dtype=torch.long)
    tlen = torch.zeros(b, dtype=torch.long); wavlen = torch.zeros(b, dtype=torch.long)
    voices = []
    for i, it in enumerate(items):
        t = it["tokens"].long(); c = it["codes"].long()
        text[i, :t.numel()] = t; codes[i, :c.numel()] = c
        tlen[i] = t.numel(); wavlen[i] = it["n_codes"] * 1024; voices.append(it["voice"])
    return (text.to(device), tlen.to(device), codes.to(device), wavlen.to(device), voices)


def cond_batch(voices, cond_map, device):
    return torch.cat([cond_map[v].to(device) for v in voices], dim=0)


def masked_kl(logits_s, logits_t, mask, T):
    # logits: (b, C, L) -> (N_valid, C)
    bs, C, L = logits_s.shape
    ls = logits_s.permute(0, 2, 1).reshape(-1, C)[mask.reshape(-1)]
    lt = logits_t.permute(0, 2, 1).reshape(-1, C)[mask.reshape(-1)]
    p_t = F.softmax(lt / T, dim=-1)
    logp_s = F.log_softmax(ls / T, dim=-1)
    return F.kl_div(logp_s, p_t, reduction="batchmean") * (T * T)


def masked_latent(lat_s, lat_t, mask):
    # lat: (b, L, d) -> (N_valid, d)
    d_s, d_t = lat_s.shape[-1], lat_t.shape[-1]
    s = lat_s.reshape(-1, d_s)[mask.reshape(-1)]
    t = lat_t.reshape(-1, d_t)[mask.reshape(-1)]
    mse = F.mse_loss(s, t)
    cos = F.cosine_similarity(s, t, dim=-1).mean()
    return mse, (1.0 - cos)


@torch.no_grad()
def evaluate(student, teacher_gpt, data, cond_s, cond_t, device, w, T, bs=16):
    student.eval()
    tot = {"ce": 0.0, "kl": 0.0, "mse": 0.0, "cosd": 0.0, "n": 0}
    for i in range(0, len(data), bs):
        items = data[i:i + bs]
        text, tlen, codes, wavlen, voices = collate(items, device)
        cs = cond_batch(voices, cond_s, device); ct = cond_batch(voices, cond_t, device)
        ce_s, lg_s, lt_s, mask = S.forward_both(student.gpt, text, tlen, codes, wavlen, cs)
        _, lg_t, lt_t, _ = S.forward_both(teacher_gpt, text, tlen, codes, wavlen, ct)
        a_s = student.adapter(lt_s)
        kl = masked_kl(lg_s, lg_t, mask, T)
        mse, cosd = masked_latent(a_s, lt_t, mask)
        n = len(items)
        tot["ce"] += float(ce_s) * n; tot["kl"] += float(kl) * n
        tot["mse"] += float(mse) * n; tot["cosd"] += float(cosd) * n; tot["n"] += n
    student.train()
    for k in ("ce", "kl", "mse", "cosd"):
        tot[k] /= max(tot["n"], 1)
    tot["score"] = w["ce"] * tot["ce"] + w["lat"] * (tot["mse"] + tot["cosd"])   # early-stop signal
    return tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-init", default="runs/rl/sub100m/student640_init.pt")
    ap.add_argument("--teacher-ckpt", default="runs/rl/round1_rft/xtts_hinglish-June-25-2026_06+39PM-0000000/best_model.pth")
    ap.add_argument("--base", default=".tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    ap.add_argument("--refs-dir", default="runs/xtts_hinglish/RELEASE/refs")
    ap.add_argument("--train-data", default="runs/rl/sub100m/distill_train.pt")
    ap.add_argument("--eval-data", default="runs/rl/sub100m/distill_eval.pt")
    ap.add_argument("--out", default="runs/rl/sub100m/student640_distilled.pt")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-ce", type=float, default=2.0)
    ap.add_argument("--w-kl", type=float, default=5.0)
    ap.add_argument("--w-lat", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--max-codes", type=int, default=500, help="drop clips longer than model capacity")
    ap.add_argument("--eval-every", type=int, default=120)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260626)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    dev = "cuda"; torch.manual_seed(args.seed)
    w = {"ce": args.w_ce, "kl": args.w_kl, "lat": args.w_lat}

    base = Path(args.base)
    cfg = XttsConfig(); cfg.load_json(str(base / "config.json")); cfg.model_args.gpt_layers = 16
    print("[load] teacher ...", flush=True)
    teacher = Xtts.init_from_config(cfg)
    teacher.load_checkpoint(cfg, checkpoint_path=args.teacher_ckpt, vocab_path=str(base / "vocab.json"), use_deepspeed=False)
    teacher.eval(); teacher.cuda()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("[load] student ...", flush=True)
    ck = torch.load(args.student_init, map_location="cpu")
    gpt = S.build_student_gpt(ck["model_args"], d_student=ck["d_student"], heads=ck["heads"])
    gpt.load_state_dict(ck["gpt"])
    student = S.Student640(gpt, d_student=ck["d_student"])
    student.adapter.load_state_dict(ck["adapter"])
    student.cuda().train()

    # baked conditioning: student 640 (from init), teacher 1024 (recompute from same refs)
    cond_s = {v: ck["voices"][v]["cond_latents"] for v in ck["voices"]}
    keep = ck["keep_idx"].long()
    cond_t = {}
    for v in ck["voices"]:
        gc, _ = teacher.get_conditioning_latents(audio_path=[str(Path(args.refs_dir) / f"{v}.wav")])
        cond_t[v] = gc.cpu()
    print(f"[cond] voices={list(cond_s)}  student{tuple(cond_s[list(cond_s)[0]].shape)} "
          f"teacher{tuple(cond_t[list(cond_s)[0]].shape)}")

    train = torch.load(args.train_data, map_location="cpu")
    ev = torch.load(args.eval_data, map_location="cpu")
    n0 = len(train)
    train = [d for d in train if 8 <= d["n_codes"] <= args.max_codes]
    ev = [d for d in ev if 8 <= d["n_codes"] <= args.max_codes]
    print(f"[data] train={len(train)} (dropped {n0-len(train)} >{args.max_codes} codes) eval={len(ev)}")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler()

    g = torch.Generator().manual_seed(args.seed)
    best = float("inf"); best_state = None; bad = 0; step = 0; t0 = time.perf_counter()
    print(f"[train] {total_steps} steps ({args.epochs}ep x {steps_per_epoch})  bs={args.batch} lr={args.lr}", flush=True)
    for ep in range(args.epochs):
        order = torch.randperm(len(train), generator=g).tolist()
        for bi in range(0, len(order), args.batch):
            items = [train[j] for j in order[bi:bi + args.batch]]
            text, tlen, codes, wavlen, voices = collate(items, dev)
            cs = cond_batch(voices, cond_s, dev); ct = cond_batch(voices, cond_t, dev)
            with torch.no_grad():
                _, lg_t, lt_t, _ = S.forward_both(teacher.gpt, text, tlen, codes, wavlen, ct)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                ce_s, lg_s, lt_s, mask = S.forward_both(student.gpt, text, tlen, codes, wavlen, cs)
                a_s = student.adapter(lt_s)
                kl = masked_kl(lg_s.float(), lg_t.float(), mask, args.T)
                mse, cosd = masked_latent(a_s.float(), lt_t.float(), mask)
                loss = w["ce"] * ce_s + w["kl"] * kl + w["lat"] * (mse + cosd)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1

            if step % 20 == 0:
                dt = time.perf_counter() - t0
                print(f"  s{step}/{total_steps} ep{ep} loss={float(loss):.3f} ce={float(ce_s):.3f} "
                      f"kl={float(kl):.3f} mse={float(mse):.3f} cosd={float(cosd):.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {step/max(dt,1):.2f}it/s", flush=True)
            if step % args.eval_every == 0:
                m = evaluate(student, teacher.gpt, ev, cond_s, cond_t, dev, w, args.T, args.batch)
                flag = ""
                if m["score"] < best - 1e-4:
                    best = m["score"]; bad = 0
                    best_state = {"gpt": {k: v.detach().cpu().clone() for k, v in student.gpt.state_dict().items()},
                                  "adapter": {k: v.detach().cpu().clone() for k, v in student.adapter.state_dict().items()}}
                    flag = " *best"
                else:
                    bad += 1
                print(f"  [eval s{step}] ce={m['ce']:.3f} kl={m['kl']:.3f} mse={m['mse']:.3f} "
                      f"cosd={m['cosd']:.3f} score={m['score']:.4f} bad={bad}{flag}", flush=True)
                if bad >= args.patience:
                    print(f"[early-stop] no eval gain for {bad} evals"); break
        else:
            continue
        break

    # final eval + save best
    if best_state is None:
        best_state = {"gpt": student.gpt.state_dict(), "adapter": student.adapter.state_dict()}
    save = dict(ck)  # keep keep_idx/head_keep/ffn_keep/model_args/voices/d_student/heads
    save["gpt"] = best_state["gpt"]; save["adapter"] = best_state["adapter"]
    save["distill"] = {"best_score": best, "steps": step, "weights": w, "T": args.T,
                       "lr": args.lr, "epochs": args.epochs, "batch": args.batch}
    torch.save(save, args.out)
    print(f"[done] best eval score={best:.4f}  saved -> {args.out}")
    print("DISTILL640_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
