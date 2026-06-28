#!/usr/bin/env python3
"""Task #14 RFT: SFT the d=640 student on its OWN best rollouts (winners) to suppress the runaway-
generation tail + lock in code-switch faithfulness. CE-only on (winner codes + distill replay anchor);
the adapter gets no gradient (CE flows GPT->mel_head only), so timbre is untouched.

Winners come from select_winners (passed ALL floors: not degenerate, UTMOS/SECS/F0/energy >= base,
duration in tol) so we never train toward a babbling/flat/voice-drifted sample. Replay anchors to the
clean distillation distribution to avoid catastrophic forgetting at this small LR."""
import argparse, math, sys, time
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import student640 as S


def collate(items, cond_map, dev):
    b = len(items)
    nt = max(it["n_text"] for it in items); nc = max(it["n_codes"] for it in items)
    text = torch.zeros(b, nt, dtype=torch.long); codes = torch.zeros(b, nc, dtype=torch.long)
    tlen = torch.zeros(b, dtype=torch.long); wavlen = torch.zeros(b, dtype=torch.long); voices = []
    for i, it in enumerate(items):
        t = it["tokens"].long(); c = it["codes"].long()
        text[i, :t.numel()] = t; codes[i, :c.numel()] = c
        tlen[i] = t.numel(); wavlen[i] = it["n_codes"] * 1024; voices.append(it["voice"])
    cond = torch.cat([cond_map[v].to(dev) for v in voices], dim=0)
    return text.to(dev), tlen.to(dev), codes.to(dev), wavlen.to(dev), cond


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="runs/rl/sub100m/student640b_distilled.pt")
    ap.add_argument("--winners", required=True, help="pre-encoded winner codes .pt (from distill_preencode on winners.csv)")
    ap.add_argument("--replay", default="runs/rl/sub100m/distill_train_full.pt")
    ap.add_argument("--replay-frac", type=float, default=1.0, help="replay clips per winner clip")
    ap.add_argument("--out", default="runs/rl/sub100m/student640b_rft.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-codes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260626)
    args = ap.parse_args()
    assert torch.cuda.is_available(); dev = "cuda"; torch.manual_seed(args.seed)

    ck = torch.load(args.student, map_location="cpu")
    gpt = S.build_student_gpt(ck["model_args"], d_student=ck["d_student"], heads=ck["heads"])
    gpt.load_state_dict(ck["gpt"])
    student = S.Student640(gpt, d_student=ck["d_student"]); student.adapter.load_state_dict(ck["adapter"])
    student.cuda().train()
    cond = {v: ck["voices"][v]["cond_latents"] for v in ck["voices"]}

    win = [d for d in torch.load(args.winners, map_location="cpu") if 8 <= d["n_codes"] <= args.max_codes]
    rep_all = [d for d in torch.load(args.replay, map_location="cpu") if 8 <= d["n_codes"] <= args.max_codes]
    g = torch.Generator().manual_seed(args.seed)
    n_rep = min(len(rep_all), int(len(win) * args.replay_frac))
    rep = [rep_all[i] for i in torch.randperm(len(rep_all), generator=g)[:n_rep].tolist()]
    data = win + rep
    print(f"[rft] winners={len(win)} replay={len(rep)} total={len(data)}  lr={args.lr} ep={args.epochs}", flush=True)

    opt = torch.optim.AdamW(student.gpt.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    total = math.ceil(len(data) / args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=args.lr * 0.1)
    scaler = torch.cuda.amp.GradScaler()
    step = 0; t0 = time.perf_counter()
    for ep in range(args.epochs):
        order = torch.randperm(len(data), generator=g).tolist()
        for bi in range(0, len(order), args.batch):
            items = [data[j] for j in order[bi:bi + args.batch]]
            text, tlen, codes, wavlen, cb = collate(items, cond, dev)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                ce, _, _, _ = S.forward_both(student.gpt, text, tlen, codes, wavlen, cb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(ce).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(student.gpt.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step % 20 == 0:
                print(f"  s{step}/{total} ep{ep} ce={float(ce):.3f} lr={sched.get_last_lr()[0]:.2e} "
                      f"{step/max(time.perf_counter()-t0,1):.2f}it/s", flush=True)

    save = dict(ck)
    save["gpt"] = student.gpt.state_dict(); save["adapter"] = student.adapter.state_dict()
    save["rft"] = {"winners": len(win), "replay": len(rep), "lr": args.lr, "epochs": args.epochs}
    torch.save(save, args.out)
    print(f"[done] saved -> {args.out}")
    print("RFT_FINETUNE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
