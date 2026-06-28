#!/usr/bin/env python3
"""Activation-based channel importance for width-pruning the 1024-dim residual stream -> top-640.
Minitron-style: run the teacher on a Hinglish calibration set, aggregate per-channel mean|activation|
across all layers + positions + data. One global hidden-channel ranking (the residual stream is shared,
so the same channel index must be kept everywhere). Output feeds the d=640 student init."""
import sys, csv, json
from pathlib import Path
import torch
sys.path.insert(0, "scripts/hinglish/rl")
from dpo_trainer import build_trainer, gpt_state_from_ckpt, load_full, loaders_for, to_dev  # noqa: E402
import glob

N_CALIB = int(sys.argv[1]) if len(sys.argv) > 1 else 384
KEEP = 640
OUT = Path("runs/rl/sub100m"); OUT.mkdir(parents=True, exist_ok=True)

dev = "cuda"
trainer, config = build_trainer()
ck = sorted(glob.glob("runs/rl/round1_rft/*/best_model*.pth"))[-1]
trainer.xtts.gpt.load_state_dict(gpt_state_from_ckpt(load_full(ck)), strict=False)
trainer.to(dev); trainer.xtts.gpt.eval()
print(f"[imp] teacher loaded from {ck}; model_dim={trainer.xtts.gpt.model_dim}")

# capture inner-GPT2 hidden states (inject output_hidden_states like distill_trainer)
store = {}
def pre(mod, args, kwargs):
    kwargs["output_hidden_states"] = True; kwargs["return_dict"] = True; return args, kwargs
def cap(mod, inp, out):
    store["hs"] = out.hidden_states
trainer.xtts.gpt.gpt.register_forward_pre_hook(pre, with_kwargs=True)
trainer.xtts.gpt.gpt.register_forward_hook(cap)

# calibration set: code-switch-leaning sample of the training corpus
rows = []
for r in csv.reader(open("data/xtts/metadata_train.csv"), delimiter="|"):
    if len(r) >= 3 and r[1].strip():
        rows.append({"text": r[1], "audio_file": r[0], "speaker_name": r[2], "root_path": "/", "language": "hi"})
rows = rows[::max(1, len(rows) // N_CALIB)][:N_CALIB]
loader = loaders_for(trainer, config, rows)
print(f"[imp] calibration utts: {len(rows)}")

imp = torch.zeros(trainer.xtts.gpt.model_dim, device=dev)
cnt = 0
with torch.no_grad():
    for i, b in enumerate(loader):
        try:
            b = trainer.format_batch_on_device(to_dev(b, dev)); b["cond_idxs"] = None
            trainer.xtts.gpt(b["text_inputs"], b["text_lengths"], b["audio_codes"], b["wav_lengths"],
                             cond_mels=b["cond_mels"], cond_idxs=None, cond_lens=b["cond_lens"])
        except Exception as e:
            print(f"  skip {i}: {type(e).__name__} {str(e)[:40]}"); continue
        for hs in store.get("hs", []):                       # each [B, S, d]
            imp += hs.abs().float().sum(dim=(0, 1))          # per-channel L1
            cnt += hs.shape[0] * hs.shape[1]
        if (i + 1) % 50 == 0:
            print(f"  {i+1} batches, cnt={cnt}")

imp = (imp / max(cnt, 1)).cpu()
order = torch.argsort(imp, descending=True)
keep = torch.sort(order[:KEEP]).values                       # keep top-640, in ascending index order
torch.save({"importance": imp, "keep_idx": keep, "n_calib": len(rows), "model_dim": int(imp.numel())},
           OUT / "channel_importance.pt")
print(f"[imp] kept {KEEP}/{imp.numel()} channels; imp range [{imp.min():.4f}, {imp.max():.4f}] "
      f"median {imp.median():.4f}")
print(f"[imp] top-640 capture {100*imp[keep].sum()/imp.sum():.1f}% of total activation mass")
print(f"[imp] saved -> {OUT/'channel_importance.pt'}")
print("CHANNEL_IMPORTANCE_DONE")
