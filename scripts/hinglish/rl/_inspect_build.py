#!/usr/bin/env python3
"""One-off: inspect channel_importance.pt + the round1 checkpoint tensor shapes
so we know exactly which weights depend on d=1024 before slicing to d=640."""
import torch, itertools, sys

CHAN = "runs/rl/sub100m/channel_importance.pt"
CKPT = "runs/rl/round1_rft/xtts_hinglish-June-25-2026_06+39PM-0000000/best_model.pth"

d = torch.load(CHAN, map_location="cpu")
print("CHAN keys:", list(d.keys()))
for k, v in d.items():
    if torch.is_tensor(v):
        print("  ", k, tuple(v.shape), v.dtype, "min", float(v.min()) if v.numel() else None,
              "max", float(v.max()) if v.numel() else None)
    else:
        print("  ", k, type(v).__name__, (len(v) if hasattr(v, "__len__") else v))

print("\n=== CKPT ===")
ck = torch.load(CKPT, map_location="cpu")
print("top keys:", list(ck.keys())[:10])
sd = ck["model"] if "model" in ck else ck
print("n tensors:", len(sd))

# group tensors by dependence on hidden dim 1024
D = 1024
print("\n--- tensors touching d=1024 (in any axis) ---")
groups = {}
for k, v in sd.items():
    if not torch.is_tensor(v):
        continue
    axes = [i for i, s in enumerate(v.shape) if s == D]
    if axes:
        # bucket by suffix pattern (strip gpt.h.N. layer index)
        import re
        key = re.sub(r"gpt\.gpt\.h\.\d+\.", "gpt.gpt.h.N.", k)
        key = re.sub(r"\.h\.\d+\.", ".h.N.", key)
        groups.setdefault(key, (tuple(v.shape), axes))

for k in sorted(groups):
    shp, axes = groups[k]
    print(f"  {k:55s} shape={shp} d-axes={axes}")

print("\n--- key singletons (embeddings / heads / norms) ---")
for k in sd:
    if any(t in k for t in ["text_embedding", "mel_embedding", "text_head", "mel_head",
                            "ln_f", "pos_embedding", "conditioning", "perceiver", "wpe", "wte"]):
        if torch.is_tensor(sd[k]):
            print(f"  {k:60s} {tuple(sd[k].shape)}")

# config inside checkpoint?
if "config" in ck:
    cfg = ck["config"]
    for kk in ("model_args", "gpt_number_text_tokens", "gpt_n_model_channels", "gpt_layers"):
        if isinstance(cfg, dict) and kk in cfg:
            print("CFG", kk, cfg[kk])
