"""Shared helpers for ONNX export of the narrowed XTTS GPT-2.

The export-killer in transformers 4.5x is `masking_utils._vmap_for_bhqkv`: building the 4D
causal mask with `torch.vmap`, which TorchScript tracing cannot handle ("unordered_map::at").

Fix (verified against modeling_gpt2.eager_attention_forward): force eager attention and make
`create_causal_mask` return None. In eager mode GPT-2 applies its own precomputed triangular
`module.bias` buffer via torch.where — exactly correct for our batch=1, no-padding, pure-causal
case (full forward, prefill, AND single-token decode all slice the right rows of the buffer).
"""
import torch
import torch.nn as nn


def prep_gpt2_for_export(gpt2_model, fp16_safe_wpe=True):
    """Force eager attention on a GPT2Model (and submodules) and neutralize the vmap mask path.

    fp16_safe_wpe: replace the nulled `wpe` (a functools.partial returning a float32 zeros constant,
    which breaks fp16 conversion by feeding a mixed-type Add) with a real zero-weight nn.Embedding.
    Adds the same zeros, but as a Gather over an initializer the fp16 converter handles cleanly.
    """
    import transformers.models.gpt2.modeling_gpt2 as g2
    # return None -> eager attention falls back to the triangular module.bias buffer (no vmap)
    g2.create_causal_mask = lambda *a, **k: None
    if fp16_safe_wpe and not isinstance(getattr(gpt2_model, "wpe", None), nn.Embedding):
        wpe = nn.Embedding(gpt2_model.config.n_positions, gpt2_model.config.n_embd)
        nn.init.zeros_(wpe.weight)
        gpt2_model.wpe = wpe
    gpt2_model.config._attn_implementation = "eager"
    if hasattr(gpt2_model, "_attn_implementation"):
        gpt2_model._attn_implementation = "eager"
    for sub in gpt2_model.modules():
        if hasattr(sub, "config"):
            sub.config._attn_implementation = "eager"
        if hasattr(sub, "_attn_implementation"):
            try:
                sub._attn_implementation = "eager"
            except Exception:
                pass
    return gpt2_model


def max_abs(a, b):
    import numpy as np
    n = min(a.shape[0], b.shape[0]) if a.ndim == 1 else None
    return float(np.abs(a - b).max())
