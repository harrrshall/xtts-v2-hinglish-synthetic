#!/usr/bin/env python3
"""Length-normalized DPO to close the last accent residual on the round-1 RFT student.

Reuses the coqui GPTTrainer machinery (DVAE, mel extractors, tokenizer, format_batch_on_device, the GPT
forward) so the per-sequence logprob is the model's OWN computation: the GPT forward returns
(loss_text, loss_mel, mel_logits) where loss_mel = mean CE over the audio codes = -mean_token_logprob.
With batch=1 that scalar IS the length-normalized sequence logprob of the audio. No fragile manual
masking/gather -> low-risk.

DPO loss (Rafailov et al., length-normalized / SimPO-style ratio):
  L = -log sigmoid( beta * [ (lp_chosen_pol - lp_chosen_ref) - (lp_rejected_pol - lp_rejected_ref) ] )
      + lambda_ce * CE_chosen_pol           # MPO-style CE anchor on the chosen (keeps audio quality)
Reference = frozen round-1 (the model we refine), so KL stays small and expressivity is protected.
"""
import argparse, copy, json
from pathlib import Path
import torch
import torch.nn.functional as F

from trainer import TrainerArgs  # noqa: F401  (ensures coqui trainer importable)
from TTS.config.shared_configs import BaseDatasetConfig  # noqa: F401
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig

CKPT = Path(".tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2")


def load_full(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d


def gpt_state_from_ckpt(ckpt):
    """Bare gpt-module state dict ('gpt.h.*') from a GPTTrainer/Xtts checkpoint (strip xtts./gpt. prefix)."""
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    out = {}
    for k, v in sd.items():
        kk = k[len("xtts."):] if k.startswith("xtts.") else k
        if kk.startswith("gpt."):
            out[kk[len("gpt."):]] = v
    return out


def seq_logp_and_ce(gpt, b):
    """Return (length-normalized logp, CE) for a batch-1 item using the GPT's own forward."""
    _, loss_mel, _ = gpt(b["text_inputs"], b["text_lengths"], b["audio_codes"], b["wav_lengths"],
                         cond_mels=b["cond_mels"], cond_idxs=b["cond_idxs"], cond_lens=b["cond_lens"])
    return -loss_mel, loss_mel


def build_trainer():
    common = dict(
        max_conditioning_length=132300, min_conditioning_length=66150,
        max_wav_length=255995, max_text_length=200,
        mel_norm_file=str(CKPT / "mel_stats.pth"), dvae_checkpoint=str(CKPT / "dvae.pth"),
        xtts_checkpoint=str(CKPT / "model.pth"), tokenizer_file=str(CKPT / "vocab.json"),
        gpt_num_audio_tokens=1026, gpt_start_audio_token=1024, gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True, gpt_layers=16)
    model_args = GPTArgs(**common)
    audio = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    config = GPTTrainerConfig(output_path="runs/rl/dpo_tmp", model_args=model_args, audio=audio,
                              run_name="dpo", batch_size=1, eval_batch_size=1, num_loader_workers=2,
                              num_eval_loader_workers=2, run_eval=True)
    return GPTTrainer.init_from_config(config), config


def loaders_for(trainer, config, samples):
    return trainer.get_data_loader(config, {}, is_eval=True, samples=samples, verbose=False, num_gpus=1)


def to_dev(b, dev):
    for k in list(b):
        if torch.is_tensor(b[k]):
            b[k] = b[k].to(dev)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round1-ckpt", required=True)
    ap.add_argument("--pairs", required=True, help="jsonl with chosen_wav/rejected_wav/ref_text/voice")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--lambda-ce", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = "cuda"
    trainer, config = build_trainer()
    # policy = round-1 weights into the trainer's gpt
    gsd = gpt_state_from_ckpt(load_full(args.round1_ckpt))
    miss, unexp = trainer.xtts.gpt.load_state_dict(gsd, strict=False)
    print(f"[dpo] policy loaded gpt tensors={len(gsd)} missing={len(miss)} unexpected={len(unexp)}")
    trainer.to(dev)
    policy = trainer.xtts.gpt
    ref = copy.deepcopy(policy).to(dev).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    # robust to concatenated/merged jsonl (shard cat without trailing newlines)
    pairs, dec, txt, i = [], json.JSONDecoder(), open(args.pairs, encoding="utf-8").read(), 0
    while i < len(txt):
        while i < len(txt) and txt[i] in " \n\r\t":
            i += 1
        if i >= len(txt):
            break
        obj, end = dec.raw_decode(txt, i)
        pairs.append(obj); i = end
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    chosen = [{"text": p["ref_text"], "audio_file": p["chosen_wav"], "speaker_name": p["voice"],
               "root_path": "/", "language": "hi"} for p in pairs]
    rejected = [{"text": p["ref_text"], "audio_file": p["rejected_wav"], "speaker_name": p["voice"],
                 "root_path": "/", "language": "hi"} for p in pairs]
    cl = loaders_for(trainer, config, chosen)
    rl = loaders_for(trainer, config, rejected)

    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr,
                            betas=(0.9, 0.96), eps=1e-8, weight_decay=1e-2)
    policy.eval()  # no dropout -> clean logp margin (policy==ref => margin 0 at init); grads still flow
    step = wins = seen = 0
    opt.zero_grad()
    for ep in range(args.epochs):
        for bc, br in zip(cl, rl):
            try:
                bc = trainer.format_batch_on_device(to_dev(bc, dev))
                br = trainer.format_batch_on_device(to_dev(br, dev))
            except Exception as e:
                print(f"[dpo] skip batch: {type(e).__name__} {str(e)[:50]}"); continue
            # conditioning supplied via perceiver cond_mels; skip the sample-unit cond-overlap masking
            # (cond_idxs in samples would mask all mel frames -> NaN CE)
            bc["cond_idxs"] = None; br["cond_idxs"] = None
            lp_c_pol, ce_c = seq_logp_and_ce(policy, bc)
            lp_r_pol, _ = seq_logp_and_ce(policy, br)
            with torch.no_grad():
                lp_c_ref, _ = seq_logp_and_ce(ref, bc)
                lp_r_ref, _ = seq_logp_and_ce(ref, br)
            margin = (lp_c_pol - lp_c_ref) - (lp_r_pol - lp_r_ref)
            loss = -F.logsigmoid(args.beta * margin) + args.lambda_ce * ce_c
            (loss / args.accum).backward()
            seen += 1
            wins += int((lp_c_pol - lp_r_pol).item() > 0)
            if seen % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step(); opt.zero_grad(); step += 1
            if seen % 20 == 0 or (args.smoke and seen <= 8):
                print(f"[dpo] ep{ep} seen={seen} step={step} loss={float(loss):.4f} "
                      f"margin={float(margin):.4f} chosen>rej={wins}/{seen} ce_c={float(ce_c):.3f}")
            if args.smoke and seen >= 8:
                break
        if args.smoke:
            break
    if seen % args.accum != 0:
        opt.step(); opt.zero_grad()

    # save in the round-1 checkpoint format with updated gpt weights (so gen_panel_ckpt can load it)
    out_ckpt = load_full(args.round1_ckpt)
    sd = out_ckpt.get("model", out_ckpt)
    new = {("gpt." + k): v.detach().cpu() for k, v in policy.state_dict().items()}
    n = 0
    for k in list(sd):
        kk = k[len("xtts."):] if k.startswith("xtts.") else k
        if kk in new and sd[k].shape == new[kk].shape:
            sd[k] = new[kk].to(sd[k].dtype); n += 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.out)
    print(f"[dpo] DONE seen={seen} steps={step} chosen>rej={wins}/{seen} updated {n} gpt tensors -> {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
