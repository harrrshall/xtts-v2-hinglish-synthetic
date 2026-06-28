#!/usr/bin/env python3
"""Pre-encode the Hinglish corpus to DVAE audio codes + text tokens (once), for the d=640 distillation.

Mirrors GPTTrainer's exact path: wav(22050) -> TorchMelSpectrogram(mel_norm_file, sr=dvae_sr)
-> dvae.get_codebook_indices -> codes.  Output: a .pt list of {tokens, codes, voice, n_codes}.
Decouples the distiller from the Coqui data loader (fully debuggable, no per-step DVAE cost)."""
import argparse, sys
from pathlib import Path
import torch, torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from TTS.tts.layers.xtts.dvae import DiscreteVAE

BASE = ".tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    base = Path(args.base)

    cfg = XttsConfig(); cfg.load_json(str(base / "config.json"))
    dvae_sr = cfg.audio.dvae_sample_rate
    mel_norm = str(base / "mel_stats.pth")
    dvae_ckpt = str(base / "dvae.pth")

    # tokenizer via a (cheap) Xtts shell — only the tokenizer is used
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_dir=str(base), use_deepspeed=False, eval=True)
    tok = model.tokenizer

    mel_fn = TorchMelSpectrogram(mel_norm_file=mel_norm, sampling_rate=dvae_sr).cuda()
    dvae = DiscreteVAE(channels=80, normalization=None, positional_dims=1,
                       num_tokens=cfg.model_args.gpt_num_audio_tokens - 2, codebook_dim=512,
                       hidden_dim=512, num_resnet_blocks=3, kernel_size=3, num_layers=2,
                       use_transposed_convs=False)
    dvae.load_state_dict(torch.load(dvae_ckpt, map_location="cpu"), strict=False)
    dvae.eval().cuda()

    rows = [l.rstrip("\n").split("|") for l in open(args.manifest, encoding="utf-8") if l.strip()]
    if args.max:
        rows = rows[:args.max]

    out, skipped = [], 0
    for i, r in enumerate(rows):
        wavp, text, voice = r[0], r[1], r[2]
        try:
            wav, sr = torchaudio.load(wavp)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != dvae_sr:
                wav = torchaudio.functional.resample(wav, sr, dvae_sr)
            with torch.no_grad():
                mel = mel_fn(wav.cuda())                       # (1, 80, T)
                codes = dvae.get_codebook_indices(mel)[0].cpu()  # (L,)
            tokens = torch.IntTensor(tok.encode(text, lang="hi"))
            out.append({"tokens": tokens, "codes": codes.short(), "voice": voice,
                        "n_codes": int(codes.numel()), "n_text": int(tokens.numel()), "text": text})
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  SKIP {wavp}: {type(e).__name__} {str(e)[:60]}")
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(rows)} encoded (skipped {skipped})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    lens = [d["n_codes"] for d in out]
    print(f"saved {len(out)} clips -> {args.out}  (skipped {skipped}) "
          f"codes min/med/max = {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
    print("PREENCODE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
