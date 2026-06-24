#!/usr/bin/env python3
"""Fine-tune XTTS-v2 on the Hinglish synthetic corpus (multi-speaker, 4 teacher voices).

Adapted from the coqui XTTS-v2 GPT fine-tune recipe. Reads our pipe-delimited metadata
(audio_path|text|speaker), conditions per-speaker, language token "hi" (Hinglish base script).

SMOKE MODE (--smoke): tiny subset + 1 short epoch to confirm the training loop runs clean on GPU
before committing to the full run. ALWAYS smoke-test first.

Run on the box (GPU pinned by the caller), e.g.:
  CUDA_VISIBLE_DEVICES=5 .venv_xtts/bin/python scripts/hinglish/train_xtts.py \
      --smoke
  CUDA_VISIBLE_DEVICES=5 .venv_xtts/bin/python scripts/hinglish/train_xtts.py \
      --epochs 8 --batch-size 8 --grad-accum 4
"""
from __future__ import annotations
import argparse, csv, os
from pathlib import Path

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig

CKPT = Path(".tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2")
LANG = "hi"


def csv_formatter(root_path, meta_file, ignored_speakers=None):
    items = []
    with open(meta_file, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) < 3:
                continue
            audio, text, spk = row[0], row[1], row[2]
            if not text.strip():
                continue
            items.append({
                # audio paths are absolute; root_path="/" so add_extra_keys' relative_to works
                "text": text, "audio_file": audio, "speaker_name": spk,
                "root_path": "/", "language": LANG,
            })
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-train", default="data/xtts/metadata_train.csv")
    ap.add_argument("--metadata-eval", default="data/xtts/metadata_eval.csv")
    ap.add_argument("--out-path", default="runs/xtts_hinglish")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-samples", type=int, default=0, help="cap train samples (0=all)")
    args = ap.parse_args()

    Path(args.out_path).mkdir(parents=True, exist_ok=True)

    dataset = BaseDatasetConfig(
        formatter="hinglish_csv", dataset_name="hinglish",
        path=str(Path(args.metadata_train).parent),
        meta_file_train=args.metadata_train, meta_file_val=args.metadata_eval, language=LANG,
    )

    model_args = GPTArgs(
        max_conditioning_length=132300, min_conditioning_length=66150,
        max_wav_length=255995, max_text_length=200,
        mel_norm_file=str(CKPT / "mel_stats.pth"),
        dvae_checkpoint=str(CKPT / "dvae.pth"),
        xtts_checkpoint=str(CKPT / "model.pth"),
        tokenizer_file=str(CKPT / "vocab.json"),
        gpt_num_audio_tokens=1026, gpt_start_audio_token=1024, gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True,
    )
    audio = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    config = GPTTrainerConfig(
        output_path=args.out_path,
        model_args=model_args, audio=audio,
        run_name="xtts_hinglish", project_name="hinglish_tts",
        run_description="XTTS-v2 fine-tune on synthetic Hinglish (4 fixed voices)",
        epochs=(1 if args.smoke else args.epochs),
        batch_size=(2 if args.smoke else args.batch_size),
        eval_batch_size=(2 if args.smoke else args.batch_size),
        batch_group_size=0 if args.smoke else 48,
        num_loader_workers=4 if args.smoke else 8,
        eval_split_max_size=64,
        print_step=5 if args.smoke else 50,
        plot_step=100, log_model_step=1000,
        save_step=200 if args.smoke else 2000,
        save_n_checkpoints=1, save_checkpoints=True,
        optimizer="AdamW", optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=args.lr, lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [900000, 2700000, 5400000], "gamma": 0.5},
    )

    train_samples, eval_samples = load_tts_samples(
        [dataset], eval_split=True, formatter=csv_formatter,
        eval_split_max_size=config.eval_split_max_size, eval_split_size=0.01,
    )
    if args.smoke:
        train_samples = train_samples[:24]
        eval_samples = eval_samples[:4]
    elif args.max_samples:
        train_samples = train_samples[:args.max_samples]
    print(f"[train_xtts] train={len(train_samples)} eval={len(eval_samples)} smoke={args.smoke} "
          f"epochs={config.epochs} bs={config.batch_size} grad_accum={args.grad_accum}")

    model = GPTTrainer.init_from_config(config)
    trainer = Trainer(
        TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False,
                    grad_accum_steps=(1 if args.smoke else args.grad_accum)),
        config, output_path=args.out_path, model=model,
        train_samples=train_samples, eval_samples=eval_samples,
    )
    trainer.fit()
    print("[train_xtts] DONE. checkpoints in", args.out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
