#!/bin/bash
# Model-averaging sweep: theta(alpha)=(1-alpha)*16L + alpha*round1. Eval each on held-out 89.
# Endpoints already measured: alpha=0 -> data/eval_16L (UTMOS 3.141 / acc 0.805); alpha=1 -> data/eval_round1 (3.104 / 0.837).
set -e
cd /mnt/data/harshal/syntts
export PYTHONUNBUFFERED=1
XD=.tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2
REFS=runs/xtts_hinglish/RELEASE/refs
MAN=data/eval_heldout89.jsonl
REFSJSON='{"kaustubh":"runs/xtts_hinglish/RELEASE/refs/kaustubh.wav","arjun":"runs/xtts_hinglish/RELEASE/refs/arjun.wav","maya":"runs/xtts_hinglish/RELEASE/refs/maya.wav","aadya":"runs/xtts_hinglish/RELEASE/refs/aadya.wav"}'
B16=runs/rl/ref_16L/model.pth
RL=$(ls -t runs/rl/round1_rft/*/best_model*.pth | head -1)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}

for A in 0.5 0.7; do
  TAG=a${A/./}
  CK=runs/rl/avg_$TAG/model.pth
  D=data/eval_avg_$TAG
  echo "=== alpha=$A : build averaged checkpoint ==="
  .venv_xtts/bin/python scripts/hinglish/rl/model_average.py --base $B16 --rl "$RL" --alpha $A --out $CK 2>&1 | tail -1
  echo "=== alpha=$A : gen panel ==="
  .venv_xtts/bin/python scripts/hinglish/compare/gen_panel_ckpt.py --base $XD --ckpt "$CK" --gpt-layers 16 \
      --eval-manifest $MAN --refs-dir $REFS --out-dir $D/wav --label cand > $D.gen.log 2>&1 || true
  echo "=== alpha=$A : score ==="
  .venv_eval/bin/python scripts/hinglish/09_objective_eval.py --manifest $D/wav/manifest.jsonl --label avg$A \
      --voice-refs "$REFSJSON" --out $D/obj.json 2>&1 | tail -1
  .venv_xtts/bin/python scripts/hinglish/10_accent_eval.py --manifest $D/wav/manifest.jsonl --label avg$A \
      --out $D/accent.json 2>&1 | tail -1
  .venv_xtts/bin/python scripts/hinglish/rl/pitch_monitor.py --cand $D/wav/manifest.jsonl \
      --ref data/eval_16L/wav/manifest.jsonl --label avg$A 2>&1 | grep -E "paired|expressivity"
done
echo "ALPHA_SWEEP_DONE"
