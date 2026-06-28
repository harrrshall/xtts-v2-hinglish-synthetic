#!/bin/bash
# Stronger ON-POLICY DPO escalation: regenerate candidates FROM round-1 (not 16L), build high-contrast
# pairs, train a stronger DPO, then certify vs the 443M teacher.
set -e
cd /mnt/data/harshal/syntts
export PYTHONUNBUFFERED=1
PY=.venv_xtts/bin/python
XD=.tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2
REFS=runs/xtts_hinglish/RELEASE/refs
R1=$(ls -t runs/rl/round1_rft/*/best_model*.pth | head -1)
D=data/rl/dpo2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
mkdir -p $D

echo "=== [1] on-policy candidates from round-1 (N=8, 360 prompts) ==="
$PY scripts/hinglish/rl/gen_candidates.py --base $XD --ckpt "$R1" --gpt-layers 16 \
    --prompts data/rl/round1/prompts.jsonl --refs-dir $REFS --out-dir $D/cand --n 8 --temp 0.95 \
    > $D/gen.log 2>&1
echo "cand wavs: $(ls $D/cand/*.wav 2>/dev/null | wc -l)"

echo "=== [2] high-contrast pairs (parallel scorer) ==="
bash scripts/hinglish/rl/score_parallel.sh $D/cand/candidates.jsonl $D 8 > $D/score.log 2>&1
echo "pairs: $(grep -c . $D/pairs.jsonl 2>/dev/null || echo 0)"

echo "=== [3] stronger DPO (beta 3.0, lambda_ce 0.05, lr 1e-6, 3 ep) from round-1 ==="
$PY scripts/hinglish/rl/dpo_trainer.py --round1-ckpt "$R1" --pairs $D/pairs.jsonl \
    --out runs/rl/dpo2/model.pth --beta 3.0 --lambda-ce 0.05 --lr 1e-6 --epochs 3 --accum 8 > $D/dpo.log 2>&1
grep -E "\[dpo\] DONE|chosen>rej" $D/dpo.log | tail -2

echo "=== [4] certify dpo2 vs 443M teacher ==="
bash scripts/hinglish/rl/cert_ckpt.sh runs/rl/dpo2/model.pth dpo2
echo "ESCALATE_DPO_DONE"
