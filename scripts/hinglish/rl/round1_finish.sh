#!/bin/bash
# Finish round-1 from already-generated candidates: parallel-score -> SFT-replay anchor -> CE fine-tune.
set -e
cd /mnt/data/harshal/syntts
export PYTHONUNBUFFERED=1
PY=.venv_xtts/bin/python
REF16=runs/rl/ref_16L/model.pth
R=data/rl/round1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}

echo "=== [3p] parallel winner selection (8 scorers on GPU $CUDA_VISIBLE_DEVICES) ==="
bash scripts/hinglish/rl/score_parallel.sh $R/cand/candidates.jsonl $R 8
NW=$(grep -vc "^$" $R/winners.csv)
echo "winners=$NW"

echo "=== [4] SFT-replay anchor ==="
awk "NR % 7 == 0" data/xtts/metadata_train.csv | head -$NW > $R/replay.csv
grep -v "^$" $R/winners.csv > $R/train.csv; cat $R/replay.csv >> $R/train.csv
echo "train rows=$(grep -vc '^$' $R/train.csv) (winners=$NW + replay=$(wc -l < $R/replay.csv))"

echo "=== [5] CE fine-tune (warm-restore 16L, lr 2e-6, 4 epochs) -> $R/train.log ==="
$PY scripts/hinglish/train_xtts.py --gpt-layers 16 --restore $REF16 \
    --metadata-train $R/train.csv --metadata-eval data/xtts/metadata_eval.csv \
    --out-path runs/rl/round1_rft --epochs 4 --batch-size 4 --grad-accum 8 --lr 2e-6 > $R/train.log 2>&1
grep -iE "EPOCH|BEST|DONE|avg_loss" $R/train.log | tail -8

CK=$(ls -t runs/rl/round1_rft/*/best_model*.pth 2>/dev/null | head -1)
echo "=== ROUND1_DONE winners=$NW ckpt=$CK ==="
test -n "$CK" && echo "ROUND1_OK" || echo "ROUND1_FAIL"
