#!/bin/bash
# Round-1 offline RFT, SINGLE GPU (GPU 6). Each step writes directly to a step log (unbuffered, no pipe)
# so progress is visible in real time. gen+select on the full prompt set, then CE fine-tune + SFT-replay anchor.
set -e
cd /mnt/data/harshal/syntts
export PYTHONUNBUFFERED=1
PY=.venv_xtts/bin/python
REF16=runs/rl/ref_16L/model.pth
XD=.tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2
REFS=runs/xtts_hinglish/RELEASE/refs
R=data/rl/round1
NPROMPTS=${NPROMPTS:-360}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
mkdir -p $R

echo "=== [1] build $NPROMPTS code-switch prompts ==="
$PY scripts/hinglish/rl/build_prompts.py --out $R/prompts.jsonl --n $NPROMPTS

echo "=== [2] roll out N=8 candidates (GPU $CUDA_VISIBLE_DEVICES) -> $R/gen.log ==="
$PY scripts/hinglish/rl/gen_candidates.py --base $XD --ckpt $REF16 --gpt-layers 16 \
    --prompts $R/prompts.jsonl --refs-dir $REFS --out-dir $R/cand --n 8 --temp 0.95 > $R/gen.log 2>&1
echo "gen done: $(ls $R/cand/*.wav 2>/dev/null | wc -l) wavs; $(grep -E 'model on|clips' $R/gen.log | tail -2)"

echo "=== [3] select winners (self-calibrated 16L floors, min-gain 0.01) -> $R/select.log ==="
$PY scripts/hinglish/rl/select_winners.py --candidates $R/cand/candidates.jsonl --refs-dir $REFS \
    --out-corpus $R/winners.csv --out-pairs $R/pairs.jsonl --out-report $R/report.json \
    --out-floors $R/floors_16L.json --min-gain 0.01 > $R/select.log 2>&1
grep -E "select" $R/select.log | tail -2
NW=$(grep -vc "^$" $R/winners.csv)
echo "winners=$NW"

echo "=== [4] SFT-replay anchor (spread sample of original corpus ~= winners) ==="
awk "NR % 7 == 0" data/xtts/metadata_train.csv | head -$NW > $R/replay.csv
grep -v "^$" $R/winners.csv > $R/train.csv; cat $R/replay.csv >> $R/train.csv
echo "train rows=$(grep -vc "^$" $R/train.csv) (winners=$NW + replay=$(wc -l < $R/replay.csv))"

echo "=== [5] CE fine-tune (warm-restore 16L, gpt-layers 16, lr 2e-6, 4 epochs) -> $R/train.log ==="
$PY scripts/hinglish/train_xtts.py --gpt-layers 16 --restore $REF16 \
    --metadata-train $R/train.csv --metadata-eval data/xtts/metadata_eval.csv \
    --out-path runs/rl/round1_rft --epochs 4 --batch-size 4 --grad-accum 8 --lr 2e-6 > $R/train.log 2>&1
grep -iE "EPOCH|BEST|DONE|avg_loss" $R/train.log | tail -8

CK=$(ls -t runs/rl/round1_rft/*/best_model*.pth 2>/dev/null | head -1)
echo "=== ROUND1_DONE winners=$NW ckpt=$CK ==="
test -n "$CK" && echo "ROUND1_OK" || echo "ROUND1_FAIL"
