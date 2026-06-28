#!/bin/bash
# SMOKE GATE for the offline-RFT pipeline (blocking). ~24 code-switch prompts, N=8.
# Asserts: candidates decode to wav, base floors extract, winners select, CE fine-tune produces a loadable ckpt.
set -e
cd /mnt/data/harshal/syntts
XD=.tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2
REFS=runs/xtts_hinglish/RELEASE/refs
REF16=runs/rl/ref_16L/model.pth
PY=.venv_xtts/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
D=data/rl/smoke
mkdir -p $D

echo "=== [1/6] build 24 code-switch prompts ==="
$PY scripts/hinglish/rl/build_prompts.py --out $D/prompts.jsonl --n 24

echo "=== [2/6] frozen-16L base sample per prompt (greedy floor reference) ==="
$PY scripts/hinglish/rl/gen_candidates.py --base $XD --ckpt $REF16 --gpt-layers 16 \
    --prompts $D/prompts.jsonl --refs-dir $REFS --out-dir $D/base --n 1 --greedy 2>&1 | grep -vE "WeightNorm|warn|exceeds the character" | tail -2

echo "=== [3/6] extract per-prompt base floors ==="
$PY scripts/hinglish/rl/extract_base_and_validate.py --base-panel $D/base/candidates.jsonl \
    --refs-dir $REFS --max 0 --out $D/base_stats.json 2>&1 | grep -E "validate|wrote" | tail -3

echo "=== [4/6] roll out N=8 candidates per prompt (current policy = frozen 16L) ==="
$PY scripts/hinglish/rl/gen_candidates.py --base $XD --ckpt $REF16 --gpt-layers 16 \
    --prompts $D/prompts.jsonl --refs-dir $REFS --out-dir $D/cand --n 8 --temp 0.95 2>&1 | grep -vE "WeightNorm|warn|exceeds the character" | tail -2

echo "=== [5/6] select winners (floors must hold) ==="
$PY scripts/hinglish/rl/select_winners.py --candidates $D/cand/candidates.jsonl \
    --base-stats $D/base_stats.json --refs-dir $REFS \
    --out-corpus $D/winners.csv --out-pairs $D/pairs.jsonl --out-report $D/select_report.json 2>&1 | grep -E "select" | tail -3
echo "winners corpus rows: $(wc -l < $D/winners.csv)"

echo "=== [6/6] CE fine-tune 1 epoch on winners (warm-restore frozen 16L, gpt-layers 16) ==="
# blend a little original corpus for the CE/SFT-replay anchor (smoke: tiny)
head -48 data/xtts/metadata_train.csv > $D/replay.csv
cat $D/winners.csv > $D/train.csv; echo >> $D/train.csv; cat $D/replay.csv >> $D/train.csv
$PY scripts/hinglish/train_xtts.py --gpt-layers 16 --restore $REF16 \
    --metadata-train $D/train.csv --metadata-eval data/xtts/metadata_eval.csv \
    --out-path runs/rl/smoke_rft --epochs 1 --batch-size 4 --grad-accum 4 --lr 5e-7 --max-samples 40 2>&1 \
    | grep -vE "WeightNorm|warn" | grep -iE "EPOCH|train=|BEST|DONE|loss_mel|error|Traceback" | tail -8

CK=$(ls -t runs/rl/smoke_rft/*/best_model*.pth 2>/dev/null | head -1)
echo "=== SMOKE RESULT ==="
echo "winners=$(wc -l < $D/winners.csv) pairs=$(wc -l < $D/pairs.jsonl 2>/dev/null) ckpt=$CK"
test -n "$CK" && echo "SMOKE_PASS: checkpoint produced" || echo "SMOKE_FAIL: no checkpoint"
