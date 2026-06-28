#!/bin/bash
# One RFT shard: generate N candidates for a prompt sub-set + select winners. GPU pinned by caller (CUDA_VISIBLE_DEVICES).
set -e
cd /mnt/data/harshal/syntts
PROMPTS=$1; OUT=$2
PY=.venv_xtts/bin/python
XD=.tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2
REFS=runs/xtts_hinglish/RELEASE/refs
REF16=runs/rl/ref_16L/model.pth
mkdir -p $OUT
echo "[shard $OUT] gen N=8 on GPU $CUDA_VISIBLE_DEVICES"
$PY scripts/hinglish/rl/gen_candidates.py --base $XD --ckpt $REF16 --gpt-layers 16 \
    --prompts $PROMPTS --refs-dir $REFS --out-dir $OUT/cand --n 8 --temp 0.95 2>&1 \
    | grep -vE "WeightNorm|warn|exceeds the character" | tail -2
echo "[shard $OUT] select winners"
$PY scripts/hinglish/rl/select_winners.py --candidates $OUT/cand/candidates.jsonl --refs-dir $REFS \
    --out-corpus $OUT/winners.csv --out-pairs $OUT/pairs.jsonl --out-report $OUT/report.json \
    --out-floors $OUT/floors.json --min-gain 0.01 2>&1 | grep -E "select" | tail -2
echo "[shard $OUT] DONE winners=$(wc -l < $OUT/winners.csv)"
