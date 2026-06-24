#!/bin/bash
# Shard the qwen filter over the full synth set across GPU 0,5,6 and merge.
# Run from the syntts repo root on the GPU box.
set -euo pipefail
cd .
export HF_HOME=<hf-cache>
QWEN=<qwen-venv>/bin/python
IDX=data/synth/synth_index.jsonl
COR=data/corpus/corpus.jsonl
mkdir -p data/filtered logs

echo "[shard] splitting $(wc -l < $IDX) rows into 3 round-robin shards"
python3 - <<'PY'
rows=open('data/synth/synth_index.jsonl').readlines()
for g in (0,1,2):
    open(f'data/synth/_shard{g}.jsonl','w').writelines(rows[g::3])
print('shard sizes:', [sum(1 for _ in open(f'data/synth/_shard{g}.jsonl')) for g in (0,1,2)])
PY

# map shard0->gpu0, shard1->gpu5, shard2->gpu6
declare -A GPU=( [0]=0 [1]=5 [2]=6 )
for s in 0 1 2; do
  g=${GPU[$s]}
  echo "[shard] launching shard $s on GPU $g"
  CUDA_VISIBLE_DEVICES=$g $QWEN scripts/hinglish/03_filter_qwen.py \
    --synth-index data/synth/_shard${s}.jsonl --corpus $COR \
    --out data/filtered/_fs${s}.jsonl > logs/filter_${s}.log 2>&1 &
done
wait
echo "[shard] all shards done; merging"
cat data/filtered/_fs0.jsonl data/filtered/_fs1.jsonl data/filtered/_fs2.jsonl > data/filtered/filter_scores.jsonl
echo "FILTER_DONE rows=$(wc -l < data/filtered/filter_scores.jsonl)"
python3 - <<'PY'
import json, statistics
from collections import Counter
rows=[json.loads(l) for l in open('data/filtered/filter_scores.jsonl')]
acc=sum(1 for r in rows if r.get('accept'))
rc=[r['filter_recall'] for r in rows if r.get('filter_recall') is not None]
rej=Counter(r.get('reject_reason') for r in rows if not r.get('accept'))
print(f"accepted {acc}/{len(rows)} = {100*acc/len(rows):.1f}%  mean_recall={statistics.mean(rc):.3f}")
print("reject reasons:", dict(rej))
PY
