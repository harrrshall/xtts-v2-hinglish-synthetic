#!/bin/bash
# Parallel winner-selection: shard candidates BY PROMPT across N concurrent scorers on one GPU.
# Self-calibrated floors are per-prompt, so each prompt's 8 candidates must stay in the same shard.
# Saturates the GPU (N concurrent whisper streams) and parallelizes the CPU pitch work across cores.
set -e
cd /mnt/data/harshal/syntts
export PYTHONUNBUFFERED=1
PY=.venv_xtts/bin/python
CAND=$1; OUT=$2; NS=${3:-8}
REFS=runs/xtts_hinglish/RELEASE/refs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
mkdir -p $OUT/shards

echo "[score_parallel] splitting $CAND into $NS prompt-shards"
$PY - "$CAND" "$OUT/shards" "$NS" <<'PY'
import json, sys, collections
cand, outd, ns = sys.argv[1], sys.argv[2], int(sys.argv[3])
by = collections.OrderedDict()
for l in open(cand):
    if l.strip():
        r = json.loads(l); by.setdefault(r["utt_id"], []).append(r)
files = [open(f"{outd}/shard{i}.jsonl", "w") for i in range(ns)]
for k, (u, g) in enumerate(by.items()):
    f = files[k % ns]
    for r in g:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
for f in files:
    f.close()
print(f"split {len(by)} prompts -> {ns} shards")
PY

echo "[score_parallel] launching $NS concurrent scorers on GPU $CUDA_VISIBLE_DEVICES"
pids=""
for i in $(seq 0 $((NS-1))); do
  $PY scripts/hinglish/rl/select_winners.py --candidates $OUT/shards/shard$i.jsonl --refs-dir $REFS \
      --out-corpus $OUT/shards/w$i.csv --out-pairs $OUT/shards/p$i.jsonl --out-report $OUT/shards/r$i.json \
      --out-floors $OUT/shards/f$i.json --min-gain 0.01 > $OUT/shards/log$i.txt 2>&1 &
  pids="$pids $!"
done
echo "[score_parallel] scorer PIDs:$pids"
wait

echo "[score_parallel] merging shard outputs"
cat $OUT/shards/w*.csv 2>/dev/null | grep -v "^$" > $OUT/winners.csv
cat $OUT/shards/p*.jsonl 2>/dev/null > $OUT/pairs.jsonl
$PY - "$OUT" "$NS" <<'PY'
import json, sys
outd, ns = sys.argv[1], int(sys.argv[2])
d = {}
for i in range(ns):
    try:
        d.update(json.load(open(f"{outd}/shards/f{i}.json")))
    except Exception:
        pass
json.dump(d, open(f"{outd}/floors_16L.json", "w"))
print("merged floors for %d prompts" % len(d))
PY
echo "[score_parallel] DONE winners=$(grep -vc '^$' $OUT/winners.csv) pairs=$(wc -l < $OUT/pairs.jsonl)"
