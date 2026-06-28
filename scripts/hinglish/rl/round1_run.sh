#!/bin/bash
# Round-1 offline RFT: dual-GPU candidate gen+select, merge winners, CE fine-tune (warm-restore 16L) + SFT-replay anchor.
set -e
cd /mnt/data/harshal/syntts
PY=.venv_xtts/bin/python
REF16=runs/rl/ref_16L/model.pth
R=data/rl/round1
NPROMPTS=${NPROMPTS:-400}
GPUA=${GPUA:-6}; GPUB=${GPUB:-3}
mkdir -p $R

echo "=== [1] build $NPROMPTS code-switch prompts ==="
$PY scripts/hinglish/rl/build_prompts.py --out $R/prompts.jsonl --n $NPROMPTS
$PY - <<PY
L=[l for l in open("$R/prompts.jsonl") if l.strip()]
open("$R/promptsA.jsonl","w").write("\n".join(L[0::2]))
open("$R/promptsB.jsonl","w").write("\n".join(L[1::2]))
print("split A=%d B=%d" % (len(L[0::2]), len(L[1::2])))
PY

echo "=== [2] parallel shards: A on GPU$GPUA, B on GPU$GPUB ==="
CUDA_VISIBLE_DEVICES=$GPUA nohup bash scripts/hinglish/rl/round1_shard.sh $R/promptsA.jsonl $R/A > /tmp/round1_A.log 2>&1 &
PA=$!
CUDA_VISIBLE_DEVICES=$GPUB nohup bash scripts/hinglish/rl/round1_shard.sh $R/promptsB.jsonl $R/B > /tmp/round1_B.log 2>&1 &
PB=$!
echo "shard PIDs A=$PA B=$PB"
wait $PA; wait $PB
echo "=== shards done: A winners=$(wc -l < $R/A/winners.csv) B winners=$(wc -l < $R/B/winners.csv) ==="

echo "=== [3] merge winners + frozen-16L floor table ==="
cat $R/A/winners.csv $R/B/winners.csv | grep -v "^$" > $R/winners.csv
cat $R/A/pairs.jsonl $R/B/pairs.jsonl > $R/pairs.jsonl
$PY - <<PY
import json
d={}
for f in ["$R/A/floors.json","$R/B/floors.json"]:
    d.update(json.load(open(f)))
json.dump(d, open("$R/floors_16L.json","w"))
print("frozen floors for %d prompts" % len(d))
PY
NW=$(wc -l < $R/winners.csv)
echo "merged winners=$NW"

echo "=== [4] SFT-replay anchor (spread sample of original corpus, ~= winners count) ==="
awk "NR % 7 == 0" data/xtts/metadata_train.csv | head -$NW > $R/replay.csv
cat $R/winners.csv > $R/train.csv; echo >> $R/train.csv; cat $R/replay.csv >> $R/train.csv
echo "train rows=$(wc -l < $R/train.csv) (winners=$NW + replay=$(wc -l < $R/replay.csv))"

echo "=== [5] CE fine-tune (warm-restore 16L, gpt-layers 16, lr 2e-6, 4 epochs) on GPU$GPUA ==="
CUDA_VISIBLE_DEVICES=$GPUA $PY scripts/hinglish/train_xtts.py --gpt-layers 16 --restore $REF16 \
    --metadata-train $R/train.csv --metadata-eval data/xtts/metadata_eval.csv \
    --out-path runs/rl/round1_rft --epochs 4 --batch-size 4 --grad-accum 8 --lr 2e-6 2>&1 \
    | grep -vE "WeightNorm|warn" | grep -iE "EPOCH|train=|BEST|DONE|avg_loss|error|Traceback" | tail -12

CK=$(ls -t runs/rl/round1_rft/*/best_model*.pth 2>/dev/null | head -1)
echo "=== ROUND1_DONE winners=$NW ckpt=$CK ==="
test -n "$CK" && echo "ROUND1_OK" || echo "ROUND1_FAIL"
