#!/bin/bash
# Powered cert round 2: add batch-2 (127 prompts) to reach accent n~280. Generate round-1 + 443M-teacher
# panels on batch-2, then REBUILD the powered panels from all 3 batches (idempotent), re-run equivalence.
set -e
cd /mnt/data/harshal/syntts
export PYTHONUNBUFFERED=1
XD=.tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2
REFS=runs/xtts_hinglish/RELEASE/refs
NEW2=data/eval_pow/new2_prompts.jsonl
REFSJSON='{"kaustubh":"runs/xtts_hinglish/RELEASE/refs/kaustubh.wav","arjun":"runs/xtts_hinglish/RELEASE/refs/arjun.wav","maya":"runs/xtts_hinglish/RELEASE/refs/maya.wav","aadya":"runs/xtts_hinglish/RELEASE/refs/aadya.wav"}'
ROUND1=$(ls -t runs/rl/round1_rft/*/best_model*.pth | head -1)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
M=scripts/hinglish/rl/merge_json.py

gen_score_batch2 () {
  local NAME=$1 CK=$2 GL=$3
  local D=data/eval_pow/${NAME}_2
  echo "=== [$NAME b2] gen $(grep -c . $NEW2) prompts (gpt-layers $GL) ==="
  .venv_xtts/bin/python scripts/hinglish/compare/gen_panel_ckpt.py --base $XD --ckpt "$CK" --gpt-layers $GL \
      --eval-manifest $NEW2 --refs-dir $REFS --out-dir $D/wav --label ${NAME}2 > $D.gen.log 2>&1 || true
  echo "[$NAME b2] wavs=$(ls $D/wav/*.wav 2>/dev/null | wc -l)"
  .venv_eval/bin/python scripts/hinglish/09_objective_eval.py --manifest $D/wav/manifest.jsonl --label ${NAME}2 \
      --voice-refs "$REFSJSON" --out $D/obj_new.json 2>&1 | tail -1
  .venv_xtts/bin/python scripts/hinglish/10_accent_eval.py --manifest $D/wav/manifest.jsonl --label ${NAME}2 \
      --out $D/accent_new.json 2>&1 | tail -1
}

# round-1 and teacher only (PRIMARY claim = round-1 vs teacher); 16L not needed for the accent PASS
gen_score_batch2 round1 "$ROUND1"                          16
gen_score_batch2 400m   runs/xtts_hinglish/RELEASE/model.pth 30

# rebuild powered panels from all 3 batches: cached-89 + batch1-98 + batch2-127  (idempotent)
rebuild () {
  local NAME=$1 CACHED=$2
  for KIND in obj accent; do
    .venv_xtts/bin/python $M $CACHED/$KIND.json data/eval_pow/$NAME/${KIND}_new.json /tmp/_m1_$NAME.json >/dev/null
    .venv_xtts/bin/python $M /tmp/_m1_$NAME.json data/eval_pow/${NAME}_2/${KIND}_new.json data/eval_pow_$NAME/$KIND.json
  done
}
rebuild round1 data/eval_round1
rebuild 400m   data/eval_400m

echo "=== POWERED CERT (n~314): round-1 (265M) vs 443M teacher ==="
.venv_eval/bin/python scripts/hinglish/12_equivalence_eval.py --ref data/eval_pow_400m --cand data/eval_pow_round1 2>&1 | tail -12
echo "POWERED_CERT2_DONE"
