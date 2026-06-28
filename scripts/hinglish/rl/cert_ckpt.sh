#!/bin/bash
# Full powered cert for ANY 16-layer checkpoint: generate on all 3 held-out batches (89+98+127=314),
# score, merge, and run the equivalence gate vs the cached 443M-teacher powered panel (eval_pow_400m).
# Usage: cert_ckpt.sh <ckpt> <name>
set -e
cd /mnt/data/harshal/syntts
export PYTHONUNBUFFERED=1
CK=$1; NAME=$2
XD=.tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2
REFS=runs/xtts_hinglish/RELEASE/refs
RJ='{"kaustubh":"runs/xtts_hinglish/RELEASE/refs/kaustubh.wav","arjun":"runs/xtts_hinglish/RELEASE/refs/arjun.wav","maya":"runs/xtts_hinglish/RELEASE/refs/maya.wav","aadya":"runs/xtts_hinglish/RELEASE/refs/aadya.wav"}'
M=scripts/hinglish/rl/merge_json.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}

gen_score () {  # manifest tag
  local MAN=$1 TAG=$2 D=data/eval_pow/${NAME}_${2}
  echo "=== [$NAME/$TAG] gen $(grep -c . $MAN) ==="
  .venv_xtts/bin/python scripts/hinglish/compare/gen_panel_ckpt.py --base $XD --ckpt "$CK" --gpt-layers 16 \
      --eval-manifest $MAN --refs-dir $REFS --out-dir $D/wav --label $NAME > $D.gen.log 2>&1 || true
  echo "[$NAME/$TAG] wavs=$(ls $D/wav/*.wav 2>/dev/null|wc -l)"
  .venv_eval/bin/python scripts/hinglish/09_objective_eval.py --manifest $D/wav/manifest.jsonl --label $NAME --voice-refs "$RJ" --out $D/obj.json 2>&1 | tail -1
  .venv_xtts/bin/python scripts/hinglish/10_accent_eval.py --manifest $D/wav/manifest.jsonl --label $NAME --out $D/accent.json 2>&1 | tail -1
}

gen_score data/eval_heldout89.jsonl        h
gen_score data/eval_pow/new_prompts.jsonl  n1
gen_score data/eval_pow/new2_prompts.jsonl n2

mkdir -p data/eval_pow_$NAME
for KIND in obj accent; do
  .venv_xtts/bin/python $M data/eval_pow/${NAME}_h/$KIND.json data/eval_pow/${NAME}_n1/$KIND.json /tmp/_c1_$NAME.json >/dev/null
  .venv_xtts/bin/python $M /tmp/_c1_$NAME.json data/eval_pow/${NAME}_n2/$KIND.json data/eval_pow_$NAME/$KIND.json >/dev/null
done

echo "=== POWERED CERT (n~314): $NAME vs 443M teacher ==="
.venv_eval/bin/python scripts/hinglish/12_equivalence_eval.py --ref data/eval_pow_400m --cand data/eval_pow_$NAME 2>&1 | tail -12
echo "=== expressivity: $NAME vs frozen 16L (batch-2 prompts) ==="
.venv_xtts/bin/python scripts/hinglish/rl/pitch_monitor.py --cand data/eval_pow/${NAME}_n2/wav/manifest.jsonl --ref data/eval_pow/16L_2/wav/manifest.jsonl --label $NAME 2>&1 | grep -E "paired|expressivity" || true
echo "CERT_CKPT_DONE_$NAME"
