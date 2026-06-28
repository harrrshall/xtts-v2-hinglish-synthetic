#!/bin/bash
# Powered (n>=150) certification: generate 16L / round-1 / 443M-teacher panels on the 98 NEW held-out
# code-switch prompts, merge with the cached 89-panel, then run the equivalence gate.
# PRIMARY claim: round-1 (final 265M model) is non-inferior to the 443M teacher on accent+UTMOS+SECS.
set -e
cd /mnt/data/harshal/syntts
export PYTHONUNBUFFERED=1
XD=.tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2
REFS=runs/xtts_hinglish/RELEASE/refs
NEW=data/eval_pow/new_prompts.jsonl
REFSJSON='{"kaustubh":"runs/xtts_hinglish/RELEASE/refs/kaustubh.wav","arjun":"runs/xtts_hinglish/RELEASE/refs/arjun.wav","maya":"runs/xtts_hinglish/RELEASE/refs/maya.wav","aadya":"runs/xtts_hinglish/RELEASE/refs/aadya.wav"}'
ROUND1=$(ls -t runs/rl/round1_rft/*/best_model*.pth | head -1)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}

# model name | checkpoint | gpt-layers | cached-panel-dir
gen_and_score () {
  local NAME=$1 CK=$2 GL=$3 CACHED=$4
  local D=data/eval_pow/$NAME
  echo "=== [$NAME] gen panel on $(grep -c . $NEW) new prompts (gpt-layers $GL) ==="
  .venv_xtts/bin/python scripts/hinglish/compare/gen_panel_ckpt.py --base $XD --ckpt "$CK" --gpt-layers $GL \
      --eval-manifest $NEW --refs-dir $REFS --out-dir $D/wav --label $NAME > $D.gen.log 2>&1 || true
  echo "[$NAME] wavs=$(ls $D/wav/*.wav 2>/dev/null | wc -l)"
  .venv_eval/bin/python scripts/hinglish/09_objective_eval.py --manifest $D/wav/manifest.jsonl --label $NAME \
      --voice-refs "$REFSJSON" --out $D/obj_new.json 2>&1 | tail -1
  .venv_xtts/bin/python scripts/hinglish/10_accent_eval.py --manifest $D/wav/manifest.jsonl --label $NAME \
      --out $D/accent_new.json 2>&1 | tail -1
  # merge cached(89) + new(98) -> powered panel dir
  mkdir -p data/eval_pow_$NAME
  .venv_xtts/bin/python scripts/hinglish/rl/merge_json.py $CACHED/obj.json $D/obj_new.json data/eval_pow_$NAME/obj.json
  .venv_xtts/bin/python scripts/hinglish/rl/merge_json.py $CACHED/accent.json $D/accent_new.json data/eval_pow_$NAME/accent.json
}

gen_and_score 16L    runs/rl/ref_16L/model.pth        16 data/eval_16L
gen_and_score round1 "$ROUND1"                        16 data/eval_round1
gen_and_score 400m   runs/xtts_hinglish/RELEASE/model.pth 30 data/eval_400m

echo "=== PRIMARY CERT: round-1 (265M) vs 443M teacher (no quality loss) ==="
.venv_eval/bin/python scripts/hinglish/12_equivalence_eval.py --ref data/eval_pow_400m --cand data/eval_pow_round1 2>&1 | tail -14
echo "=== SECONDARY: round-1 vs 16L (RL did not hurt vs pre-RL student) ==="
.venv_eval/bin/python scripts/hinglish/12_equivalence_eval.py --ref data/eval_pow_16L --cand data/eval_pow_round1 2>&1 | tail -10
echo "=== expressivity: round-1 vs 16L pitch-SD on the NEW prompts ==="
.venv_xtts/bin/python scripts/hinglish/rl/pitch_monitor.py --cand data/eval_pow/round1/wav/manifest.jsonl \
    --ref data/eval_pow/16L/wav/manifest.jsonl --label round1_new 2>&1 | grep -E "paired|expressivity"
echo "POWERED_CERT_DONE"
