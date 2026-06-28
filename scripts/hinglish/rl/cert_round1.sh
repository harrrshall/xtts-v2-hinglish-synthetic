#!/bin/bash
# Certify the round-1 RFT checkpoint vs the frozen 16L (and 443M teacher) on the held-out panel:
# accent recovery + UTMOS/SECS hold + pitch-SD expressivity monitor. Paired seeds with data/eval_16L.
set -e
cd /mnt/data/harshal/syntts
export PYTHONUNBUFFERED=1
XD=.tts_models/tts/tts_models--multilingual--multi-dataset--xtts_v2
REFS=runs/xtts_hinglish/RELEASE/refs
MAN=data/eval_heldout89.jsonl
REFSJSON='{"kaustubh":"runs/xtts_hinglish/RELEASE/refs/kaustubh.wav","arjun":"runs/xtts_hinglish/RELEASE/refs/arjun.wav","maya":"runs/xtts_hinglish/RELEASE/refs/maya.wav","aadya":"runs/xtts_hinglish/RELEASE/refs/aadya.wav"}'
CK=$(ls -t runs/rl/round1_rft/*/best_model*.pth | head -1)
D=data/eval_round1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}

echo "=== gen round1 panel (gpt-layers 16, paired seeds) ckpt=$CK ==="
.venv_xtts/bin/python scripts/hinglish/compare/gen_panel_ckpt.py --base $XD --ckpt "$CK" --gpt-layers 16 \
    --eval-manifest $MAN --refs-dir $REFS --out-dir $D/wav --label cand > $D.gen.log 2>&1 || true
echo "wavs=$(ls $D/wav/*.wav 2>/dev/null | wc -l)"

echo "=== UTMOS+SECS ==="
.venv_eval/bin/python scripts/hinglish/09_objective_eval.py --manifest $D/wav/manifest.jsonl --label round1 \
    --voice-refs "$REFSJSON" --out $D/obj.json 2>&1 | tail -1
echo "=== accent ==="
.venv_xtts/bin/python scripts/hinglish/10_accent_eval.py --manifest $D/wav/manifest.jsonl --label round1 \
    --out $D/accent.json 2>&1 | tail -1
echo "=== pitch-SD expressivity monitor (round1 vs frozen 16L) ==="
.venv_xtts/bin/python scripts/hinglish/rl/pitch_monitor.py --cand $D/wav/manifest.jsonl \
    --ref data/eval_16L/wav/manifest.jsonl --label round1 2>&1 | grep -vE "WeightNorm|warn"
echo "=== also pitch-SD of the 16L base (context) ==="
.venv_xtts/bin/python scripts/hinglish/rl/pitch_monitor.py --cand data/eval_16L/wav/manifest.jsonl --label 16L_base 2>&1 | grep -vE "WeightNorm|warn"
echo "CERT_DONE"
