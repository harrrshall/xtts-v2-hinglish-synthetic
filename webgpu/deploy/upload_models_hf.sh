#!/usr/bin/env bash
# Upload the 3 ONNX models to the Hugging Face Hub so the browser app can stream them from the CDN.
# Run it yourself after `huggingface-cli login` (keeps your token private).
#
#   REPO=harrrshall/syntts-webgpu bash webgpu/deploy/upload_models_hf.sh
#
# Then set "modelBase" in webgpu/app/config.json to:
#   https://huggingface.co/<REPO>/resolve/main/
set -euo pipefail

REPO="${REPO:-harrrshall/syntts-webgpu}"
APP_DIR="$(cd "$(dirname "$0")/../app" && pwd)"
MODELS_DIR="$APP_DIR/models"

for f in gpt_prefill.onnx gpt_decode.onnx vocoder.onnx; do
  test -f "$MODELS_DIR/$f" || { echo "missing $MODELS_DIR/$f (run the export pipeline first)"; exit 1; }
done

command -v huggingface-cli >/dev/null || { echo "pip install -U 'huggingface_hub[cli]' and 'huggingface-cli login' first"; exit 1; }

echo "creating repo $REPO (ok if it already exists) ..."
huggingface-cli repo create "$REPO" --type model -y || true

echo "uploading $(du -sh "$MODELS_DIR" | cut -f1) of models to $REPO/models ..."
huggingface-cli upload "$REPO" "$MODELS_DIR" models --repo-type=model

echo
echo "done. set this in webgpu/app/config.json:"
echo "  \"modelBase\": \"https://huggingface.co/$REPO/resolve/main/\""
