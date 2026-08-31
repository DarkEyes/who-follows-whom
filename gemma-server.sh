#!/bin/bash
set -e
cd "$(dirname "$0")"
MODEL="$(find models -name "*.gguf" ! -name "mmproj*" | head -1)"
MMPROJ="$(find models -name "mmproj*.gguf" | head -1)"
if [ -z "$MODEL" ] || [ -z "$MMPROJ" ]; then
  echo "No GGUF model + mmproj pair found under models/"
  echo "Place a vision GGUF (model + mmproj files) in models/<any-folder>/"
  exit 1
fi
echo "model:  $MODEL"
echo "mmproj: $MMPROJ"
exec llama-server -m "$MODEL" --mmproj "$MMPROJ" --port 8080
