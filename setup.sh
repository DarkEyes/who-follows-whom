#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p models
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -r requirements.txt
if [ ! -f models/yolo11s.pt ]; then
  curl -L -o models/yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt
fi
echo "setup complete"
