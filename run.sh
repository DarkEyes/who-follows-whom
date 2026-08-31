#!/bin/bash
set -e
if [ -z "$1" ]; then
  echo "usage: ./run.sh <video.mp4> [--no-gemma] [--stage 1|2]"
  exit 1
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
VIDEO="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
shift
cd "$HERE/vjepa-gemma-demo"
exec "$HERE/.venv/bin/python3" run_pipeline.py "$VIDEO" "$@"
