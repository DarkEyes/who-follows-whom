"""Print tracking diagnostics for one video: raw fragments, merge
constraints, cluster coverages, and per-person track summaries."""

import sys

sys.path.insert(0, ".")
import numpy as np

from src.config import Config
from src.tracking import read_video_meta, track_people

video = sys.argv[1] if len(sys.argv) > 1 else "../game_zoom.mp4"
cfg = Config()
tracks = track_people(video, cfg)
meta = read_video_meta(video)
n = meta["n_frames"]
print()
for pid, t in tracks.items():
    xs = [(b[0] + b[2]) / 2 for b in t.values()]
    ys = [(b[1] + b[3]) / 2 for b in t.values()]
    ws = [b[2] - b[0] for b in t.values()]
    hs = [b[3] - b[1] for b in t.values()]
    print(f"{pid}: cov={len(t)/n:.2f} "
          f"center=({np.mean(xs):.0f},{np.mean(ys):.0f}) "
          f"box={np.mean(ws):.0f}x{np.mean(hs):.0f}")
