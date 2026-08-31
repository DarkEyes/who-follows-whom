#!/usr/bin/env python3
"""Render who-is-who images from saved tracks.

Usage:
    python people_snapshot.py <video> [work_dir]

Reads tracks.npy from the work dir (default: results/<video-name>/),
writes into <work_dir>/ids/:
    scene_<t>s.jpg   full frames with labeled boxes, several timestamps
    P1.jpg, P2.jpg   one representative crop per person
"""

import os
import sys

import cv2
import numpy as np

from src.config import Config
from src.tracking import box_at, crop_person, load_frames, read_video_meta


def main():
    video = sys.argv[1]
    name = os.path.splitext(os.path.basename(video))[0]
    work = sys.argv[2] if len(sys.argv) > 2 else os.path.join("results", name)
    tracks = np.load(os.path.join(work, "tracks.npy"),
                     allow_pickle=True).item()
    out = os.path.join(work, "ids")
    os.makedirs(out, exist_ok=True)

    cfg = Config()
    meta = read_video_meta(video)
    fps = meta["fps"]
    ts = [meta["duration"] * f for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    idx = np.array([int(t * fps) for t in ts])
    frames = load_frames(video, idx)

    colors = [(0, 255, 0), (0, 165, 255), (255, 0, 0),
              (0, 0, 255), (255, 0, 255), (0, 255, 255)]

    for t, fi, frame in zip(ts, idx, frames):
        vis = frame.copy()
        for k, (pid, tr) in enumerate(sorted(tracks.items())):
            x1, y1, x2, y2 = (int(v) for v in box_at(tr, int(fi)))
            c = colors[k % len(colors)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
            cv2.putText(vis, pid, (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2)
        p = os.path.join(out, f"scene_{int(t)}s.jpg")
        cv2.imwrite(p, vis)
        print(p)

    for pid, tr in sorted(tracks.items()):
        fi = int(np.median(sorted(tr.keys())))
        frame = load_frames(video, np.array([fi]))[0]
        crop = crop_person(frame, box_at(tr, fi), cfg.crop_expand, 224)
        p = os.path.join(out, f"{pid}.jpg")
        cv2.imwrite(p, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        print(p)


if __name__ == "__main__":
    main()
