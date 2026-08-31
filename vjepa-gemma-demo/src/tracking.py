"""Person detection + tracking (YOLO11 + ByteTrack via ultralytics).

ByteTrack fragments IDs on small, mostly-static seated people, so raw
fragments are merged afterwards. Merging is constrained by temporal
co-occurrence: fragments sharing frames with non-overlapping boxes are
different people (cannot-link); fragments sharing frames with heavily
overlapping boxes are duplicate detections of one person (must-link);
fragments that never share frames merge by spatial proximity, but never
across a cannot-link constraint.

Output: {pid: {frame_idx: (x1, y1, x2, y2)}} with pid P1..PN left-to-right.
"""

from collections import defaultdict

import cv2
import numpy as np


def read_video_meta(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "n_frames": n_frames, "width": w, "height": h,
            "duration": n_frames / fps if fps else 0.0}


def sample_frame_indices(meta: dict, window_sec: float, frames_per_window: int):
    """Uniform frame indices per window. Returns list of (t0, t1, [idx...])."""
    fps, n = meta["fps"], meta["n_frames"]
    duration = n / fps
    windows = []
    t = 0.0
    while t + window_sec <= duration + 1e-6:
        f0, f1 = int(t * fps), min(int((t + window_sec) * fps), n - 1)
        idx = np.linspace(f0, f1, frames_per_window).round().astype(int)
        windows.append((t, t + window_sec, idx))
        t += window_sec
    return windows


def load_frames(path: str, indices: np.ndarray):
    """Read specific frames (BGR). Sequential read; fine for short videos."""
    cap = cv2.VideoCapture(path)
    wanted = set(int(i) for i in indices)
    frames = {}
    i = 0
    ok = True
    while ok and wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = frame
            wanted.discard(i)
        i += 1
    cap.release()
    return [frames[int(i)] for i in indices if int(i) in frames]


def _overlap(b1, b2):
    """(IoU, intersection / smaller box area)."""
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    if inter == 0:
        return 0.0, 0.0
    return inter / (a1 + a2 - inter), inter / min(a1, a2)


def merge_fragments(raw: dict, n_frames: int, frame_diag: float,
                    min_cooccur: int = 10):
    """Constrained fragment merging (see module docstring)."""
    frags = list(raw.values())
    n = len(frags)
    if n == 0:
        return {}
    fsets = [set(d.keys()) for d in frags]
    centers = np.array([
        [np.mean([(b[0] + b[2]) / 2 for b in d.values()]),
         np.mean([(b[1] + b[3]) / 2 for b in d.values()])]
        for d in frags])

    mean_boxes = [tuple(np.mean([b[k] for b in d.values()])
                        for k in range(4)) for d in frags]

    # Size quarantine: fragments whose mean box is far larger than the
    # median fragment are boxes around several people; exclude them from
    # all constraint building so they cannot bridge individuals.
    areas = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in mean_boxes])
    med = float(np.median(areas))
    oversized = {i for i in range(n) if areas[i] > 2.0 * med}
    if oversized:
        print(f"[track] quarantined {len(oversized)} oversized fragments "
              f"(area > 2x median)")

    must, cannot, near = [], set(), []
    for i in range(n):
        for j in range(i + 1, n):
            if i in oversized or j in oversized:
                continue
            common = fsets[i] & fsets[j]
            if len(common) >= min_cooccur:
                sample = sorted(common)[:: max(1, len(common) // 50)]
                ious, conts = zip(*(_overlap(frags[i][f], frags[j][f])
                                    for f in sample))
                if np.mean(conts) > 0.8:
                    must.append((i, j))
                else:
                    cannot.add((i, j))
            else:
                # Never co-occurring: same person iff their mean boxes
                # overlap heavily or nest. Scale-free (no pixel radius):
                # zooming changes distances but not overlap geometry.
                iou, cont = _overlap(mean_boxes[i], mean_boxes[j])
                if iou > 0.4 or cont > 0.6:
                    near.append((-max(iou, cont), i, j))

    # Joint-detection quarantine: a fragment must-linked to two fragments
    # that cannot-link with each other is a box around several people;
    # remove it from all constraints so it cannot bridge real people.
    partners = defaultdict(set)
    for i, j in must:
        partners[i].add(j)
        partners[j].add(i)
    joint = set()
    for c, ms in partners.items():
        ms = sorted(ms)
        for a in range(len(ms)):
            for b in range(a + 1, len(ms)):
                if (ms[a], ms[b]) in cannot:
                    joint.add(c)
    if joint:
        print(f"[track] quarantined {len(joint)} joint-detection fragments")
        must = [(i, j) for i, j in must
                if i not in joint and j not in joint]
        near = [(dd, i, j) for dd, i, j in near
                if i not in joint and j not in joint]
        cannot = {(i, j) for i, j in cannot
                  if i not in joint and j not in joint}

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    members = {i: {i} for i in range(n)}

    def violates(ra, rb):
        for a in members[ra]:
            for b in members[rb]:
                if (min(a, b), max(a, b)) in cannot:
                    return True
        return False

    def union(i, j, forced=False):
        ra, rb = find(i), find(j)
        if ra == rb:
            return
        if not forced and violates(ra, rb):
            return
        parent[ra] = rb
        members[rb] |= members.pop(ra)

    for i, j in must:
        union(i, j, forced=True)
    for _, i, j in sorted(near):
        union(i, j)

    groups = defaultdict(dict)
    for i, d in enumerate(frags):
        g = groups[find(i)]
        for fi, box in d.items():
            g.setdefault(fi, box)

    merged = dict(enumerate(groups.values()))
    cov = sorted((len(f) / n_frames for f in merged.values()), reverse=True)
    print(f"[track] merged into {len(merged)} clusters "
          f"({len(must)} must-links, {len(cannot)} cannot-links, "
          f"overlap-based); coverage of top 8: "
          f"{[round(c, 2) for c in cov[:8]]}")
    return merged


def track_people(path: str, cfg):
    """Run YOLO+ByteTrack, then constrained fragment merging.

    Returns:
        tracks: {pid: {frame_idx: (x1, y1, x2, y2)}}  with pid in {"P1", ...}
    """
    from ultralytics import YOLO

    model = YOLO(cfg.yolo_model)
    raw = defaultdict(dict)  # track_id -> frame_idx -> box
    frame_idx = 0
    for res in model.track(source=path, stream=True, persist=True,
                           classes=[0], tracker="bytetrack.yaml",
                           conf=cfg.det_conf, verbose=False):
        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.int().tolist()
            boxes = res.boxes.xyxy.cpu().numpy()
            for tid, box in zip(ids, boxes):
                raw[tid][frame_idx] = tuple(float(v) for v in box)
        frame_idx += 1

    n_frames = frame_idx
    cov = sorted((len(d) / n_frames for d in raw.values()), reverse=True)
    print(f"[track] {len(raw)} raw tracks over {n_frames} frames; "
          f"coverage of top 8: "
          f"{[round(c, 2) for c in cov[:8]] if cov else 'none'}")

    meta = read_video_meta(path)
    frame_diag = float(np.hypot(meta["width"], meta["height"]))
    raw = merge_fragments(raw, n_frames, frame_diag)

    kept = [(tid, d) for tid, d in raw.items()
            if len(d) >= cfg.min_track_coverage * n_frames]
    kept.sort(key=lambda kv: -len(kv[1]))
    kept = kept[: cfg.max_people]

    def mean_x(d):
        return float(np.mean([(b[0] + b[2]) / 2 for b in d.values()]))

    kept.sort(key=lambda kv: mean_x(kv[1]))
    return {f"P{i + 1}": d for i, (_, d) in enumerate(kept)}


def box_at(track: dict, frame_idx: int):
    """Nearest available box for a frame index (tracks can have gaps)."""
    if frame_idx in track:
        return track[frame_idx]
    keys = np.array(sorted(track.keys()))
    j = keys[np.argmin(np.abs(keys - frame_idx))]
    return track[int(j)]


def crop_person(frame, box, expand: float, size: int):
    """Square, expanded, resized crop (RGB uint8)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + 2 * expand) / 2
    x1, x2 = int(max(0, cx - half)), int(min(w, cx + half))
    y1, y2 = int(max(0, cy - half)), int(min(h, cy + half))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
