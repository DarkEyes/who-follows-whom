#!/usr/bin/env python3
"""V-JEPA + Gemma feasibility demo.

Stage 1 (V-JEPA):  track -> window -> crop -> encode -> lagged coordination
                   -> save embeddings + scores, unload model.
Stage 2 (Gemma):   describe top pair (and control pair) per window, unload.

Usage:
    python run_pipeline.py game.mp4
    python run_pipeline.py game.mp4 --window 5 --no-gemma
    python run_pipeline.py game.mp4 --stage 2          # rerun Gemma only
"""

import argparse
import json
import os
import sys

import numpy as np

from src.config import Config
from src import tracking
from src.coordination import score_window, aggregate_network


def fmt_t(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def stage1(video: str, cfg: Config, work: str):
    from src.jepa import JepaEncoder

    meta = tracking.read_video_meta(video)
    print(f"[meta] {meta['duration']:.1f}s @ {meta['fps']:.1f}fps "
          f"{meta['width']}x{meta['height']}")

    print("[stage1] tracking people ...")
    tracks = tracking.track_people(video, cfg)
    if len(tracks) < 2:
        sys.exit(f"only {len(tracks)} persistent track(s); need >= 2")
    print(f"[stage1] kept tracks: {', '.join(sorted(tracks))}")

    windows = tracking.sample_frame_indices(meta, cfg.window_sec,
                                            cfg.frames_per_window)
    print(f"[stage1] {len(windows)} windows of {cfg.window_sec}s")

    print(f"[stage1] loading V-JEPA ({cfg.jepa_backend}) ...")
    enc = JepaEncoder(cfg)
    rng = np.random.default_rng(0)

    results = []
    for w_id, (t0, t1, idx) in enumerate(windows):
        frames = tracking.load_frames(video, idx)
        if len(frames) < cfg.frames_per_window:
            continue
        z_by_pid = {}
        for pid, tr in tracks.items():
            clip = np.stack([
                tracking.crop_person(f, tracking.box_at(tr, int(i)),
                                     cfg.crop_expand, cfg.crop_size)
                for f, i in zip(frames, idx)
            ])
            z_by_pid[pid] = enc.encode_clip(clip)
        edges, directed = score_window(z_by_pid, cfg, rng)
        results.append({"start": t0, "end": t1,
                        "frame_indices": [int(i) for i in idx],
                        "edges": edges,
                        "directed": directed})
        top = edges[0]
        print(f"  {fmt_t(t0)}-{fmt_t(t1)}  top {top['leader']}->"
              f"{top['follower']}  z={top['z']}  lag={top['lag_steps']}")

    enc.unload()
    np.save(os.path.join(work, "tracks.npy"), tracks, allow_pickle=True)
    with open(os.path.join(work, "stage1.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[stage1] done -> {work}/stage1.json")
    return results, tracks


def stage2(video: str, cfg: Config, work: str, results, tracks, args):
    from src.gemma_describe import GemmaDescriber

    print("[stage2] loading Gemma ...")
    desc = GemmaDescriber(cfg)
    for w in results:
        idx = np.asarray(w["frame_indices"])
        sel = np.linspace(0, len(idx) - 1, cfg.gemma_frames).round().astype(int)
        frames = tracking.load_frames(video, idx[sel])
        dur = w["end"] - w["start"]

        top = w["edges"][0]
        if args.blind:
            top["event_blind"] = desc.describe(
                frames, tracks, idx[sel], top["pair"], dur, blind=True)
            print(f"  {fmt_t(w['start'])}  {'/'.join(top['pair'])} "
                  f"[blind]: {top['event_blind']}")
        else:
            top["event"] = desc.describe(frames, tracks, idx[sel],
                                         top["pair"], dur)
            print(f"  {fmt_t(w['start'])}  {'/'.join(top['pair'])}: "
                  f"{top['event']}")

        if cfg.run_control_pair and len(w["edges"]) > 1:
            ctrl = w["edges"][-1]
            if args.blind:
                ctrl["event_control_blind"] = desc.describe(
                    frames, tracks, idx[sel], ctrl["pair"], dur, blind=True)
            else:
                ctrl["event_control"] = desc.describe(
                    frames, tracks, idx[sel], ctrl["pair"], dur)
    desc.unload()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--window", type=float, default=None)
    ap.add_argument("--backend", choices=["hf20", "hub21"], default=None)
    ap.add_argument("--no-gemma", action="store_true")
    ap.add_argument("--production", action="store_true",
                    help="report-only mode: no control descriptions, "
                         "compact console output")
    ap.add_argument("--blind", action="store_true",
                    help="stage 2 with neutral prompt; writes event_blind "
                         "fields alongside existing ones")
    ap.add_argument("--stage", type=int, choices=[1, 2], default=None,
                    help="run only one stage (2 requires stage1.json)")
    args = ap.parse_args()

    cfg = Config()
    if args.production:
        cfg.run_control_pair = False
    if args.window:
        cfg.window_sec = args.window
    if args.backend:
        cfg.jepa_backend = args.backend

    name = os.path.splitext(os.path.basename(args.video))[0]
    work = os.path.join(cfg.out_dir, name)
    os.makedirs(work, exist_ok=True)

    if args.stage == 2:
        with open(os.path.join(work, "stage1.json")) as f:
            results = json.load(f)
        tracks = np.load(os.path.join(work, "tracks.npy"),
                         allow_pickle=True).item()
    else:
        results, tracks = stage1(args.video, cfg, work)

    # Preserve descriptions from earlier stage-2 passes (e.g. framed run
    # before a --blind run): merge event* fields from an existing
    # game.json into the freshly loaded results.
    prev_path = os.path.join(work, f"{name}.json")
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            prev = {w["start"]: w["edges"] for w in json.load(f)["windows"]}
        for w in results:
            if w["start"] not in prev:
                continue
            for old_e in prev[w["start"]]:
                for new_e in w["edges"]:
                    if new_e["pair"] == old_e["pair"]:
                        for k, v in old_e.items():
                            if k.startswith("event") and k not in new_e:
                                new_e[k] = v

    if not args.no_gemma and args.stage != 1:
        results = stage2(args.video, cfg, work, results, tracks, args)

    cols = sorted({k for w in results for k in w.get("directed", {})})
    z_rows, lag_rows = [], []
    for w in results:
        dd = w.get("directed", {})
        z_rows.append([dd[c]["z"] if c in dd else None for c in cols])
        lag_rows.append([dd[c]["lag_steps"] if c in dd else None
                         for c in cols])
    z_matrix = {
        "columns": cols,
        "starts": [w["start"] for w in results],
        "z": z_rows,
        "lag_steps": lag_rows,
    }
    csv_path = os.path.join(work, "z_matrix.csv")
    with open(csv_path, "w") as f:
        f.write("start," + ",".join(cols) + "\n")
        for st, row in zip(z_matrix["starts"], z_rows):
            f.write(f"{st}," + ",".join("" if v is None else str(v)
                                        for v in row) + "\n")
    print(f"[matrix] {csv_path}  ({len(z_rows)} intervals x {len(cols)} edges)")

    import math

    def phi(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    stouffer, follow = {}, {}
    for k, c in enumerate(cols):
        vals = [r[k] for r in z_rows if r[k] is not None]
        if vals:
            stouffer[c] = round(sum(vals) / math.sqrt(len(vals)), 3)
            follow[c] = round(sum(phi(v) for v in vals) / len(vals), 3)
    stouffer = dict(sorted(stouffer.items(), key=lambda kv: -kv[1]))
    follow = dict(sorted(follow.items(), key=lambda kv: -kv[1]))
    net = {}
    for c in cols:
        i, j = c.split("->")
        rev = f"{j}->{i}"
        if c in stouffer and rev in stouffer and stouffer[c] >= stouffer[rev]:
            net[c] = round(stouffer[c] - stouffer[rev], 3)
    net = dict(sorted(net.items(), key=lambda kv: -kv[1]))

    out = {
        "video": args.video,
        "z_matrix": z_matrix,
        "aggregate": {
            "stouffer_z": stouffer,
            "following_F": follow,
            "net_leadership": net,
            "n_windows": len(z_rows),
            "note": "stouffer_z ~ N(0,1) per edge under the null "
                    "(approximate; windows/edges not fully independent). "
                    "following_F = mean Phi(z) in [0,1]: probability that "
                    "i->j alignment beats timing-scrambled coincidence in a "
                    "random interval; 0.5 = chance, > 0.5 following, "
                    "< 0.5 anti-alignment. Effect size only -- judge "
                    "reliability with stouffer_z and n_windows. "
                    "net_leadership = S(i->j) - S(j->i), reported in the "
                    "stronger direction.",
        },
        "window_sec": cfg.window_sec,
        "people": sorted(tracks),
        "windows": [
            {"start": w["start"], "end": w["end"],
             "edges": [w["edges"][0]]
             + ([w["edges"][-1]]
                if ("event_control" in w["edges"][-1]
                    or "event_control_blind" in w["edges"][-1]) else [])}
            for w in results
        ],
        "following_network": aggregate_network(results),
    }
    people = sorted(tracks)

    def write_adj(fname, values):
        p = os.path.join(work, fname)
        with open(p, "w") as f:
            f.write("," + ",".join(people) + "\n")
            for i in people:
                row = [str(values.get(f"{i}->{j}", 0)) if i != j else "0"
                       for j in people]
                f.write(i + "," + ",".join(row) + "\n")
        print(f"[aggregate] {p}")
        return p

    write_adj("F_matrix.csv", follow)
    write_adj("S_matrix.csv", stouffer)

    out_path = os.path.join(work, f"{name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] {out_path}")
    if args.production:
        top_edges = list(out["aggregate"]["following_F"].items())[:3]
        summary = ", ".join(f"{e} (F={v})" for e, v in top_edges)
        print(f"[summary] strongest following: {summary}")
    else:
        print("[stouffer]", json.dumps(out["aggregate"]["stouffer_z"],
                                       indent=2))
        print("[following-F]", json.dumps(out["aggregate"]["following_F"],
                                          indent=2))
        print("[net-leadership]",
              json.dumps(out["aggregate"]["net_leadership"], indent=2))


if __name__ == "__main__":
    main()
