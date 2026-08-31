# Setup and Experiment Guide

Local pipeline (no cloud, no training) for a video of people around a
table: who coordinates with whom, when, and what visibly happened.

Per 5-second interval: V-JEPA-based scoring ranks all directed person
pairs by calibrated coordination (z); Gemma then describes the top
pair's observable behavior from the frames.

This guide takes you from the zip to completed Stage 1 and Stage 2
experiments. Everything runs on an Apple Silicon Mac (16 GB
recommended). All commands are copy-paste-safe and run from the
project root unless stated otherwise.

## 1. Unpack

Unzip and enter the folder. This folder is called the ROOT below.

```bash
cd who-follows-whom
```

Layout after full setup:

```text
who-follows-whom/
├── GUIDE.md
├── setup.sh
├── run.sh
├── gemma-server.sh
├── .venv/                     created by setup.sh
├── models/
│   ├── yolo11s.pt             downloaded by setup.sh
│   ├── vjepa2-vitl/           you download (step 4)
│   └── gemma/                 you download (step 5)
├── game.mp4                   you create (step 6)
└── vjepa-gemma-demo/          pipeline code
```

If a `.venv/` folder somehow came with your copy, delete it first;
virtual environments do not survive copying between machines:

```bash
rm -rf .venv
```

## 2. Prerequisites (one time)

```bash
python3 --version
```

Need 3.10 or newer. If older, or if `brew` / `ffmpeg` / `llama-server`
are missing:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11 ffmpeg llama.cpp
```

## 3. Base setup (one time)

```bash
chmod +x setup.sh run.sh gemma-server.sh
./setup.sh
```

Creates `.venv`, installs Python packages, downloads YOLO weights.
A few minutes.

## 4. Download V-JEPA (~1.3 GB)

```bash
.venv/bin/pip install -U huggingface_hub
.venv/bin/hf download facebook/vjepa2-vitl-fpc64-256 --local-dir models/vjepa2-vitl
```

If `hf` is not found, use `huggingface-cli` with the same arguments.
The pipeline finds `models/vjepa2-vitl/` automatically; if the folder
is absent it downloads from HuggingFace into a cache on first run
instead.

## 5. Download Gemma (~4.5 GB)

The pipeline needs a vision GGUF pair: one model file plus one
`mmproj-*` file. The pair this project was developed with:

```bash
.venv/bin/hf download HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf --local-dir models/gemma
```

Any other vision GGUF pair also works: put both files anywhere under
`models/` and `gemma-server.sh` finds them automatically. An official
Google Gemma build is preferable when one with an mmproj file is
available for llama.cpp; note that the model above is a
community finetune with a "reasoning" style (it thinks before
answering), which is why `gemma_max_tokens` in the config is large and
Stage 2 is slow.

## 6. Get the test video

The demo input is a segment of the AMI Meeting Corpus (free for
research), Corner camera, meeting ES2010a: four people around a table.

```bash
curl -O https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2010a/video/ES2010a.Corner_orig.avi
ffmpeg -i ES2010a.Corner_orig.avi -ss 00:03:00 -t 00:02:00 -c:v libx264 -pix_fmt yuv420p game.mp4
```

The second command cuts a 2-minute mp4 starting at minute 3. Any other
segment works; pick one with active discussion.

## 7. Stage 1: tracking + coordination scores

```bash
./run.sh game.mp4 --no-gemma
```

Runtime: several minutes. Watch the `[track]` lines: the "kept tracks"
count must equal the number of real people (4 for ES2010a). Then
verify identities visually:

```bash
cd vjepa-gemma-demo
../.venv/bin/python3 people_snapshot.py ../game.mp4
cd ..
```

Open `vjepa-gemma-demo/results/game/ids/scene_*.jpg`: one box per
person, no box covering two people (a box far larger than the others
means a fusion). A correct result looks like `docs/tracking_example.jpg`
at the project root: four colored boxes, one per person, labeled P1-P4. If the count is wrong:

```bash
cd vjepa-gemma-demo
../.venv/bin/python3 diagnose_tracks.py ../game.mp4
cd ..
```

and send that output to your advisor. Small adjustments live in
`vjepa-gemma-demo/src/config.py`: `min_track_coverage` (lower keeps
sparser tracks) and `det_conf` (lower detects smaller people).

## 8. Stage 2: Gemma descriptions

Terminal 1, from the ROOT, leave it running:

```bash
./gemma-server.sh
```

Wait until it reports listening on port 8080. Test it (optional):

```bash
cd vjepa-gemma-demo
../.venv/bin/python3 test_gemma.py
cd ..
```

Terminal 2, from the ROOT:

```bash
./run.sh game.mp4 --stage 2
```

This is the FRAMED condition: the prompt tells Gemma which pair was
detected as coordinated. With a reasoning model expect roughly a
minute per interval (24 intervals, plus 24 control descriptions).

Then run the BLIND condition (neutral prompt, same frames and pairs;
results are added next to the framed ones, nothing is overwritten):

```bash
./run.sh game.mp4 --stage 2 --blind
```

## 8b. Production mode (results only)

For users who just want the output, without the control descriptions
and experiment statistics:

```bash
./run.sh game.mp4 --production
```

One pass, video to results: per-interval top pair with its Gemma
description, the z/F/S matrices, and a one-line summary of the
strongest following edges. Roughly half the Stage 2 runtime. The
statistical calibration behind z is always on (it is what makes the
numbers meaningful and costs almost nothing); what production mode
drops is the validation apparatus (control pair, blind condition,
aggregate dumps). For any result you intend to report or publish, run
the full mode instead.

## 9. The blind-vs-framed comparison

Quick automatic answer (keyword-based):

```bash
cd vjepa-gemma-demo
../.venv/bin/python3 rating_sheet.py auto results/game/game.json
cd ..
```

Rigorous human-rated version: export a shuffled, condition-hidden
sheet, rate each description 0 (no interaction described), 1 (some),
or 2 (clear coordinated interaction) in the `rating` column, then
score:

```bash
cd vjepa-gemma-demo
../.venv/bin/python3 rating_sheet.py export results/game/game.json
open results/game/rating_sheet.csv
../.venv/bin/python3 rating_sheet.py score results/game/game.json
cd ..
```

Do not open `rating_key.json` until rating is finished, and ideally
the rater has not seen the videos or scores.

## 10. Outputs (vjepa-gemma-demo/results/game/)

| File | Content |
|---|---|
| `game.json` | Everything: per-interval top pair, z, lag, framed text (`event`, `event_control`), blind text (`event_blind`, `event_control_blind`), aggregates |
| `z_matrix.csv` | Row = interval, column = directed edge, value = z |
| `F_matrix.csv` | Adjacency matrix of F (row leads column) |
| `S_matrix.csv` | Adjacency matrix of Stouffer S |
| `ids/` | Who-is-who images |
| `rating_sheet.csv` | Blinded description rating sheet (step 9) |

## 11. How to read the numbers

- **z** (per pair, per interval): coordination beyond that interval's
  coincidence baseline, in SD units. The baseline (null) is built by
  replaying one person's behavior sequence with scrambled relative
  timing (circular shifts) and rescoring; z near 0 means the observed
  alignment is what accidental timing produces, z above about 2 is
  notable. Only z is comparable across intervals; the raw `score`
  (a cosine) is not.
- **S** (per directed edge, whole video): Stouffer combination of that
  edge's z over all intervals, approximately N(0,1) if there is no
  coordination. |S| above 2 is worth attention. Negative S =
  systematic anti-alignment.
- **F** (per directed edge, whole video): effect size in [0,1] -- the
  probability that the edge beats timing-scrambled coincidence in a
  randomly chosen interval. 0.5 = chance, toward 1 = consistent
  following (row person leads, column person follows), below 0.5 =
  anti-alignment. Judge reliability with S and the number of
  intervals; F alone from few intervals is noise. Diagonal 0s in
  `F_matrix.csv` are placeholders, not measurements.
- **Corpus-level check**: across all pair-intervals about 5% of z
  values would exceed 1.645 by chance; a substantially larger
  fraction (this project observed 14-22% on AMI segments) is the
  evidence that some real coordination exists somewhere.
- **event vs event_control** (framed): descriptions of the top and the
  bottom pair under a prompt that asserts coordination. If both read
  equally coordinated, the describer is narrating the prompt.
- **blind fields**: the fair test. Discrimination
  Delta = coordination-content(top) minus coordination-content(control)
  under the blind prompt. Delta clearly above 0 means descriptions are
  grounded in the video; Delta near 0 means the describer cannot
  resolve the behavior (the result obtained on AMI footage at this
  resolution).
- All quantities measure temporal coupling of visible behavior. None
  of them measure friendship, intention, emotion, or group
  membership; treat such interpretations as hypotheses for separate
  validation.

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: cv2` | Wrong Python found. Use `./run.sh`, or the explicit `.venv/bin/python3` path |
| `permission denied` running a .py file | Scripts run via Python: `.venv/bin/python3 script.py arguments` |
| `only 0 persistent track(s)` on first run | Run the same command again (a dependency installed itself mid-run) |
| Kept tracks fewer than real people | Step 7 diagnostics; try lowering `min_track_coverage` |
| Stage 2: server not reachable | Start `./gemma-server.sh` first, wait for port 8080 |
| Stage 2 descriptions empty or `[unfinished-reasoning]` | Increase `gemma_max_tokens` in `vjepa-gemma-demo/src/config.py` |
| `hf: command not found` | Use `huggingface-cli` with the same arguments |
| Command not found: brew / ffmpeg / llama-server | Section 2 |

## 13. Rules

- Run commands from the ROOT unless the block says otherwise.
- Never copy or commit `.venv/`; recreate it with `./setup.sh`.
- `vjepa-gemma-demo/` is replaceable code. `models/`, videos, and
  `results/` are the persistent data around it.
- `results/game/` accumulates framed and blind runs for the same
  video; archive with a copy, not a rename.
