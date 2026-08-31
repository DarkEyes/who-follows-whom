# Method Notes

Design rationale, test footage, acceptance criteria, and the phased plan.
Operational instructions live in [GUIDE.md](GUIDE.md).

## Method notes

- **Dense temporal features:** one V-JEPA forward pass per person-window
  yields 16 temporal token positions (32 frames / tubelet 2), spatially
  pooled → 15 Δz samples. No 4-chunk splitting; correlation over 3 samples
  would be noise.
- **Global-mode removal:** per timestep, the cross-person mean Δz is
  subtracted before scoring. Removes camera/lighting/scene-wide common
  causes that otherwise inflate every pair.
- **Directed lagged score:** max over lag τ ∈ [0, 4 steps] (~1.25 s) of
  mean cosine between Δz sequences, both directions. τ > 0 with i leading
  gives the following edge i → j.
- **Null calibration:** circular-shift null (500 samples/window) → each
  pair reported as a z-score. Raw cosines are uninterpretable.
- **Confabulation control:** Gemma also describes the *lowest*-z pair with
  the identical prompt (`event_control` field). If top and control
  descriptions read the same, the description stage is narrating the
  prompt, not the video.

## Test footage

Run order: staged clip → AMI segment → Werewolf Among Us games.

**Staged clip (first — only input with a known correct answer).**
Phone on tripod, 60–90 s, one person deliberately mirrors another at
~1 s lag, one person acts independently.

**AMI Meeting Corpus (plausible-signal test).** Fixed Corner camera,
4 people around a table, free for research. Direct download (any meeting
ID follows the same pattern; take `_orig`, not the low-bitrate version):

```
https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2010a/video/ES2010a.Corner_orig.avi
```

Browse: `https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/`
Download page: `https://groups.inf.ed.ac.uk/ami/download/`

```bash
ffmpeg -i ES2010a.Corner_orig.avi -ss 00:03:00 -t 00:02:00 \
       -c:v libx264 -pix_fmt yuv420p game.mp4
```

Pick a segment with active discussion (turn-taking, gaze shifts), not
silent writing. Cameras are PAL-era (~352×288): crops will be small.
Do not switch to the Overhead view (top-down kills appearance features).

**Werewolf Among Us (Lai et al., ACL 2023) — for Phases 1–2.**
199 social-deduction game videos with game-outcome annotations
(= faction ground truth).
Dataset: `https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us`
Code: `https://github.com/SALT-NLP/PersuationGames`
YouTube portion ships as video ids (`Youtube/youtube_urls_released.json`);
download via yt-dlp. Videos are 640×320; games are One Night Ultimate
Werewolf; skip the Ego4D portion (separate license); expect some dead ids.

**CMU Panoptic "Haggling" sequences** (`domedb.perception.cs.cmu.edu`) —
later sanity check: ground-truth 3D pose lets you validate V-JEPA-derived
coordination against pose-derived coordination directly.

## Acceptance criteria (what "technically possible" means)

Test on a staged clip first: 2 people where one deliberately mirrors the
other at ~1 s lag, plus 1 independent person, 60–90 s.

1. Staged pair ranked #1 (by z) in the majority of windows, lag sign correct.
2. Staged pair z ≫ 0; uninvolved pairs z ≈ 0.
3. Gemma's top-pair description matches the clip on human inspection and
   is distinguishable from the control description.

Pass all three → the claim "frozen V-JEPA + lagged correlation surfaces
following episodes without training, and a local VLM can narrate them"
is demonstrated.

## Plan for the larger task

**Phase 0 — this demo.** Staged clip → cooperative game/meeting clip.
Deliverable: acceptance criteria above.

**Phase 1 — werewolf corpus.** Record or collect N ≥ 10 games with a fixed
camera and post-game role labels (the video analogue of AIWolf server
logs: free ground truth). Per game: full pairwise (z, τ) matrices per
window; aggregate F(i→j) and lag-sign consistency.

**Phase 2 — faction evaluation.** Test edge scores against the
same-faction indicator, **two-sided** (deception predicts suppressed or
inverted coordination between werewolves; anomalously low coordination
between verbally interacting players is also a cue). AUC + permutation
test. One 5-player game = 10 pairs — no faction claims from a single
video; power comes from the corpus.

**Phase 3 — only if Phase 2 shows signal.** Learned probe heads on frozen
V-JEPA features for interaction-type classification; audio/diarization;
comparison against gaze-based baselines. Everything on the original
"what not to add yet" list stays out until here.

Risk ordering: Stage 1 (does the ranking beat the null?) is the only
scientifically risky component; tracking, Gemma, and the network
aggregation are engineering.
