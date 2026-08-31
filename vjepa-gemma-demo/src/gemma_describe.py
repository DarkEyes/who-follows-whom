"""Gemma description stage. Two backends:

  llama : any vision GGUF served by llama.cpp's llama-server
          (OpenAI-compatible; needs -m model.gguf --mmproj mmproj.gguf).
          Also works with any other OpenAI-compatible local server
          by pointing cfg.llama_url at it.
  mlx      : mlx-vlm with an mlx-community Gemma 3n conversion.

Feeds ID-annotated representative frames of the FULL scene for the selected
interval and asks for observable behavior only.

Confabulation control: with cfg.run_control_pair, the lowest-z pair of the
same window is described with the identical prompt. If top-pair and control
descriptions are indistinguishable to a human, the description stage is
narrating the prompt, not the video. This check matters MORE with community
finetunes, which tend toward embellishment.
"""

import base64
import os
import tempfile

import cv2

from .tracking import box_at

PROMPT = (
    "You see {gap} frames sampled from a {dur:.0f}-second video interval. "
    "People are labeled {labels}. "
    "{a} and {b} were detected as having coordinated behavioral change in "
    "this interval. Describe ONLY the observable behavior involving {a} and "
    "{b}: posture, gaze direction, gestures, movements, and their timing "
    "relative to each other. Do not infer friendship, intention, "
    "personality, emotion, or group membership. If nothing notable is "
    "visible, say so. Two sentences maximum."
)

PROMPT_BLIND = (
    "You see {gap} frames, in temporal order, from a {dur:.0f}-second "
    "video. People are labeled {labels}. "
    "Describe the observable behavior of {a} and of {b} across these "
    "frames: posture, gaze direction, gestures, movements, and any "
    "changes over time. State only what is visible; if their behavior is "
    "unremarkable or unchanged, say exactly that. Do not infer "
    "friendship, intention, personality, emotion, or group membership. "
    "Two sentences maximum."
)


def annotate(frame, tracks: dict, frame_idx: int):
    out = frame.copy()
    for pid, tr in tracks.items():
        x1, y1, x2, y2 = (int(v) for v in box_at(tr, frame_idx))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, pid, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return out


def _build_prompt(n_frames, duration, tracks, pair, blind=False):
    tpl = PROMPT_BLIND if blind else PROMPT
    return tpl.format(
        gap=n_frames, dur=duration,
        labels=", ".join(sorted(tracks)),
        a=pair[0], b=pair[1],
    )


class GemmaDescriber:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = cfg.gemma_backend
        if self.backend == "mlx":
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
            self.model, self.processor = load(cfg.gemma_model)
            self.model_config = load_config(cfg.gemma_model)
        elif self.backend == "llama":
            import requests
            self._requests = requests
            # fail fast if the server isn't up
            try:
                requests.get(cfg.llama_url.rsplit("/chat", 1)[0]
                             + "/models", timeout=3)
            except Exception as e:
                raise RuntimeError(
                    f"llama-server not reachable at {cfg.llama_url}. "
                    "Start llama-server with the model and --mmproj "
                    "(see README)."
                ) from e
        else:
            raise ValueError(f"unknown gemma backend {self.backend}")

    # ---------- shared frame prep ----------
    def _annotated_jpegs(self, frames, tracks, frame_indices):
        tmpdir = tempfile.mkdtemp()
        paths = []
        for k, (frame, fi) in enumerate(zip(frames, frame_indices)):
            p = os.path.join(tmpdir, f"f{k}.jpg")
            cv2.imwrite(p, annotate(frame, tracks, int(fi)))
            paths.append(p)
        return paths

    # ---------- backends ----------
    def _describe_mlx(self, paths, prompt):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        formatted = apply_chat_template(
            self.processor, self.model_config, prompt, num_images=len(paths))
        out = generate(self.model, self.processor, formatted, paths,
                       max_tokens=self.cfg.gemma_max_tokens, verbose=False)
        return (out.text if hasattr(out, "text") else str(out)).strip()

    def _describe_llama(self, paths, prompt):
        content = [{"type": "text", "text": prompt}]
        for p in paths:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        payload = {
            "model": self.cfg.llama_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.cfg.gemma_max_tokens,
            "temperature": 0.1,
        }
        r = self._requests.post(self.cfg.llama_url, json=payload,
                                timeout=600)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        text = (msg.get("content") or "").strip()
        if text:
            return text
        # Reasoning models put thinking into reasoning_content and may
        # exhaust max_tokens before emitting the final answer. Salvage
        # the tail of the reasoning as a last resort, clearly marked.
        reasoning = (msg.get("reasoning_content") or "").strip()
        if reasoning:
            tail = reasoning[-400:].replace("\n", " ")
            return f"[unfinished-reasoning] ...{tail}"
        return "[empty response]"

    # ---------- public ----------
    def describe(self, frames, tracks, frame_indices, pair, duration,
                 blind=False):
        paths = self._annotated_jpegs(frames, tracks, frame_indices)
        prompt = _build_prompt(len(paths), duration, tracks, pair, blind)
        if self.backend == "mlx":
            return self._describe_mlx(paths, prompt)
        return self._describe_llama(paths, prompt)

    def unload(self):
        if self.backend == "mlx":
            self.model = None
            self.processor = None
