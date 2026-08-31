"""V-JEPA encoder wrapper.

Two backends:
  hf20  : V-JEPA 2.0 ViT-L via transformers (stable today, recommended first run)
  hub21 : V-JEPA 2.1 ViT-B via torch.hub (2.1 loads only through torch.hub;
          verify the entrypoint name with torch.hub.list(cfg.hub_repo))

encode_clip() returns a TEMPORAL SEQUENCE of embeddings, one per tubelet
position (spatial mean-pool), NOT a single pooled vector. With 32 frames and
tubelet size 2 this yields 16 temporal positions -> 15 delta vectors/window.
"""

import numpy as np
import torch


def _pick_device(pref: str) -> str:
    if pref == "mps" and torch.backends.mps.is_available():
        return "mps"
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class JepaEncoder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = _pick_device(cfg.device)
        self.backend = cfg.jepa_backend
        if self.backend == "hf20":
            from transformers import AutoModel, AutoVideoProcessor
            self.processor = AutoVideoProcessor.from_pretrained(cfg.hf_model)
            self.model = AutoModel.from_pretrained(
                cfg.hf_model, torch_dtype=torch.float16
            ).to(self.device).eval()
        elif self.backend == "hub21":
            # 2.1 checkpoints: torch.hub only (no transformers support yet).
            self.model = torch.hub.load(
                cfg.hub_repo, cfg.hub_entrypoint
            ).to(self.device).eval()
            self.processor = None
        else:
            raise ValueError(f"unknown backend {self.backend}")

    @torch.no_grad()
    def encode_clip(self, frames: np.ndarray) -> np.ndarray:
        """frames: (T, H, W, 3) RGB uint8 -> (T_tokens, D) float32."""
        if self.backend == "hf20":
            inputs = self.processor(
                list(frames), return_tensors="pt"
            ).to(self.device)
            out = self.model(**inputs)
            tok = out.last_hidden_state[0].float()          # (N, D)
        else:
            x = torch.from_numpy(frames).permute(3, 0, 1, 2)  # (3,T,H,W)
            x = (x.float() / 255.0).unsqueeze(0).to(self.device)
            tok = self.model(x)[0].float()                   # (N, D)

        # tokens are (T/2) * (H/16) * (W/16); recover temporal axis
        t_pos = frames.shape[0] // 2
        n, d = tok.shape
        assert n % t_pos == 0, f"token count {n} not divisible by {t_pos}"
        tok = tok.reshape(t_pos, n // t_pos, d).mean(dim=1)  # spatial pool
        return tok.cpu().numpy()

    def unload(self):
        del self.model
        self.model = None
        if self.device == "mps":
            torch.mps.empty_cache()
