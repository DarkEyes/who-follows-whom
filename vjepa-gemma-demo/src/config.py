"""Central configuration. All paths are resolved relative to the project
root (the folder containing models/ and vjepa-gemma-demo/), so the whole
tree can be copied anywhere and run by anyone."""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


def _vjepa_model() -> str:
    local = MODELS / "vjepa2-vitl"
    if (local / "config.json").exists():
        return str(local)
    return "facebook/vjepa2-vitl-fpc64-256"


@dataclass
class Config:
    # --- windowing ---
    window_sec: float = 5.0
    frames_per_window: int = 32        # sampled uniformly inside each window
    # V-JEPA tubelet size is 2 frames -> 32 frames yield 16 temporal positions
    # -> 15 delta vectors per window (enough for lagged correlation).

    # --- tracking ---
    yolo_model: str = str(MODELS / "yolo11s.pt")
    max_people: int = 6
    min_track_coverage: float = 0.10   # after constrained fragment merging
    crop_expand: float = 0.15          # expand person box by 15% each side
    det_conf: float = 0.15             # low threshold for small seated people

    # --- V-JEPA backend ---
    # "hf20":  V-JEPA 2.0 ViT-L; uses models/vjepa2-vitl if present,
    #          otherwise downloads once from HuggingFace.
    # "hub21": V-JEPA 2.1 ViT-B via torch.hub (verify entrypoint name with
    #          torch.hub.list('facebookresearch/vjepa2')).
    jepa_backend: str = "hf20"
    hf_model: str = field(default_factory=_vjepa_model)
    hub_repo: str = "facebookresearch/vjepa2"
    hub_entrypoint: str = "vjepa2_1_vit_base"
    crop_size: int = 256               # 384 for the 2.1 checkpoints

    # --- coordination ---
    max_lag: int = 4                   # temporal-token steps (~1.25 s)
    n_null: int = 500                  # circular-shift null samples
    remove_global_mode: bool = True

    # --- Gemma ---
    # "llama": local GGUF via llama.cpp llama-server (OpenAI-compatible);
    #          start it with gemma-server.sh at the project root.
    # "mlx":   mlx-vlm with an mlx-community conversion.
    gemma_backend: str = "llama"
    llama_url: str = "http://localhost:8080/v1/chat/completions"
    llama_model: str = "gemma"         # llama-server ignores the name
    gemma_model: str = "mlx-community/gemma-3n-E4B-it-4bit"  # mlx only
    gemma_frames: int = 5              # representative frames per interval
    gemma_max_tokens: int = 1200       # reasoning finetunes think before answering
    run_control_pair: bool = True      # also describe lowest-scoring pair
                                       # (confabulation check)

    device: str = "mps"                # falls back to cpu automatically
    out_dir: str = "results"
