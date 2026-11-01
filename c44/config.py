"""Benchmark specification, paths and environment settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CELLS = ("RC", "RR", "GR", "GV")
CELL_NAMES = {
    "RC": "real scene, camera capture",
    "RR": "real scene, 3DGS render",
    "GR": "generated world, 3DGS render",
    "GV": "generated video, video diffusion (reference)",
}


@dataclass(frozen=True)
class Spec:
    """Every clip in every cell is normalised to this spec so that cells differ only in origin/formation."""

    width: int = 704          # FlashWorld's native generation width
    height: int = 480
    n_frames: int = 48
    fps: float = 30.0         # nominal frame rate written into every clip
    crf: int = 18             # libx264 quality of the clean benchmark version
    preset: str = "slow"
    keyframes: int = 24       # cameras given to FlashWorld per generated world
    background: tuple = (1.0, 1.0, 1.0)  # white, FlashWorld's default; identical for RR and GR
    min_coverage: float = 0.98            # QC: mean rendered alpha per frame
    max_native_fps: float = 45.0          # real video above this rate: keep every second frame
    seed: int = 0


SPEC = Spec()

# Pinned upstream code (verified 2026-10-31)
FLASHWORLD_REPO = "https://github.com/imlixinyang/FlashWorld.git"
FLASHWORLD_COMMIT = "424db4487de308a39e5137afbe85e7568a96193f"
FLASHWORLD_DIFFUSERS = "git+https://github.com/huggingface/diffusers.git@447e8322f76efea55d4769cd67c372edbf0715b8"
NPR_REPO = "https://github.com/chuangchuangtan/NPR-DeepfakeDetection.git"
NPR_COMMIT = "781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a"
UNIVFD_FC_URL = ("https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/"
                 "030495aea3300a8b54c0ec37ec7fe1dd7e63c619/pretrained_weights/fc_weights.pth")
VGGT_PIP = "git+https://github.com/facebookresearch/vggt.git@a288dd0f14786c93483e45524328726ab7b1b4ce"
GSPLAT_VERSION = "1.5.3"


def work_dir() -> Path:
    """Scratch directory for clips, metadata and caches (ephemeral on molab; persisted via storage.Store)."""
    p = Path(os.environ.get("C44_WORK", str(Path.home() / "c44_work")))
    p.mkdir(parents=True, exist_ok=True)
    return p


def clip_path(cell: str, clip_id: str) -> Path:
    p = work_dir() / "clips" / cell / f"{clip_id}.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def meta_path(cell: str, clip_id: str) -> Path:
    p = work_dir() / "meta" / cell / f"{clip_id}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_dotenv() -> None:
    """Export HF_TOKEN / C44_* from a .env next to the code (or the working dir) if not already set.
    molab's Secrets panel writes exactly such a file; reading it ourselves makes the token reach every library."""
    for d in (Path.cwd(), Path(__file__).resolve().parent.parent):
        p = d / ".env"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            key, sep, val = line.strip().partition("=")
            if sep and (key == "HF_TOKEN" or key.startswith("C44_")) and not os.environ.get(key):
                os.environ[key] = val.strip().strip('"').strip("'")


def hf_token() -> str | None:
    load_dotenv()
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def hf_repo() -> str | None:
    """Private HF dataset repo used to persist outputs across 12-hour molab sessions, e.g. 'you/c44-w2v'."""
    load_dotenv()
    return normalize_repo_id(os.environ.get("C44_HF_REPO"))


def normalize_repo_id(repo: str | None) -> str | None:
    """Accept 'user/name', 'datasets/user/name' or a full huggingface.co URL (with or without trailing '/' or '/tree/..')."""
    if not repo:
        return None
    r = repo.strip()
    for prefix in ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/"):
        if r.startswith(prefix):
            r = r[len(prefix):]
    r = r.removeprefix("datasets/").strip("/")
    return "/".join(r.split("/")[:2]) or None


load_dotenv()
