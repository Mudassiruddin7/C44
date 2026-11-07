"""Frozen-backbone clip embeddings for the retrained-probe baselines and the physics + learned fusion detector.

  results/clip_embeddings.npz   CLIP ViT-L/14 image embedding (768-d, the UnivFD backbone), mean over 8 frames
  results/npr_embeddings.npz    pooled features (512-d) of the NPR network (the NPR backbone), mean over 8 frames

The released UnivFD / NPR weights are evaluated zero-shot elsewhere; here the same backbones get a probe trained on the
benchmark's own dev scenes, which is the fair comparison for a method that is also trained on them. Resumable (finished
clips are kept) and incremental. Each backbone is optional: a failure is logged and the others still run.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import pipeline, setup_env, video
from .detectors.base import sample_frames
from .storage import Store

N_FRAMES = 8
CLIP_ID = "openai/clip-vit-large-patch14"


class ClipEmbedder:
    name, dim = "clip", 768

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        self.torch, self.device = torch, device
        self.proc = CLIPImageProcessor.from_pretrained(CLIP_ID)
        self.model = CLIPVisionModelWithProjection.from_pretrained(CLIP_ID, dtype=torch.float16).to(device).eval()

    def prep(self, frames: np.ndarray):
        return self.proc(images=list(sample_frames(frames, N_FRAMES)), return_tensors="pt")["pixel_values"]

    def encode(self, x):
        with self.torch.no_grad():
            return self.model(pixel_values=x.to(self.device, self.torch.float16)).image_embeds.float().cpu().numpy()


class NPREmbedder:
    """Pooled features of the NPR network: the input of its final linear layer (hooked on `avgpool`)."""
    name, dim = "npr", 512

    def __init__(self, device: str = "cuda"):
        from .detectors.npr import NPR
        det = NPR(repo_dir=setup_env.code_dir() / "NPR", device=device)
        self.torch, self.device, self.model, self.tf = det.torch, device, det.model, det.tf
        self._feat = None
        self.model.avgpool.register_forward_hook(lambda _m, _i, out: setattr(self, "_feat", out.flatten(1)))

    def prep(self, frames: np.ndarray):
        from PIL import Image
        return self.torch.stack([self.tf(Image.fromarray(f)) for f in sample_frames(frames, N_FRAMES)])

    def encode(self, x):
        with self.torch.no_grad():
            self.model(x.to(self.device))
        return self._feat.float().cpu().numpy()


def _save(path: Path, ids: list[str], emb: list[np.ndarray]) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, ids=np.array(ids), emb=np.stack(emb).astype(np.float32))
    tmp.replace(path)


def embed_clips(embedder, clips, path: Path, log=print, chunk: int = 64, workers: int = 4, push=None) -> int:
    """Embed every clip in `clips` that is not yet in `path` (mean over its sampled frames); returns the number added."""
    ids, emb = [], []
    if path.exists():
        with np.load(path, allow_pickle=True) as z:
            ids, emb = list(z["ids"]), list(z["emb"])
    have = set(ids)
    todo = [c for c in clips if c.clip_id not in have]
    log(f"[embed:{embedder.name}] {len(todo)} clips to embed ({len(have)} cached)")

    def load(c):
        return embedder.prep(video.read_video(pipeline._abs(c.path)))

    added = 0
    with ThreadPoolExecutor(workers) as ex:
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            xs = list(ex.map(load, part))                       # decode + preprocess in threads, encode in one batch
            feats = embedder.encode(embedder.torch.cat(xs) if hasattr(embedder, "torch") else np.concatenate(xs))
            sizes = np.cumsum([0] + [len(x) for x in xs])
            for c, a, b in zip(part, sizes[:-1], sizes[1:]):
                ids.append(c.clip_id)
                emb.append(feats[a:b].mean(0))
            added += len(part)
            _save(path, ids, emb)
            if (i // chunk) % 8 == 7:
                log(f"[embed:{embedder.name}] {min(i + chunk, len(todo))}/{len(todo)}")
                if push:
                    push()                                      # keep partial embeddings on HF in case the session is reset
    return added


def valid_clips(with_weak: bool = False):
    """Every ok clip except retired attack variants (same rule as the physics features); with_weak adds the c44.weak cells."""
    extra = set()
    if with_weak:
        from . import weak
        extra = set(weak.VARIANTS)
    return [c for c in pipeline.manifest().load().values()
            if c.status == "ok" and ("@" not in c.cell or c.cell.split("@")[1] in pipeline.VARIANTS or c.cell.split("@")[1] in extra)
            and pipeline._abs(c.path).exists()]


def run(store: Store | None = None, log=print, backbones=("clip", "npr"), with_weak: bool = False) -> dict:
    out = {}
    clips = valid_clips(with_weak)
    for name in backbones:
        path = pipeline.results_dir() / f"{name}_embeddings.npz"
        try:
            emb = ClipEmbedder() if name == "clip" else NPREmbedder()
            out[name] = embed_clips(emb, clips, path, log=log,
                                    push=lambda p=path, n=name: pipeline._flush(store, [p], f"c44: {n} embeddings (partial)"))
            del emb
            pipeline._release_gpu()
        except Exception as e:                                  # noqa: BLE001 - one backbone failing must not stop the other
            log(f"[embed:{name}] FAILED: {e!r}")
            out[name] = -1
        pipeline._flush(store, [path], f"c44: {name} embeddings")
    return out


def load(name: str = "clip", path=None):
    """(ids, embeddings) of a backbone, or None if it was not computed."""
    p = Path(path) if path else pipeline.results_dir() / f"{name}_embeddings.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as z:
        return z["ids"], z["emb"]
