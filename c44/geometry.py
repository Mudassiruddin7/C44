"""Per-clip camera geometry with VGGT, plus numpy geometry used by CPS and the consistency detector.

VGGT (facebookresearch/vggt) predicts OpenCV world-to-camera extrinsics (first camera = identity), pixel
intrinsics and depth maps at its input resolution (width 518, height a multiple of 14).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from .cameras import invert, to44


@dataclass
class Geometry:
    w2c: np.ndarray        # [T,4,4] OpenCV world-to-camera
    K: np.ndarray          # [T,3,3] intrinsics in model (depth-map) pixels
    depth: np.ndarray      # [T,h,w] float32
    conf: np.ndarray       # [T,h,w] float32
    image_hw: tuple        # (H, W) of the original frames
    y0: int = 0            # rows cropped from the resized image before the model (crop mode)

    @property
    def model_hw(self) -> tuple:
        return tuple(self.depth.shape[1:])

    def _scales(self) -> tuple[float, float]:
        H, W = self.image_hw
        h, w = self.model_hw
        nh = h + 2 * self.y0
        return W / w, H / nh

    def K_image(self) -> np.ndarray:
        """Intrinsics in original-image pixels."""
        sx, sy = self._scales()
        A = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, (self.y0 + 0.5) * sy - 0.5], [0, 0, 1]])
        return A[None] @ self.K

    def save(self, path) -> None:
        np.savez_compressed(path, w2c=self.w2c, K=self.K, depth=self.depth.astype(np.float16),
                            conf=self.conf.astype(np.float16), image_hw=np.array(self.image_hw), y0=self.y0)

    @classmethod
    def load(cls, path) -> "Geometry":
        d = np.load(path)
        return cls(d["w2c"], d["K"], d["depth"].astype(np.float32), d["conf"].astype(np.float32),
                   tuple(int(x) for x in d["image_hw"]), int(d["y0"]))


def model_size(H: int, W: int, target: int = 518) -> tuple[int, int, int]:
    """VGGT 'crop' preprocessing: width -> 518, height rounded to a multiple of 14, centre-cropped to <= 518."""
    nh = int(round(H * (target / W) / 14.0) * 14)
    h = min(nh, target)
    return h, target, (nh - h) // 2


def resize_for_model(frames: np.ndarray) -> np.ndarray:
    """Frames [T,H,W,3] uint8 -> float32 [T,h,w,3] in [0,1], exactly as VGGT's load_and_preprocess_images."""
    from PIL import Image
    T, H, W, _ = frames.shape
    h, w, y0 = model_size(H, W)
    nh = h + 2 * y0
    out = np.empty((T, h, w, 3), np.float32)
    for t, f in enumerate(frames):
        r = np.asarray(Image.fromarray(f).resize((w, nh), Image.Resampling.BICUBIC), np.float32) / 255.0
        out[t] = r[y0:y0 + h]
    return out


class VGGTGeometry:
    """Lazy VGGT wrapper (GPU)."""

    def __init__(self, device: str = "cuda", model_id: str = "facebook/VGGT-1B"):
        import torch
        from vggt.models.vggt import VGGT
        self.torch = torch
        self.device = device
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.model = VGGT.from_pretrained(model_id).to(device).eval()

    def __call__(self, frames: np.ndarray) -> Geometry:
        torch = self.torch
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        T, H, W, _ = frames.shape
        h, w, y0 = model_size(H, W)
        x = torch.from_numpy(resize_for_model(frames)).permute(0, 3, 1, 2)[None].to(self.device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=self.dtype):
                tokens, ps_idx = self.model.aggregator(x)
            pose_enc = self.model.camera_head(tokens)[-1]
            extr, intr = pose_encoding_to_extri_intri(pose_enc, x.shape[-2:])
            depth, conf = self.model.depth_head(tokens, x, ps_idx)
        return Geometry(w2c=to44(extr[0].float().cpu().numpy()), K=intr[0].float().cpu().numpy().astype(np.float64),
                        depth=depth[0, ..., 0].float().cpu().numpy(), conf=conf[0].float().cpu().numpy(),
                        image_hw=(H, W), y0=y0)


# ----------------------------------------------------------------------------- numpy geometry
def project(geom: Geometry, t: int, t2: int, stride: int = 8, conf_q: float = 30.0):
    """Project a pixel grid of frame t (with its depth) into frame t2. Returns model-pixel uv0, uv1, depth z1."""
    h, w = geom.model_hw
    vs, us = np.mgrid[stride // 2:h:stride, stride // 2:w:stride]
    d = geom.depth[t][vs, us]
    c = geom.conf[t][vs, us]
    ok = (d > 1e-6) & (c >= np.percentile(geom.conf[t], conf_q))
    pix = np.stack([us[ok], vs[ok], np.ones(ok.sum())]).astype(np.float64)
    Xc = (np.linalg.inv(geom.K[t]) @ pix) * d[ok]
    c2w = invert(geom.w2c[t])
    Xw = c2w[:3, :3] @ Xc + c2w[:3, 3:]
    w2 = to44(geom.w2c[t2])
    X2 = w2[:3, :3] @ Xw + w2[:3, 3:]
    z = X2[2]
    front = z > 1e-6
    uv1 = np.full((2, z.size), np.nan)
    uv1[:, front] = (geom.K[t2] @ X2[:, front])[:2] / z[front]
    return pix[:2].T, uv1.T, z


def motion(geom: Geometry, stride: int = 8, grid: tuple[int, int] = (6, 8)) -> dict:
    """Per-frame image motion to the next frame in original-image pixels: median magnitude m [T],
    mean unit direction [T,2] and a per-cell median flow grid [T,gh,gw,2]."""
    T = len(geom.depth)
    sx, sy = geom._scales()
    h, w = geom.model_hw
    gh, gw = grid
    m = np.full(T, np.nan)
    direction = np.full((T, 2), np.nan)
    flow_grid = np.full((T, gh, gw, 2), np.nan)
    for t in range(T - 1):
        uv0, uv1, _ = project(geom, t, t + 1, stride)
        ok = np.all(np.isfinite(uv1), axis=1)
        if ok.sum() < 10:
            continue
        f = (uv1[ok] - uv0[ok]) * np.array([sx, sy])
        m[t] = float(np.median(np.linalg.norm(f, axis=1)))
        mean = f.mean(0)
        if np.linalg.norm(mean) > 1e-9:
            direction[t] = mean / np.linalg.norm(mean)
        ci = np.clip((uv0[ok, 1] * gh / h).astype(int), 0, gh - 1)
        cj = np.clip((uv0[ok, 0] * gw / w).astype(int), 0, gw - 1)
        for i in range(gh):
            for j in range(gw):
                sel = (ci == i) & (cj == j)
                if sel.sum() >= 3:
                    flow_grid[t, i, j] = np.median(f[sel], axis=0)
    if T > 1:
        m[-1], direction[-1], flow_grid[-1] = m[-2], direction[-2], flow_grid[-2]
    return {"m": m, "direction": direction, "flow_grid": flow_grid}


def _gray(frames_model: np.ndarray) -> np.ndarray:
    return frames_model @ np.array([0.299, 0.587, 0.114], np.float32)


def reprojection_error(frames: np.ndarray, geom: Geometry, k: int = 4, stride: int = 4,
                       occl_tol: float = 0.1) -> float:
    """Multi-view photometric consistency (GenWorld-style criterion): warp frame t+k into frame t with depth and
    poses, compare standardised intensities on visible pixels. Higher = less geometrically consistent."""
    gray = _gray(resize_for_model(frames))
    T = len(gray)
    h, w = geom.model_hw
    errs = []
    for t in range(0, T - k, max(1, k // 2)):
        uv0, uv1, z1 = project(geom, t, t + k, stride)
        ok = np.all(np.isfinite(uv1), axis=1) & (uv1[:, 0] >= 0) & (uv1[:, 0] <= w - 1) & \
            (uv1[:, 1] >= 0) & (uv1[:, 1] <= h - 1)
        if ok.sum() < 50:
            continue
        coords = uv1[ok][:, ::-1].T                               # (row, col)
        d2 = map_coordinates(geom.depth[t + k], coords, order=1)
        vis = np.abs(d2 - z1[ok]) / np.maximum(z1[ok], 1e-6) < occl_tol
        if vis.sum() < 50:
            continue
        a = gray[t][uv0[ok][vis, 1].astype(int), uv0[ok][vis, 0].astype(int)]
        b = map_coordinates(gray[t + k], coords[:, vis], order=1)
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        errs.append(float(np.median(np.abs(a - b))))
    return float(np.mean(errs)) if errs else float("nan")
