"""Video I/O. Decode with PyAV; encode with the static ffmpeg bundled in imageio-ffmpeg (libx264)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def probe(path: str | Path) -> dict:
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        rate = s.average_rate or s.guessed_rate or s.base_rate
        fps = float(rate) if rate else 0.0
        n = int(s.frames or 0)
        if n == 0 and s.duration is not None and fps:
            n = int(round(float(s.duration * s.time_base) * fps))
        return {"fps": fps, "n_frames": n, "width": s.codec_context.width, "height": s.codec_context.height}


def crop_box(w: int, h: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Centre crop (x0, y0, cw, ch) of a w x h image that has the aspect ratio width/height."""
    target = width / height
    if w / h > target:
        cw, ch = int(round(h * target)), h
    else:
        cw, ch = w, int(round(w / target))
    return (w - cw) // 2, (h - ch) // 2, cw, ch


def crop_resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Centre-crop to the target aspect ratio, then resize (area filter when shrinking)."""
    h, w = img.shape[:2]
    x0, y0, cw, ch = crop_box(w, h, width, height)
    img = img[y0:y0 + ch, x0:x0 + cw]
    if (cw, ch) != (width, height):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA if cw > width else cv2.INTER_CUBIC)
    return np.ascontiguousarray(img)


def crop_resize_K(K: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    """Intrinsics after crop_resize (pixel centres at integer coordinates, OpenCV convention)."""
    (w, h), (W, H) = src_wh, dst_wh
    x0, y0, cw, ch = crop_box(w, h, W, H)
    K = np.array(K, dtype=np.float64).copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    sx, sy = W / cw, H / ch
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K


def iter_frames(path: str | Path, start: int = 0, step: int = 1, count: int | None = None,
                size: tuple[int, int] | None = None) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames start, start+step, ... (sequential decode; optional crop_resize to size=(W, H))."""
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        emitted = 0
        for i, fr in enumerate(c.decode(s)):
            if i < start or (i - start) % step:
                continue
            img = fr.to_ndarray(format="rgb24")
            yield crop_resize(img, *size) if size else img
            emitted += 1
            if count is not None and emitted >= count:
                return


def read_video(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    return np.stack(list(iter_frames(path, size=size)))


def write_mp4(frames: np.ndarray, path: str | Path, fps: float, crf: int = 18, preset: str = "slow") -> Path:
    """Encode RGB uint8 frames [T,H,W,3] as H.264/yuv420p (H and W must be even)."""
    import imageio_ffmpeg
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, _ = frames.shape
    w = imageio_ffmpeg.write_frames(str(path), (W, H), fps=fps, codec="libx264", pix_fmt_in="rgb24",
                                    pix_fmt_out="yuv420p", quality=None, macro_block_size=1,
                                    output_params=["-crf", str(crf), "-preset", preset])
    w.send(None)
    for f in frames:
        w.send(np.ascontiguousarray(f, dtype=np.uint8))
    w.close()
    return path


def reencode(src: str | Path, dst: str | Path, crf: int, scale: float = 1.0, fps: float | None = None) -> Path:
    """Re-compress (and optionally rescale) a clip, e.g. for the codec-robustness sweep."""
    frames = read_video(src)
    if scale != 1.0:
        H, W = frames.shape[1:3]
        nw, nh = int(round(W * scale / 2) * 2), int(round(H * scale / 2) * 2)
        frames = np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in frames])
    return write_mp4(frames, dst, fps or probe(src)["fps"] or 30.0, crf=crf)
