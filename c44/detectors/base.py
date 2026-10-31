from __future__ import annotations

import numpy as np


class Detector:
    name = "base"
    needs_geometry = False

    def score(self, frames: np.ndarray, geom=None) -> float:   # frames: uint8 [T,H,W,3]
        raise NotImplementedError


def sample_frames(frames: np.ndarray, n: int) -> np.ndarray:
    idx = np.unique(np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int))
    return frames[idx]
