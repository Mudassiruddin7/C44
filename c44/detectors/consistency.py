"""Multi-view consistency detector: the criterion GenWorld's SpannDetector builds on (its code is not public),
implemented as photometric reprojection error under VGGT depth and poses. Renders of explicit 3D worlds should
satisfy it by construction, which is the hypothesis C44 tests."""

from __future__ import annotations

import numpy as np

from ..geometry import reprojection_error
from .base import Detector


class ReprojectionConsistency(Detector):
    name = "consistency"
    needs_geometry = True

    def __init__(self, k: int = 4, stride: int = 4):
        self.k, self.stride = k, stride

    def score(self, frames: np.ndarray, geom=None) -> float:
        if geom is None:
            raise ValueError("consistency detector needs the clip's Geometry")
        return reprojection_error(frames, geom, k=self.k, stride=self.stride)
