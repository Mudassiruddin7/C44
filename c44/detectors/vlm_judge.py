"""Zero-shot vision-language judge: asks for P(not real camera footage) from 8 frames."""

from __future__ import annotations

import numpy as np

from ..vlm import JUDGE_PROMPT, VLM, parse_probability
from .base import Detector, sample_frames


class VLMJudge(Detector):
    name = "vlm_judge"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda", n_frames: int = 8):
        self.vlm, self.n = VLM(model_id, device), n_frames

    def score(self, frames: np.ndarray, geom=None) -> float:
        return parse_probability(self.vlm.ask(list(sample_frames(frames, self.n)), JUDGE_PROMPT, max_new_tokens=8))
