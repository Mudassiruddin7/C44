"""Detectors. Every detector returns a per-clip score where higher means 'more likely not real camera footage'."""

from .base import Detector, sample_frames


def make(name: str, **kw) -> Detector:
    if name == "univfd":
        from .univfd import UnivFD
        return UnivFD(**kw)
    if name == "npr":
        from .npr import NPR
        return NPR(**kw)
    if name == "consistency":
        from .consistency import ReprojectionConsistency
        return ReprojectionConsistency(**kw)
    if name == "vlm_judge":
        from .vlm_judge import VLMJudge
        return VLMJudge(**kw)
    raise ValueError(f"unknown detector {name}")


__all__ = ["Detector", "sample_frames", "make"]
