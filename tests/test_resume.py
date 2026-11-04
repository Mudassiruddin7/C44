"""Resuming in a fresh session: variant clips come back from the HF repo, their geometry files do not."""

from c44 import config, pipeline
from c44.manifest import Clip


def test_variant_geometry_falls_back_to_the_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("C44_WORK", str(tmp_path))
    var = Clip("GR_ab@blurcal", "GR@blurcal", "ab", "clips/GR@blurcal/GR_ab@blurcal.mp4", 48, 30.0, 704, 480, parent="GR_ab")
    assert pipeline._meta_of(var) == config.meta_path("GR", "GR_ab")                   # no own copy after a resume
    own = config.meta_path("GR@blurcal", "GR_ab@blurcal")
    own.parent.mkdir(parents=True, exist_ok=True)
    own.write_bytes(b"x")
    assert pipeline._meta_of(var) == own                                               # its own copy wins when it exists
    base = Clip("GR_ab", "GR", "ab", "clips/GR/GR_ab.mp4", 48, 30.0, 704, 480)
    assert pipeline._meta_of(base) == config.meta_path("GR", "GR_ab")
