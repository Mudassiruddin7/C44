"""Resuming in a fresh session: variant clips come back from the HF repo, their geometry files do not."""

import pytest

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


def test_lite_pull_skips_geometry_and_lite_stage_order(monkeypatch):
    from c44 import cli, molab, setup_env
    assert not any(p.startswith("meta/") for p in cli.PULL_LITE) and "meta/RC/*" in cli.PULL_FULL
    assert "clips/*@nz-*/*" in cli.PULL_LITE and "clips/*@aecal/*" in cli.PULL_LITE
    calls = []
    for name in ("require_gpu", "install_base", "install_npr"):
        monkeypatch.setattr(setup_env, name, lambda n=name: calls.append(n))
    monkeypatch.setattr(molab, "_need", lambda args: calls.append(" ".join(args)))
    monkeypatch.setattr(molab, "_check_imports", lambda: (True, ""))
    molab.run("lite")
    assert calls == ["require_gpu", "install_base", "install_npr", "pull --lite"]
    monkeypatch.setattr(molab, "_check_imports", lambda: (False, "ImportError: huggingface-hub>=1.5.0,<2.0 is required"))
    calls.clear()
    with pytest.raises(RuntimeError, match="cannot import"):
        molab.run("lite")
    assert "pull --lite" not in calls                                   # fails before anything long starts
