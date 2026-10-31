"""The molab smoke-test orchestration: reports every failing component at once (subprocess stages are faked)."""

import json

import pandas as pd
import pytest

from c44 import config, molab, pipeline
from c44.manifest import Clip


@pytest.fixture()
def work(tmp_path, monkeypatch):
    monkeypatch.setenv("C44_WORK", str(tmp_path))
    return tmp_path


def _clip(cell, status="ok", **meta):
    pipeline.manifest().add(Clip(f"{cell}_s1", cell, "s1", f"clips/{cell}/x.mp4", 48, 30.0, 704, 480, status=status,
                                 meta=meta))


def _fake_stage(fail=()):
    def stage(args):
        name = args[0] if args[0] != "detect" else f"detect:{args[2]}"
        if name in fail:
            return 1, "Traceback...\nValueError: boom\n"
        if name == "rc":
            _clip("RC")
        elif name == "captions":
            (config.work_dir() / "captions.json").write_text(json.dumps({"RC_s1": "a kitchen with a table"}))
        elif name == "gr":
            _clip("GR") if "gr-clip-error" not in fail else _clip("GR", status="error", error="KeyError('x') @ app.py:9")
        elif name == "rr":
            _clip("RR", status="qc_fail")
        elif name.startswith("detect:"):
            p = config.work_dir() / "results"
            p.mkdir(exist_ok=True)
            old = pd.read_csv(p / "scores.csv") if (p / "scores.csv").exists() else pd.DataFrame()
            pd.concat([old, pd.DataFrame([{"clip_id": "RC_s1", "detector": args[2], "score": 0.3}])]).to_csv(
                p / "scores.csv", index=False)
        return 0, ""
    return stage


def test_smoke_passes(work, monkeypatch, capsys):
    monkeypatch.setattr(molab, "_stage", _fake_stage())
    molab._smoke()
    out = capsys.readouterr().out
    assert "SMOKE TEST PASSED" in out and "[FAIL]" not in out and "low coverage" in out


def test_smoke_lists_all_failures_at_once(work, monkeypatch):
    monkeypatch.setattr(molab, "_stage", _fake_stage(fail=("captions", "detect:npr", "gr-clip-error")))
    with pytest.raises(RuntimeError) as e:
        molab._smoke()
    msg = str(e.value)
    assert "captions" in msg and "detector npr" in msg and "GR" in msg and "KeyError('x') @ app.py:9" in msg
    assert "detector univfd" not in msg and "detector consistency" not in msg      # independent steps still pass


def test_smoke_skips_dependents_when_rc_fails(work, monkeypatch, capsys):
    monkeypatch.setattr(molab, "_stage", _fake_stage(fail=("rc",)))
    with pytest.raises(RuntimeError) as e:
        molab._smoke()
    assert "RC" in str(e.value) and "[SKIP]" in capsys.readouterr().out
