"""c44.weak / c44.weak_analysis / c44.bootstats: operators, the attacker ladder, the feature stage, and the analysis on fabricated tables."""

import numpy as np
import pandas as pd
import pytest

from c44 import analysis, bootstats, cps, physics, pipeline, video, weak, weak_analysis
from c44.config import SPEC
from c44.manifest import Clip
from tests.synthetic import clip


def _frames(kind, seed=1, T=24):
    out = clip(kind, seed, T=T, h=192, w=256)
    return np.asarray(out[0] if isinstance(out, tuple) else out)


@pytest.fixture(scope="module")
def render():
    return _frames("render")


@pytest.fixture(scope="module")
def mo(render):
    return physics.flow_motion(render)


@pytest.mark.parametrize("name", list(weak._OPS))
def test_every_operator_keeps_shape_dtype_and_changes_the_clip(name, render, mo):
    out, params = weak._OPS[name](render, np.random.default_rng(3), mo)
    assert out.shape == render.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - render.astype(int)).mean() > 0.05 and params
    out2, _ = weak._OPS[name](render, np.random.default_rng(3), mo)
    assert np.array_equal(out, out2)                                   # deterministic in the seed


def test_tnr_reduces_temporal_noise_and_chroma_reduces_chroma_noise():
    noisy = _frames("camera")
    mo = physics.flow_motion(noisy)
    out, _ = weak.op_tnr(noisy, np.random.default_rng(0), mo)
    d = lambda x: np.abs(np.diff(x.astype(float), axis=0)).mean()      # noqa: E731 - frame-to-frame change is dominated by the noise here
    assert d(out) < d(noisy)
    ch, _ = weak.op_chroma(noisy, np.random.default_rng(0))
    ycc = lambda x: np.stack([__import__("cv2").cvtColor(f, __import__("cv2").COLOR_RGB2YCrCb)[..., 1] for f in x]).astype(float)   # noqa: E731
    assert ycc(ch).std() < ycc(noisy).std()


def test_apply_ops_order_params_and_random3(render, mo):
    out, p = weak.apply_ops(render, ["ae", "tnr", "blur", "optics"], np.random.default_rng(1), 5, mo)
    assert p["ops"] == ["optics", "blur", "tnr", "ae"] and {"blur_exposure", "gain_sigma", "tnr_keep", "optics_k1"} <= set(p)
    assert out.shape == render.shape
    _, p3 = weak.apply_variant(render, {"ops": "random3"}, 9)
    assert len(p3["ops"]) == 3 and set(p3["ops"]) <= set(weak.ALL_FAMS)
    assert weak.PARTS["nz-mix"] == set(weak.ALL_FAMS) and set(weak.PARTS) == set(weak.NZ_KINDS)


def test_joint_features_are_finite_on_a_camera_clip_and_never_raise_on_junk():
    f = weak.joint_features(_frames("camera"))
    assert set(f) == set(weak.JOINT) and np.isfinite(list(f.values())).sum() >= 5
    for junk in (np.zeros((24, 192, 256, 3), np.uint8), _frames("camera", T=4)):       # constant clip, too-short clip
        g = weak.joint_features(junk)
        assert set(g) == set(weak.JOINT)


def test_features_plus_shares_the_flow_and_matches_clip_physics(render):
    a, b = weak.clip_features_plus(render), physics.clip_physics(render)
    assert all(np.isclose(a[k], b[k], equal_nan=True) for k in physics.ALL)


@pytest.fixture(autouse=True)
def _never_touch_the_real_hub(monkeypatch):
    """c44/.env holds a real HF token: no test may fall back to it, whatever C44_HF_REPO says in the shell."""
    from c44 import config as _config
    monkeypatch.setattr(_config, "hf_repo", lambda: None)
    monkeypatch.setattr(_config, "hf_token", lambda: None)


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setenv("C44_WORK", str(tmp_path))
    monkeypatch.setattr(pipeline, "_default_workers", lambda: 1)
    return tmp_path


def _world(work, n=3, T=24):
    """n scenes with RC (camera), GR and RR (renders); ids as in the real manifest."""
    man = pipeline.manifest()
    for i in range(n):
        scene = f"{i:02x}" + "cd" * 31
        sid = scene[:16]
        for cell, kind, seed in (("RC", "camera", i), ("GR", "render", i + 50), ("RR", "render", i + 90)):
            cid = f"{cell}_{sid}"
            path = config_path(work, cell, cid)
            video.write_mp4(_frames(kind, seed, T), path, SPEC.fps, 23, "ultrafast")
            man.add(Clip(cid, cell, scene, path.relative_to(work).as_posix(), T, SPEC.fps, 256, 192,
                         parent="" if cell == "RC" else f"RC_{sid}"))
    return man


def config_path(work, cell, cid):
    p = work / "clips" / cell / f"{cid}.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def test_build_variants_for_weak_kinds_pairs_parents_and_runs_without_geometry(work):
    _world(work)
    assert not (work / "meta").exists()                                    # as after `pull --lite`: no geometry at all
    made = pipeline.build_variants("nz-tnr", cells=("RC", "RR", "GR"), n_jobs=1)
    assert len(made) == 9 and {c.cell for c in pipeline.manifest().load().values() if "@nz-tnr" in c.cell} == {"RC@nz-tnr", "RR@nz-tnr", "GR@nz-tnr"}
    assert pipeline.build_variants("nz-tnr", cells=("RC", "RR", "GR"), n_jobs=1) == []              # resumable
    lad = pipeline.build_variants("wk-L1", cells=("RR", "GR"), n_jobs=1)                             # fakes only, parent = the real clip
    assert len(lad) == 6 and not any(c.cell == "RC@wk-L1" for c in pipeline.manifest().load().values())
    m = pipeline.manifest().load()[lad[0]]
    assert m.parent.startswith(("GR_", "RR_")) and {"noise_strength", "gain_sigma", "ladder"} <= set(m.meta)


def test_ladder_rung_matches_the_real_noise_level_and_each_rung_runs(work):
    _world(work, n=1)
    rc, gr = _frames("camera", 0), _frames("render", 50)
    target = cps.resid_level(rc[2:weak.CAL_FRAMES])
    for level in (1, 3):
        out, extra = weak._ladder(gr, level, 7, rc)
        assert out.shape == gr.shape and out.dtype == np.uint8 and extra["ladder"] == level
        assert 0.6 < extra["resid_got"] / target < 1.4, (level, extra["resid_got"], target)         # calibrated like the main attacker
    out2, _ = weak._ladder(gr, 2, 7, rc)
    assert out2.shape == gr.shape
    with pytest.raises(ValueError):
        weak.apply_variant(gr, {"ladder": 1}, 0, None)


def test_feature_stage_is_resumable_and_writes_both_tables(work):
    _world(work)
    pipeline.build_variants("nz-chroma", cells=("RC", "GR"), n_jobs=1)
    out = weak.run_features(store=None, n_jobs=1)
    res = pipeline.results_dir()
    j, p = pd.read_csv(res / "joint_features.csv"), pd.read_csv(res / "physics_features_weak.csv")
    assert len(j) == 9 + 6 and len(p) == 6 and set(p.clip_id) <= set(j.clip_id) and "gain_mad" in p and "jt_rho_fs" in j
    assert out == {"joint": 15, "physics_weak": 6}
    assert weak.run_features(store=None, n_jobs=1) == out                                          # nothing redone


# ----------------------------------------------------------------------------- the analysis on fabricated tables
def _fabricate(res, n_scenes=70, seed=0, embeddings=False):
    """Features where fakes are shifted in a few CPS-2 cues; ladder rungs and operator kinds partly erase the shift."""
    rng = np.random.default_rng(seed)
    kinds = ["", "crf35", "blurcal+noiseadapt+aecal", *weak.NZ_KINDS, *weak.LADDER]
    rows = []
    for s in range(n_scenes):
        scene = f"{s:02x}" + "ab" * 7                                                           # 16 hex chars
        base = rng.normal(0, 1, len(physics.ALL))
        for cell in ("RC", "RR", "GR"):
            for k in kinds:
                if k in weak.LADDER and cell == "RC":
                    continue
                if k in ("crf35", "blurcal+noiseadapt+aecal") and False:
                    continue
                shift = 0.0 if cell == "RC" else (1.4 if cell == "GR" else 1.0)
                if k in weak.LADDER:
                    shift *= 0.5 ** (weak.LADDER.index(k))
                elif k == "blurcal+noiseadapt+aecal":
                    shift *= 0.3
                vec = base + rng.normal(0, 1, len(physics.ALL))
                vec[:6] += shift
                cid = f"{cell}_{scene}" + (f"@{k}" if k else "")
                rows.append({"clip_id": cid, **dict(zip(physics.ALL, vec)), "motion_med": 2.0,
                             **{j: rng.normal(0, 1) + (0.5 * shift if i < 3 else 0) for i, j in enumerate(weak.JOINT)}})
    df = pd.DataFrame(rows)
    df[["clip_id"] + physics.ALL + ["motion_med"]].to_csv(res / "physics_features.csv", index=False)
    df[["clip_id"] + weak.JOINT].to_csv(res / "joint_features.csv", index=False)
    if embeddings:
        E = rng.normal(0, 1, (len(df), 16)).astype(np.float32)
        E[:, 0] += (~df.clip_id.str.startswith("RC")).to_numpy() * 1.5
        np.savez(res / "clip_embeddings.npz", ids=df.clip_id.to_numpy(), emb=E)
    return df


@pytest.mark.parametrize("embeddings", [False, True])
def test_weak_analysis_end_to_end_on_fabricated_tables(work, embeddings):
    res = pipeline.results_dir()
    _fabricate(res, embeddings=embeddings)
    text = weak_analysis.run(store=None, B=300, n_jobs=1, log=lambda *_: None)
    out = res / "weak"
    names = {p.name for p in out.glob("*")}
    assert {"ladder.csv", "lofo.csv", "lofo_false_alarms.csv", "lofo_mean.csv", "paired.csv", "summary.md"} <= names
    lad = pd.read_csv(out / "ladder.csv")
    assert {"CPS-2", "CPS-3"} <= set(lad.detector) and (("CLIP probe" in set(lad.detector)) == embeddings)
    assert {"clean", "sym3", "sym-wide", "rat-wide"} <= set(lad.protocol) and {"ladder L1", "ladder L2", "ladder L3"} <= set(lad.setting)
    clean = lad[(lad.detector == "CPS-2") & (lad.protocol == "clean") & (lad.fake == "GR")].set_index("setting").auc
    assert clean["clean"] > 0.8 and clean["ladder L3"] < clean["ladder L1"] < clean["clean"] + 1e-9      # the fabricated shift is halved per rung
    lofo = pd.read_csv(out / "lofo.csv")
    assert set(lofo.family) == set(weak.ALL_FAMS) and {"attack", "symmetric"} == set(lofo.contrast)
    assert (lofo[lofo.protocol.str.startswith("lofo-")].apply(lambda r: r.protocol == f"lofo-{r.family}", axis=1)).all()
    mean = pd.read_csv(out / "lofo_mean.csv")
    assert "lofo (family unseen)" in set(mean.protocol) and mean.families.eq(len(weak.ALL_FAMS)).all()
    assert "Findings" in text and "Traceback" not in text


def test_weak_analysis_survives_missing_cells(work):
    """Only clean cells + one operator kind: the tables that cannot be built are empty, nothing raises."""
    res = pipeline.results_dir()
    df = _fabricate(res)
    keep = df.clip_id.str.match(r"^(RC|RR|GR)_[0-9a-f]+$")
    df[keep][["clip_id"] + physics.ALL + ["motion_med"]].to_csv(res / "physics_features.csv", index=False)
    df[keep][["clip_id"] + weak.JOINT].to_csv(res / "joint_features.csv", index=False)
    text = weak_analysis.run(store=None, B=200, n_jobs=1, log=lambda *_: None)
    assert (res / "weak" / "summary.md").exists() and "Weakness experiments" in text


def test_bootstats_matches_brute_force_resampling():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    sc = np.array([f"s{i}" for i in range(25)])
    neg, pos = rng.normal(0, 1, 25), rng.normal(0.7, 1, 25)
    sb = bootstats.SceneBoot(sc, 100, 2)
    st = sb.auc(sc, neg, sc, pos)
    counts = np.array([int(sb.W[7, sb.ix[n]]) for n in sc])
    idx = np.repeat(np.arange(25), counts)
    y = np.r_[np.zeros(len(idx)), np.ones(len(idx))]
    assert abs(st.boot[7] - roc_auc_score(y, np.r_[neg[idx], pos[idx]])) < 1e-12
    lo, hi = (st / 2 + 0.25).ci()
    assert lo < st.point / 2 + 0.25 < hi


def test_store_upload_never_raises_without_a_repo(tmp_path, monkeypatch):
    from c44 import config
    from c44.storage import Store
    monkeypatch.setattr(config, "hf_repo", lambda: None)                  # never read a real token / repo from a local .env in a test
    monkeypatch.setattr(config, "hf_token", lambda: None)
    f = tmp_path / "x.zip"
    f.write_bytes(b"zip")
    ok, msg = Store(repo_id=None, token=None).upload_file(f, "weak/x.zip")
    assert not ok and "no HF repo" in msg
    ok, msg = Store(repo_id="a/b", token="t").upload_file(tmp_path / "missing.zip", "weak/x.zip")
    assert not ok and "does not exist" in msg


# ----------------------------------------------------------------------------- notebook-facing wiring (molab stages, zip, HF upload)
def _fake_results(work):
    res = work / "results"
    (res / "weak").mkdir(parents=True, exist_ok=True)
    (res / "weak" / "summary.md").write_text("# Weakness experiments\n", encoding="utf-8")
    (res / "weak" / "ladder.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (res / "joint_features.csv").write_text("clip_id\nx\n", encoding="utf-8")
    (res / "physics_features_weak.csv").write_text("clip_id\nx\n", encoding="utf-8")


def test_weak_analyse_stage_zips_everything_and_uploads_to_hf(work, monkeypatch, tmp_path):
    import zipfile

    from c44 import molab
    from c44.storage import Store
    _fake_results(work)
    run_dir = tmp_path / "cwd"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)
    calls = []
    monkeypatch.setattr(molab, "_need", lambda args: calls.append(args))
    monkeypatch.setattr(Store, "upload_file", lambda self, local, path_in_repo, retries=3: (calls.append(("upload", path_in_repo)) or True, f"uploaded {path_in_repo}"))
    zp = molab.run("weak_analyse")
    assert calls[0] == ["weak-analyse"] and [c[1] for c in calls[1:]] == [
        next(c[1] for c in calls[1:] if c[1].startswith("weak/c44_weak_results_v18")), "weak/c44_weak_results_latest.zip"]
    names = set(zipfile.ZipFile(zp).namelist())
    assert {"weak/summary.md", "weak/ladder.csv", "joint_features.csv", "physics_features_weak.csv"} <= names
    assert len(molab.LAST_UPLOAD) == 2 and all(m.startswith("OK: ") for m in molab.LAST_UPLOAD)


def test_weak_analyse_stage_survives_a_failed_hf_upload(work, monkeypatch, tmp_path):
    from c44 import molab
    from c44.storage import Store
    _fake_results(work)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(molab, "_need", lambda args: None)
    monkeypatch.setattr(Store, "upload_file", lambda self, *a, **k: (False, "upload failed after 3 attempts: boom"))
    zp = molab.run("weak_analyse")                                        # must not raise: the local zip is the fallback
    assert zp.endswith(".zip") and molab.LAST_UPLOAD and molab.LAST_UPLOAD[0].startswith("NOT UPLOADED")


def test_weak_stage_arguments_and_cli_parsing(monkeypatch):
    from c44 import cli, molab
    seen = []
    monkeypatch.setattr(molab, "_need", lambda args: seen.append(args))
    molab.run("weak_attack", n_scenes=6)
    molab.run("weak_features")
    molab.run("weak_embed")
    assert seen == [["weak-attack", "--n", "6"], ["weak-features"], ["embed", "--weak"]]
    for cmd in (["weak-attack", "--n", "6", "--kinds", "nz-tnr"], ["weak-features", "--no-old-j"], ["weak-analyse", "--boot", "500", "--no-lofo"],
                ["embed", "--weak"], ["pull", "--weak"]):
        with pytest.raises(SystemExit) as e:
            cli.main(cmd + ["--help"])                                    # argparse accepted the arguments (--help exits 0 after parsing)
        assert e.value.code == 0
    assert cli.PULL_WEAK and all(p in cli.PULL_WEAK for p in ("clips/*@nz-*/*", "clips/*@wk-*/*", "results/*", "manifest.jsonl"))


# ----------------------------------------------------------------------------- the detached job (survives a notebook refresh)
def test_job_runs_detached_logs_and_a_second_start_attaches(work):
    import sys

    from c44 import weak_job
    cmd = [sys.executable, "-u", "-c", "import time; print('hello from the job', flush=True); time.sleep(6); print('bye', flush=True)"]
    st = weak_job.start(150, cmd=cmd)
    assert st["state"] == "running" and weak_job.alive(st["pid"]) and weak_job.is_running()
    again = weak_job.start(150, cmd=["python", "-c", "raise SystemExit('must not start a second job')"])
    assert again["pid"] == st["pid"]                                      # attached to the live job instead of starting another
    lines = []
    final = weak_job.follow(poll=0.5, out=lines.append)
    text = "\n".join(lines)
    assert "hello from the job" in text and "bye" in text
    assert final["state"] == "ended" and "recycled" in final["error"]     # this stand-in job never wrote done/failed: reported, not hung
    assert not weak_job.is_running()


def test_job_state_survives_a_new_process_view_and_zombie_is_not_alive(work):
    from c44 import weak_job
    weak_job.write_state(state="running", pid=2 ** 22 + 12345, step="weak_features")       # a pid that does not exist
    assert not weak_job.is_running() and weak_job.read_state()["step"] == "weak_features"
    assert not weak_job.alive(None) and not weak_job.alive(0)


def test_run_steps_skips_a_failing_nonfatal_step_and_stops_on_a_fatal_one(work):
    from c44 import weak_job
    seen = []

    def runner(stage, **kw):
        seen.append((stage, kw))
        if stage == "weak_embed":
            raise RuntimeError("CLIP failed")
        return "/x/c44_weak_results.zip" if stage == "weak_analyse" else None

    weak_job.write_state(n_scenes=6)
    st = weak_job.run_steps(runner, log=lambda *_: None)
    assert [s for s, _ in seen] == ["weak_attack", "weak_features", "weak_embed", "weak_analyse"] and seen[0][1] == {"n_scenes": 6}
    assert st["state"] == "done" and st["zip"].endswith("c44_weak_results.zip")

    seen.clear()

    def boom(stage, **kw):
        seen.append(stage)
        if stage == "weak_features":
            raise RuntimeError("disk full")

    st = weak_job.run_steps(boom, log=lambda *_: None)
    assert st["state"] == "failed" and "weak_features" in st["error"] and seen == ["weak_attack", "weak_features"]


def test_job_stage_in_molab_returns_the_zip_and_reports_upload_status(work, monkeypatch):
    from c44 import molab, weak_job
    monkeypatch.setattr(weak_job, "start", lambda n, cmd=None: {"state": "running"})
    monkeypatch.setattr(weak_job, "follow", lambda: {"state": "done", "zip": "/z.zip", "upload": ["OK: uploaded"]})
    assert molab.run("weak_job", n_scenes=6) == "/z.zip" and molab.LAST_UPLOAD == ["OK: uploaded"]
    monkeypatch.setattr(weak_job, "follow", lambda: {"state": "failed", "error": "weak_features: boom"})
    with pytest.raises(RuntimeError, match="boom"):
        molab.run("weak_job", n_scenes=6)


RATE_LIMIT = ("429 Too Many Requests for url: https://huggingface.co/api/datasets/x/y/commit/main. You have exceeded the rate limit for "
              "repository commit creation (128 per hour). You can retry this action in about 1 hour.")


class _Hub:
    """A Store whose commits can be scripted: a list of outcomes (None = success, or an exception to raise)."""

    @staticmethod
    def make(outcomes):
        from c44.storage import Store

        class Scripted(Store):
            def __init__(self):
                super().__init__(repo_id="a/b", token="t")
                self.calls, self.sent = 0, []

            def flush(self, message="x"):
                self.calls += 1
                o = outcomes.pop(0) if outcomes else None
                if o is not None:
                    raise o
                self.sent.append((message, [p for _, p in self._pending]))
                self._pending.clear()
                return 1
        return Scripted()


@pytest.fixture
def hub_state(monkeypatch):
    from c44 import storage
    storage._reset_push_state()
    monkeypatch.setattr("time.sleep", lambda s: None)
    yield storage
    storage._reset_push_state()


def test_retry_after_reads_the_hubs_wait(hub_state):
    ra = hub_state.retry_after
    assert ra(RuntimeError(RATE_LIMIT)) == 3600
    assert ra(RuntimeError("429 Too Many Requests ... You can retry this action in 44 minutes.")) == 44 * 60
    assert ra(RuntimeError("rate limit, retry in an hour")) == 3600
    assert ra(RuntimeError("429 Too Many Requests")) == 600                 # no figure given: a default
    assert ra(RuntimeError("connection reset by peer")) is None             # not a rate limit: the caller treats it as a network error

    class WithHeader(Exception):
        response = type("R", (), {"status_code": 429, "headers": {"Retry-After": "90"}})()
    assert ra(WithHeader("x")) == 90


def test_a_429_is_not_hammered_and_nothing_is_lost(work, hub_state):
    from c44 import pipeline
    s = _Hub.make([RuntimeError(RATE_LIMIT)])
    (work / "a.csv").write_text("x\n", encoding="utf-8")
    assert pipeline._flush(s, [work / "a.csv"], "m1") == "backoff"          # must not raise, and not retry 3 times
    assert s.calls == 1 and s._pending                                      # the file is still queued
    assert pipeline._flush(s, [work / "a.csv"], "m2") == "backoff" and s.calls == 1       # inside the pause: no request at all
    assert hub_state.backing_off()
    # the end of a stage waits out the (capped) pause and then gets through; everything staged goes out in that one commit
    assert pipeline._flush(s, [work / "a.csv"], "end", force=True) == "pushed"
    assert s.calls == 2 and s.sent[-1][0] == "end" and "a.csv" in s.sent[-1][1] and not s._pending
    assert not hub_state.backing_off()


def test_progress_pushes_are_spaced_out_and_capped_but_the_end_always_goes(work, hub_state):
    from c44 import pipeline
    s = _Hub.make([])
    (work / "a.csv").write_text("x\n", encoding="utf-8")
    assert pipeline._flush(s, [work / "a.csv"], "p1") == "pushed"
    for i in range(30):                                                     # 30 progress pushes right after: all skipped, files stay queued
        assert pipeline._flush(s, [work / "a.csv"], f"p{i + 2}") == "throttled"
    assert s.calls == 1 and s._pending
    assert pipeline._flush(s, [work / "a.csv"], "end", force=True) == "pushed" and s.calls == 2

    storage = hub_state                                                     # the hourly budget
    storage._PUSH["times"] = [storage.time.time() - 60] * storage.PUSH_PER_HOUR
    storage._PUSH["last"] = 0.0
    assert pipeline._flush(s, [work / "a.csv"], "q") == "throttled"
    assert pipeline._flush(s, [work / "a.csv"], "end2", force=True) == "pushed"


def test_a_network_error_gets_one_quick_retry_then_stays_queued(work, hub_state):
    from c44 import pipeline
    (work / "a.csv").write_text("x\n", encoding="utf-8")
    s = _Hub.make([OSError("reset"), OSError("reset")])
    assert pipeline._flush(s, [work / "a.csv"], "m") == "failed" and s.calls == 2 and s._pending
    s2 = _Hub.make([OSError("reset")])
    assert pipeline._flush(s2, [work / "a.csv"], "m", force=True) == "pushed" and s2.calls == 2


def test_no_repo_means_local_only_and_never_raises(work, hub_state):
    from c44 import pipeline
    from c44.storage import Store
    s = Store(repo_id=None, token=None)
    (work / "a.csv").write_text("x\n", encoding="utf-8")
    assert pipeline._flush(s, [work / "a.csv"], "m", force=True) == "nothing" and not s._pending
    assert pipeline._flush(None, [], "m") == "nothing"


def test_upload_waits_out_a_rate_limit_instead_of_failing(work, hub_state, monkeypatch, tmp_path):
    hf = pytest.importorskip("huggingface_hub")
    from c44.storage import Store
    f = tmp_path / "r.zip"
    f.write_bytes(b"zip")
    outcomes = [RuntimeError(RATE_LIMIT), None]
    seen = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, *a, **k):
            pass

        def upload_file(self, **k):
            seen.append(k["path_in_repo"])
            o = outcomes.pop(0)
            if o is not None:
                raise o

        def list_repo_files(self, *a, **k):
            return ["weak/r.zip"]

    monkeypatch.setattr(hf, "HfApi", FakeApi)
    ok, msg = Store(repo_id="a/b", token="t").upload_file(f, "weak/r.zip", retries=1)       # ONE attempt, yet the 429 does not use it up
    assert ok and "uploaded" in msg and len(seen) == 2
    outcomes[:] = [RuntimeError(RATE_LIMIT)]
    ok, msg = Store(repo_id="a/b", token="t").upload_file(f, "weak/r.zip", retries=1, max_wait=0)     # a log push never waits
    assert not ok and "429" in msg


def test_watchdog_is_rearmed_by_progress_and_disk_free_ignores_fake_figures(monkeypatch):
    import collections
    import faulthandler
    import shutil

    from c44 import config, pipeline
    calls = []
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **k: calls.append("arm"))
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later", lambda: calls.append("cancel"))
    pipeline._WATCHDOG["on"] = False
    pipeline.watchdog_ping()
    assert calls == []                                                      # not armed (tests, library use): silent
    try:
        pipeline.arm_watchdog()
        pipeline.watchdog_ping()
        assert calls == ["arm", "cancel", "arm"]
    finally:
        pipeline._WATCHDOG["on"] = False
    Usage = collections.namedtuple("Usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(2 ** 63, 0, 2 ** 63))      # molab's gVisor
    assert config.disk_free_gb() is None and "disk free" not in pipeline._resources()
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(100 * 2 ** 30, 0, 40 * 2 ** 30))
    assert config.disk_free_gb() == 40.0 and "disk free 40 GiB" in pipeline._resources()


# ----------------------------------------------------------------------------- the job's own log on the Hub (so a crash leaves a trace)
def test_flush_carries_the_job_log_in_the_same_commit(work, monkeypatch):
    log = work / "weak_job.log"
    log.write_text("hello\n", encoding="utf-8")
    monkeypatch.setenv("C44_PUSH_LOG", f"{log}|logs/weak_job_X.log")
    staged = []

    class Rec:
        def stage(self, local, path_in_repo=None):
            staged.append((str(local), path_in_repo))

        def push(self, msg, force=False):
            staged.append(("push", msg, force))
            return "pushed"

    pipeline._flush(Rec(), [], "m", force=True)
    assert (str(log), "logs/weak_job_X.log") in staged and staged[-1] == ("push", "m", True)


def test_monitor_writes_resource_lines_and_pushes_the_log(monkeypatch, capsys):
    import threading
    import time

    from c44 import weak_job
    pushed = []
    monkeypatch.setattr(weak_job, "push_log", lambda why="": pushed.append(why) or True)
    stop = threading.Event()
    t = threading.Thread(target=weak_job.monitor, args=(stop, 0.05, 0.12), daemon=True)
    t.start()
    time.sleep(0.6)
    stop.set()
    t.join(2)
    out = capsys.readouterr().out
    assert out.count("[mon ") >= 3 and "RAM" in out and pushed


def test_each_run_gets_its_own_remote_log_and_the_job_sees_it(work):
    import sys

    from c44 import weak_job
    st = weak_job.start(6, cmd=[sys.executable, "-c", "import os; print('PUSHLOG=' + os.environ['C44_PUSH_LOG'])"])
    assert st["remote_log"].startswith("logs/weak_job_") and st["remote_log"].endswith(".log")
    lines = []
    weak_job.follow(poll=0.3, out=lines.append)
    assert "PUSHLOG=" in "\n".join(lines) and st["remote_log"] in "\n".join(lines)


def test_main_pushes_the_log_at_the_end_and_on_a_crash(work, monkeypatch, capsys):
    import signal

    from c44 import molab, weak_job
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    pushed = []
    monkeypatch.setattr(weak_job, "push_log", lambda why="": pushed.append(why) or True)
    try:
        monkeypatch.setattr(molab, "run", lambda stage, **kw: "/z.zip" if stage == "weak_analyse" else None)
        assert weak_job.main(["--n", "6"]) == 0 and pushed == ["end"]
        assert weak_job.read_state()["state"] == "done"
        pushed.clear()

        def boom(stage, **kw):
            raise KeyboardInterrupt()
        monkeypatch.setattr(molab, "run", boom)
        with pytest.raises(KeyboardInterrupt):
            weak_job.main(["--n", "6"])
        assert pushed == ["end"] and weak_job.read_state()["state"] == "failed"
    finally:
        for s, h in saved.items():
            signal.signal(s, h)
    out = capsys.readouterr().out
    assert "[job] python" in out and "[job] end" in out and "[job] crashed" in out
