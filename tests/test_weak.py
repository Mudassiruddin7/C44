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
