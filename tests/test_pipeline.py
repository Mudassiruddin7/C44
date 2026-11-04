"""End-to-end orchestration test with fake heavy components (VGGT, FlashWorld, DL3DV downloads, gsplat).

Real: video encode/decode, manifest bookkeeping, resumption, camera-path resampling, CPS features, out-of-fold
scoring and the report. Fake: anything that needs the network or a big GPU.
"""

import json

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from c44 import dl3dv, geometry, gs_render, pipeline, video
from c44.config import SPEC
from tests.synthetic import texture

N_SCENES = 15


class FakeGeo:
    """Sideways-translating camera (variable speed) over a plane at depth 3, at VGGT model resolution."""

    def __call__(self, frames):
        T, H, W, _ = frames.shape
        h, w, y0 = geometry.model_size(H, W)
        K = np.array([[0.8 * w, 0, w / 2], [0, 0.8 * w, h / 2], [0, 0, 1.0]])
        w2c = np.stack([np.eye(4) for _ in range(T)])
        w2c[:, 0, 3] = -np.cumsum(0.01 * (1.2 + np.sin(np.arange(T))))
        return geometry.Geometry(w2c=w2c, K=np.repeat(K[None], T, 0), depth=np.full((T, h, w), 3.0, np.float32),
                                 conf=np.ones((T, h, w), np.float32), image_hw=(H, W), y0=y0)


class FakeFW:
    def generate(self, image, cam11, text, image_index=0, seed=0):
        assert image.shape == (SPEC.height, SPEC.width, 3) and cam11.shape == (SPEC.keyframes, 11)
        return image, np.eye(4), 1.0, 0.01

    def render_path(self, scene, ref_w2c, T_norm, cam11_path, W, H, background):
        assert cam11_path.shape == (SPEC.n_frames, 11)
        frames = np.stack([np.roll(scene, 2 * t, axis=1) for t in range(len(cam11_path))])
        return frames, {"coverage": np.ones(len(frames)), "holes": np.zeros(len(frames))}


def _fake_video(path, seed):
    tex = texture(300, 900, seed)
    frames = np.stack([(np.clip(tex[20:260, 3 * t:3 * t + 352] * 255, 0, 255)).astype(np.uint8) for t in range(150)])
    video.write_mp4(frames, path, fps=30, crf=18, preset="veryfast")
    return path


def _fake_gs(root):
    rng = np.random.default_rng(0)
    N = 2000
    cols = {k: rng.normal(size=N) for k in ("x", "y")}
    cols["z"] = rng.normal(5, 0.5, N)
    for i in range(3):
        cols[f"f_dc_{i}"] = rng.normal(size=N)
        cols[f"scale_{i}"] = np.full(N, -3.0)
    cols["opacity"] = np.zeros(N)
    for i, v in enumerate((1.0, 0, 0, 0)):
        cols[f"rot_{i}"] = np.full(N, v)
    arr = np.rec.fromarrays([c.astype(np.float32) for c in cols.values()], names=list(cols))
    (root / "point_cloud" / "iteration_30000").mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(root / "point_cloud" / "iteration_30000" / "point_cloud.ply"))
    cams = [{"id": i, "img_name": f"frame_{i:05d}", "width": 960, "height": 540, "position": [0.05 * i, 0, 0],
             "rotation": np.eye(3).tolist(), "fx": 800.0, "fy": 800.0} for i in range(30)]
    (root / "cameras.json").write_text(json.dumps(cams))
    return dl3dv.locate_gs_files(root)


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setenv("C44_WORK", str(tmp_path))
    index = {f"{i:02d}" + "ab" * 15: {"video": f"x/{i}/video.mp4", "gs": f"3DGS/x/{i}.zip"} for i in range(N_SCENES)}
    monkeypatch.setattr(dl3dv, "build_index", lambda *a, **k: index)
    monkeypatch.setattr(dl3dv, "fetch_video",
                        lambda scene, idx, token=None: _fake_video(tmp_path / f"{scene}.mp4", int(scene[:2])))
    monkeypatch.setattr(dl3dv, "fetch_gs", lambda scene, idx, token=None: _fake_gs(tmp_path / "gs" / scene))
    monkeypatch.setattr(gs_render, "gs_to_torch", lambda gs, device="cuda": gs)

    def fake_render(gs, c2w_cv, K, width, height, background=(1, 1, 1), **kw):
        tex = texture(height + 20, width + 400, 3)
        fr = np.stack([(np.clip(tex[10:10 + height, 3 * t:3 * t + width] * 255, 0, 255)).astype(np.uint8)
                       for t in range(len(c2w_cv))])
        return fr, {"coverage": np.ones(len(fr)), "holes": np.zeros(len(fr))}

    monkeypatch.setattr(gs_render, "render", fake_render)
    return tmp_path


def test_full_pipeline_with_fakes(work, monkeypatch):
    geo = FakeGeo()
    made = pipeline.build_rc(N_SCENES, store=None, geo=geo)
    assert len(made) == N_SCENES
    assert pipeline.build_rc(N_SCENES, store=None, geo=geo) == []          # resumable: nothing redone
    rc = pipeline.manifest().by_cell("RC")
    info = video.probe(work / rc[0].path)
    assert (info["width"], info["height"], info["n_frames"]) == (SPEC.width, SPEC.height, SPEC.n_frames)

    assert len(pipeline.build_gr(FakeFW(), store=None, geo=geo)) == N_SCENES
    assert len(pipeline.build_rr(store=None, geo=geo, device="cpu")) == N_SCENES
    man = pipeline.manifest().load()
    gr = [c for c in man.values() if c.cell == "GR"]
    assert all(c.parent.startswith("RC_") and c.status == "ok" for c in gr)
    rr = [c for c in man.values() if c.cell == "RR"]
    assert all(c.meta["coverage"] == 1.0 for c in rr)

    scores = pipeline.run_cps(n_folds=3, store=None)
    assert set(scores.clip_id) == {c.clip_id for c in man.values() if c.status == "ok"}
    rep = pipeline.report(n_boot=50)
    assert set(rep.pair) == {"RC vs GR", "RC vs RR"} and np.isfinite(rep.auc).all()

    # robustness variants and camera-simulating attacks become derived cells, paired like-for-like
    assert len(pipeline.build_variants("crf28", cells=("RC", "GR"), n_jobs=1)) == 2 * N_SCENES
    assert len(pipeline.build_variants("blurcal", cells=("RC", "GR"), n_jobs=1)) == N_SCENES     # attacks skip real clips
    assert pipeline.build_variants("blurcal", cells=("RC", "GR"), n_jobs=1) == []                 # resumable
    monkeypatch.setenv("C44_RECALIBRATE", "1")                                         # exercise the measurement itself
    assert 0 < pipeline.blur_exposure() <= 0.5
    nc = pipeline.build_variants("noiseadapt", cells=("RC", "GR"), limit_scenes=3, n_jobs=1)     # attacks run on a subset
    assert len(nc) == 3 and all("noise_strength" in pipeline.manifest().load()[i].meta for i in nc)
    ae = pipeline.build_variants("aecal", cells=("RC", "GR"), limit_scenes=3, n_jobs=2)             # parallel worker pool
    assert len(ae) == 3 and all("gain_sigma" in pipeline.manifest().load()[i].meta for i in ae)
    nz = pipeline.build_variants("nz-all", cells=("RC", "GR"), limit_scenes=3, n_jobs=1)           # randomised training clips: fakes only
    assert len(nz) == 6 and all({"blur_exposure", "noise_strength", "gain_sigma"} <= set(pipeline.manifest().load()[i].meta) for i in nz)
    assert {c.cell for c in pipeline.manifest().load().values() if "@nz-all" in c.cell} == {"RC@nz-all", "GR@nz-all"}     # real too
    assert pipeline.scored_variant("GR@blurcal") and not pipeline.scored_variant("GR@nz-all") and not pipeline.scored_variant("GR@aeadapt")
    pipeline.run_cps(n_folds=0, min_train=5, store=None)                                # leave-one-scene-out
    rep = pipeline.report(n_boot=50)
    assert {"RC@crf28 vs GR@crf28", "RC vs GR@blurcal"} <= set(rep.pair)
    fa = pipeline.feature_auc()
    assert {"GR", "RR", "GR@blurcal"} <= set(fa.cell) and fa.drop(columns="cell").stack().between(0, 1).all()

    # resolution control: supersampled twins of GR/RR on a subset, same calibrated path as the originals
    ctl = pipeline.subset_scenes(2, need=("RC", "GR", "RR"))
    assert len(ctl) == 2
    assert len(pipeline.build_gr(FakeFW(), store=None, geo=geo, cell="GRs", ss=2, scenes=ctl)) == 2
    assert len(pipeline.build_rr(store=None, geo=geo, device="cpu", cell="RRs", ss=2, scenes=ctl)) == 2
    man = pipeline.manifest().load()
    for sc in ctl:
        assert man[f"GRs_{sc[:16]}"].meta["path_speed"] == man[f"GR_{sc[:16]}"].meta["path_speed"]
        assert man[f"RRs_{sc[:16]}"].meta["start_key"] == man[f"RR_{sc[:16]}"].meta["start_key"]
        assert man[f"GRs_{sc[:16]}"].meta["ss"] == 2
    assert pipeline.build_gr(FakeFW(), store=None, geo=geo, cell="GRs", ss=2, scenes=ctl) == []   # resumable


def test_calibrate_matches_target_and_ends_at_returned_value():
    calls = []

    def measure(f):
        calls.append(f)
        return 2.8 * f ** 0.95                       # slightly sub-linear, like real parallax/occlusion

    f, m = pipeline.calibrate_scale(measure, target=1.0)
    assert abs(m - 1.0) < 0.1 and calls[-1] == f and f < 0.5
    assert pipeline.calibrate_scale(lambda f: 1.05, 1.0)[0] == 1.0                 # already matched
    assert pipeline.calibrate_scale(lambda f: float("nan"), 1.0)[0] == 1.0         # degenerate
    n = []
    f, m = pipeline.calibrate_scale(lambda f: n.append(f) or 0.5, 1.0)             # too slow: cannot speed up
    assert f == 1.0 and len(n) == 1


def test_slow_path_stays_on_the_path():
    from scipy.spatial.transform import Rotation
    T = 9
    c2w = np.tile(np.eye(4), (T, 1, 1))
    c2w[:, :3, :3] = Rotation.from_rotvec(np.outer(np.radians(np.linspace(0, 40, T)), [0, 1, 0])).as_matrix()
    c2w[:, :3, 3] = np.linspace(0, 1, T)[:, None] * np.array([2.0, 0.5, 0.0]) + np.array([1.0, 1.0, 1.0])
    assert np.allclose(pipeline.slow_path(c2w, 1.0), c2w, atol=1e-9)
    half = pipeline.slow_path(c2w, 0.5)
    assert np.allclose(half[0], c2w[0]) and np.allclose(half[-1], c2w[T // 2], atol=1e-9)
    assert np.allclose(half[:, :3, :3] @ np.swapaxes(half[:, :3, :3], 1, 2), np.eye(3), atol=1e-9)
    step = np.linalg.norm(np.diff(half[:, :3, 3], axis=0), axis=1)
    assert np.allclose(step, step[0]) and step[0] < np.linalg.norm(c2w[1, :3, 3] - c2w[0, :3, 3])


def test_noise_calibration_hits_target_after_encoding():
    from c44 import attacks
    from tests.synthetic import clip
    base = clip("render", 1, T=12, h=192, w=256)
    frames = np.asarray(base[0] if isinstance(base, tuple) else base)
    seed = 5
    noisy = lambda s: attacks.scaled_noise(frames, s, seed)                      # noqa: E731
    for true_strength in (0.05, 1.0):            # a quiet and a typical "real" camera: both must be reachable
        ref = pipeline._rt_resid(attacks.scaled_noise(frames, true_strength, 1))
        s, got = pipeline.calibrate_noise_strength(noisy, ref)
        assert abs(got / ref - 1) < 0.1, (true_strength, got, ref)
    # a clip already noisier than the target gets (almost) no extra noise
    s2, _ = pipeline.calibrate_noise_strength(noisy, ref * 0.1)
    assert s2 <= 2e-3
