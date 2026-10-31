import cv2
import numpy as np

from c44 import geometry
from tests.synthetic import texture

f, D, b = 400.0, 3.0, 0.02            # focal (px), plane depth (m), camera step (m/frame) -> f*b/D = 2.667 px/frame
H, W, T = 350, 518, 16                # VGGT model size for 518-wide frames, so no resampling in the test


def _scene(consistent: bool = True):
    tex = texture(H + 40, W + 200, seed=7)
    s = f * b / D
    frames, rng = [], np.random.default_rng(0)
    for t in range(T):
        src = tex if consistent else texture(H + 40, W + 200, seed=100 + t)
        M = np.float32([[1, 0, -(20 + t * s)], [0, 1, -20]])
        img = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        frames.append(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8))
    K = np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1]])
    w2c = np.stack([np.eye(4) for _ in range(T)])
    w2c[:, 0, 3] = -b * np.arange(T)
    geom = geometry.Geometry(w2c=w2c, K=np.repeat(K[None], T, 0), depth=np.full((T, H, W), D, np.float32),
                             conf=np.ones((T, H, W), np.float32), image_hw=(H, W), y0=0)
    return np.stack(frames), geom


def test_model_size_matches_vggt_rule():
    assert geometry.model_size(480, 704) == (350, 518, 0)
    assert geometry.model_size(1080, 1920) == (294, 518, 0)
    h, w, y0 = geometry.model_size(1920, 1080)
    assert (h, w) == (518, 518) and y0 > 0


def test_motion_matches_ground_truth():
    _, geom = _scene()
    mo = geometry.motion(geom)
    assert np.allclose(mo["m"], f * b / D, rtol=1e-6)
    assert np.allclose(mo["direction"], [[-1.0, 0.0]] * T, atol=1e-6)
    assert np.isfinite(mo["flow_grid"]).all()


def test_reprojection_error_separates_consistent_from_inconsistent():
    good, geom = _scene(consistent=True)
    bad, _ = _scene(consistent=False)
    e_good = geometry.reprojection_error(good, geom)
    e_bad = geometry.reprojection_error(bad, geom)
    assert e_good < 0.2, e_good
    assert e_bad > 0.6, e_bad


def test_geometry_save_load(tmp_path):
    _, geom = _scene()
    geom.save(tmp_path / "g.npz")
    g2 = geometry.Geometry.load(tmp_path / "g.npz")
    assert g2.image_hw == geom.image_hw and np.allclose(g2.K, geom.K) and np.allclose(g2.depth, geom.depth)
    assert np.allclose(g2.K_image(), geom.K)          # no resampling in this setup
