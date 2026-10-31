import numpy as np

from c44 import attacks, cps, metrics
from tests import synthetic


def _features(kind, seeds):
    out = []
    for s in seeds:
        f, m, d = synthetic.clip(kind, s)
        out.append(cps.clip_features(f, m, direction=d))
    return out


def test_blur_coupling_present_only_for_camera():
    cam = _features("camera", range(4))
    ren = _features("render", range(4))
    rc = np.array([r["rho_blur"] for r in cam])
    rr = np.array([r["rho_blur"] for r in ren])
    assert rc.mean() > 0.6, rc
    assert abs(rr.mean()) < 0.35, rr
    assert np.mean([r["rho_dir"] for r in cam]) > 0.5
    assert np.mean([r["sharp_ratio"] for r in cam]) < np.mean([r["sharp_ratio"] for r in ren]) - 0.1


def test_fixed_pattern_noise_detected():
    cam = _features("camera", range(4, 8))
    ren = _features("render", range(4, 8))
    assert all(r["n_fpn_pairs"] > 0 for r in cam + ren)
    assert np.mean([r["fpn_corr"] for r in cam]) > np.mean([r["fpn_corr"] for r in ren]) + 0.02


def test_one_class_score_separates_without_seeing_renders():
    train = _features("camera", range(100, 112))
    model = cps.OneClassCPS().fit(train)
    test_cam = _features("camera", range(200, 206))
    test_ren = _features("render", range(200, 206))
    s = model.score(test_cam + test_ren)
    y = np.array([0] * len(test_cam) + [1] * len(test_ren))
    assert metrics.roc_auc(y, s) > 0.9
    again = cps.OneClassCPS.from_json(model.to_json()).score(test_cam + test_ren)
    assert np.allclose(again, s)


def test_attacks_restore_some_camera_cues():
    f, m, d = synthetic.clip("render", 300)
    base = cps.clip_features(f, m, direction=d)
    blurred = cps.clip_features(attacks.motion_blur(f, m, d, exposure=0.6), m, direction=d)
    # strong but plausible PRNU (5%) under low shot noise, so a single 48-frame clip can show the effect
    noisy = cps.clip_features(attacks.sensor_noise(f, gain=0.001, read_std=0.002, prnu_std=0.05, seed=1), m,
                              direction=d)
    assert blurred["rho_blur"] > base["rho_blur"] + 0.3      # the smart blur attack fakes the coupling
    assert noisy["fpn_corr"] > base["fpn_corr"]              # injected PRNU fakes the fixed pattern
