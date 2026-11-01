"""CPS-2: feature extraction, the exposure attack, the classifiers and the held-out analysis protocol."""

import numpy as np
import pandas as pd

from c44 import analysis, attacks, physics
from tests.synthetic import clip


def _frames(kind, seed=1):
    out = clip(kind, seed, T=24, h=192, w=256)
    return np.asarray(out[0] if isinstance(out, tuple) else out)


def test_clip_physics_returns_all_features():
    f = physics.clip_physics(_frames("camera"))
    assert set(physics.ALL) <= set(f)
    assert np.isfinite([f[k] for k in physics.ALL]).sum() >= len(physics.ALL) - 3     # a few may be NaN on a tiny clip


def test_camera_noise_vs_noise_free_render():
    cam, ren = physics.clip_physics(_frames("camera")), physics.clip_physics(_frames("render"))
    assert cam["res_ac1"] < 0.5 * ren["res_ac1"]                      # sensor noise is white; a render's residual is structured
    assert cam["tn_ratio"] > 2 * ren["tn_ratio"]                     # independent noise per frame vs. a stable render


def test_exposure_jitter_attack_hits_its_target_and_is_measured():
    base = _frames("render")
    assert physics.gain_targets(base)[0] < 5e-3                     # a render has no exposure dynamics
    sigma = 0.01
    att = attacks.exposure_jitter(base, gain_sigma=sigma, wb_sigma=0.004, seed=3)
    gm, wm = physics.gain_targets(att)
    assert 0.4 * sigma < gm < 2.5 * sigma and wm > 5e-4              # measured jitter is the injected one (within estimator slack)


def _toy_features(rng, n_scenes=50):
    """Fabricated feature table: fakes shifted in a few features, recompression and attacks as extra cells."""
    rows = []
    for s in range(n_scenes):
        scene = f"{s:02x}" + "ab" * 6
        for cell, shift in (("RC", 0.0), ("GR", 1.2), ("RR", 0.9)):
            for var in ("", "crf28", "blurcal"):
                if var == "blurcal" and (cell == "RC" or s >= 40):
                    continue
                if var == "crf28" and s >= 45:
                    continue
                cid = f"{cell}_{scene}" + (f"@{var}" if var else "")
                f = rng.normal(0, 1, len(physics.ALL)) + shift * (np.arange(len(physics.ALL)) % 3 == 0)
                rows.append({"clip_id": cid, **dict(zip(physics.ALL, f))})
    return analysis._parse(pd.DataFrame(rows))


def test_classifiers_separate_a_planted_shift():
    pf = _toy_features(np.random.default_rng(0))
    pf = pf[pf["var"] == ""]
    real, fake = pf[pf.base == "RC"], pf[pf.base != "RC"]
    sup = physics.PhysicsClassifier().fit(pf, (pf.base != "RC").astype(int))
    oc = physics.PhysicsOneClass().fit(real)
    for model in (sup, oc):
        assert model.score(fake).mean() > model.score(real).mean()


def test_analysis_protocol_on_toy_features():
    pf = _toy_features(np.random.default_rng(1), n_scenes=80)
    sw = pf[["clip_id", "cell", "scene"]].assign(univfd=np.random.default_rng(2).random(len(pf)))
    t = analysis.physics_tables(pf, sw)
    assert {"cps2_main", "cps2_families", "cps2_transfer", "cps2_features"} <= set(t)
    main = t["cps2_main"]
    assert {"RC vs GR", "RC vs RR", "RC@crf28 vs GR@crf28", "RC vs GR@blurcal"} <= set(main.pair)
    auc = lambda s: float(s.split()[0])                              # noqa: E731
    row = main[main.pair == "RC vs GR"].iloc[0]
    assert auc(row["CPS-2 supervised"]) > 0.8 and 0.3 < auc(row["univfd"]) < 0.7
    assert all(str(c).startswith(("only", "minus", "all")) for c in t["cps2_families"].features)
    # test scenes are a fixed hash split and nothing from them is used to fit
    tests = {s for s in pf.scene.unique() if analysis.is_test(s)}
    assert 0.15 < len(tests) / pf.scene.nunique() < 0.45
