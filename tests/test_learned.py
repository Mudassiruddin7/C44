"""Retrained probes, physics + CLIP fusion, the embedding stage and their place in the analysis tables (fake backbones)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from c44 import analysis, embed, learned, physics, video
from tests.test_physics import _toy_features

D = 12


def _with_embeddings(pf, rng, shift=0.8):
    """Toy 'backbone' embeddings: fakes shifted along a few dimensions (a second, independent cue source)."""
    e = rng.normal(0, 1, (len(pf), D)) + shift * (pf.base != "RC").to_numpy()[:, None] * (np.arange(D) % 4 == 0)
    cols = [f"clip{i}" for i in range(D)]
    return pd.concat([pf.reset_index(drop=True), pd.DataFrame(e, columns=cols)], axis=1), {"CLIP": cols}


def test_probe_and_fusion_separate_planted_shifts():
    rng = np.random.default_rng(0)
    pf, emb_cols = _with_embeddings(_toy_features(rng), rng)
    pf = pf[pf["var"] == ""].reset_index(drop=True)
    y = (pf.base != "RC").to_numpy().astype(int)
    real, fake = pf[y == 0], pf[y == 1]
    probe = learned.ProbeClassifier(emb_cols["CLIP"]).fit(pf, y)
    assert probe.score(fake).mean() > probe.score(real).mean()
    for mode in ("mean", "stack"):
        fusion = learned.make_models(emb_cols, mode)["CPS-2 + CLIP"]().fit(pf, y)
        s = fusion.score(pf)
        assert s[y == 1].mean() > s[y == 0].mean()
        assert abs(fusion.mu).max() < 50 and (fusion.sd > 0).all()
    mean_fusion = learned.make_models(emb_cols)["CPS-2 + CLIP"]().fit(pf, y)
    z = mean_fusion._z(real)
    assert abs(float(np.median(z))) < 1.5                              # real clips sit near 0 on the common scale (out-of-fold)


def test_analysis_tables_include_learned_models():
    rng = np.random.default_rng(1)
    pf, emb_cols = _with_embeddings(_toy_features(rng, n_scenes=70, nz=True), rng)
    main = analysis.physics_tables(pf, pf[["clip_id", "cell", "scene"]], emb_cols)["cps2_main"]
    assert {"CLIP probe", analysis.FUSION, f"{analysis.FUSION} + aug"} <= set(main.columns)
    att = main[main.setting == "blurcal"]
    assert len(att) == 2 and att[analysis.FUS_AUG["components"]].notna().all()
    assert att[list(analysis.RAT_COLS.values()) + list(analysis.SYM_COLS.values())].notna().all().all()   # RAT and SYM columns
    ops = analysis.physics_tables(pf, pf[["clip_id", "cell", "scene"]], emb_cols)["cps2_operators"]
    assert {"clean-trained", "RAT", "SYM"} <= set(ops.training) and ops["FPR real+all"].between(0, 1).all()
    assert (ops["TPR GR full attacker"].isna() | ops["TPR GR full attacker"].between(0, 1)).all()
    assert not main.setting.str.startswith("nz-").any()                              # training-only clips are never evaluated
    cv = analysis.cv_table(pf, emb_cols=emb_cols)
    assert {"CLIP probe", analysis.FUSION, analysis.FUS_AUG["seen"], *analysis.RAT_COLS.values(), *analysis.SYM_COLS.values()} <= set(cv.columns)
    assert analysis._parts("nz-blur+noise") == {"blur", "noise"} == analysis._parts("blurcal+noiseadapt")
    assert analysis._train_kinds([], "rat-components", "noiseadapt", rat=["nz-blur", "nz-noise", "nz-all"]) == ["nz-blur"]


def test_attach_embeddings_needs_full_coverage(tmp_path, monkeypatch):
    monkeypatch.setenv("C44_WORK", str(tmp_path))
    rng = np.random.default_rng(2)
    pf = _toy_features(rng, n_scenes=6)
    ids = pf.clip_id.to_numpy()
    path = embed.pipeline.results_dir() / "clip_embeddings.npz"
    np.savez(path, ids=ids[:-3], emb=rng.normal(size=(len(ids) - 3, D)).astype(np.float32))      # three clips missing
    logs = []
    out, cols = analysis.attach_embeddings(pf, log=logs.append)
    assert cols == {} and any("lack clip embeddings" in m for m in logs)
    np.savez(path, ids=ids, emb=rng.normal(size=(len(ids), D)).astype(np.float32))
    out, cols = analysis.attach_embeddings(pf, log=logs.append)
    assert list(cols) == ["CLIP"] and len(cols["CLIP"]) == D and out[cols["CLIP"]].notna().all().all()


class _FakeEmbedder:
    name = "fake"

    def prep(self, frames):
        return np.stack([f.mean((0, 1)) for f in frames[::12]]).astype(np.float32)        # (n_frames, 3)

    def encode(self, x):
        return np.asarray(x, np.float32)


def test_embed_clips_is_resumable(tmp_path, monkeypatch):
    monkeypatch.setenv("C44_WORK", str(tmp_path))
    clips = []
    for k in range(5):
        p = tmp_path / f"c{k}.mp4"
        video.write_mp4(np.full((24, 64, 96, 3), 40 * k, np.uint8), p, fps=30, crf=18, preset="veryfast")
        clips.append(SimpleNamespace(clip_id=f"RC_0{k}", path=str(p)))
    path = tmp_path / "emb.npz"
    assert embed.embed_clips(_FakeEmbedder(), clips[:3], path, log=lambda *_: None, chunk=2, workers=2) == 3
    assert embed.embed_clips(_FakeEmbedder(), clips, path, log=lambda *_: None, chunk=2, workers=2) == 2   # only the new ones
    ids, e = embed.load(path=path)
    assert list(ids) == [c.clip_id for c in clips] and e.shape == (5, 3)
    assert e[4].mean() > e[0].mean()                                    # brighter clips embed brighter: order is preserved


def test_analyse_stage_end_to_end(tmp_path, monkeypatch):
    """The whole `analyse` step on toy data: every section must run (the driver swallows exceptions, so look for SKIPPED)."""
    from c44 import pipeline
    monkeypatch.setenv("C44_WORK", str(tmp_path))
    rng = np.random.default_rng(5)
    pf = _toy_features(rng, n_scenes=60, nz=True)
    res = pipeline.results_dir()
    pf[["clip_id", *physics.ALL]].to_csv(res / "physics_features.csv", index=False)
    fake = (pf.base != "RC").to_numpy()
    emb = rng.normal(size=(len(pf), D)) + 0.8 * fake[:, None] * (np.arange(D) % 4 == 0)
    np.savez(res / "clip_embeddings.npz", ids=pf.clip_id.to_numpy(), emb=emb.astype(np.float32))
    pd.DataFrame({"clip_id": pf.clip_id, "detector": "univfd", "score": rng.random(len(pf))}).to_csv(res / "scores.csv", index=False)
    text = analysis.run(store=None)
    assert "SKIPPED" not in text, text[-2500:]
    out = res / "analysis"
    for name in ("cps2_main", "cps2_operators", "cps2_cv", "cps2_operators_cv", "cps2_families", "attack_calibration"):
        assert (out / f"{name}.csv").exists(), name
    tex = (out / "tables.tex").read_text()
    assert "tab:operators" in tex and "tab:attacks-cv" in tex and tex.count(r"\begin{table}") == tex.count(r"\end{table}")
