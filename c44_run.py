# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "pandas",
#     "scikit-learn",
#     "opencv-python-headless",
#     "av",
#     "imageio-ffmpeg",
#     "pillow",
#     "plyfile",
#     "huggingface_hub",
# ]
# ///

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys

    import marimo as mo

    _root = str(mo.notebook_dir())
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from c44 import config, pipeline, setup_env
    from c44.storage import Store

    store = Store()
    return config, mo, pipeline, setup_env, store


@app.cell
def _(mo):
    mo.md(
        r"""
        # C44 · world-rendered fakes vs. video forensics

        **Before you start**
        1. Attach the RTX PRO 6000 GPU (notebook specs button).
        2. Secrets panel: add `HF_TOKEN` (free Hugging Face token, type *Read* or *Write*, not fine-grained).
           Optional: `C44_HF_REPO` (e.g. `yourname/c44-w2v`, needs a *Write* token) keeps outputs between sessions;
           without it, use the download button at the bottom before the session ends.
        3. On huggingface.co, open `DL3DV/DL3DV-ALL-video` and `DL3DV/DL3DV-GS-960P` and accept their terms.

        Then press the buttons top to bottom. Every step resumes where it stopped.
        """
    )
    return


@app.cell
def _(mo, setup_env):
    mo.ui.table([{"item": k, "value": str(v)} for k, v in setup_env.report().items()], selection=None)
    return


@app.cell
def _(mo):
    setup_btn = mo.ui.run_button(label="1 · Set up GPU stack")
    setup_btn
    return (setup_btn,)


@app.cell
def _(mo, setup_btn, setup_env):
    mo.stop(not setup_btn.value, mo.md("**Step 1** installs the CUDA compiler, gsplat (compiled for Blackwell), VGGT, NPR and FlashWorld."))
    setup_env.ensure_nvcc()
    setup_env.install_gsplat()
    setup_env.install_vggt()
    setup_env.install_npr()
    _fw = setup_env.install_flashworld()
    mo.md(f"Setup finished. FlashWorld code at `{_fw}`. gsplat smoke test passed.")
    return


@app.cell
def _(mo):
    n_scenes = mo.ui.slider(4, 250, value=6, step=1, label="scenes")
    build_btn = mo.ui.run_button(label="2 · Build clips (RC → captions → GR → RR)")
    mo.hstack([n_scenes, build_btn], justify="start")
    return build_btn, n_scenes


@app.cell
def _(build_btn, mo, n_scenes, pipeline, setup_env, store):
    mo.stop(not build_btn.value, mo.md("**Step 2** builds the benchmark. Start with 6 scenes (pilot), then raise the slider."))
    from c44.flashworld_gen import FlashWorld
    from c44.geometry import VGGTGeometry

    store.ensure_repo()
    _geo = VGGTGeometry()
    pipeline.build_rc(n_scenes.value, store=store, geo=_geo)
    pipeline.build_captions(store=store)
    pipeline._release_gpu()
    _fw = FlashWorld(setup_env.code_dir() / "FlashWorld")
    pipeline.build_gr(_fw, store=store, geo=_geo)
    del _fw
    pipeline._release_gpu()
    pipeline.build_rr(store=store, geo=_geo)
    _df = pipeline.manifest().to_frame()
    _cols = [c for c in ["clip_id", "cell", "status", "meta.motion_med", "meta.coverage", "meta.error"] if c in _df.columns]
    mo.ui.table(_df[_cols], selection=None)
    return


@app.cell
def _(mo):
    eval_btn = mo.ui.run_button(label="3 · Detectors + CPS + report")
    eval_btn
    return (eval_btn,)


@app.cell
def _(eval_btn, mo, pipeline, store):
    mo.stop(not eval_btn.value, mo.md("**Step 3** scores every clip with UnivFD, NPR, multi-view consistency and CPS."))
    pipeline.run_detectors(("univfd", "npr", "consistency"), store=store)
    _n_rc = len(pipeline.manifest().by_cell("RC"))
    pipeline.run_cps(n_folds=0 if _n_rc < 30 else 5, min_train=4 if _n_rc < 30 else 10, store=store)
    report = pipeline.report(n_boot=500)
    mo.vstack([
        mo.md("### Detection AUC: real footage (RC) vs. each cell, 95% scene-bootstrap CI"),
        mo.ui.table(report.round(3), selection=None),
        mo.md("### Which camera cue carries the signal (per-feature AUC vs. RC)"),
        mo.ui.table(pipeline.feature_auc().round(3), selection=None),
    ])
    return (report,)


@app.cell
def _(mo, pipeline):
    _df = pipeline.manifest().to_frame()
    _scenes = sorted(_df[_df.cell == "GR"].scene.unique()) if len(_df) else []
    scene_pick = mo.ui.dropdown(_scenes, value=_scenes[0] if _scenes else None, label="preview scene")
    scene_pick
    return (scene_pick,)


@app.cell
def _(config, mo, pipeline, scene_pick):
    mo.stop(scene_pick.value is None)
    _clips = sorted((c for c in pipeline.manifest().load().values()
                     if c.scene == scene_pick.value and c.status == "ok" and "@" not in c.cell), key=lambda c: c.cell)
    mo.hstack([mo.vstack([mo.md(f"**{c.cell}**"),
                          mo.video(src=str(config.work_dir() / c.path), autoplay=True, loop=True, muted=True, width=340)])
               for c in _clips])
    return


@app.cell
def _(config, mo):
    import io
    import zipfile

    def _results_zip():
        _buf, _root = io.BytesIO(), config.work_dir()
        with zipfile.ZipFile(_buf, "w", zipfile.ZIP_DEFLATED) as _z:
            for _p in [_root / "manifest.jsonl", _root / "captions.json", *sorted((_root / "results").glob("*.csv"))]:
                if _p.exists():
                    _z.write(_p, _p.relative_to(_root).as_posix())
        return _buf.getvalue()

    mo.vstack([
        mo.md("Without `C44_HF_REPO`, outputs vanish when the session ends: download the tables before you stop."),
        mo.download(data=_results_zip, filename="c44_results.zip", label="Download results (tables + manifest)"),
    ])
    return


if __name__ == "__main__":
    app.run()
