"""One-call stages for a single molab cell:  from c44 import molab; molab.run("check" | "setup" | "pilot" | "evaluate" | "full")."""

from __future__ import annotations

import shutil
from pathlib import Path

from . import config, pipeline, setup_env
from .storage import Store


def _check_access() -> list[str]:
    from huggingface_hub import HfApi
    out, api = [], HfApi(token=config.hf_token())
    for repo in ("DL3DV/DL3DV-ALL-video", "DL3DV/DL3DV-GS-960P"):
        try:
            api.auth_check(repo, repo_type="dataset")
            out.append(f"{repo}: access OK")
        except Exception as e:                    # noqa: BLE001
            out.append(f"{repo}: NO ACCESS ({type(e).__name__})")
    return out


def _build(n: int, store: Store) -> "object":
    from .flashworld_gen import FlashWorld
    from .geometry import VGGTGeometry
    store.ensure_repo()
    geo = VGGTGeometry()
    pipeline.build_rc(n, store=store, geo=geo)
    pipeline.build_captions(store=store)
    pipeline._release_gpu()
    fw = FlashWorld(setup_env.code_dir() / "FlashWorld")
    pipeline.build_gr(fw, store=store, geo=geo)
    del fw
    pipeline._release_gpu()
    pipeline.build_rr(store=store, geo=geo)
    df = pipeline.manifest().to_frame()
    cols = [c for c in ("clip_id", "cell", "status", "meta.coverage", "meta.motion_med", "meta.error") if c in df]
    return df[cols]


def run(stage: str, n_scenes: int | None = None):
    store = Store()
    if stage == "check":
        print("\n".join(f"{k}: {v}" for k, v in setup_env.report().items()))
        print("\n".join(_check_access()))
        return None
    if stage == "setup":
        setup_env.ensure_nvcc()
        setup_env.install_gsplat()
        setup_env.install_vggt()
        setup_env.install_npr()
        setup_env.install_flashworld()
        print("SETUP DONE")
        return setup_env.report()
    if stage in ("pilot", "full"):
        return _build(n_scenes or (6 if stage == "pilot" else 150), store)
    if stage == "evaluate":
        pipeline.run_detectors(("univfd", "npr", "consistency"), store=store)
        n_rc = len(pipeline.manifest().by_cell("RC"))
        pipeline.run_cps(n_folds=0 if n_rc < 30 else 5, min_train=4 if n_rc < 30 else 10, store=store)
        rep = pipeline.report(n_boot=500)
        zip_path = shutil.make_archive(str(Path.cwd() / "c44_results"), "zip", config.work_dir(),
                                       base_dir="results")
        print(f"results saved to {zip_path} - download it from the file browser before the session ends")
        print(pipeline.feature_auc().round(3).to_string(index=False))
        return rep.round(3)
    raise ValueError("stage must be check, setup, pilot, evaluate or full")
