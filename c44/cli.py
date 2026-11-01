"""Command line: python -m c44 <stage> [options]. Stages are resumable; see README for the order."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="c44")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env", help="print the environment report")
    s = sub.add_parser("setup", help="CUDA compiler, gsplat, VGGT, FlashWorld, NPR")
    s.add_argument("--no-runfile", action="store_true", help="never download the 5.4 GB CUDA toolkit")
    sub.add_parser("pull", help="restore outputs from the HF repo into the work dir")
    s = sub.add_parser("rc", help="real camera clips")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    sub.add_parser("captions", help="VLM captions of RC first frames (FlashWorld prompts)")
    s = sub.add_parser("gr", help="FlashWorld world renders")
    s.add_argument("--limit", type=int)
    s.add_argument("--seed", type=int, default=0)
    sub.add_parser("rr", help="DL3DV 3DGS renders")
    s = sub.add_parser("gv", help="video-diffusion reference (experimental)")
    s.add_argument("--limit", type=int)
    s = sub.add_parser("detect", help="run detectors")
    s.add_argument("--detectors", default="univfd,npr,consistency")
    sub.add_parser("cps", help="capture-physics score (out-of-fold)")
    sub.add_parser("features", help="CPS per-clip features only (CPU)")
    sub.add_parser("evaluate", help="detectors + CPS + report tables (what the molab notebook runs)")
    s = sub.add_parser("robust", help="recompression + camera-simulating attack variants, rescored")
    s.add_argument("--n", type=int, default=75, help="scenes for the (expensive) calibrated attacks")
    s = sub.add_parser("control", help="supersampled GR/RR renders (resolution control), rescored")
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--ss", type=int, default=3)
    sub.add_parser("physics", help="CPS-2 physics features of every clip (CPU, parallel)")
    sub.add_parser("analyse", help="final tables: detectors, supervised physics score, attacks, control")
    sub.add_parser("report", help="metrics table")
    a = p.parse_args(argv)

    if a.cmd == "physics":                                # one BLAS/OpenMP thread per worker process
        import os
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[v] = "1"
    from . import pipeline, setup_env
    from .storage import Store
    store = Store()
    if a.cmd == "env":
        print(json.dumps(setup_env.report(), indent=1))
    elif a.cmd == "setup":
        setup_env.ensure_nvcc(allow_runfile=not a.no_runfile)
        setup_env.install_gsplat()
        setup_env.install_vggt()
        setup_env.install_npr()
        setup_env.install_flashworld()
        print(json.dumps(setup_env.report(), indent=1))
    elif a.cmd == "pull":
        # the manifest, tables, base clips and geometry; not the thousands of recompression/attack variant clips
        print(store.pull(["manifest.jsonl", "captions.json", "results/*", "meta/RC/*", "meta/GR/*", "meta/RR/*",
                          "clips/RC/*", "clips/GR/*", "clips/RR/*", "clips/GRs/*", "clips/RRs/*",
                          "clips/*@crf*/*", "clips/*@blurcal*/*", "clips/*@noiseadapt*/*", "clips/*@aeadapt*/*"]))
    elif a.cmd == "rc":
        store.ensure_repo()
        pipeline.build_rc(a.n, a.seed, store)
    elif a.cmd == "captions":
        pipeline.build_captions(store=store)
    elif a.cmd == "gr":
        from .flashworld_gen import FlashWorld
        fw = FlashWorld(setup_env.code_dir() / "FlashWorld")
        pipeline.build_gr(fw, a.seed, store, limit=a.limit)
    elif a.cmd == "rr":
        pipeline.build_rr(store=store)
    elif a.cmd == "gv":
        from .gen_video import WanI2V
        pipeline.build_gv(WanI2V(), store=store, limit=a.limit)
    elif a.cmd == "detect":
        pipeline.run_detectors(a.detectors.split(","), store=store)
    elif a.cmd == "cps":
        pipeline.run_cps(store=store)
    elif a.cmd == "features":
        df = pipeline.cps_features()
        print(df.describe().T.round(3).to_string() if len(df) else "no clips")
        if not len(df) or not df.drop(columns=["clip_id"]).notna().any().any():
            raise SystemExit("CPS features are empty/NaN")
    elif a.cmd in ("evaluate", "robust", "control"):
        if a.cmd == "robust":
            for kind in pipeline.VARIANTS:
                pipeline.build_variants(kind, store=store, limit_scenes=a.n)
        if a.cmd == "control":
            from .flashworld_gen import FlashWorld
            ctl = pipeline.subset_scenes(a.n, need=("RC", "GR", "RR"))
            fw = FlashWorld(setup_env.code_dir() / "FlashWorld")
            pipeline.build_gr(fw, store=store, cell="GRs", ss=a.ss, scenes=ctl)
            del fw
            pipeline._release_gpu()
            pipeline.build_rr(store=store, cell="RRs", ss=a.ss, scenes=ctl)
        cells = sorted(c for c in pipeline.manifest().to_frame().cell.unique()
                       if "@" not in c or c.split("@")[1] in pipeline.VARIANTS)   # skip retired attack variants
        pipeline.run_detectors(("univfd", "npr", "consistency"), cells=cells, store=store)
        n_rc = len(pipeline.manifest().by_cell("RC"))
        pipeline.run_cps(n_folds=0 if n_rc < 30 else 5, min_train=4 if n_rc < 30 else 10, store=store)
        pipeline.report(n_boot=500)
        pipeline.feature_auc().to_csv(pipeline.results_dir() / "feature_auc.csv", index=False)
    elif a.cmd == "physics":
        df = pipeline.physics_features(store=store)
        print(f"{len(df)} clips with physics features")
    elif a.cmd == "analyse":
        from . import analysis
        print(analysis.run(store=store))
    elif a.cmd == "report":
        print(pipeline.report().to_string(index=False))


if __name__ == "__main__":
    main()
