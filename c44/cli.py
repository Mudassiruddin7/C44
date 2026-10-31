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
    sub.add_parser("report", help="metrics table")
    a = p.parse_args(argv)

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
        print(store.pull())
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
    elif a.cmd == "report":
        print(pipeline.report().to_string(index=False))


if __name__ == "__main__":
    main()
