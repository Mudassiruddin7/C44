"""One-call stages for a single molab cell:  from c44 import molab; molab.run("check" | "setup" | "pilot" | "evaluate" | "full").

Every GPU stage runs in a *fresh subprocess* (`python -m c44 <stage>`): the notebook process imported torch /
transformers before setup installed more packages, and Python caches those imports and "is X installed" checks, so
in-process stages can fail with errors like "requires accelerate" even though it is installed. A fresh process sees
the final environment. `pilot`/`full` first run a one-scene smoke test that exercises every component and reports
all failures together.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from . import config, pipeline, setup_env
from .storage import Store

VERSION = "v12-scale-ready"


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


def _stage(args: list[str]) -> tuple[int, str]:
    """Run `python -m c44 <args>` in a fresh process, streaming its output. Returns (returncode, last lines)."""
    code_root = str(Path(__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TQDM_DISABLE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
           "PYTHONPATH": os.pathsep.join(p for p in (code_root, os.environ.get("PYTHONPATH", "")) if p)}
    print(f"\n$ c44 {' '.join(args)}", flush=True)
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-u", "-m", "c44", *args], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(Path.cwd()))
    tail: deque = deque(maxlen=40)
    for line in p.stdout:
        print(line, end="", flush=True)
        tail.append(line)
    rc = p.wait()
    print(f"[c44 {args[0]}: {(time.time() - t0) / 60:.1f} min, exit {rc}]", flush=True)
    return rc, "".join(tail)


def _need(args: list[str]) -> None:
    rc, tail = _stage(args)
    if rc != 0:
        raise RuntimeError(f"stage `{' '.join(args)}` failed (exit {rc}). Last output:\n{tail}")


def _cell_errors(cell: str) -> list[str]:
    return [f"{c.clip_id}: {c.meta.get('error', c.meta.get('reason', c.status))}"
            for c in pipeline.manifest().load().values() if c.cell == cell and c.status != "ok"]


def _smoke() -> None:
    """One scene through every component. Each step is independent, so one run lists every problem at once."""
    import pandas as pd
    print("\n===== SMOKE TEST: 1 scene through every component =====")
    report: list[tuple[str, str, str]] = []          # (component, PASS/FAIL/SKIP, detail)

    def step(name: str, args: list[str], ok_fn, skip: bool = False):
        if skip:
            report.append((name, "SKIP", "depends on a failed step"))
            return False
        rc, tail = _stage(args)
        try:
            ok, detail = ok_fn()
        except Exception as e:                        # noqa: BLE001
            ok, detail = False, repr(e)
        if rc != 0:
            ok, detail = False, f"exit {rc}: " + " | ".join(tail.strip().splitlines()[-6:])
        report.append((name, "PASS" if ok else "FAIL", detail))
        return ok

    def cell_ok(cell):
        def f():
            cl = pipeline.manifest().by_cell(cell)
            if cl:
                return True, f"{len(cl)} clip(s)"
            errs = _cell_errors(cell)
            return False, "; ".join(errs[:2]) or "no clip produced"
        return f

    def cell_any(cell):
        def f():
            ok = [c for c in pipeline.manifest().load().values() if c.cell == cell and c.status in ("ok", "qc_fail")]
            if ok:
                warn = "" if all(c.status == "ok" for c in ok) else " (low coverage: qc_fail)"
                return True, f"{len(ok)} clip(s){warn}"
            return False, "; ".join(_cell_errors(cell)[:2]) or "no clip produced"
        return f

    def captions_ok():
        import json
        path = config.work_dir() / "captions.json"
        caps = json.loads(path.read_text()) if path.exists() else {}
        rc_ids = [c.clip_id for c in pipeline.manifest().by_cell("RC")]
        have = [k for k in rc_ids if caps.get(k)]
        return (bool(have), f"e.g. {caps[have[0]][:70]!r}" if have else "no caption written")

    def det_ok(name):
        def f():
            p = config.work_dir() / "results" / "scores.csv"
            df = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["detector", "score"])
            s = df[df.detector == name].score
            return (len(s) > 0 and s.notna().all(), f"{len(s)} scores, mean {s.mean():.3f}" if len(s) else "no scores")
        return f

    rc_ok = step("RC: DL3DV video + VGGT", ["rc", "--n", "1"], cell_ok("RC"))
    step("captions: Qwen2.5-VL", ["captions"], captions_ok, skip=not rc_ok)
    step("GR: FlashWorld + gsplat", ["gr", "--limit", "1"], cell_any("GR"), skip=not rc_ok)
    step("RR: DL3DV 3DGS + gsplat", ["rr"], cell_any("RR"), skip=not rc_ok)
    for d in ("univfd", "npr", "consistency"):
        step(f"detector {d}", ["detect", "--detectors", d], det_ok(d), skip=not rc_ok)
    step("CPS features", ["features"], lambda: (True, "ok"), skip=not rc_ok)

    print("\n===== SMOKE TEST RESULT =====")
    for name, status, detail in report:
        print(f"[{status}] {name}" + (f"  -> {detail}" if detail else ""))
    bad = [r for r in report if r[1] == "FAIL"]
    if bad:
        raise RuntimeError("SMOKE TEST FAILED (nothing long was started). Failing components:\n" +
                           "\n".join(f"- {n}: {d}" for n, _, d in bad))
    print("SMOKE TEST PASSED - starting the full build\n")


def _build(n: int, smoke: bool = True) -> "object":
    store = Store()
    store.ensure_repo()
    if store.enabled:
        print(f"resuming from private HF repo {store.repo_id} (if it has earlier work)")
        _need(["pull"])
    if smoke:
        _smoke()
    _need(["rc", "--n", str(n)])
    _need(["captions"])
    _need(["gr"])
    _need(["rr"])
    df = pipeline.manifest().to_frame()
    cols = [c for c in ("clip_id", "cell", "status", "meta.coverage", "meta.motion_med", "meta.error") if c in df]
    return df[cols]


def run(stage: str, n_scenes: int | None = None):
    if stage == "check":
        print(f"c44 code version: {VERSION}")
        print("\n".join(f"{k}: {v}" for k, v in setup_env.report().items()))
        print("\n".join(_check_access()))
        return None
    if stage == "setup":
        setup_env.require_gpu()
        # independent installs first, so one failure doesn't block the rest
        setup_env.install_base()
        setup_env.install_vggt()
        setup_env.install_npr()
        setup_env.install_flashworld()
        setup_env.remove_shadow_torch()
        setup_env.ensure_nvcc()
        setup_env.install_gsplat()
        print("SETUP DONE")
        return setup_env.report()
    if stage == "smoke":
        _smoke()
        return None
    if stage in ("pilot", "full"):
        return _build(n_scenes or (6 if stage == "pilot" else 150))
    if stage == "robust":
        import pandas as pd
        _need(["robust"])
        zip_path = shutil.make_archive(str(Path.cwd() / "c44_results"), "zip", config.work_dir(), base_dir="results")
        print(f"results (with robustness variants) saved to {zip_path}")
        return pd.read_csv(config.work_dir() / "results" / "report.csv").round(3)
    if stage == "evaluate":
        import pandas as pd
        _need(["evaluate"])
        res = config.work_dir() / "results"
        zip_path = shutil.make_archive(str(Path.cwd() / "c44_results"), "zip", config.work_dir(), base_dir="results")
        print(f"results saved to {zip_path} - download it from the file browser before the session ends")
        print(pd.read_csv(res / "feature_auc.csv").round(3).to_string(index=False))
        return pd.read_csv(res / "report.csv").round(3)
    raise ValueError("stage must be check, setup, smoke, pilot, evaluate, robust or full")
