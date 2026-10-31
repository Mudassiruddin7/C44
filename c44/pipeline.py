"""Benchmark build and evaluation stages.

Every stage is resumable: finished clips are skipped, errored ones retried, and outputs are pushed to the
private HF repo every `batch` clips, so a 12-hour molab shutdown loses at most one batch.
Pairing: RR and GR clips record the RC clip they were derived from in `parent` (same scene, matched motion).
"""

from __future__ import annotations

import json
import shutil
import traceback
import zlib
from pathlib import Path
from typing import Callable

import numpy as np

from . import cameras, config, dl3dv, geometry, video
from .config import CELLS, SPEC
from .manifest import Clip, Manifest
from .storage import Store

Log = Callable[[str], None]


def manifest() -> Manifest:
    return Manifest(config.work_dir() / "manifest.jsonl")


def results_dir() -> Path:
    p = config.work_dir() / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _rel(p: Path) -> str:
    return Path(p).resolve().relative_to(config.work_dir().resolve()).as_posix()


def _abs(rel: str) -> Path:
    return config.work_dir() / rel


def _flush(store: Store | None, extra: list[Path] = (), msg: str = "c44") -> None:
    if store is None:
        return
    for p in [config.work_dir() / "manifest.jsonl", *extra]:
        if Path(p).exists():
            store.stage(p)
    store.flush(msg)


def _finished(cell: str, key: str = "scene") -> set[str]:
    return {getattr(c, key) for c in manifest().load().values() if c.cell == cell and c.status in ("ok", "qc_fail")}


def _where(e: BaseException) -> str:
    tb = traceback.extract_tb(e.__traceback__)
    return f"{Path(tb[-1].filename).name}:{tb[-1].lineno} in {tb[-1].name}" if tb else ""


def _why(e: BaseException) -> str:
    """repr + innermost frame + last traceback lines, for the log."""
    tail = " | ".join(traceback.format_exc().strip().splitlines()[-6:])
    return f"{e!r} @ {_where(e)} :: {tail}"


def _fail(cell: str, cid: str, scene: str, status: str, parent: str = "", **meta) -> None:
    manifest().add(Clip(cid, cell, scene, "", 0, 0.0, 0, 0, parent=parent, status=status, meta=meta))


def _add_clip(cell: str, cid: str, scene: str, frames: np.ndarray, geo, generator: str, parent: str = "",
              status: str = "ok", store: Store | None = None, **meta) -> Clip:
    """Encode, estimate geometry on the *encoded* clip (what detectors see), record and stage for upload."""
    out = video.write_mp4(frames, config.clip_path(cell, cid), SPEC.fps, SPEC.crf, SPEC.preset)
    g = geo(video.read_video(out))
    g.save(config.meta_path(cell, cid))
    meta["motion_med"] = float(np.nanmedian(geometry.motion(g)["m"]))
    clip = Clip(cid, cell, scene, _rel(out), len(frames), SPEC.fps, SPEC.width, SPEC.height, generator=generator,
                parent=parent, status=status, meta=meta)
    manifest().add(clip)
    if store is not None:
        store.stage(out)
        store.stage(config.meta_path(cell, cid))
    return clip


# ----------------------------------------------------------------------------- RC
def build_rc(n: int, seed: int = 0, store: Store | None = None, log: Log = print, geo=None,
             keep_video: bool = False, batch: int = 10) -> list[str]:
    """Real camera clips: one 48-frame window per sampled DL3DV scene (scenes with a 3DGS model only)."""
    index = dl3dv.build_index()
    scenes = dl3dv.sample_scenes(index, n, seed)
    done, made = _finished("RC"), []
    for k, scene in enumerate(scenes):
        if scene in done:
            continue
        cid, vpath = f"RC_{scene[:16]}", None
        try:
            vpath = dl3dv.fetch_video(scene, index)
            info = video.probe(vpath)
            step = 2 if info["fps"] > SPEC.max_native_fps else 1
            need, nat = SPEC.n_frames * step, info["n_frames"]
            if nat < need + 20:
                _fail("RC", cid, scene, "qc_fail", reason="video too short", **info)
                continue
            rng = np.random.default_rng([seed, k])
            lo = int(0.1 * nat)
            start = int(rng.integers(lo, max(lo + 1, int(0.9 * nat) - need)))
            frames = np.stack(list(video.iter_frames(vpath, start, step, SPEC.n_frames, (SPEC.width, SPEC.height))))
            if len(frames) < SPEC.n_frames:
                raise RuntimeError(f"decoded only {len(frames)} frames")
            geo = geo or geometry.VGGTGeometry()
            _add_clip("RC", cid, scene, frames, geo, "camera", store=store, native_fps=info["fps"],
                      native_wh=[info["width"], info["height"]], start=start, step=step)
            made.append(cid)
            log(f"[RC] {len(made)} done, scene {k + 1}/{len(scenes)}: {cid}")
        except Exception as e:                    # noqa: BLE001 - keep building, record the error, retry later
            log(f"[RC] {scene[:12]} failed: {_why(e)}")
            _fail("RC", cid, scene, "error", error=f"{e!r} @ {_where(e)}")
        finally:
            if vpath is not None and not keep_video and Path(vpath).exists():
                Path(vpath).unlink()
        if made and len(made) % batch == 0:
            _flush(store, msg=f"c44: RC x{len(made)}")
    _flush(store, msg="c44: RC")
    return made


# ----------------------------------------------------------------------------- captions
def build_captions(vlm=None, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", store: Store | None = None,
                   log: Log = print) -> dict:
    path = config.work_dir() / "captions.json"
    caps = json.loads(path.read_text()) if path.exists() else {}
    for rc in manifest().by_cell("RC"):
        if rc.clip_id in caps:
            continue
        if vlm is None:
            from .vlm import VLM
            vlm = VLM(model_id)
        first = next(video.iter_frames(_abs(rc.path), count=1))
        caps[rc.clip_id] = vlm.caption(first)
        path.write_text(json.dumps(caps, indent=1))
        log(f"[caption] {rc.clip_id}: {caps[rc.clip_id][:80]}")
    _flush(store, [path], "c44: captions")
    return caps


# ----------------------------------------------------------------------------- GR
def build_gr(fw, seed: int = 0, store: Store | None = None, log: Log = print, geo=None, limit: int | None = None,
             batch: int = 10) -> list[str]:
    """FlashWorld world per RC clip: first frame + 24 keyframes of its camera path; rendered along all 48 poses."""
    caps_path = config.work_dir() / "captions.json"
    caps = json.loads(caps_path.read_text()) if caps_path.exists() else {}
    done = _finished("GR", "parent")
    todo = [c for c in manifest().by_cell("RC") if c.clip_id not in done][:limit]
    made = []
    for i, rc in enumerate(todo):
        cid = f"GR_{rc.scene[:16]}"
        try:
            frames = video.read_video(_abs(rc.path))
            g = geometry.Geometry.load(config.meta_path("RC", rc.clip_id))
            c2w, K = cameras.invert(g.w2c), g.K_image().mean(0)
            key = np.unique(np.linspace(0, len(frames) - 1, SPEC.keyframes).round().astype(int))
            cam_key = cameras.to_fw_cam11(c2w[key], K, SPEC.width, SPEC.height)
            cam_all = cameras.to_fw_cam11(c2w, K, SPEC.width, SPEC.height)
            text = caps.get(rc.clip_id, "")
            scene, ref, Tn, secs = fw.generate(frames[0], cam_key, text, image_index=0, seed=seed)
            out_frames, qc = fw.render_path(scene, ref, Tn, cam_all, SPEC.width, SPEC.height, SPEC.background)
            del scene
            cov = float(qc["coverage"].mean())
            geo = geo or geometry.VGGTGeometry()
            _add_clip("GR", cid, rc.scene, out_frames, geo, "flashworld", parent=rc.clip_id,
                      status="ok" if cov >= SPEC.min_coverage else "qc_fail", store=store, coverage=cov,
                      holes=float(qc["holes"].mean()), gen_seconds=secs, caption=text, T_norm=Tn)
            made.append(cid)
            log(f"[GR] {i + 1}/{len(todo)} {cid} coverage={cov:.3f} gen={secs:.1f}s")
        except Exception as e:                    # noqa: BLE001
            log(f"[GR] {rc.clip_id} failed: {_why(e)}")
            _fail("GR", cid, rc.scene, "error", parent=rc.clip_id, error=f"{e!r} @ {_where(e)}")
        if made and len(made) % batch == 0:
            _flush(store, msg=f"c44: GR x{len(made)}")
    _flush(store, msg="c44: GR")
    return made


# ----------------------------------------------------------------------------- RR
def build_rr(seed: int = 0, store: Store | None = None, log: Log = print, geo=None, device: str = "cuda",
             keep_gs: bool = False, batch: int = 10) -> list[str]:
    """DL3DV 3DGS of the RC scene, rendered along its training-camera path resampled so that the per-frame image
    motion reproduces the RC clip's motion profile (VGGT); identical rasteriser settings to GR."""
    from . import gs_render
    index = dl3dv.build_index()
    done = _finished("RR", "parent")
    todo = [c for c in manifest().by_cell("RC") if c.clip_id not in done]
    made = []
    for i, rc in enumerate(todo):
        cid = f"RR_{rc.scene[:16]}"
        gs_dir = config.work_dir() / "gs" / rc.scene
        try:
            files = dl3dv.fetch_gs(rc.scene, index)
            gs_np, cams = dl3dv.load_gs_ply(files["ply"]), dl3dv.load_cameras(files["cameras"])
            c2ws = np.stack([c["c2w"] for c in cams])
            K = video.crop_resize_K(cams[0]["K"], (cams[0]["width"], cams[0]["height"]), (SPEC.width, SPEC.height))
            depth = dl3dv.scene_depth(gs_np, cams)
            m = geometry.motion(geometry.Geometry.load(config.meta_path("RC", rc.clip_id)))["m"][:-1]
            m = np.where(np.isfinite(m), m, np.nanmedian(m))
            rng, path = np.random.default_rng([seed, i]), None
            for start in rng.permutation(max(1, len(c2ws) - 2)):
                path = cameras.resample_by_profile(c2ws, K, SPEC.width, SPEC.height, depth, m, int(start))
                if path is not None:
                    break
            if path is None:
                _fail("RR", cid, rc.scene, "qc_fail", parent=rc.clip_id, reason="camera path too short for profile")
                continue
            frames, qc = gs_render.render(gs_render.gs_to_torch(gs_np, device), path, K, SPEC.width, SPEC.height,
                                          SPEC.background)
            cov = float(qc["coverage"].mean())
            geo = geo or geometry.VGGTGeometry()
            _add_clip("RR", cid, rc.scene, frames, geo, "dl3dv-3dgs", parent=rc.clip_id,
                      status="ok" if cov >= SPEC.min_coverage else "qc_fail", store=store, coverage=cov,
                      holes=float(qc["holes"].mean()), start_key=int(start), scene_depth=depth)
            made.append(cid)
            log(f"[RR] {i + 1}/{len(todo)} {cid} coverage={cov:.3f}")
        except Exception as e:                    # noqa: BLE001
            log(f"[RR] {rc.clip_id} failed: {_why(e)}")
            _fail("RR", cid, rc.scene, "error", parent=rc.clip_id, error=f"{e!r} @ {_where(e)}")
        finally:
            if not keep_gs:
                shutil.rmtree(gs_dir, ignore_errors=True)
                shutil.rmtree(config.work_dir() / "downloads" / "3DGS", ignore_errors=True)
        if made and len(made) % batch == 0:
            _flush(store, msg=f"c44: RR x{len(made)}")
    _flush(store, msg="c44: RR")
    return made


# ----------------------------------------------------------------------------- GV (experimental)
def build_gv(gen, seed: int = 0, store: Store | None = None, log: Log = print, geo=None, limit: int | None = None,
             batch: int = 10) -> list[str]:
    """Video-diffusion reference: image-to-video from the RC first frame + caption (see gen_video.py caveats)."""
    caps_path = config.work_dir() / "captions.json"
    caps = json.loads(caps_path.read_text()) if caps_path.exists() else {}
    done = _finished("GV", "parent")
    todo = [c for c in manifest().by_cell("RC") if c.clip_id not in done][:limit]
    made = []
    for i, rc in enumerate(todo):
        cid = f"GV_{rc.scene[:16]}"
        try:
            first = next(video.iter_frames(_abs(rc.path), count=1))
            prompt = (caps.get(rc.clip_id, "") + " The camera moves slowly and smoothly through the scene.").strip()
            frames = gen(first, prompt, SPEC.n_frames, (SPEC.width, SPEC.height), seed=seed)
            geo = geo or geometry.VGGTGeometry()
            _add_clip("GV", cid, rc.scene, frames, geo, "wan2.2-ti2v-5b", parent=rc.clip_id, store=store, prompt=prompt)
            made.append(cid)
            log(f"[GV] {i + 1}/{len(todo)} {cid}")
        except Exception as e:                    # noqa: BLE001
            log(f"[GV] {rc.clip_id} failed: {_why(e)}")
            _fail("GV", cid, rc.scene, "error", parent=rc.clip_id, error=f"{e!r} @ {_where(e)}")
        if made and len(made) % batch == 0:
            _flush(store, msg=f"c44: GV x{len(made)}")
    _flush(store, msg="c44: GV")
    return made


# ----------------------------------------------------------------------------- geometry for any cell
def ensure_geometry(cells=CELLS, geo=None, log: Log = print) -> int:
    n = 0
    for cell in cells:
        for c in manifest().by_cell(cell):
            if not config.meta_path(cell, c.clip_id).exists():
                geo = geo or geometry.VGGTGeometry()
                geo(video.read_video(_abs(c.path))).save(config.meta_path(cell, c.clip_id))
                n += 1
    log(f"[geometry] computed {n}")
    return n


# ----------------------------------------------------------------------------- detectors
def run_detectors(names=("univfd", "npr", "consistency"), cells=CELLS, store: Store | None = None,
                  log: Log = print) -> "object":
    import pandas as pd
    from . import detectors, setup_env
    out = results_dir() / "scores.csv"
    df = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["clip_id", "detector", "score"])
    have = set(zip(df.clip_id, df.detector))
    clips = [c for cell in cells for c in manifest().by_cell(cell)]
    for name in names:
        kw = {"repo_dir": setup_env.code_dir() / "NPR"} if name == "npr" else {}
        det, rows = detectors.make(name, **kw), []
        for j, c in enumerate(clips):
            if (c.clip_id, name) in have:
                continue
            geom = geometry.Geometry.load(config.meta_path(c.cell, c.clip_id)) if det.needs_geometry else None
            rows.append({"clip_id": c.clip_id, "detector": name, "score": det.score(video.read_video(_abs(c.path)), geom)})
            if len(rows) % 50 == 0:
                log(f"[{name}] {j + 1}/{len(clips)}")
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(out, index=False)
        del det
        _release_gpu()
        log(f"[{name}] scored {len(rows)} new clips")
    _flush(store, [out], "c44: detector scores")
    return df


def _release_gpu() -> None:
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except ImportError:
        pass


# ----------------------------------------------------------------------------- CPS
def cps_features(cells=CELLS, log: Log = print) -> "object":
    import pandas as pd
    from . import cps
    out = results_dir() / "cps_features.csv"
    df = pd.read_csv(out) if out.exists() else pd.DataFrame()
    have = set(df.clip_id) if len(df) else set()
    rows = []
    for cell in cells:
        for c in manifest().by_cell(cell):
            if c.clip_id in have:
                continue
            mo = geometry.motion(geometry.Geometry.load(config.meta_path(cell, c.clip_id)))
            f = cps.clip_features(video.read_video(_abs(c.path)), mo["m"], mo["direction"], mo["flow_grid"])
            rows.append({"clip_id": c.clip_id, **f})
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(out, index=False)
    log(f"[cps] features for {len(rows)} new clips")
    return df


def run_cps(n_folds: int = 5, seed: int = 0, store: Store | None = None, log: Log = print,
            min_train: int = 10, cells=None) -> "object":
    """Out-of-fold CPS: folds are scene-disjoint; each fold's model is fit on real (RC) clips of the other scenes.
    n_folds=0 means leave-one-scene-out (useful for small pilots, together with a lower min_train)."""
    import pandas as pd
    from . import cps
    man = manifest().to_frame()[["clip_id", "cell", "scene", "status"]]
    cells = cells or sorted(man.cell.unique())
    feats = cps_features(cells=cells, log=log)
    df = feats.merge(man[man.status == "ok"], on="clip_id")
    scenes = np.array(sorted(df.scene.unique()))
    n_folds = n_folds or len(scenes)
    fold_of = dict(zip(scenes, np.random.default_rng(seed).permutation(len(scenes)) % n_folds))
    df["fold"] = df.scene.map(fold_of)
    scores = []
    for k in range(n_folds):
        train = df[(df.fold != k) & (df.cell == "RC")]
        test = df[df.fold == k]
        if len(train) < min_train or test.empty:
            continue
        model = cps.OneClassCPS().fit(train.to_dict("records"))
        scores.append(pd.DataFrame({"clip_id": test.clip_id, "detector": "cps",
                                    "score": model.score(test.to_dict("records"))}))
    out = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame(columns=["clip_id", "detector", "score"])
    path = results_dir() / "scores.csv"
    allsc = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=out.columns)
    allsc = pd.concat([allsc[allsc.detector != "cps"], out], ignore_index=True)
    allsc.to_csv(path, index=False)
    _flush(store, [path, results_dir() / "cps_features.csv"], "c44: CPS")
    log(f"[cps] scored {len(out)} clips out-of-fold")
    return out


# ----------------------------------------------------------------------------- robustness / adaptive attacks
VARIANTS = {
    "crf23": ("compress", 23), "crf28": ("compress", 28), "crf35": ("compress", 35),
    "blur": ("blur", 0.5), "noise": ("noise", 0.01), "blur+noise": ("blur+noise", 0.5),
}


def build_variants(kind: str, cells=("RC", "RR", "GR"), store: Store | None = None, log: Log = print) -> list[str]:
    """Derived cells '<cell>@<kind>' (e.g. RC@crf28, GR@blur): recompression for every cell, camera-simulating
    attacks (motion blur along the measured motion, sensor noise with PRNU) for render cells. Geometry is reused
    from the parent clip, since camera motion is unchanged."""
    from . import attacks
    op, param = VARIANTS[kind]
    made = []
    done = _finished_parents()
    for cell in cells:
        if op != "compress" and cell == "RC":
            continue
        vcell = f"{cell}@{kind}"
        for c in manifest().by_cell(cell):
            if (vcell, c.clip_id) in done:
                continue
            cid = f"{c.clip_id}@{kind}"
            frames = video.read_video(_abs(c.path))
            out = config.clip_path(vcell, cid)
            if op == "compress":
                video.write_mp4(frames, out, SPEC.fps, crf=int(param), preset=SPEC.preset)
            else:
                mo = geometry.motion(geometry.Geometry.load(config.meta_path(cell, c.clip_id)))
                d = np.where(np.isfinite(mo["direction"]), mo["direction"], [1.0, 0.0])
                m = np.where(np.isfinite(mo["m"]), mo["m"], 0.0)
                if "blur" in op:
                    frames = attacks.motion_blur(frames, m, d, exposure=float(param))
                if "noise" in op:
                    frames = attacks.sensor_noise(frames, prnu_std=0.01, seed=zlib.crc32(cid.encode()))
                video.write_mp4(frames, out, SPEC.fps, SPEC.crf, SPEC.preset)
            shutil.copy(config.meta_path(cell, c.clip_id), config.meta_path(vcell, cid))
            manifest().add(Clip(cid, vcell, c.scene, _rel(out), c.n_frames, c.fps, c.width, c.height,
                                generator=c.generator, parent=c.clip_id, meta={"variant": kind}))
            if store is not None:
                store.stage(out)
            made.append(cid)
    log(f"[variants] {kind}: {len(made)} clips")
    _flush(store, msg=f"c44: variants {kind}")
    return made


def _finished_parents() -> set[tuple[str, str]]:
    return {(c.cell, c.parent) for c in manifest().load().values() if "@" in c.cell and c.status == "ok"}


# ----------------------------------------------------------------------------- report
def report(pairs=None, n_boot: int = 1000) -> "object":
    """AUC [95% scene-bootstrap CI], AP and TPR@1%FPR of every detector for each (negative, positive) cell pair.
    Default pairs: RC vs every other base cell, and each recompressed/attacked cell vs its matching real cell."""
    import pandas as pd
    from .metrics import compare_cells
    sc = pd.read_csv(results_dir() / "scores.csv")
    man = manifest().to_frame()
    df = sc.merge(man[man.status == "ok"][["clip_id", "cell", "scene"]], on="clip_id")
    present = set(df.cell)
    if pairs is None:
        pairs = [("RC", c) for c in CELLS if c != "RC" and c in present]
        for c in sorted(present):
            if "@" in c:
                base, kind = c.split("@")
                neg = f"RC@{kind}" if VARIANTS[kind][0] == "compress" else "RC"
                if base != "RC" and neg in present:
                    pairs.append((neg, c))
    tables = []
    for name, d in df.groupby("detector"):
        for neg, pos in pairs:
            t = compare_cells(d, "score", neg_cell=neg, pos_cells=[pos], n_boot=n_boot)
            if len(t):
                tables.append(t.assign(detector=name))
    out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    out.to_csv(results_dir() / "report.csv", index=False)
    return out


def feature_auc(neg: str = "RC") -> "object":
    """Per-CPS-feature AUC of each cell against `neg` (0.5 = no signal; <0.5 = feature lower for that cell)."""
    import pandas as pd
    from .cps import FEATURES
    from .metrics import roc_auc
    feats = pd.read_csv(results_dir() / "cps_features.csv")
    man = manifest().to_frame()
    df = feats.merge(man[man.status == "ok"][["clip_id", "cell"]], on="clip_id")
    rows = []
    for cell in sorted(set(df.cell) - {neg}):
        sub = df[df.cell.isin([neg, cell])]
        y = (sub.cell == cell).to_numpy()
        rows.append({"cell": cell, **{f: roc_auc(y[np.isfinite(sub[f])], sub[f][np.isfinite(sub[f])])
                                      for f in FEATURES + ["motion_med"]}})
    return pd.DataFrame(rows)
