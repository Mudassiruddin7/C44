"""Work on the weak side of the paper: the camera-aware attacker that defeats CPS-2, and defences that do not generalise.

What the main run showed (results/analysis): a blur + noise + exposure attacker, matched to the paired real clip, takes CPS-2 to
0.69/0.45 and the CPS-2 + CLIP fusion to 0.82/0.64; defences (SYM, RAT) only help against the operator families they were trained on.
This module builds the experiments that probe that:

  1. Operator FAMILIES beyond blur / noise / exposure, each a real camera-pipeline effect a renderer lacks: temporal noise reduction
     (tnr), chroma denoising, sharpening, luma denoising, rolling shutter (rs), optics (barrel distortion + chromatic aberration +
     vignette). Applied at random strength to real AND fake clips ('nz-<family>' cells) they are the training/test material for a
     leave-one-family-out evaluation of SYM / RAT, which is the first honest measurement of whether a defence transfers.
  2. An ATTACKER LADDER ('wk-L1'..'wk-L3', fakes only), each rung calibrated to its paired real clip like the main attacker:
         L1  blur (local exposure 0.5 x motion) + matched noise + matched exposure jitter      (the main attacker, flow-based)
         L2  L1 preceded by optics + rolling shutter
         L3  L2 with the ISP chain (denoise, temporal NR, chroma NR, sharpen) applied AFTER the noise, which is then matched post-ISP
     Detectability versus attacker effort is the "cost of camera realism" curve.
  3. JOINT cues (family J, 8 cues) that couple quantities inside a frame / across cues, which a global-parameter attacker does not:
     local blur vs local flow, local noise vs local flow, blur-vs-noise co-variation. CPS-3 = the 23 CPS-2 cues + J.

Everything here is CPU (opencv / numpy) and runs through the existing variant machinery (`pipeline.build_variants`, spawn process pool,
chunked pushes to the HF repo). Only the CLIP / NPR embeddings of the new clips need the GPU (`embed --weak`).
"""

from __future__ import annotations

import zlib

import cv2
import numpy as np

from . import cps

WEAK_VERSION = "w1.1"
_EPS = 1e-8
GRID = (6, 8)
JOINT = ["jt_rho_fs", "jt_rho_fd", "jt_rho_fn", "jt_slope_fs", "jt_blur_resid", "jt_tnr", "jt_rho_sn", "jt_noise_cv"]

OLD_FAMS = ["blur", "noise", "ae"]
NEW_FAMS = ["tnr", "chroma", "sharpen", "rs", "optics", "denoise"]
ALL_FAMS = OLD_FAMS + NEW_FAMS
ORDER = ["optics", "rs", "blur", "noise", "denoise", "tnr", "chroma", "sharpen", "ae"]       # physical order of the capture chain
ISP = ["denoise", "tnr", "chroma", "sharpen"]
CAL_FRAMES = 12

# registered with pipeline.build_variants (looked up after pipeline.VARIANTS). 'real': also built for real clips (symmetric training/test).
VARIANTS = {f"nz-{f}": ("wk", {"ops": [f], "real": True}) for f in NEW_FAMS}
VARIANTS.update({
    "nz-isp": ("wk", {"ops": list(ISP), "real": True}),
    "nz-geo": ("wk", {"ops": ["optics", "rs"], "real": True}),
    "nz-mix": ("wk", {"ops": "random3", "real": True}),
    "wk-L1": ("wk", {"ladder": 1}), "wk-L2": ("wk", {"ladder": 2}), "wk-L3": ("wk", {"ladder": 3}),
})
LADDER = ["wk-L1", "wk-L2", "wk-L3"]
# families contained in every operator-training kind, old and new (a leave-one-family-out fold drops every kind that contains the family)
PARTS = {"nz-blur": {"blur"}, "nz-noise": {"noise"}, "nz-ae": {"ae"}, "nz-blur+noise": {"blur", "noise"}, "nz-all": {"blur", "noise", "ae"},
         **{f"nz-{f}": {f} for f in NEW_FAMS}, "nz-isp": set(ISP), "nz-geo": {"optics", "rs"}, "nz-mix": set(ALL_FAMS)}
NZ_KINDS = list(PARTS)


# ----------------------------------------------------------------------------- joint cues (family J)
def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or np.ptp(a[ok]) < _EPS or np.ptp(b[ok]) < _EPS:
        return float("nan")
    return cps._spearman(a[ok], b[ok])


def _med(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) >= 4 else float("nan")


def _cell_slices(h: int, w: int, gh: int, gw: int, margin: int = 4):
    for i in range(gh):
        for j in range(gw):
            ys, ye = i * h // gh + margin, (i + 1) * h // gh - margin
            xs, xe = j * w // gw + margin, (j + 1) * w // gw - margin
            yield i, j, slice(ys, ye), slice(xs, xe)


def joint_features(frames: np.ndarray, mo: dict | None = None, step: int = 2) -> dict:
    """Family J of one clip (uint8 RGB [T,H,W,3]). A failure of any part yields NaN for that cue only, never an exception."""
    out = {k: float("nan") for k in JOINT}
    try:
        from . import physics
        mo = mo or physics.flow_motion(frames)
        g = cps.to_gray(frames)
        T, H, W = g.shape
        gh, gw = GRID
        rows = []
        for t in range(0, T - 1, step):
            fl = mo["grid"][t]                                           # [gh, gw, 2] px/frame at full resolution
            fmag = np.linalg.norm(fl, axis=-1)
            if not np.isfinite(fmag).all():
                continue
            x = g[t]
            g1, g2, g3 = cps._blur(x, 0.7), cps._blur(x, 1.5), cps._blur(x, 3.0)
            hi, mid = g1 - g2, g2 - g3
            gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
            r, mask = cps.noise_residual(x)
            f_l, s_l, d_l, n_l, tex = [], [], [], [], []
            for i, j, ys, xs in _cell_slices(H, W, gh, gw):
                f = float(fmag[i, j])
                hv, mv = float(hi[ys, xs].var()), float(mid[ys, xs].var())
                s_l.append(np.log(hv / (mv + _EPS) + _EPS))
                tex.append(mv)
                f_l.append(f)
                u = fl[i, j] / (f + _EPS)
                cx, cy = gx[ys, xs], gy[ys, xs]
                par = float(((cx * u[0] + cy * u[1]) ** 2).sum())
                perp = float(((-cx * u[1] + cy * u[0]) ** 2).sum())
                d_l.append(np.log(perp / (par + _EPS) + _EPS) if f >= 0.5 else np.nan)
                mk = mask[ys, xs]
                n_l.append(float(r[ys, xs][mk].std()) if mk.sum() >= 150 else np.nan)
            f_a, s_a, d_a, n_a, tex_a = map(np.array, (f_l, s_l, d_l, n_l, tex))
            textured = tex_a >= np.median(tex_a)                         # blur is only measurable where there is texture
            row = {"m": float(np.nanmedian(fmag)), "s": _med(s_a[textured]), "n": _med(n_a)}
            ok = textured & (f_a > 0.3)
            if ok.sum() >= 10:
                lf = np.log(f_a[ok] + 0.1)
                row["rho_fs"] = _spearman(f_a[ok], s_a[ok])
                row["rho_fd"] = _spearman(f_a[ok], d_a[ok])
                if np.ptp(lf) > 0.2:
                    A = np.stack([np.ones(ok.sum()), lf], 1)
                    coef, *_ = np.linalg.lstsq(A, s_a[ok], rcond=None)
                    row["slope_fs"] = float(coef[1])
                    row["blur_resid"] = float((s_a[ok] - A @ coef).std())
            okn = np.isfinite(n_a)
            if okn.sum() >= 12:
                row["rho_fn"] = _spearman(f_a[okn], n_a[okn])
                lo, hi_ = np.quantile(f_a[okn], [1 / 3, 2 / 3])
                a, b = n_a[okn & (f_a <= lo)], n_a[okn & (f_a >= hi_)]
                if len(a) >= 3 and len(b) >= 3 and np.median(b) > _EPS:
                    row["tnr"] = float(np.log(np.median(a) / np.median(b) + _EPS))
                row["noise_cv"] = float(np.nanstd(n_a) / (np.nanmean(n_a) + _EPS))
            rows.append(row)
        if len(rows) >= 6:
            col = lambda k: [r.get(k, np.nan) for r in rows]                # noqa: E731
            out["jt_rho_fs"], out["jt_rho_fd"] = _med(col("rho_fs")), _med(col("rho_fd"))
            out["jt_rho_fn"], out["jt_slope_fs"] = _med(col("rho_fn")), _med(col("slope_fs"))
            out["jt_blur_resid"], out["jt_tnr"] = _med(col("blur_resid")), _med(col("tnr"))
            out["jt_noise_cv"] = _med(col("noise_cv"))
            m, s, n = np.array(col("m")), np.array(col("s")), np.log(np.array(col("n")) + _EPS)
            ok = np.isfinite(m) & np.isfinite(s) & np.isfinite(n)
            if ok.sum() >= 10:                                              # exposure trades blur against noise
                A = np.stack([np.ones(ok.sum()), m[ok]], 1)
                rs = s[ok] - A @ np.linalg.lstsq(A, s[ok], rcond=None)[0]
                rn = n[ok] - A @ np.linalg.lstsq(A, n[ok], rcond=None)[0]
                out["jt_rho_sn"] = _spearman(rs, rn)
    except Exception:                                                    # noqa: BLE001 - one family never kills the clip
        pass
    return out


def clip_features_plus(frames: np.ndarray) -> dict:
    """CPS-2 (23 cues + motion) and J of one clip, sharing the optical flow."""
    from . import physics
    mo = physics.flow_motion(frames)
    f = physics.clip_physics(frames, mo)
    f.update(joint_features(frames, mo))
    return f


# ----------------------------------------------------------------------------- operator families (uint8 RGB [T,H,W,3] -> same)
def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


def op_tnr(frames: np.ndarray, rng: np.random.Generator, mo: dict) -> tuple[np.ndarray, dict]:
    """Motion-compensated recursive temporal noise reduction (the ISP block of every phone): y_t = (1-k) x_t + k W(y_{t+1}), run backwards in
    time so that the available forward flows can warp the already filtered neighbour; k falls where the warped neighbour disagrees (no ghosts)."""
    from . import physics
    a, gate = float(rng.uniform(0.45, 0.8)), float(rng.uniform(0.04, 0.12))
    T, H, W, _ = frames.shape
    x = frames.astype(np.float32)
    y = np.empty_like(x)
    y[-1] = x[-1]
    for t in range(T - 2, -1, -1):
        if t >= len(mo["flows"]):
            y[t] = x[t]
            continue
        fl = cv2.resize(mo["flows"][t], (W, H)) * 2.0
        w, ok = physics._warp(y[t + 1], fl)
        err = np.abs(x[t] - w).mean(-1, keepdims=True) / 255.0
        k = (1.0 - a) * np.exp(-(err / gate) ** 2) * ok[..., None]
        y[t] = (1.0 - k) * x[t] + k * w
    return _u8(y), {"tnr_keep": a, "tnr_gate": gate}


def op_chroma(frames: np.ndarray, rng: np.random.Generator, mo=None) -> tuple[np.ndarray, dict]:
    """Chroma noise reduction: low-pass Cr / Cb (ISPs suppress chroma noise far more than luma noise)."""
    s = float(rng.uniform(1.0, 3.0))
    out = np.empty_like(frames)
    for t, f in enumerate(frames):
        y = cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb)
        y[..., 1] = cv2.GaussianBlur(y[..., 1], (0, 0), s)
        y[..., 2] = cv2.GaussianBlur(y[..., 2], (0, 0), s)
        out[t] = cv2.cvtColor(y, cv2.COLOR_YCrCb2RGB)
    return out, {"chroma_sigma": s}


def op_sharpen(frames: np.ndarray, rng: np.random.Generator, mo=None) -> tuple[np.ndarray, dict]:
    """ISP sharpening (unsharp mask): halos and re-amplified noise."""
    amt, rad = float(rng.uniform(0.3, 1.0)), float(rng.uniform(1.0, 2.0))
    out = np.empty_like(frames)
    for t, f in enumerate(frames):
        out[t] = cv2.addWeighted(f, 1.0 + amt, cv2.GaussianBlur(f, (0, 0), rad), -amt, 0)
    return out, {"sharpen_amount": amt, "sharpen_radius": rad}


def op_denoise(frames: np.ndarray, rng: np.random.Generator, mo=None) -> tuple[np.ndarray, dict]:
    """Spatial luma noise reduction (edge-preserving)."""
    sc, ss = float(rng.uniform(8, 20)), float(rng.uniform(1.0, 2.0))
    out = np.empty_like(frames)
    for t, f in enumerate(frames):
        out[t] = cv2.bilateralFilter(f, 5, sc, ss)
    return out, {"denoise_color": sc, "denoise_space": ss}


def op_rs(frames: np.ndarray, rng: np.random.Generator, mo: dict) -> tuple[np.ndarray, dict]:
    """Rolling shutter: rows are exposed one after another, so a moving camera shears the frame. Row y of frame t is shifted by the
    image motion accumulated until it is read out (readout fraction rho of the frame interval)."""
    rho, gain = float(rng.uniform(0.3, 0.95)), float(rng.uniform(1.0, 2.5))
    T, H, W, _ = frames.shape
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    ys = (np.arange(H, dtype=np.float32) / H - 0.5)[:, None]
    out = np.empty_like(frames)
    for t, f in enumerate(frames):
        fl = mo["grid"][min(t, len(mo["grid"]) - 1)].reshape(-1, 2)
        v = np.nanmean(fl, axis=0)
        v = np.where(np.isfinite(v), v, 0.0)
        mx = (xx - gain * rho * float(v[0]) * ys).astype(np.float32)               # cv2.remap needs float32 maps
        my = (yy - gain * rho * float(v[1]) * ys).astype(np.float32)
        out[t] = cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out, {"rs_readout": rho, "rs_gain": gain}


def op_optics(frames: np.ndarray, rng: np.random.Generator, mo=None) -> tuple[np.ndarray, dict]:
    """Lens: barrel / pincushion distortion, lateral chromatic aberration (R and B scaled in opposite directions), vignetting."""
    k1, ca, vig = float(rng.uniform(-0.10, 0.04)), float(rng.uniform(0.0, 0.0025)), float(rng.uniform(0.05, 0.25))
    T, H, W, _ = frames.shape
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    xn, yn = (xx - W / 2) / (W / 2), (yy - H / 2) / (W / 2)
    r2 = xn ** 2 + yn ** 2
    maps = []
    for scale in (1.0 + ca, 1.0, 1.0 - ca):                                # R, G, B
        s = (1.0 + k1 * r2) * scale
        maps.append((((xn * s) * (W / 2) + W / 2).astype(np.float32), ((yn * s) * (W / 2) + H / 2).astype(np.float32)))
    gain = (1.0 - vig * r2 / max(float(r2.max()), _EPS)).astype(np.float32)[..., None]
    out = np.empty_like(frames)
    for t, f in enumerate(frames):
        ch = [cv2.remap(f[..., c], maps[c][0], maps[c][1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for c in range(3)]
        out[t] = _u8(np.stack(ch, -1).astype(np.float32) * gain)
    return out, {"optics_k1": k1, "optics_ca": ca, "optics_vignette": vig}


_OPS = {"tnr": op_tnr, "chroma": op_chroma, "sharpen": op_sharpen, "denoise": op_denoise, "rs": op_rs, "optics": op_optics}
_NEEDS_FLOW = {"blur", "tnr", "rs"}


def apply_ops(frames: np.ndarray, ops, rng: np.random.Generator, seed: int, mo: dict | None = None) -> tuple[np.ndarray, dict]:
    """Apply a set of operator families in physical capture order, each at a random strength. blur / noise / ae use the same operators
    and strength ranges as attacks.nuisance."""
    from . import attacks, physics
    ops = [o for o in ORDER if o in set(ops)]
    if mo is None and any(o in _NEEDS_FLOW for o in ops):
        mo = physics.flow_motion(frames)
    out, params = frames, {"ops": ops}
    for o in ops:
        if o in _OPS:
            out, p = _OPS[o](out, rng, mo)
            params.update(p)
        elif o == "blur":
            m = np.where(np.isfinite(mo["m"]), mo["m"], 0.0)
            d = np.where(np.isfinite(mo["direction"]), mo["direction"], [1.0, 0.0])
            params["blur_exposure"] = float(rng.uniform(0.15, 0.6))
            out = attacks.motion_blur(out, m, d, exposure=params["blur_exposure"])
        elif o == "noise":
            params["noise_strength"] = float(np.exp(rng.uniform(np.log(0.02), np.log(0.6))))
            out = attacks.scaled_noise(out, params["noise_strength"], seed)
        elif o == "ae":
            sg = float(np.exp(rng.uniform(np.log(2e-4), np.log(8e-3))))
            sw = float(sg * np.exp(rng.normal(0.0, 0.5)))
            params.update(gain_sigma=sg, wb_sigma=sw)
            out = attacks.exposure_jitter(out, sg, sw, seed + 1)
    return out, params


# ----------------------------------------------------------------------------- attacker ladder
def _isp_chain(frames: np.ndarray, mo: dict, rng_seed: int) -> np.ndarray:
    """The ISP blocks applied after the sensor noise, with parameters fixed by `rng_seed` (so the noise calibration probes see the same chain)."""
    out = frames
    for o in ISP:
        out, _ = _OPS[o](out, np.random.default_rng(rng_seed + zlib.crc32(o.encode()) % 1000), mo)
    return out


def _ladder(frames: np.ndarray, level: int, seed: int, parent: np.ndarray) -> tuple[np.ndarray, dict]:
    """One attacker-ladder rung on a render clip; noise level and exposure jitter are matched to the paired real clip `parent`."""
    from . import attacks, physics, pipeline
    mo = physics.flow_motion(frames)
    x = frames
    extra = {"ladder": level}
    if level >= 2:
        rng = np.random.default_rng(seed + 11)
        x, p1 = op_optics(x, rng, mo)
        x, p2 = op_rs(x, rng, mo)
        extra.update(p1, **p2)
    m = np.where(np.isfinite(mo["m"]), mo["m"], 0.0)
    d = np.where(np.isfinite(mo["direction"]), mo["direction"], [1.0, 0.0])
    x = attacks.motion_blur(x, m, d, exposure=0.5)                       # the measured exposure of the main attacker (data/blur_exposure.json)
    target = cps.resid_level(parent[2:CAL_FRAMES])
    isp_seed = seed + 29
    mo_cal = {**mo, "flows": mo["flows"][:CAL_FRAMES - 1], "grid": mo["grid"][:CAL_FRAMES]}
    if level >= 3:
        noisy = lambda s: _isp_chain(attacks.scaled_noise(x[:CAL_FRAMES], s, seed), mo_cal, isp_seed)       # noqa: E731
    else:
        noisy = lambda s: attacks.scaled_noise(x[:CAL_FRAMES], s, seed)                                      # noqa: E731
    strength, got = pipeline.calibrate_noise_strength(noisy, target)
    x = attacks.scaled_noise(x, strength, seed)
    if level >= 3:
        x = _isp_chain(x, mo, isp_seed)
    gm, wm = physics.gain_targets(parent)
    sg, sw = pipeline.calibrate_exposure_jitter(x, gm, wm, seed + 1)
    x = attacks.exposure_jitter(x, sg, sw, seed + 1)
    extra.update(noise_strength=strength, resid_target=target, resid_got=got, gain_target=gm, wb_target=wm, gain_sigma=sg, wb_sigma=sw)
    return x, extra


def apply_variant(frames: np.ndarray, param: dict, seed: int, parent_frames: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Entry point used by pipeline._variant_worker for the 'wk' operator."""
    if "ladder" in param:
        if parent_frames is None:
            raise ValueError("an attacker-ladder rung needs its paired real clip")
        return _ladder(frames, int(param["ladder"]), seed, parent_frames)
    rng = np.random.default_rng(seed)
    ops = param["ops"]
    if ops == "random3":
        ops = [str(o) for o in rng.choice(ALL_FAMS, 3, replace=False)]
    return apply_ops(frames, ops, rng, seed)


# ----------------------------------------------------------------------------- feature stage
def _feat_worker(job: tuple[str, str, bool]) -> dict:
    from . import video
    cid, path, full = job
    try:
        frames = video.read_video(path)
        return {"clip_id": cid, "full": full, **(clip_features_plus(frames) if full else joint_features(frames))}
    except Exception as e:                                                # noqa: BLE001
        return {"clip_id": cid, "error": repr(e)}


def valid_cells(cell: str) -> tuple[bool, bool]:
    """(feature-worthy, is a weak-only cell). Clean cells and main-run variants get J only; weak cells get CPS-2 and J."""
    from . import pipeline
    if "@" not in cell:
        return True, False
    var = cell.split("@")[1]
    return (var in pipeline.VARIANTS or var in VARIANTS), (var in VARIANTS)


def run_features(store=None, log=print, n_jobs: int | None = None, joint_old: bool = True) -> dict:
    """results/joint_features.csv (J of every clip) and results/physics_features_weak.csv (CPS-2 of the weak cells). Resumable; partial
    results are pushed to the HF repo every 100 clips."""
    import pandas as pd
    from . import physics, pipeline
    res = pipeline.results_dir()
    pj, pp = res / "joint_features.csv", res / "physics_features_weak.csv"
    dj = pd.read_csv(pj) if pj.exists() else pd.DataFrame(columns=["clip_id"])
    dp = pd.read_csv(pp) if pp.exists() else pd.DataFrame(columns=["clip_id"])
    have_j, have_p = set(dj.clip_id), set(dp.clip_id)
    jobs = []
    for c in pipeline.manifest().load().values():
        ok, weak_cell = valid_cells(c.cell)
        if c.status != "ok" or not ok or not pipeline._abs(c.path).exists():
            continue
        if weak_cell:
            if c.clip_id not in have_p or c.clip_id not in have_j:
                jobs.append((c.clip_id, str(pipeline._abs(c.path)), True))
        elif joint_old and c.clip_id not in have_j:
            jobs.append((c.clip_id, str(pipeline._abs(c.path)), False))
    jobs.sort(key=lambda j: (not j[2], j[0]))                              # weak cells first: the analysis needs them most
    log(f"[weak:features] {len(jobs)} clips to measure ({len(have_j)} with J, {len(have_p)} with CPS-2 cached)")
    rows_j, rows_p = [], []

    def save(final=False):
        a = pd.concat([dj, pd.DataFrame(rows_j)], ignore_index=True) if rows_j else dj
        b = pd.concat([dp, pd.DataFrame(rows_p)], ignore_index=True) if rows_p else dp
        a.to_csv(pj, index=False)
        b.to_csv(pp, index=False)
        pipeline._flush(store, [pj, pp], "c44: weak features" + ("" if final else f" (partial, {len(rows_j)})"))
        return a, b

    def keep(r):
        if "error" in r:
            log(f"[weak:features] {r['clip_id']} failed: {r['error']}")
            return
        rows_j.append({"clip_id": r["clip_id"], **{k: r.get(k, np.nan) for k in JOINT}})
        if r.get("full"):
            rows_p.append({"clip_id": r["clip_id"], **{k: r.get(k, np.nan) for k in list(physics.ALL) + ["motion_med"]}})
        if len(rows_j) % 100 == 0:
            save()

    if jobs:
        n = n_jobs or min(pipeline._default_workers(), 8)
        if n <= 1:                                                         # one worker: no pool (nothing to hang), same bookkeeping
            for j in jobs:
                keep(_feat_worker(j))
        else:
            pipeline.run_pool(_feat_worker, jobs, n, log=log, label="weak features", every=50, on_result=keep)
    dj, dp = save(final=True)
    return {"joint": len(dj), "physics_weak": len(dp)}


def run_attacks(store=None, n_scenes: int = 150, log=print, kinds=None) -> dict:
    """Build the operator-family clips and the ladder rungs (CPU, parallel, resumable, pushed every 100 clips)."""
    from . import pipeline
    out = {}
    for kind in (kinds or list(VARIANTS)):
        cells = ("RR", "GR") if kind in LADDER else ("RC", "RR", "GR")
        out[kind] = len(pipeline.build_variants(kind, cells=cells, store=store, limit_scenes=n_scenes, log=log))
    return out
