"""CPS-2 features: image-formation cues of a physical camera, measured from the pixels alone.

Everything is computed from the clip itself: image motion comes from optical flow (no depth model, no VGGT), so the
detector can be applied to any video. Two groups:

  v1 (cps.py, now driven by flow-based motion)
      rho_blur, rho_dir, sharp_ratio   exposure integrates motion: faster frames are blurrier, along the motion
      nlf_slope, nlf_slope_cv          signal-dependent shot noise
      fpn_corr, resid_level, resid_cv  sensor fixed pattern and noise level

  v2 (new): temporal behaviour and noise structure, which a renderer lacks by construction
      gain_mad, gain_range      auto-exposure: motion-compensated frame-to-frame luminance gain jitters on a camera
      wb_mad                    auto-white-balance: the same for the red/blue gain ratio
      tn_ratio                  temporal noise std (after motion compensation) / spatial noise std
      tn_corr                   correlation of noise residuals between motion-compensated neighbouring frames
      res_ac1                   whiteness of the noise residual (lag-1 autocorrelation)
      chroma_ratio              chroma-to-luma noise ratio (camera pipelines suppress chroma noise)
      clean_patch_frac          fraction of patches with (almost) no noise: a sensor never produces those
      psd_slope, psd_hf         radially averaged power spectrum slope and high/mid-frequency energy ratio
      acutance                  edge acutance (fine-scale / coarse-scale gradient at strong edges)
      flat_frac                 fraction of pixels with ~zero gradient
      vignette                  corner-to-centre luminance ratio of the temporal median image
      sharp_resid_cv, sharp_pump  sharpness fluctuation not explained by motion; periodicity of that fluctuation
Each family is wrapped so a failure yields NaN for that family only.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import cps

V1 = cps.FEATURES
V2 = ["gain_mad", "gain_range", "wb_mad", "tn_ratio", "tn_corr", "res_ac1", "chroma_ratio", "clean_patch_frac",
      "psd_slope", "psd_hf", "acutance", "flat_frac", "vignette", "sharp_resid_cv", "sharp_pump"]
ALL = V1 + V2
_EPS = 1e-8


# ----------------------------------------------------------------------------- motion from optical flow
def flow_motion(frames: np.ndarray, grid: tuple[int, int] = (6, 8)) -> dict:
    """DIS optical flow between consecutive frames at half resolution. Returns per-frame median image motion m [T]
    (px/frame at full resolution), mean direction [T,2], per-cell flow grid [T,gh,gw,2] and the raw flows."""
    T, H, W, _ = frames.shape
    small = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
                      for f in frames])
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gh, gw = grid
    flows = np.zeros((T - 1, H // 2, W // 2, 2), np.float32)
    m = np.full(T, np.nan)
    direction = np.full((T, 2), np.nan)
    cells = np.full((T, gh, gw, 2), np.nan)
    for t in range(T - 1):
        fl = dis.calc(small[t], small[t + 1], None)
        flows[t] = fl
        mag = np.linalg.norm(fl, axis=-1)
        m[t] = 2.0 * float(np.median(mag))
        mean = fl.reshape(-1, 2).mean(0)
        if np.linalg.norm(mean) > 1e-6:
            direction[t] = mean / np.linalg.norm(mean)
        h, w = fl.shape[:2]
        for i in range(gh):
            for j in range(gw):
                cells[t, i, j] = 2.0 * np.median(fl[i * h // gh:(i + 1) * h // gh, j * w // gw:(j + 1) * w // gw].reshape(-1, 2), axis=0)
    m[-1], direction[-1], cells[-1] = m[-2], direction[-2], cells[-2]
    return {"m": m, "direction": direction, "grid": cells, "flows": flows, "small": small}


def _warp(img: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample `img` (frame t+1) at x + flow(x): the content of frame t. Returns (warped, in-bounds mask)."""
    h, w = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = xs + flow[..., 0], ys + flow[..., 1]
    out = cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    ok = (mx >= 2) & (mx <= w - 3) & (my >= 2) & (my <= h - 3)
    return out, ok


def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


# ----------------------------------------------------------------------------- v2 families
def _gain_features(frames: np.ndarray, mo: dict) -> dict:
    """Motion-compensated luminance / white-balance gain between consecutive frames (half resolution, RGB)."""
    T, H, W, _ = frames.shape
    small = np.stack([cv2.resize(f, (W // 2, H // 2), interpolation=cv2.INTER_AREA) for f in frames]).astype(np.float32) / 255.0
    lg, lwb = [], []
    for t in range(T - 1):
        w, ok = _warp(small[t + 1], mo["flows"][t])
        a = small[t]
        ya, yw = a.mean(-1), w.mean(-1)
        err = np.abs(ya - yw)
        sel = ok & (ya > 0.12) & (ya < 0.88) & (err < max(2.0 * np.median(err[ok]), 1e-3))
        if sel.sum() < 2000:
            lg.append(np.nan), lwb.append(np.nan)
            continue
        g = np.array([np.median(w[..., c][sel] / np.maximum(a[..., c][sel], 1e-3)) for c in range(3)])
        lg.append(float(np.log(np.mean(g))))
        lwb.append(float(np.log(g[0] / g[2])))
    lg, lwb = np.array(lg), np.array(lwb)
    cum = np.nancumsum(lg)
    return {"gain_mad": _mad(lg), "gain_range": float(np.ptp(cum)) if np.isfinite(cum).all() else float("nan"),
            "wb_mad": _mad(lwb)}


def _noise_features(frames: np.ndarray, mo: dict, step: int = 3) -> dict:
    T = len(frames)
    g = cps.to_gray(frames)
    ts, tc, ac, cr, cp = [], [], [], [], []
    res_cache = {}

    def res(t):
        if t not in res_cache:
            res_cache[t] = cps.noise_residual(g[t])
        return res_cache[t]

    for t in range(0, T - 1, step):
        r, mask = res(t)
        if mask.sum() < 2000:
            continue
        sig_s = float(r[mask].std())
        # temporal: motion-compensated difference with the next frame, at full resolution
        fl = cv2.resize(mo["flows"][t], (g.shape[2], g.shape[1])) * 2.0
        gw, ok = _warp(g[t + 1], fl)
        sel = mask & ok
        if sel.sum() > 2000 and sig_s > _EPS:
            d = (g[t] - gw)[sel]
            ts.append(float(np.std(d) / (np.sqrt(2.0) * sig_s)))
            r1, _ = res(t + 1)
            rw, _ = _warp(r1, fl)
            a, b = r[sel], rw[sel]
            if a.std() > _EPS and b.std() > _EPS:
                tc.append(float(np.corrcoef(a, b)[0, 1]))
        # whiteness: lag-1 autocorrelation (average of horizontal and vertical) inside the flat mask
        mx = mask[:, 1:] & mask[:, :-1]
        my = mask[1:] & mask[:-1]
        if mx.sum() > 1000 and my.sum() > 1000:
            cx = np.corrcoef(r[:, 1:][mx], r[:, :-1][mx])[0, 1]
            cy = np.corrcoef(r[1:][my], r[:-1][my])[0, 1]
            ac.append(float(0.5 * (cx + cy)))
        # chroma vs luma noise
        ycc = cv2.cvtColor(frames[t], cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
        rc = [(ycc[..., k] - cps._blur(ycc[..., k], 1.0))[mask].std() for k in (1, 2)]
        if sig_s > _EPS:
            cr.append(float(np.mean(rc) / sig_s))
        # noise-free patches: 32x32 blocks with almost no residual energy inside otherwise valid, non-saturated areas
        h, w = r.shape
        bs = 32
        e = np.sqrt((r[:h // bs * bs, :w // bs * bs] ** 2).reshape(h // bs, bs, w // bs, bs).mean((1, 3)))
        lum = g[t][:h // bs * bs, :w // bs * bs].reshape(h // bs, bs, w // bs, bs).mean((1, 3))
        valid = (lum > 0.08) & (lum < 0.92)
        if valid.sum() > 20:
            cp.append(float((e[valid] < 0.2 * np.median(e[valid])).mean()))
    f = lambda v: float(np.median(v)) if len(v) else float("nan")  # noqa: E731
    return {"tn_ratio": f(ts), "tn_corr": f(tc), "res_ac1": f(ac), "chroma_ratio": f(cr), "clean_patch_frac": f(cp)}


def _spectral_features(frames: np.ndarray, step: int = 4) -> dict:
    g = cps.to_gray(frames)
    n = int(min(256, 2 ** np.floor(np.log2(min(g.shape[1:])))))              # 256 for the 704x480 benchmark clips
    win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    rad = np.sqrt(fx * fx + fy * fy)
    sl, hf, ac, fl = [], [], [], []
    for t in range(0, len(g), step):
        y0, x0 = (g.shape[1] - n) // 2, (g.shape[2] - n) // 2
        p = g[t, y0:y0 + n, x0:x0 + n]
        P = np.abs(np.fft.fft2((p - p.mean()) * win)) ** 2
        bands = lambda lo, hi: float(P[(rad >= lo) & (rad < hi)].mean() + _EPS)  # noqa: E731
        edges = np.linspace(0.04, 0.20, 9)
        xs = np.log(0.5 * (edges[1:] + edges[:-1]))
        ys = np.log([bands(a, b) for a, b in zip(edges[:-1], edges[1:])])
        sl.append(float(np.polyfit(xs, ys, 1)[0]))
        hf.append(float(np.log(bands(0.30, 0.50)) - np.log(bands(0.10, 0.20))))
        gx = cv2.Sobel(g[t], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g[t], cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        fl.append(float((mag < 0.004).mean()))
        m1 = np.sqrt(sum(cv2.Sobel(cps._blur(g[t], 0.7), cv2.CV_32F, *d, ksize=3) ** 2 for d in ((1, 0), (0, 1))))
        m2 = np.sqrt(sum(cv2.Sobel(cps._blur(g[t], 2.0), cv2.CV_32F, *d, ksize=3) ** 2 for d in ((1, 0), (0, 1))))
        top = m2 >= np.quantile(m2, 0.95)
        ac.append(float(np.mean(m1[top] / (m2[top] + _EPS))))
    f = lambda v: float(np.median(v)) if len(v) else float("nan")  # noqa: E731
    return {"psd_slope": f(sl), "psd_hf": f(hf), "acutance": f(ac), "flat_frac": f(fl)}


def _vignette(frames: np.ndarray) -> float:
    med = cps.to_gray(frames[::4]).mean(0)
    med = cps._blur(med, 8.0)
    h, w = med.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2) / np.sqrt(2)
    return float(np.log(med[r > 0.75].mean() + _EPS) - np.log(med[r < 0.25].mean() + _EPS))


def _sharp_features(frames: np.ndarray, m: np.ndarray) -> dict:
    g = cps.to_gray(frames)
    s = np.log(np.array([cps.sharpness(x) for x in g]) + _EPS)
    ok = np.isfinite(s) & np.isfinite(m)
    if ok.sum() < 12:
        return {"sharp_resid_cv": float("nan"), "sharp_pump": float("nan")}
    A = np.stack([np.ones(ok.sum()), m[ok]], 1)
    coef, *_ = np.linalg.lstsq(A, s[ok], rcond=None)
    resid = s[ok] - A @ coef
    spec = np.abs(np.fft.rfft(resid - resid.mean()))[1:]
    return {"sharp_resid_cv": float(resid.std()), "sharp_pump": float(spec.max() / (spec.mean() + _EPS))}


def clip_physics(frames: np.ndarray, mo: dict | None = None) -> dict:
    """All CPS-2 features of one clip (uint8 RGB [T,H,W,3]). `mo` = flow_motion(frames) when the caller already has it."""
    out = {k: float("nan") for k in ALL}
    mo = mo if mo is not None else flow_motion(frames)
    parts = [
        lambda: cps.clip_features(frames, mo["m"], mo["direction"], mo["grid"]),
        lambda: _gain_features(frames, mo),
        lambda: _noise_features(frames, mo),
        lambda: _spectral_features(frames),
        lambda: {"vignette": _vignette(frames)},
        lambda: _sharp_features(frames, mo["m"]),
    ]
    for p in parts:
        try:
            out.update({k: v for k, v in p().items() if k in out or k in ("motion_med",)})
        except Exception:                                                # noqa: BLE001 - one family never kills the clip
            pass
    out["motion_med"] = float(np.nanmedian(mo["m"]))
    return out


# ----------------------------------------------------------------------------- detectors on top of the features
def gain_targets(frames: np.ndarray) -> tuple[float, float]:
    """(gain_mad, wb_mad) of a clip: what an exposure-aware attacker must reproduce."""
    mo = flow_motion(frames)
    g = _gain_features(frames, mo)
    return g["gain_mad"], g["wb_mad"]


def _prep(features):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import QuantileTransformer
    return SimpleImputer(strategy="median"), QuantileTransformer(n_quantiles=100, output_distribution="normal", random_state=0)


class PhysicsClassifier:
    """CPS-2 (supervised): rank-normalised physics features -> L2 logistic regression, real (0) vs rendered/generated (1).
    Rank normalisation tames the heavy tails of several features (nlf_slope_cv reaches 300)."""

    def __init__(self, features=None, C: float = 0.3, class_weight="balanced"):
        self.features, self.C, self.class_weight = list(features or ALL), C, class_weight

    def fit(self, df, y):
        from sklearn.linear_model import LogisticRegression
        self.imp, self.qt = _prep(self.features)
        X = self.imp.fit_transform(df[self.features].to_numpy())
        self.qt.n_quantiles = min(100, len(X))
        self.lr = LogisticRegression(C=self.C, max_iter=3000, class_weight=self.class_weight)
        self.lr.fit(self.qt.fit_transform(X), np.asarray(y).astype(int))
        return self

    def score(self, df) -> np.ndarray:
        X = self.qt.transform(self.imp.transform(df[self.features].to_numpy()))
        return self.lr.predict_proba(X)[:, 1]


class PhysicsOneClass:
    """CPS-2 (one-class): Gaussian on rank-normalised features of REAL clips only; score = Mahalanobis distance."""

    def __init__(self, features=None):
        self.features = list(features or ALL)

    def fit(self, df_real):
        from sklearn.covariance import LedoitWolf
        self.imp, self.qt = _prep(self.features)
        X = self.imp.fit_transform(df_real[self.features].to_numpy())
        self.qt.n_quantiles = min(100, len(X))
        Z = self.qt.fit_transform(X)
        self.lw = LedoitWolf().fit(Z)
        return self

    def score(self, df) -> np.ndarray:
        return self.lw.mahalanobis(self.qt.transform(self.imp.transform(df[self.features].to_numpy())))
