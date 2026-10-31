"""Capture-physics score (CPS): image-formation cues of a physical camera, fit on real footage only.

Per clip we measure
  rho_blur      Spearman(-sharpness_t, motion_t): exposure integrates motion, so faster frames are blurrier
  rho_dir       Spearman(directional_ratio_t, motion_t): blur suppresses gradients along the motion direction
  sharp_ratio   log(median sharpness of the fastest quarter / slowest quarter of frames)
  nlf_slope     slope of residual variance vs. intensity (signal-dependent shot noise), scale-free
  nlf_slope_cv  temporal stability of that slope
  fpn_corr      correlation of noise residuals at the same pixels across frames whose content has moved
                (fixed-pattern noise is pinned to the sensor, scene texture is not)
  resid_level   median residual std in flat regions
  resid_cv      temporal coefficient of variation of the residual std
A renderer samples an instantaneous, noise-free pinhole image, so these cues are absent unless simulated.
The one-class model (robust standardisation + Ledoit-Wolf Mahalanobis) never sees generated or rendered data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.stats import spearmanr

FEATURES = ["rho_blur", "rho_dir", "sharp_ratio", "nlf_slope", "nlf_slope_cv", "fpn_corr", "resid_level", "resid_cv"]
_EPS = 1e-8


def to_gray(frames: np.ndarray) -> np.ndarray:
    f = frames.astype(np.float32) / 255.0
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _blur(g: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(g, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def sharpness(g: np.ndarray, border: int = 8) -> float:
    """Band-energy ratio (sigma 0.7-1.5 over 1.5-3.0); falls when the frame is motion- or defocus-blurred.

    The finest band (below sigma ~0.7) is deliberately excluded: sensor noise dominates it and does not
    change with blur, which would flip the sign of the blur-motion coupling on noisy footage.
    """
    g1, g2, g3 = _blur(g, 0.7), _blur(g, 1.5), _blur(g, 3.0)
    hi, mid = (g1 - g2)[border:-border, border:-border], (g2 - g3)[border:-border, border:-border]
    return float(hi.var() / (mid.var() + _EPS))


def directional_ratio(g: np.ndarray, flow: np.ndarray | None, direction: np.ndarray | None,
                      min_flow: float = 0.5) -> float:
    """Gradient energy perpendicular / parallel to the local motion direction (>1 when blurred along motion).

    flow: [gh, gw, 2] per-cell image motion (px); direction: fallback global unit vector [2].
    """
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    H, W = g.shape
    if flow is None:
        if direction is None or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) < _EPS:
            return float("nan")
        u = np.asarray(direction, np.float32) / (np.linalg.norm(direction) + _EPS)
        par = (gx * u[0] + gy * u[1]) ** 2
        perp = (-gx * u[1] + gy * u[0]) ** 2
        return float(perp.sum() / (par.sum() + _EPS))
    gh, gw = flow.shape[:2]
    e_par = e_perp = 0.0
    for i in range(gh):
        for j in range(gw):
            f = flow[i, j]
            n = float(np.linalg.norm(f))
            if not np.isfinite(n) or n < min_flow:
                continue
            u = f / n
            ys, ye = i * H // gh, (i + 1) * H // gh
            xs, xe = j * W // gw, (j + 1) * W // gw
            cx, cy = gx[ys:ye, xs:xe], gy[ys:ye, xs:xe]
            e_par += float(((cx * u[0] + cy * u[1]) ** 2).sum())
            e_perp += float(((-cx * u[1] + cy * u[0]) ** 2).sum())
    return float(e_perp / e_par) if e_par > 0 else float("nan")


def noise_residual(g: np.ndarray, sigma: float = 1.0, flat_q: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    """Residual of a Gaussian denoiser and a mask of flat, unsaturated pixels where it is mostly noise."""
    r = g - _blur(g, sigma)
    m1 = _blur(g, 3.0)
    local_std = np.sqrt(np.maximum(_blur(g * g, 3.0) - m1 * m1, 0.0))
    thr = np.percentile(local_std, flat_q)
    mask = (local_std <= thr) & (m1 > 0.06) & (m1 < 0.94)
    mask[:8], mask[-8:], mask[:, :8], mask[:, -8:] = False, False, False, False
    return r, mask


def nlf_slope(g: np.ndarray, r: np.ndarray, mask: np.ndarray, bins: int = 8) -> float:
    """Scale-free slope of residual variance vs. mean intensity (noise-level function)."""
    I, rv = _blur(g, 3.0)[mask], r[mask]
    if I.size < 200:
        return float("nan")
    edges = np.quantile(I, np.linspace(0, 1, bins + 1))
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (I >= a) & (I <= b)
        if sel.sum() >= 30:
            xs.append(float(I[sel].mean()))
            ys.append(float(rv[sel].var()))
    if len(xs) < 3 or np.ptp(xs) < 1e-3:
        return float("nan")
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope / (np.mean(ys) + _EPS))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 6 or np.ptp(a[ok]) < _EPS or np.ptp(b[ok]) < _EPS:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


def clip_features(frames: np.ndarray, m: np.ndarray, direction: np.ndarray | None = None,
                  flow_grid: np.ndarray | None = None, lag: int | None = None, min_shift_px: float = 5.0) -> dict:
    """CPS features of one clip. frames: uint8 [T,H,W,3]; m: per-frame image motion (px/frame) [T];
    direction: [T,2] unit flow directions; flow_grid: [T,gh,gw,2] per-cell flow (optional)."""
    g = to_gray(frames)
    T = len(g)
    m = np.asarray(m, float)
    s = np.array([sharpness(x) for x in g])
    d = np.array([directional_ratio(g[t], None if flow_grid is None else flow_grid[t],
                                    None if direction is None else direction[t]) for t in range(T)])
    res, masks = zip(*(noise_residual(x) for x in g))
    nlf = np.array([nlf_slope(g[t], res[t], masks[t]) for t in range(T)])
    lvl = np.array([float(res[t][masks[t]].std()) if masks[t].sum() > 100 else np.nan for t in range(T)])

    lag = lag or max(2, T // 4)
    cum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(m[:-1]))])
    fpn = []
    for t in range(0, T - lag, 2):
        if cum[t + lag] - cum[t] < min_shift_px:   # content barely moved: correlation would be texture, not sensor
            continue
        cm = masks[t] & masks[t + lag]
        if cm.sum() > 500:
            a, b = res[t][cm], res[t + lag][cm]
            if a.std() > _EPS and b.std() > _EPS:
                fpn.append(float(np.corrcoef(a, b)[0, 1]))

    fast, slow = m >= np.nanquantile(m, 0.75), m <= np.nanquantile(m, 0.25)
    sr = (np.log(np.median(s[fast]) + _EPS) - np.log(np.median(s[slow]) + _EPS)) if fast.any() and slow.any() else np.nan
    nlf_ok = nlf[np.isfinite(nlf)]
    lvl_ok = lvl[np.isfinite(lvl)]
    return {
        "rho_blur": _spearman(-s, m),
        "rho_dir": _spearman(d, m),
        "sharp_ratio": float(sr),
        "nlf_slope": float(np.median(nlf_ok)) if nlf_ok.size else float("nan"),
        "nlf_slope_cv": float(nlf_ok.std() / (abs(nlf_ok.mean()) + _EPS)) if nlf_ok.size > 2 else float("nan"),
        "fpn_corr": float(np.mean(fpn)) if fpn else float("nan"),
        "resid_level": float(np.median(lvl_ok)) if lvl_ok.size else float("nan"),
        "resid_cv": float(lvl_ok.std() / (lvl_ok.mean() + _EPS)) if lvl_ok.size > 2 else float("nan"),
        "motion_med": float(np.nanmedian(m)),
        "n_fpn_pairs": len(fpn),
    }


@dataclass
class OneClassCPS:
    """Robust-standardised Ledoit-Wolf Mahalanobis distance to real camera footage (higher = less camera-like)."""

    features: tuple = tuple(FEATURES)
    center: np.ndarray | None = None
    scale: np.ndarray | None = None
    precision: np.ndarray | None = None
    fill: np.ndarray | None = None

    def _matrix(self, rows: list[dict]) -> np.ndarray:
        X = np.array([[r.get(f, np.nan) for f in self.features] for r in rows], dtype=float)
        if self.fill is not None:
            X = np.where(np.isfinite(X), X, self.fill)
        return X

    def fit(self, real_rows: list[dict]) -> "OneClassCPS":
        from sklearn.covariance import LedoitWolf
        X = np.array([[r.get(f, np.nan) for f in self.features] for r in real_rows], dtype=float)
        self.fill = np.nanmedian(X, axis=0)
        self.fill = np.where(np.isfinite(self.fill), self.fill, 0.0)
        X = np.where(np.isfinite(X), X, self.fill)
        self.center = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.center), axis=0) * 1.4826
        self.scale = np.where(mad > _EPS, mad, X.std(axis=0) + _EPS)
        Z = (X - self.center) / self.scale
        self.precision = LedoitWolf().fit(Z).precision_
        return self

    def score(self, rows: list[dict]) -> np.ndarray:
        Z = (self._matrix(rows) - self.center) / self.scale
        return np.sqrt(np.einsum("ij,jk,ik->i", Z, self.precision, Z))

    def to_json(self) -> str:
        return json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else list(v) if isinstance(v, tuple) else v)
                           for k, v in self.__dict__.items()})

    @classmethod
    def from_json(cls, s: str) -> "OneClassCPS":
        d = json.loads(s)
        return cls(features=tuple(d["features"]), **{k: np.array(v) for k, v in d.items() if k != "features"})
