"""Scene-cluster bootstrap statistics (BCa intervals, paired AUC differences, Wilson, Holm) - the same code, and the same unit tests,
as improve_numbers.py in the project root (that script is the stand-alone version; this copy lets the molab notebooks, which only
receive the c44 package, use it).

AUC is a U-statistic over (fake, real) pairs, so a scene-cluster bootstrap resample is fully described by how many times each scene was
drawn: AUC* = w_pos' M w_neg / (sum w_pos * sum w_neg). 10,000 resamples are two matrix products, and every statistic computed with the
same SceneBoot shares its resamples, which makes differences between detectors exactly paired.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

SQ2 = math.sqrt(2.0)
EQUIV = 0.03

# =============================================================================================================== 1. statistics
def wilson(k: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (k may be fractional: ties count one half)."""
    if n <= 0:
        return float("nan"), float("nan")
    z = float(sps.norm.ppf(1 - (1 - level) / 2))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def holm(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaN stays NaN)."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m, run = len(order), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def mcnemar_exact(flag_a, flag_b) -> tuple[int, int, float]:
    """Exact McNemar test on paired binary outcomes. Returns (a-only, b-only, two-sided p)."""
    a, b = np.asarray(flag_a, bool), np.asarray(flag_b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = 1.0 if n10 + n01 == 0 else float(sps.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)
    return n10, n01, p


def hm_se(A: float, n1: int, n0: int) -> float:
    """Hanley-McNeil (1982) standard error of an AUC (independent samples, a mild under-estimate for paired designs)."""
    A = min(max(A, 1e-6), 1 - 1e-6)
    q1, q2 = A / (2 - A), 2 * A * A / (1 + A)
    var = (A * (1 - A) + (n1 - 1) * (q1 - A * A) + (n0 - 1) * (q2 - A * A)) / (n1 * n0)
    return math.sqrt(max(var, 0.0))


def n_for_halfwidth(A: float, h: float, level: float = 0.95) -> int:
    """Smallest per-class n with a Hanley-McNeil half-width <= h."""
    z = float(sps.norm.ppf(1 - (1 - level) / 2))
    lo, hi = 5, 200000
    while lo < hi:
        mid = (lo + hi) // 2
        if z * hm_se(A, mid, mid) <= h:
            hi = mid
        else:
            lo = mid + 1
    return lo


def n_for_proportion(p: float, h: float, level: float = 0.95) -> int:
    z = float(sps.norm.ppf(1 - (1 - level) / 2))
    return int(math.ceil(z * z * p * (1 - p) / (h * h)))


def dprime(auc):
    """Equal-variance binormal separation implied by an AUC: AUC = Phi(d'/sqrt2)."""
    return SQ2 * sps.norm.ppf(np.clip(auc, 0.005, 0.995))


@dataclass
class Stat:
    """A statistic with its scene-bootstrap replicates (boot) and leave-one-scene-out values (jack): enough for BCa intervals,
    bootstrap p-values, and for arithmetic between statistics that share the same resamples (paired differences)."""
    point: float
    boot: np.ndarray
    jack: np.ndarray

    def __sub__(self, o):
        return Stat(self.point - o.point, self.boot - o.boot, self.jack - o.jack) if isinstance(o, Stat) \
            else Stat(self.point - o, self.boot - o, self.jack - o)

    def __add__(self, o):
        return Stat(self.point + o.point, self.boot + o.boot, self.jack + o.jack) if isinstance(o, Stat) \
            else Stat(self.point + o, self.boot + o, self.jack + o)

    def __mul__(self, k):
        return Stat(self.point * k, self.boot * k, self.jack * k)

    __rmul__ = __mul__

    def __truediv__(self, k):
        return Stat(self.point / k, self.boot / k, self.jack / k)

    def apply(self, f) -> "Stat":
        return Stat(float(f(self.point)), f(self.boot), f(self.jack))

    def _z0a(self) -> tuple[float, float]:
        b = self.boot[np.isfinite(self.boot)]
        B = len(b)
        below = (b < self.point).mean() + 0.5 * (b == self.point).mean()
        z0 = float(sps.norm.ppf(min(max(below, 1 / (B + 1)), B / (B + 1))))
        j = self.jack[np.isfinite(self.jack)]
        d = j.mean() - j
        s2 = float((d ** 2).sum())
        a = float((d ** 3).sum() / (6 * s2 ** 1.5)) if s2 > 0 else 0.0
        return z0, a

    def ci(self, level: float = 0.95, method: str = "bca") -> tuple[float, float]:
        b = self.boot[np.isfinite(self.boot)]
        if len(b) < 50 or not np.isfinite(self.point):
            return float("nan"), float("nan")
        alpha = 1 - level
        qs = [alpha / 2, 1 - alpha / 2]
        if method == "bca":
            z0, a = self._z0a()
            adj = []
            for q in qs:
                z = float(sps.norm.ppf(q))
                t, den = z0 + z, 1 - a * (z0 + z)
                adj.append(float(sps.norm.cdf(z0 + t / den)) if den > 1e-9 else q)
            qs = adj
        lo, hi = np.quantile(b, qs)
        return float(lo), float(hi)

    def p(self, null: float = 0.0) -> float:
        """Two-sided bootstrap p-value by inverting the BCa interval (reduces to the percentile p when z0 = a = 0)."""
        b = self.boot[np.isfinite(self.boot)]
        B = len(b)
        if B < 50 or not np.isfinite(self.point):
            return float("nan")
        z0, a = self._z0a()
        u = ((b < null).sum() + 0.5 * (b == null).sum()) / B
        u = min(max(u, 0.5 / B), 1 - 0.5 / B)
        t = float(sps.norm.ppf(u)) - z0
        den = 1 + a * t
        q = float(sps.norm.cdf((t - z0 - a * t * z0) / den)) if den > 1e-9 else u
        return float(min(1.0, 2 * min(q, 1 - q)))


class SceneBoot:
    """Scene-cluster bootstrap for AUC. All clips of a scene are resampled together. Because AUC is a U-statistic over (fake, real)
    pairs, a resample is fully described by how many times each scene was drawn: AUC* = w_pos' M w_neg / (sum w_pos * sum w_neg),
    with M[i, j] = 1[s_pos_i > s_neg_j] + 0.5 * 1[=]. That makes 10,000+ resamples a pair of matrix products, and every statistic
    that uses the same SceneBoot shares its resamples, so paired differences are exact."""

    def __init__(self, scenes, B: int, seed: int):
        self.scenes = sorted(set(scenes))
        self.S, self.B = len(self.scenes), B
        self.ix = {s: i for i, s in enumerate(self.scenes)}
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(self.S, np.full(self.S, 1.0 / self.S), size=B).astype(np.float64)
        self.J = 1.0 - np.eye(self.S)                                             # leave-one-scene-out weights

    def auc(self, neg_sc, neg_s, pos_sc, pos_s, ok_neg=None, ok_pos=None) -> Stat:
        neg_s, pos_s = np.asarray(neg_s, float), np.asarray(pos_s, float)
        okn = np.isfinite(neg_s) if ok_neg is None else (np.asarray(ok_neg, bool) & np.isfinite(neg_s))
        okp = np.isfinite(pos_s) if ok_pos is None else (np.asarray(ok_pos, bool) & np.isfinite(pos_s))
        if okn.sum() == 0 or okp.sum() == 0:
            return Stat(float("nan"), np.full(self.B, np.nan), np.full(self.S, np.nan))
        d = pos_s[:, None] - neg_s[None, :]
        M = (d > 0).astype(np.float64) + 0.5 * (d == 0)
        wp, wn = okp.astype(np.float64), okn.astype(np.float64)
        ip = np.array([self.ix[s] for s in pos_sc])
        in_ = np.array([self.ix[s] for s in neg_sc])
        point = float(wp @ M @ wn / (wp.sum() * wn.sum()))
        with np.errstate(divide="ignore", invalid="ignore"):
            Wp, Wn = self.W[:, ip] * wp, self.W[:, in_] * wn
            boot = ((Wp @ M) * Wn).sum(1) / (Wp.sum(1) * Wn.sum(1))
            Jp, Jn = self.J[:, ip] * wp, self.J[:, in_] * wn
            jack = ((Jp @ M) * Jn).sum(1) / (Jp.sum(1) * Jn.sum(1))
        return Stat(point, boot, jack)


_BOOTS: dict = {}


def get_boot(scenes, B: int, seed: int) -> SceneBoot:
    """Identical scene sets share identical resamples (deterministic in the seed), which is what makes paired comparisons valid."""
    key = (tuple(sorted(set(scenes))), B, seed)
    if key not in _BOOTS:
        if len(_BOOTS) > 24:
            _BOOTS.clear()
        _BOOTS[key] = SceneBoot(list(key[0]), B, (seed + zlib.crc32("|".join(key[0]).encode())) % (2 ** 31))
    return _BOOTS[key]


class Contrast:
    """Real clips N vs fake clips P (DataFrames whose index addresses full-length score arrays)."""

    def __init__(self, N: pd.DataFrame, P: pd.DataFrame, B: int, seed: int):
        self.ni, self.pi = N.index.to_numpy(), P.index.to_numpy()
        self.nsc, self.psc = N.scene.to_numpy(), P.scene.to_numpy()
        self.n_neg, self.n_pos = len(N), len(P)
        self.n_scenes = len(set(self.nsc) | set(self.psc))
        self.boot = get_boot(set(self.nsc) | set(self.psc), B, seed)

    def auc(self, s, ok_neg=None, ok_pos=None) -> Stat:
        s = np.asarray(s, float)
        return self.boot.auc(self.nsc, s[self.ni], self.psc, s[self.pi], ok_neg, ok_pos)

    def pair(self, a, b) -> tuple[Stat, Stat]:
        """AUCs of two detectors on exactly the clips both scored (so their difference is a paired statistic)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        okn = np.isfinite(a[self.ni]) & np.isfinite(b[self.ni])
        okp = np.isfinite(a[self.pi]) & np.isfinite(b[self.pi])
        return self.auc(a, okn, okp), self.auc(b, okn, okp)

    def valid(self, s) -> tuple[int, int]:
        s = np.asarray(s, float)
        return int(np.isfinite(s[self.ni]).sum()), int(np.isfinite(s[self.pi]).sum())


def fmt_stat(st: Stat, nd: int = 3, method: str = "bca") -> str:
    if not np.isfinite(st.point):
        return "n/a"
    lo, hi = st.ci(method=method)
    return f"{st.point:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def verdict_of(d: Stat, p_adj: float, delta: float = EQUIV) -> str:
    """Decision for a paired difference: significant after Holm / equivalent within +/-delta (90% CI) / inconclusive."""
    lo90, hi90 = d.ci(0.90)
    if np.isfinite(p_adj) and p_adj < 0.05:
        return "A better" if d.point > 0 else "B better"
    if np.isfinite(lo90) and lo90 > -delta and hi90 < delta:
        return f"equivalent (+/-{delta})"
    return "inconclusive"


