"""Detection metrics with scene-level cluster-bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC via the Mann-Whitney U statistic (ties count one half)."""
    from scipy.stats import rankdata
    y, s = np.asarray(y, bool), np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    y = np.asarray(y, bool)
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float = 0.01) -> float:
    """Fraction of positives scoring above the (1-fpr) quantile of negatives."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    thr = np.quantile(s[~y], 1 - fpr, method="higher")
    return float((s[y] > thr).mean())


def cluster_bootstrap(y, s, groups, fn=roc_auc, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Point estimate and (1-alpha) percentile CI, resampling whole groups (scenes) with replacement."""
    y, s, groups = np.asarray(y), np.asarray(s, float), np.asarray(groups)
    est = fn(y, s)
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        v = fn(y[pick], s[pick])
        if np.isfinite(v):
            stats.append(v)
    if not stats:
        return est, float("nan"), float("nan")
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


def compare_cells(df, score_col: str, neg_cell: str = "RC", pos_cells=("RR", "GR", "GV"),
                  group_col: str = "scene", n_boot: int = 1000, seed: int = 0):
    """One row per positive cell: AUC [95% CI], AP, TPR@1%FPR of `score_col` (higher = more fake) vs. neg_cell."""
    import pandas as pd
    rows = []
    neg = df[df["cell"] == neg_cell]
    for pc in pos_cells:
        pos = df[df["cell"] == pc]
        if pos.empty or neg.empty:
            continue
        sub = pd.concat([neg, pos])
        sub = sub[np.isfinite(sub[score_col].astype(float))]
        y = (sub["cell"] == pc).to_numpy()
        s = sub[score_col].to_numpy(float)
        auc, lo, hi = cluster_bootstrap(y, s, sub[group_col].to_numpy(), roc_auc, n_boot, seed)
        rows.append({"score": score_col, "pair": f"{neg_cell} vs {pc}", "n_neg": int((~y).sum()), "n_pos": int(y.sum()),
                     "auc": auc, "auc_lo": lo, "auc_hi": hi, "ap": average_precision(y, s),
                     "tpr@1%fpr": tpr_at_fpr(y, s, 0.01)})
    return pd.DataFrame(rows)
