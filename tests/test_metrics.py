import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from c44 import metrics


def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500).astype(bool)
    s = np.round(rng.standard_normal(500) + y * 0.8, 1)
    assert abs(metrics.roc_auc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_tpr_at_fpr_extremes():
    y = np.array([0] * 100 + [1] * 100, bool)
    s = np.concatenate([np.zeros(100), np.ones(100)])
    assert metrics.tpr_at_fpr(y, s, 0.01) == 1.0
    assert metrics.tpr_at_fpr(y, -s, 0.01) == 0.0


def test_cluster_bootstrap_and_compare():
    rng = np.random.default_rng(1)
    rows = []
    for sc in range(30):
        rows.append({"cell": "RC", "scene": f"s{sc}", "score": rng.normal(0, 1)})
        rows.append({"cell": "GR", "scene": f"s{sc}", "score": rng.normal(1.5, 1)})
        rows.append({"cell": "RR", "scene": f"s{sc}", "score": rng.normal(0, 1)})
    df = pd.DataFrame(rows)
    out = metrics.compare_cells(df, "score", pos_cells=("RR", "GR"), n_boot=300)
    gr = out[out.pair == "RC vs GR"].iloc[0]
    rr = out[out.pair == "RC vs RR"].iloc[0]
    assert gr.auc_lo <= gr.auc <= gr.auc_hi and gr.auc > 0.75
    assert rr.auc_lo < 0.5 < rr.auc_hi
