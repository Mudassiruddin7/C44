"""Final analysis tables, CPU only, from results/scores.csv + results/cps_features.csv (+ the manifest).

  1. detectors   : AUC [95% scene-bootstrap CI] of UnivFD / NPR / consistency / one-class CPS, real vs each fake cell,
                   clean, recompressed and under the camera-aware attacks.
  2. features    : AUC of each single physics feature.
  3. supervised  : an 8-feature physics classifier, scene-grouped cross-validation (never sees the test scenes), incl.
                   cross-cell transfer, compression and attacks, with and without attack-augmented training.
  4. attack_calibration : how well the calibrated-noise attack matched the paired real clip's noise level.
  5. control     : GR/RR vs their supersampled twins GRs/RRs (does the signal survive matching the real clips' resampling?).
Everything lands in results/analysis/*.csv and summary.md; each section is independent and never aborts the run.
"""

from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

from . import config, pipeline
from .cps import FEATURES

DETECTORS = ["univfd", "npr", "consistency", "cps"]
ID_RE = r"^(RC|RRs|RR|GRs|GR)_([0-9a-f]+)(?:@(.+))?$"


def _roc_auc(y, score) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def _parse(df: pd.DataFrame) -> pd.DataFrame:
    m = df.clip_id.str.extract(ID_RE)
    df = df.assign(base=m[0], scene=m[1], var=m[2].fillna(""))
    df = df[df.base.notna()]
    df = df[(df["var"] == "") | df["var"].isin(pipeline.VARIANTS)]          # drop retired attack variants
    return df.assign(cell=df.base + np.where(df["var"] != "", "@" + df["var"], ""))


def auc_ci(neg: pd.DataFrame, pos: pd.DataFrame, col: str, n_boot: int = 300, seed: int = 0):
    """AUC (higher `col` = more fake) with a scene-level bootstrap 95% CI."""
    d = pd.concat([neg.assign(y=0), pos.assign(y=1)])
    d = d[np.isfinite(d[col])]
    if d.y.nunique() < 2:
        return float("nan"), float("nan"), float("nan")
    a = _roc_auc(d.y, d[col])
    rng = np.random.default_rng(seed)
    groups = {k: v for k, v in d.groupby("scene")}
    keys = np.array(list(groups))
    bs = []
    for _ in range(n_boot):
        b = pd.concat([groups[k] for k in rng.choice(keys, len(keys))])
        if b.y.nunique() == 2:
            bs.append(_roc_auc(b.y, b[col]))
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (float("nan"), float("nan"))
    return a, float(lo), float(hi)


def _fmt(t) -> str:
    return "n/a" if not np.isfinite(t[0]) else f"{t[0]:.3f} [{t[1]:.2f},{t[2]:.2f}]"


# ----------------------------------------------------------------------------- 1. detectors
def detectors_table(sw: pd.DataFrame) -> pd.DataFrame:
    cells = set(sw.cell)
    vars_ = sorted({c.split("@")[1] for c in cells if "@" in c})
    pairs = [("RC", "GR", "clean"), ("RC", "RR", "clean")]
    for k in vars_:
        if k.startswith("crf"):
            pairs += [(f"RC@{k}", f"GR@{k}", k), (f"RC@{k}", f"RR@{k}", k)]
        else:
            pairs += [("RC", f"GR@{k}", k), ("RC", f"RR@{k}", k)]
    rows = []
    for neg, pos, label in pairs:
        if neg not in cells or pos not in cells:
            continue
        P = sw[sw.cell == pos]
        N = sw[sw.cell == neg]
        if "@" in pos and not pos.split("@")[1].startswith("crf"):          # attacks cover a scene subset:
            N = N[N.scene.isin(set(P.scene))]                              # compare on the same scenes
        row = {"pair": f"{neg} vs {pos}", "setting": label, "n_neg": len(N), "n_pos": len(P)}
        for d in DETECTORS:
            if d in sw:
                row[d] = _fmt(auc_ci(N, P, d))
        rows.append(row)
        if "@" in pos and not pos.split("@")[1].startswith("crf"):          # clean baseline on that subset
            Pc = sw[(sw.cell == pos.split("@")[0]) & sw.scene.isin(set(P.scene))]
            r2 = {"pair": f"{neg} vs {pos.split('@')[0]} (same scenes)", "setting": "clean-subset", "n_neg": len(N),
                  "n_pos": len(Pc)}
            for d in DETECTORS:
                if d in sw:
                    r2[d] = _fmt(auc_ci(N, Pc, d))
            rows.append(r2)
    return pd.DataFrame(rows).drop_duplicates(subset="pair")


# ----------------------------------------------------------------------------- 2. features
def features_table(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in sorted(set(f.cell)):
        base, _, var = c.partition("@")
        if base == "RC":
            continue
        neg = f[f.cell == ("RC@" + var if var.startswith("crf") else "RC")]
        pos = f[f.cell == c]
        if (var and not var.startswith("crf")) or base in ("GRs", "RRs"):      # subsets: compare on the same scenes
            neg = neg[neg.scene.isin(set(pos.scene))]
        d = {"cell": c, "n": len(pos)}
        for ft in FEATURES + ["motion_med"]:
            x = pd.concat([neg.assign(y=0), pos.assign(y=1)])
            x = x[np.isfinite(x[ft])]
            d[ft] = round(_roc_auc(x.y, x[ft]), 3) if x.y.nunique() == 2 else float("nan")
        rows.append(d)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 3. supervised physics score
def _model(kind: str):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    imp = SimpleImputer(strategy="median")
    if kind == "lr":
        return make_pipeline(imp, StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))
    return make_pipeline(imp, HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05))


def oof_scores(F: pd.DataFrame, neg_cells, pos_cells, kind: str, n_folds: int = 5, seed: int = 0) -> pd.Series:
    """Train real-vs-fake on some cells of the other folds' scenes, score EVERY clip of the held-out scenes."""
    scenes = np.array(sorted(F.scene.unique()))
    n_folds = max(2, min(n_folds, len(scenes)))
    order = np.random.default_rng(seed).permutation(len(scenes))
    fold = dict(zip(scenes[order], np.arange(len(scenes)) % n_folds))
    k_of = F.scene.map(fold)
    out = pd.Series(np.nan, index=F.index)
    for k in range(n_folds):
        tr = F[(k_of != k) & F.cell.isin(list(neg_cells) + list(pos_cells))]
        if tr.cell.isin(neg_cells).sum() < 5 or tr.cell.isin(pos_cells).sum() < 5:
            continue
        m = _model(kind).fit(tr[FEATURES], tr.cell.isin(pos_cells).astype(int))
        te = F[k_of == k]
        out[te.index] = m.predict_proba(te[FEATURES])[:, 1]
    return out


def supervised_table(f: pd.DataFrame) -> pd.DataFrame:
    F = f.reset_index(drop=True)
    cells = set(F.cell)
    atk = [c for c in cells if "@" in c and not c.split("@")[1].startswith("crf") and c.split("@")[0] in ("GR", "RR")]
    rows = []

    def ev(oof, neg, pos, setting, kind, same_scenes=False):
        if neg not in cells or pos not in cells:
            return
        a = F.assign(score=oof)
        N, P = a[a.cell == neg], a[a.cell == pos]
        if same_scenes:
            N = N[N.scene.isin(set(P.scene))]
        rows.append({"model": kind, "setting": setting, "pair": f"{neg} vs {pos}", "auc": _fmt(auc_ci(N, P, "score"))})

    for kind in ("lr", "gb"):
        o = oof_scores(F, ["RC"], ["GR", "RR"], kind)
        ev(o, "RC", "GR", "train clean (GR+RR)", kind); ev(o, "RC", "RR", "train clean (GR+RR)", kind)
        for k in ("crf28", "crf35"):
            ev(o, f"RC@{k}", f"GR@{k}", "clean-trained, recompressed", kind)
            ev(o, f"RC@{k}", f"RR@{k}", "clean-trained, recompressed", kind)
        for c in sorted(atk):
            ev(o, "RC", c, "clean-trained, attacked", kind, same_scenes=True)
        og = oof_scores(F, ["RC"], ["GR"], kind)
        ev(og, "RC", "RR", "transfer: train GR -> test RR", kind)
        orr = oof_scores(F, ["RC"], ["RR"], kind)
        ev(orr, "RC", "GR", "transfer: train RR -> test GR", kind)
        if atk:                                                     # defender who has seen the calibrated attacks
            oa = oof_scores(F, ["RC"], ["GR", "RR"] + sorted(atk), kind)
            for c in sorted(atk):
                ev(oa, "RC", c, "attack-augmented training, attacked", kind, same_scenes=True)
            ev(oa, "RC", "GR", "attack-augmented training, clean", kind); ev(oa, "RC", "RR", "attack-augmented training, clean", kind)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 4. attack calibration
def attack_calibration(f: pd.DataFrame) -> pd.DataFrame:
    rc = f[f.cell == "RC"].set_index("scene")
    rows = []
    for c in sorted(set(f.cell)):
        if "@" in c and "noiseadapt" in c:
            a = f[f.cell == c]
            ref = rc.reindex(a.scene)
            ratio = (a.resid_level.to_numpy() / ref.resid_level.to_numpy())
            ratio = ratio[np.isfinite(ratio)]
            if len(ratio):
                rows.append({"cell": c, "n": len(ratio), "resid_level ratio median": round(float(np.median(ratio)), 3),
                             "q25": round(float(np.percentile(ratio, 25)), 3), "q75": round(float(np.percentile(ratio, 75)), 3)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 5. resolution control
def control_table(sw: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    ctl = {c for c in set(sw.cell) if c in ("GRs", "RRs")}
    if not ctl:
        return pd.DataFrame()
    scenes = set(sw[sw.cell.isin(ctl)].scene)
    F = f.reset_index(drop=True)
    rows = []
    sup = oof_scores(F, ["RC"], ["GR", "RR"], "lr")
    Fa = F.assign(score=sup)
    for name, table, col in [(d, sw, d) for d in DETECTORS if d in sw] + [("supervised LR", Fa, "score")] + \
                            [(ft, F, ft) for ft in ("sharp_ratio", "rho_blur", "resid_level", "fpn_corr")]:
        row = {"measure": name}
        N = table[(table.cell == "RC") & table.scene.isin(scenes)]
        for c in ("GR", "GRs", "RR", "RRs"):
            P = table[(table.cell == c) & table.scene.isin(scenes)]
            row[c] = _fmt(auc_ci(N, P, col)) if len(P) else "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- driver
def run(store=None) -> str:
    res = pipeline.results_dir()
    out = res / "analysis"
    out.mkdir(exist_ok=True)
    s = _parse(pd.read_csv(res / "scores.csv"))
    f = _parse(pd.read_csv(res / "cps_features.csv"))
    sw = s.pivot_table(index=["clip_id", "cell", "scene"], columns="detector", values="score").reset_index()
    text, files = ["# C44 analysis\n"], []
    sections = [
        ("1. detectors (AUC, higher score = fake)", "detectors", lambda: detectors_table(sw)),
        ("2. single physics features (AUC fake vs real)", "features", lambda: features_table(f)),
        ("3. supervised physics score (scene-grouped CV)", "supervised", lambda: supervised_table(f)),
        ("4. attack calibration (attacked / paired real noise level)", "attack_calibration", lambda: attack_calibration(f)),
        ("5. resolution control (same scenes; GRs/RRs = supersampled renders)", "control", lambda: control_table(sw, f)),
    ]
    for title, name, fn in sections:
        try:
            df = fn()
        except Exception as e:                                          # noqa: BLE001 - never abort the run
            df = pd.DataFrame()
            text.append(f"## {title}\n\nSKIPPED: {e!r}\n```\n{traceback.format_exc()[-600:]}\n```\n")
            continue
        df.to_csv(out / f"{name}.csv", index=False)
        files.append(out / f"{name}.csv")
        text.append(f"## {title}\n\n```\n{df.to_string(index=False) if len(df) else '(nothing to report)'}\n```\n")
    summary = "\n".join(text)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    files.append(out / "summary.md")
    if store is not None:
        for p in files:
            store.stage(p)
        store.flush("c44: analysis")
    return summary
