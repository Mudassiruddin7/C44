"""Final analysis tables, CPU only, from results/scores.csv (generic detectors, CPS-1) and results/physics_features.csv (CPS-2).

Protocol. Scenes are split once by a hash into dev (70%) and test (30%) (`is_test`); nothing was designed on the test scenes.
Every CPS-2 model is trained on dev scenes only and evaluated on the test scenes; confidence intervals are scene-level
bootstrap. Generic detectors (UnivFD, NPR, consistency) need no training and are scored on the same test scenes.

  cps2_main     test scenes: every detector vs. real, in every setting (clean, recompression, attacks, resolution control).
                'aug' is trained with recompressed and attacked fakes; for an attack it is trained on all the OTHER attack kinds
  cps2_families which cue family carries the signal (only / minus), on test scenes
  cps2_transfer train on one fake type, test on the other
  cps2_features single-feature AUC on dev scenes
  attack_calibration  did the attacks reproduce the real clip's noise level / exposure jitter?
  detectors     the generic detectors and the one-class CPS-1, all scenes, clean / recompressed / attacked
Each section is independent and never aborts the run.
"""

from __future__ import annotations

import traceback
import zlib

import numpy as np
import pandas as pd

from . import pipeline, physics

DETECTORS = ["univfd", "npr", "consistency", "cps"]          # 'cps' in scores.csv is the one-class CPS-1 (VGGT-based)
ID_RE = r"^(RC|RRs|RR|GRs|GR)_([0-9a-f]+)(?:@(.+))?$"
FAMILIES = {
    "AE/WB dynamics": ["gain_mad", "gain_range", "wb_mad"],
    "noise structure": ["tn_ratio", "tn_corr", "res_ac1", "chroma_ratio", "clean_patch_frac"],
    "spectrum/edges": ["psd_slope", "psd_hf", "acutance", "flat_frac"],
    "blur-motion coupling": ["rho_blur", "rho_dir", "sharp_ratio", "sharp_resid_cv", "sharp_pump"],
    "noise level": ["nlf_slope", "nlf_slope_cv", "fpn_corr", "resid_level", "resid_cv"],
    "vignette": ["vignette"],
}


def is_test(scene: str) -> bool:
    """Fixed 30% of scenes by hash; used only for the final evaluation."""
    return zlib.crc32(scene.encode()) % 10 < 3


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


def _is_attack(var: str) -> bool:
    return bool(var) and not var.startswith("crf")


# ----------------------------------------------------------------------------- generic detectors (all scenes)
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
        P, N = sw[sw.cell == pos], sw[sw.cell == neg]
        if _is_attack(pos.partition("@")[2]):
            N = N[N.scene.isin(set(P.scene))]                              # attacks cover a scene subset
        row = {"pair": f"{neg} vs {pos}", "setting": label, "n_neg": len(N), "n_pos": len(P)}
        for d in DETECTORS:
            if d in sw:
                row[d] = _fmt(auc_ci(N, P, d))
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- CPS-2
def _train_set(dev: pd.DataFrame, compress: bool, attack_kinds) -> tuple[pd.DataFrame, np.ndarray]:
    kinds = list(attack_kinds)
    keep = dev.base.isin(["RC", "GR", "RR"]) & (
        (dev["var"] == "") | (compress & dev["var"].str.startswith("crf")) | dev["var"].isin(kinds))
    d = dev[keep & ~((dev.base == "RC") & dev["var"].isin(kinds))]       # attacks exist for fakes only
    return d, (d.base != "RC").to_numpy().astype(int)


def physics_tables(pf: pd.DataFrame, sw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pf = pf.assign(test=pf.scene.map(is_test))
    dev, test = pf[~pf.test], pf[pf.test]
    attacks = sorted({v for v in pf["var"].unique() if _is_attack(v)})

    m_oc = physics.PhysicsOneClass().fit(dev[dev.cell == "RC"])
    d0, y0 = _train_set(dev, compress=False, attack_kinds=[])
    m_clean = physics.PhysicsClassifier().fit(d0, y0)
    aug = {}

    def aug_model(exclude):
        if exclude not in aug:
            d, y = _train_set(dev, compress=True, attack_kinds=[a for a in attacks if a != exclude])
            aug[exclude] = physics.PhysicsClassifier().fit(d, y)
        return aug[exclude]

    pairs = [("RC", "GR", "clean", ""), ("RC", "RR", "clean", "")]
    for k in sorted({v for v in pf["var"].unique() if v.startswith("crf")}):
        pairs += [(f"RC@{k}", f"GR@{k}", k, k), (f"RC@{k}", f"RR@{k}", k, k)]
    for k in attacks:
        pairs += [("RC", f"GR@{k}", k, k), ("RC", f"RR@{k}", k, k)]
    for c in ("GRs", "RRs"):
        if c in set(pf.cell):
            pairs += [("RC", c[:2], f"same scenes as {c}", ""), ("RC", c, f"supersampled {c}", "")]

    base_scores = sw.set_index("clip_id")
    rows = []
    for neg, pos, label, kind in pairs:
        P, N = test[test.cell == pos], test[test.cell == neg]
        if label.startswith(("same scenes", "supersampled")):
            ctl_scenes = set(test[test.cell == pos[:2] + "s"].scene) if label.startswith("same") else set(P.scene)
            P, N = P[P.scene.isin(ctl_scenes)], N[N.scene.isin(ctl_scenes)]
        elif _is_attack(kind):
            N = N[N.scene.isin(set(P.scene))]
        if len(P) < 5 or len(N) < 5:
            continue
        row = {"pair": f"{neg} vs {pos}", "setting": label, "n_test_pos": len(P)}
        for d in DETECTORS:
            if d in base_scores:
                a = N.assign(s=base_scores[d].reindex(N.clip_id).to_numpy())
                b = P.assign(s=base_scores[d].reindex(P.clip_id).to_numpy())
                row[{"cps": "CPS-1 (one-class)"}.get(d, d)] = _fmt(auc_ci(a, b, "s"))
        models = [("CPS-2 one-class", m_oc), ("CPS-2 supervised", m_clean),
                  ("CPS-2 + aug (unseen attack)" if _is_attack(kind) else "CPS-2 + aug", aug_model(kind if _is_attack(kind) else None))]
        if _is_attack(kind):
            models.append(("CPS-2 + aug (attack seen)", aug_model(None)))      # a defender who trained on this attack kind
        for name, model in models:
            a, b = N.assign(s=model.score(N)), P.assign(s=model.score(P))
            row[name] = _fmt(auc_ci(a, b, "s"))
        rows.append(row)
    main = pd.DataFrame(rows)

    # which cue family carries the signal (clean-trained, test scenes)
    fam_rows = []

    def fam_row(name, feats):
        m = physics.PhysicsClassifier(features=feats).fit(d0, y0)
        r = {"features": name, "n": len(feats)}
        for c in ("GR", "RR", "GRs", "RRs"):
            P = test[test.cell == c]
            if len(P) >= 5:
                N = test[(test.cell == "RC") & test.scene.isin(set(P.scene))] if c.endswith("s") else test[test.cell == "RC"]
                r[c] = _fmt(auc_ci(N.assign(s=m.score(N)), P.assign(s=m.score(P)), "s"))
        fam_rows.append(r)

    fam_row(f"all {len(physics.ALL)}", physics.ALL)
    for k, v in FAMILIES.items():
        fam_row("only " + k, v)
    for k, v in FAMILIES.items():
        fam_row("minus " + k, [f for f in physics.ALL if f not in v])
    fams = pd.DataFrame(fam_rows)

    # transfer between fake types
    tr_rows = []
    for src, dst in (("GR", "RR"), ("RR", "GR")):
        d = dev[dev.cell.isin(["RC", src])]
        m = physics.PhysicsClassifier().fit(d, (d.base != "RC").astype(int))
        N, P = test[test.cell == "RC"], test[test.cell == dst]
        tr_rows.append({"train on": src, "test on": dst,
                        "AUC": _fmt(auc_ci(N.assign(s=m.score(N)), P.assign(s=m.score(P)), "s"))})
    transfer = pd.DataFrame(tr_rows)

    # single-feature AUC on dev scenes (<0.5: the feature is lower in fakes)
    f_rows = []
    for ft in physics.ALL:
        r = {"feature": ft}
        for c in ("GR", "RR"):
            d = dev[dev.cell.isin(["RC", c])]
            r[c] = round(_roc_auc((d.cell == c).astype(int), d[ft].fillna(d[ft].median())), 3)
        f_rows.append(r)
    return {"cps2_main": main, "cps2_families": fams, "cps2_transfer": transfer, "cps2_features": pd.DataFrame(f_rows)}


def attack_calibration(pf: pd.DataFrame) -> pd.DataFrame:
    rc = pf[pf.cell == "RC"].set_index("scene")
    rows = []
    for c in sorted(set(pf.cell)):
        base, _, var = c.partition("@")
        if base not in ("GR", "RR") or not _is_attack(var):
            continue
        a = pf[pf.cell == c]
        ref = rc.reindex(a.scene)
        r = {"cell": c, "n": len(a)}
        for ft, tag in (("resid_level", "noise level"), ("gain_mad", "exposure jitter"), ("wb_mad", "white-balance jitter")):
            ratio = a[ft].to_numpy() / ref[ft].to_numpy()
            ratio = ratio[np.isfinite(ratio)]
            r[f"{tag} / real (median)"] = round(float(np.median(ratio)), 2) if len(ratio) else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- driver
def run(store=None) -> str:
    res = pipeline.results_dir()
    out = res / "analysis"
    out.mkdir(exist_ok=True)
    s = _parse(pd.read_csv(res / "scores.csv"))
    sw = s.pivot_table(index=["clip_id", "cell", "scene"], columns="detector", values="score").reset_index()
    pf_path = res / "physics_features.csv"
    pf = _parse(pd.read_csv(pf_path)) if pf_path.exists() else pd.DataFrame()
    if len(pf):
        pf = pf[pf[physics.ALL].notna().any(axis=1)].reset_index(drop=True)
    n_test = int(pf[pf.scene.map(is_test)].scene.nunique()) if len(pf) else 0
    text, files = ["# C44 analysis\n", f"Held-out protocol: scenes split by hash, 30% test ({n_test} test scenes).\n"], []

    def section(title, name, df_fn):
        try:
            df = df_fn()
        except Exception as e:                                          # noqa: BLE001 - never abort the run
            text.append(f"## {title}\n\nSKIPPED: {e!r}\n```\n{traceback.format_exc()[-700:]}\n```\n")
            return
        df.to_csv(out / f"{name}.csv", index=False)
        files.append(out / f"{name}.csv")
        text.append(f"## {title}\n\n```\n{df.to_string(index=False) if len(df) else '(nothing to report)'}\n```\n")

    tables, err = {}, None
    if len(pf):
        try:
            tables = physics_tables(pf, sw)
        except Exception as e:                                          # noqa: BLE001
            err = f"## CPS-2\n\nSKIPPED: {e!r}\n```\n{traceback.format_exc()[-900:]}\n```\n"
    else:
        err = "## CPS-2\n\nSKIPPED: results/physics_features.csv is missing (run `python -m c44 physics`).\n"
    if err:
        text.append(err)
    titles = {"cps2_main": "1. every detector vs. real, test scenes (AUC, higher score = fake)",
              "cps2_families": "2. which cue family carries the signal (clean-trained CPS-2, test scenes)",
              "cps2_transfer": "3. transfer between fake types (test scenes)",
              "cps2_features": "4. single-feature AUC on dev scenes (<0.5: lower in fakes)"}
    for name, title in titles.items():
        if name in tables:
            section(title, name, lambda n=name: tables[n])
    section("5. attack calibration (attacked / paired real clip)", "attack_calibration",
            lambda: attack_calibration(pf) if len(pf) else pd.DataFrame())
    section("6. generic detectors and CPS-1 on all scenes", "detectors", lambda: detectors_table(sw))
    summary = "\n".join(text)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    files.append(out / "summary.md")
    if store is not None:
        for p in files:
            store.stage(p)
        store.flush("c44: analysis")
    return summary
