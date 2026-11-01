"""Final analysis tables, CPU only, from results/scores.csv (generic detectors, CPS-1) and results/physics_features.csv (CPS-2).

Protocol. Scenes are split once by a hash into dev (70%) and test (30%) (`is_test`); nothing was designed on the test scenes.
Every CPS-2 model is trained on dev scenes only and evaluated on the test scenes; confidence intervals are scene-level
bootstrap. Generic detectors (UnivFD, NPR, consistency) need no training and are scored on the same test scenes.

  cps2_main     test scenes: every detector vs. real, in every setting (clean, recompression, attacks, resolution control).
                'aug' is trained with recompressed and attacked fakes; for an attack, every attack sharing a component with it is left out
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


def _parts(kind: str) -> set:
    return set(kind.split("+")) if kind else set()


def _train_kinds(attacks, mode: str, kind: str | None):
    """Which attack kinds a defender has seen. 'seen': all (incl. the evaluated one); 'combination': all but the evaluated one
    (its components still appear in other attacks); 'components': nothing that shares a component with it."""
    if mode == "none":
        return []
    if mode == "seen":
        return list(attacks)
    if mode == "combination":
        return [a for a in attacks if a != kind]
    return [a for a in attacks if not (_parts(a) & _parts(kind))]


AUG_COLS = {"seen": "CPS-2 + aug (attack seen)", "combination": "CPS-2 + aug (unseen combination)",
            "components": "CPS-2 + aug (unseen components)"}


def _settings(pf: pd.DataFrame):
    attacks = sorted({v for v in pf["var"].unique() if _is_attack(v)})
    pairs = [("RC", "GR", "clean", ""), ("RC", "RR", "clean", "")]
    for k in sorted({v for v in pf["var"].unique() if v.startswith("crf")}):
        pairs += [(f"RC@{k}", f"GR@{k}", k, k), (f"RC@{k}", f"RR@{k}", k, k)]
    for k in attacks:
        pairs += [("RC", f"GR@{k}", k, k), ("RC", f"RR@{k}", k, k)]
    for c in ("GRs", "RRs"):
        if c in set(pf.cell):
            pairs += [("RC", c[:2], f"same scenes as {c}", ""), ("RC", c, f"supersampled {c}", "")]
    return attacks, pairs


def _pair_frames(df: pd.DataFrame, neg: str, pos: str, label: str, kind: str):
    """Real / fake clips of one setting; attacks and the resolution control cover a scene subset, so compare on those scenes."""
    P, N = df[df.cell == pos], df[df.cell == neg]
    if label.startswith(("same scenes", "supersampled")):
        ctl = set(df[df.cell == pos[:2] + "s"].scene) if label.startswith("same") else set(P.scene)
        P, N = P[P.scene.isin(ctl)], N[N.scene.isin(ctl)]
    elif _is_attack(kind):
        N = N[N.scene.isin(set(P.scene))]
    return N, P


def physics_tables(pf: pd.DataFrame, sw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pf = pf.assign(test=pf.scene.map(is_test))
    dev, test = pf[~pf.test], pf[pf.test]
    attacks, pairs = _settings(pf)

    m_oc = physics.PhysicsOneClass().fit(dev[dev.cell == "RC"])
    d0, y0 = _train_set(dev, compress=False, attack_kinds=[])
    m_clean = physics.PhysicsClassifier().fit(d0, y0)
    cache = {}

    def aug_model(mode, kind=None):
        key = (mode, kind if mode in ("combination", "components") else None)
        if key not in cache:
            d, y = _train_set(dev, True, _train_kinds(attacks, mode, kind))
            cache[key] = physics.PhysicsClassifier().fit(d, y)
        return cache[key]

    base_scores = sw.set_index("clip_id")
    rows = []
    for neg, pos, label, kind in pairs:
        N, P = _pair_frames(test, neg, pos, label, kind)
        if len(P) < 5 or len(N) < 5:
            continue
        row = {"pair": f"{neg} vs {pos}", "setting": label, "n_test_pos": len(P)}
        for d in DETECTORS:
            if d in base_scores:
                a = N.assign(s=base_scores[d].reindex(N.clip_id).to_numpy())
                b = P.assign(s=base_scores[d].reindex(P.clip_id).to_numpy())
                row[{"cps": "CPS-1 (one-class)"}.get(d, d)] = _fmt(auc_ci(a, b, "s"))
        models = [("CPS-2 one-class", m_oc), ("CPS-2 supervised", m_clean)]
        if _is_attack(kind):
            models += [(AUG_COLS[m], aug_model(m, kind)) for m in ("seen", "combination", "components")]
        else:
            models.append(("CPS-2 + aug", aug_model("seen")))
        for name, model in models:
            row[name] = _fmt(auc_ci(N.assign(s=model.score(N)), P.assign(s=model.score(P)), "s"))
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


def cv_table(pf: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """Pre-specified (nothing tuned) scene-grouped K-fold cross-validation over ALL scenes: each scene's clips are scored by a model
    that never saw that scene. Same settings and defender protocols as the held-out table, with ~3x the sample."""
    pf = pf.reset_index(drop=True)
    fold = pf.scene.map(lambda sc: zlib.crc32(sc.encode()) % n_folds).to_numpy()
    attacks, pairs = _settings(pf)
    specs = [("none", None, False), ("seen", None, True)]
    for k in attacks:
        specs += [("combination", k, True), ("components", k, True)]
    scores = {sp: np.full(len(pf), np.nan) for sp in specs}
    for f in range(n_folds):
        train, idx = pf[fold != f], np.where(fold == f)[0]
        te = pf.iloc[idx]
        for sp in specs:
            d, y = _train_set(train, sp[2], _train_kinds(attacks, sp[0], sp[1]))
            if len(np.unique(y)) == 2:
                scores[sp][idx] = physics.PhysicsClassifier().fit(d, y).score(te)
    rows = []
    for neg, pos, label, kind in pairs:
        N, P = _pair_frames(pf, neg, pos, label, kind)
        if len(P) < 5 or len(N) < 5:
            continue

        def auc_for(sp):
            return _fmt(auc_ci(N.assign(s=scores[sp][N.index]), P.assign(s=scores[sp][P.index]), "s", n_boot=200))

        row = {"pair": f"{neg} vs {pos}", "setting": label, "n_pos": len(P), "CPS-2 supervised": auc_for(("none", None, False))}
        if _is_attack(kind):
            row[AUG_COLS["seen"]] = auc_for(("seen", None, True))
            row[AUG_COLS["combination"]] = auc_for(("combination", kind, True))
            row[AUG_COLS["components"]] = auc_for(("components", kind, True))
        else:
            row["CPS-2 + aug"] = auc_for(("seen", None, True))
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- paper tables
def _tex_num(s) -> str:
    import re
    if not isinstance(s, str):
        return "--"
    m = re.match(r"([0-9.]+) \[([0-9.]+),([0-9.]+)\]", s)
    return f"{float(m[1]):.2f}" + r"{\,\scriptsize[" + f"{m[2]},{m[3]}" + "]}" if m else "--"


_SETTING = {"clean": "clean", "crf23": "CRF 23", "crf28": "CRF 28", "crf35": "CRF 35", "blurcal": "blur", "noiseadapt": "noise",
            "aeadapt": "exposure jitter", "blurcal+noiseadapt": "blur + noise", "blurcal+noiseadapt+aeadapt": "blur + noise + exposure"}


def _row_name(pair: str, setting: str) -> str:
    fake = pair.split(" vs ")[1].split("@")[0]
    who = {"GR": "Generated world", "RR": "Real-scene render", "GRs": "Generated, 3x supersampled", "RRs": "Real-scene, 3x supersampled"}.get(fake, fake)
    if setting.startswith("same scenes"):
        return f"{who} (control scenes)"
    return f"{who}, {_SETTING.get(setting, 'supersampled' if setting.startswith('supersampled') else setting)}"


def _tex_table(df: pd.DataFrame, cols: dict, caption: str, label: str) -> str:
    out = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
           " & ".join(["Setting"] + list(cols.values())) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        out.append(" & ".join([_row_name(r["pair"], r["setting"])] + [_tex_num(r.get(c)) for c in cols]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}", ""]
    return "\n".join(out)


def write_latex(tables: dict[str, pd.DataFrame], path) -> None:
    """booktabs tables for the paper (AUC with 95% scene-bootstrap CI). Needs \\usepackage{booktabs}."""
    parts = ["% generated by c44.analysis.write_latex; requires \\usepackage{booktabs}\n"]
    if "cps2_main" in tables and len(tables["cps2_main"]):
        m = tables["cps2_main"]
        base = m[~m.setting.map(analysis_is_attack_label)]
        parts.append(_tex_table(base, {"univfd": "UnivFD", "npr": "NPR", "consistency": "Consistency", "CPS-1 (one-class)": "CPS-1",
                                       "CPS-2 one-class": "CPS-2 (one-class)", "CPS-2 supervised": "CPS-2"},
                                "Real vs. fake AUC on held-out scenes: clean, recompressed and resolution-control settings.", "tab:main"))
        att = m[m.setting.map(analysis_is_attack_label)]
        if len(att):
            parts.append(_tex_table(att, {"univfd": "UnivFD", "CPS-2 supervised": "CPS-2", AUG_COLS["seen"]: "+aug (seen)",
                                          AUG_COLS["combination"]: "+aug (new combo.)", AUG_COLS["components"]: "+aug (new comp.)"},
                                    "Camera-aware attacks on held-out scenes. +aug: CPS-2 trained with attacks; the attack is seen, a new "
                                    "combination of seen components, or built from components never seen.", "tab:attacks"))
    if "cps2_cv" in tables and len(tables["cps2_cv"]):
        c = tables["cps2_cv"]
        att = c[c.setting.map(analysis_is_attack_label)]
        if len(att):
            parts.append(_tex_table(att, {"CPS-2 supervised": "CPS-2", AUG_COLS["seen"]: "+aug (seen)", AUG_COLS["combination"]: "+aug (new combo.)",
                                          AUG_COLS["components"]: "+aug (new comp.)"},
                                    "Attacks, scene-grouped 5-fold cross-validation over all scenes (larger sample).", "tab:attacks-cv"))
    if "cps2_families" in tables and len(tables["cps2_families"]):
        f = tables["cps2_families"]
        lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lcccc}", r"\toprule",
                 r"Features & GR & RR & GR supersampled & RR supersampled \\", r"\midrule"]
        for _, r in f.iterrows():
            lines.append(" & ".join([str(r["features"]).replace("&", r"\&")] + [_tex_num(r.get(k)) for k in ("GR", "RR", "GRs", "RRs")]) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Which cue family carries the signal (held-out scenes).}", r"\label{tab:families}", r"\end{table}", ""]
        parts.append("\n".join(lines))
    path.write_text("\n".join(parts), encoding="utf-8")


def analysis_is_attack_label(setting: str) -> bool:
    return _is_attack(setting) and not setting.startswith(("same scenes", "supersampled")) and setting != "clean"


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
            with np.errstate(divide="ignore", invalid="ignore"):
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
              "cps2_families": "3. which cue family carries the signal (clean-trained CPS-2, test scenes)",
              "cps2_transfer": "4. transfer between fake types (test scenes)",
              "cps2_features": "5. single-feature AUC on dev scenes (<0.5: lower in fakes)"}
    for name in ("cps2_main",):
        if name in tables:
            section(titles[name], name, lambda n=name: tables[n])
    if len(pf):
        def _cv():
            tables["cps2_cv"] = cv_table(pf)
            return tables["cps2_cv"]
        section("2. pre-specified 5-fold scene-grouped CV over ALL scenes (same settings, ~3x the sample)", "cps2_cv", _cv)
    for name in ("cps2_families", "cps2_transfer", "cps2_features"):
        if name in tables:
            section(titles[name], name, lambda n=name: tables[n])
    section("6. attack calibration (attacked / paired real clip; 1.0 = matched)", "attack_calibration",
            lambda: attack_calibration(pf) if len(pf) else pd.DataFrame())
    section("7. generic detectors and CPS-1 on all scenes", "detectors", lambda: detectors_table(sw))
    if tables:
        try:
            write_latex(tables, out / "tables.tex")
            files.append(out / "tables.tex")
        except Exception as e:                                          # noqa: BLE001
            text.append(f"## LaTeX tables\n\nSKIPPED: {e!r}\n")
    summary = "\n".join(text)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    files.append(out / "summary.md")
    if store is not None:
        for p in files:
            store.stage(p)
        store.flush("c44: analysis")
    return summary
