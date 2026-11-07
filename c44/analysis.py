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

from . import embed, learned, pipeline, physics

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
    d = pd.concat([neg[["scene", col]].assign(y=0), pos[["scene", col]].assign(y=1)])      # only what the AUC needs: callers pass wide frames
    d = d[np.isfinite(d[col])]
    if d.y.nunique() < 2:
        return float("nan"), float("nan"), float("nan")
    a = _roc_auc(d.y, d[col])
    rng = np.random.default_rng(seed)
    groups = {k: (v.y.to_numpy(), v[col].to_numpy()) for k, v in d.groupby("scene")}
    keys = np.array(list(groups))
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(keys, len(keys))
        y = np.concatenate([groups[k][0] for k in pick])
        sc = np.concatenate([groups[k][1] for k in pick])
        if len(np.unique(y)) == 2:
            bs.append(_roc_auc(y, sc))
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (float("nan"), float("nan"))
    return a, float(lo), float(hi)


def _fmt(t) -> str:
    return "n/a" if not np.isfinite(t[0]) else f"{t[0]:.3f} [{t[1]:.2f},{t[2]:.2f}]"


def _is_attack(var: str) -> bool:
    return bool(var) and not var.startswith(("crf", "nz-"))            # 'nz-*' are training-only clips, never evaluated


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
def _train_set(dev: pd.DataFrame, compress: bool, attack_kinds, real_ops: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """Training clips: clean RC / GR / RR, plus (compress) the recompressed ones and the listed attack kinds. Calibrated attacks
    exist for fakes only. 'nz-*' (randomised operators) are fakes-only too (RAT), unless real_ops: then the REAL clips that went
    through the same operators are kept as real (symmetric training, SYM)."""
    kinds = list(attack_kinds)
    keep = dev.base.isin(["RC", "GR", "RR"]) & (
        (dev["var"] == "") | (compress & dev["var"].str.startswith("crf")) | dev["var"].isin(kinds))
    drop_real = (dev.base == "RC") & dev["var"].isin(kinds)
    if real_ops:
        drop_real &= ~dev["var"].str.startswith("nz-")
    d = dev[keep & ~drop_real]
    return d, (d.base != "RC").to_numpy().astype(int)


_COMPONENT = {"blurcal": "blur", "noiseadapt": "noise", "aecal": "ae"}


def _parts(kind: str) -> set:
    """Components (blur / noise / ae) of an attack or of a randomised training operator ('nz-blur+noise')."""
    if not kind:
        return set()
    out = set()
    for p in kind.removeprefix("nz-").split("+"):
        out |= {"blur", "noise", "ae"} if p == "all" else {_COMPONENT.get(p, p)}                # 'nz-all' = all three operators
    return out


def _train_kinds(attacks, mode: str, kind: str | None, rat=()):
    """Which attack kinds a defender has seen.
    'seen': all calibrated attacks (incl. the evaluated one); 'combination': all but the evaluated one (its components still
    appear in other attacks); 'components': nothing that shares a component with it.
    'rat' (randomised-attacker training, fakes only) and 'sym' (the same operators on real clips too) use only the randomly
    parameterised clips, never a calibrated attack; '*-components': without any operator sharing a component with the attack."""
    if mode == "none":
        return []
    if mode in ("rat", "sym"):
        return list(rat)
    if mode in ("rat-components", "sym-components"):
        return [a for a in rat if not (_parts(a) & _parts(kind))]
    if mode == "seen":
        return list(attacks)
    if mode == "combination":
        return [a for a in attacks if a != kind]
    return [a for a in attacks if not (_parts(a) & _parts(kind))]


AUG_COLS = {"seen": "CPS-2 + aug (attack seen)", "combination": "CPS-2 + aug (unseen combination)",
            "components": "CPS-2 + aug (unseen components)"}
FUSION = "CPS-2 + CLIP"                       # physics cues + retrained CLIP probe (equal-weight z-score fusion)
FUS_AUG = {k: v.replace("CPS-2", FUSION) for k, v in AUG_COLS.items()}
RAT_COLS = {"rat": f"{FUSION} + RAT", "rat-components": f"{FUSION} + RAT (components unseen)"}      # randomised-attacker training
SYM_COLS = {"sym": f"{FUSION} + SYM", "sym-components": f"{FUSION} + SYM (components unseen)"}      # same, real clips processed too
PER_ATTACK = ("combination", "components", "rat-components", "sym-components")


def attach_embeddings(pf: pd.DataFrame, log=print) -> tuple[pd.DataFrame, dict]:
    """Add the backbone embeddings of every clip as columns. A backbone is used only if it covers every clip."""
    emb_cols, parts = {}, [pf.reset_index(drop=True)]
    for key, name, norm in (("clip", "CLIP", True), ("npr", "NPR", False)):
        got = embed.load(key)
        if got is None:
            log(f"[analysis] no {key} embeddings: the {name} probe and fusion are skipped")
            continue
        ids, e = got
        e = e / np.linalg.norm(e, axis=1, keepdims=True) if norm else e
        cols = [f"{key}{i}" for i in range(e.shape[1])]
        frame = pd.DataFrame(e, index=ids, columns=cols).reindex(pf.clip_id)
        gone = frame.iloc[:, 0].isna().to_numpy()
        miss = int(gone.sum())
        if miss:
            where = pf.cell[gone].value_counts().head(6).to_dict()
            log(f"[analysis] {miss}/{len(pf)} clips lack {key} embeddings (cells: {where}): the {name} probe and fusion are skipped")
            continue
        parts.append(frame.reset_index(drop=True))
        emb_cols[name] = cols
    return pd.concat(parts, axis=1), emb_cols


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


# ---- scoring every (model, training protocol) once, in parallel; every table is read from these scores
def _spec(name: str, mode: str, kind: str | None = None) -> tuple:
    return (name, mode, kind if mode in PER_ATTACK else None)


def _needed_specs(attacks, rat, registry) -> list:
    fused = FUSION in registry
    names = ["CPS-2"] + ([FUSION] if fused else [])
    specs = [_spec(n, "none") for n in ["CPS-2"] + [m for m in registry if m.endswith(" probe")] + ([FUSION] if fused else [])]
    specs += [_spec(n, "seen") for n in names]
    for k in attacks:
        for n in names:
            specs += [_spec(n, "combination", k), _spec(n, "components", k)]
        if rat and fused:
            specs += [_spec(FUSION, "rat-components", k), _spec(FUSION, "sym-components", k)]
    if rat:
        specs += [_spec("CPS-2", "rat"), _spec("CPS-2", "sym")]
        if fused:
            specs += [_spec(FUSION, "rat"), _spec(FUSION, "sym")]
    return list(dict.fromkeys(specs))


def _train_for(train: pd.DataFrame, mode: str, kind, attacks, rat):
    return _train_set(train, mode != "none", _train_kinds(attacks, mode, kind, rat), real_ops=mode.startswith("sym"))


def _fit_score(make, d, y, te):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):                                      # one BLAS thread per worker: no oversubscription
        return make().fit(d, y).score(te)


def _fit_many(jobs, n_jobs: int | None = None) -> list:
    import os
    n = n_jobs or max(1, min(16, (os.cpu_count() or 2) - 1))
    if n <= 1 or len(jobs) <= 1:
        return [_fit_score(*j) for j in jobs]
    try:
        from joblib import Parallel, delayed
        return Parallel(n_jobs=n)(delayed(_fit_score)(*j) for j in jobs)
    except Exception:                                                      # noqa: BLE001 - a worker-pool problem must not lose the run
        return [_fit_score(*j) for j in jobs]


def score_protocols(pf: pd.DataFrame, specs, registry, attacks, rat, split: str = "heldout", n_folds: int = 5,
                    n_jobs: int | None = None) -> dict:
    """Score every clip with every (model, protocol, attack) spec. split='heldout': train on the dev scenes, score the test
    scenes. split='cv': scene-grouped K-fold over ALL scenes (each clip is scored by a model that never saw its scene).
    Returns spec -> array over pf (NaN where a clip was not scored). `pf` must have a default RangeIndex."""
    if split == "heldout":
        te_all = pf.scene.map(is_test).to_numpy()
        folds = [(~te_all, te_all)]
    else:
        fold = pf.scene.map(lambda sc: zlib.crc32(sc.encode()) % n_folds).to_numpy()
        folds = [(fold != f, fold == f) for f in range(n_folds)]
    out = {s: np.full(len(pf), np.nan) for s in specs}
    jobs, where = [], []
    for tr, te in folds:
        train, test, idx = pf[tr], pf[te], np.where(te)[0]
        sets = {}
        for spec in specs:
            name, mode, kind = spec
            if (mode, kind) not in sets:
                sets[(mode, kind)] = _train_for(train, mode, kind, attacks, rat)
            d, y = sets[(mode, kind)]
            if len(np.unique(y)) < 2:
                continue
            jobs.append((registry[name], d, y, test))
            where.append((spec, idx))
    for (spec, idx), sc in zip(where, _fit_many(jobs, n_jobs)):
        out[spec][idx] = sc
    return out


def _auc(scores: dict, spec: tuple, N: pd.DataFrame, P: pd.DataFrame, n_boot: int = 300) -> str:
    s = scores[spec]
    return _fmt(auc_ci(N.assign(s=s[N.index]), P.assign(s=s[P.index]), "s", n_boot=n_boot))


def main_table(pf, scores, ev, registry, attacks, rat, extra=None, n_boot: int = 300, n_col: str = "n_pos") -> pd.DataFrame:
    """One row per setting (clean / recompression / attack / resolution control); a column per detector and training protocol."""
    df = pf[ev]
    probes = [n for n in registry if n.endswith(" probe")]
    fused = FUSION in registry
    rows = []
    for neg, pos, label, kind in _settings(pf)[1]:
        N, P = _pair_frames(df, neg, pos, label, kind)
        if len(P) < 5 or len(N) < 5:
            continue
        row = {"pair": f"{neg} vs {pos}", "setting": label, n_col: len(P)}
        if extra:
            row.update(extra(N, P))
        row["CPS-2 supervised"] = _auc(scores, _spec("CPS-2", "none"), N, P, n_boot)
        for n in probes + ([FUSION] if fused else []):
            row[n] = _auc(scores, _spec(n, "none"), N, P, n_boot)
        if _is_attack(kind):
            for n, cols in (("CPS-2", AUG_COLS), (FUSION, FUS_AUG)):
                if n == FUSION and not fused:
                    continue
                for mode in ("seen", "combination", "components"):
                    row[cols[mode]] = _auc(scores, _spec(n, mode, kind), N, P, n_boot)
            if rat and fused:
                for mode, cols in (("rat", RAT_COLS), ("rat-components", RAT_COLS), ("sym", SYM_COLS), ("sym-components", SYM_COLS)):
                    row[cols[mode]] = _auc(scores, _spec(FUSION, mode, kind), N, P, n_boot)
        else:
            row["CPS-2 + aug"] = _auc(scores, _spec("CPS-2", "seen"), N, P, n_boot)
            if fused:
                row[f"{FUSION} + aug"] = _auc(scores, _spec(FUSION, "seen"), N, P, n_boot)
        rows.append(row)
    return pd.DataFrame(rows)


OP_KINDS = ("blur", "noise", "ae", "all")


def operator_table(pf, scores, ev, registry, n_boot: int = 200) -> pd.DataFrame:
    """The price of a defence. A detector trained on randomised attacks (RAT) may simply have learned 'processed = fake', which
    would also flag REAL clips that went through the same blur / noise / exposure operators. For each model: the false-alarm rate on
    processed real clips (threshold = 5% false alarms on clean real clips), the detection rate on the full camera-aware attacker at
    that threshold, and the AUC of processed-real vs processed-fake (the symmetric test, where the operators carry no information).
    SYM trains with the processed real clips as real, which removes that shortcut."""
    df = pf[ev]
    real = df[df.cell == "RC"]
    rows = []
    for name, mode, proto in (("CPS-2", "none", "clean-trained"), (FUSION, "none", "clean-trained"), ("CPS-2", "rat", "RAT"),
                              (FUSION, "rat", "RAT"), ("CPS-2", "sym", "SYM"), (FUSION, "sym", "SYM")):
        spec = _spec(name, mode)
        if name not in registry or spec not in scores or not np.isfinite(scores[spec][real.index]).any():
            continue
        s = scores[spec]
        t = float(np.nanpercentile(s[real.index], 95))
        r = {"model": name, "training": proto}
        for fk in ("GR", "RR"):
            r[f"AUC {fk} clean"] = _auc(scores, spec, real, df[df.cell == fk], n_boot)
        for k in OP_KINDS:
            x = df[df.cell == f"RC@nz-{k}"]
            r[f"FPR real+{k}"] = round(float((s[x.index] > t).mean()), 3) if len(x) else float("nan")
        for fk in ("GR", "RR"):
            P = df[df.cell == f"{fk}@blurcal+noiseadapt+aecal"]
            r[f"TPR {fk} full attacker"] = round(float((s[P.index] > t).mean()), 3) if len(P) else float("nan")
        for fk in ("GR", "RR"):
            RX, FX = df[df.cell == "RC@nz-all"], df[df.cell == f"{fk}@nz-all"]
            r[f"AUC real+op vs {fk}+op"] = _auc(scores, spec, RX, FX, n_boot) if len(RX) >= 5 and len(FX) >= 5 else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


def physics_tables(pf: pd.DataFrame, sw: pd.DataFrame, emb_cols: dict | None = None) -> dict[str, pd.DataFrame]:
    pf = pf.reset_index(drop=True)
    pf = pf.assign(test=pf.scene.map(is_test))
    dev, test = pf[~pf.test], pf[pf.test]
    attacks, _ = _settings(pf)
    registry = learned.make_models(emb_cols or {})              # CPS-2, '<backbone> probe', 'CPS-2 + CLIP'
    rat = sorted(v for v in pf["var"].unique() if v.startswith("nz-"))
    scores = score_protocols(pf, _needed_specs(attacks, rat, registry), registry, attacks, rat, "heldout")
    m_oc = physics.PhysicsOneClass().fit(dev[dev.cell == "RC"])
    d0, y0 = _train_set(dev, compress=False, attack_kinds=[])
    base_scores = sw.set_index("clip_id")

    def extra(N, P):
        row = {}
        for d in DETECTORS:
            if d in base_scores:
                a = N.assign(s=base_scores[d].reindex(N.clip_id).to_numpy())
                b = P.assign(s=base_scores[d].reindex(P.clip_id).to_numpy())
                row[{"cps": "CPS-1 (one-class)"}.get(d, d)] = _fmt(auc_ci(a, b, "s"))
        row["CPS-2 one-class"] = _fmt(auc_ci(N.assign(s=m_oc.score(N)), P.assign(s=m_oc.score(P)), "s"))
        return row

    ev = pf.test.to_numpy()
    main = main_table(pf, scores, ev, registry, attacks, rat, extra=extra, n_col="n_test_pos")
    ops = operator_table(pf, scores, ev, registry)

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
    return {"cps2_main": main, "cps2_operators": ops, "cps2_families": fams, "cps2_transfer": transfer,
            "cps2_features": pd.DataFrame(f_rows)}


def cv_tables(pf: pd.DataFrame, n_folds: int = 5, emb_cols: dict | None = None) -> dict[str, pd.DataFrame]:
    """Pre-specified (nothing tuned) scene-grouped K-fold cross-validation over ALL scenes: each scene's clips are scored by a model
    that never saw that scene. Same settings and defender protocols as the held-out tables, with ~3x the sample."""
    pf = pf.reset_index(drop=True)
    attacks, _ = _settings(pf)
    registry = learned.make_models(emb_cols or {})
    rat = sorted(v for v in pf["var"].unique() if v.startswith("nz-"))
    scores = score_protocols(pf, _needed_specs(attacks, rat, registry), registry, attacks, rat, "cv", n_folds)
    ev = np.ones(len(pf), bool)
    return {"cps2_cv": main_table(pf, scores, ev, registry, attacks, rat, n_boot=200),
            "cps2_operators_cv": operator_table(pf, scores, ev, registry)}


def cv_table(pf: pd.DataFrame, n_folds: int = 5, emb_cols: dict | None = None) -> pd.DataFrame:
    return cv_tables(pf, n_folds, emb_cols)["cps2_cv"]


# ----------------------------------------------------------------------------- paper tables
def _tex_num(s) -> str:
    import re
    if not isinstance(s, str):
        return "--"
    m = re.match(r"([0-9.]+) \[([0-9.]+),([0-9.]+)\]", s)
    return f"{float(m[1]):.2f}" + r"{\,\scriptsize[" + f"{m[2]},{m[3]}" + "]}" if m else "--"


_SETTING = {"clean": "clean", "crf23": "CRF 23", "crf28": "CRF 28", "crf35": "CRF 35", "blurcal": "blur", "noiseadapt": "noise",
            "aecal": "exposure jitter", "blurcal+noiseadapt": "blur + noise", "blurcal+noiseadapt+aecal": "blur + noise + exposure"}


def _row_name(pair: str, setting: str) -> str:
    fake = pair.split(" vs ")[1].split("@")[0]
    who = {"GR": "Generated world", "RR": "Real-scene render", "GRs": "Generated, 3x supersampled", "RRs": "Real-scene, 3x supersampled"}.get(fake, fake)
    if setting.startswith("same scenes"):
        return f"{who} (control scenes)"
    return f"{who}, {_SETTING.get(setting, 'supersampled' if setting.startswith('supersampled') else setting)}"


ATTACK_COLS = {"univfd": "UnivFD", "CLIP probe": "UnivFD$^\\dagger$", "CPS-2 supervised": "CPS-2", FUSION: "CPS-2+CLIP",
               RAT_COLS["rat"]: "+RAT", RAT_COLS["rat-components"]: "+RAT (new comp.)", SYM_COLS["sym"]: "+SYM",
               FUS_AUG["seen"]: "+aug (attack seen)"}
ATTACK_CAPTION = ("Camera-aware attacks, {split}. $^\\dagger$: probe retrained on the dev scenes. +RAT: CPS-2+CLIP trained with "
                  "randomised-strength blur / noise / exposure operators on the fakes (never the calibrated attack; ``new comp.'': without any "
                  "operator sharing a component with the attack). +aug (attack seen): trained on the calibrated attack itself.")


def _tex_operators(df: pd.DataFrame, split: str, label: str) -> str:
    """False alarms of each training protocol on REAL clips that went through the attack operators, vs. detection of the full attacker."""
    heads = ["Model", "AUC GR", "AUC RR", "FPR +blur", "FPR +noise", "FPR +exp.", "FPR +all", "TPR GR", "TPR RR"]
    out = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{l" + "c" * (len(heads) - 1) + "}", r"\toprule", " & ".join(heads) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        f = lambda c: "--" if pd.isna(r.get(c)) else f"{float(r[c]):.2f}"          # noqa: E731
        out.append(" & ".join([f"{r['model']} ({r['training']})", _tex_num(r.get("AUC GR clean")), _tex_num(r.get("AUC RR clean")),
                               f("FPR real+blur"), f("FPR real+noise"), f("FPR real+ae"), f("FPR real+all"),
                               f("TPR GR full attacker"), f("TPR RR full attacker")]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}",
            f"\\caption{{Cost of attack-aware training, {split}. FPR: real clips processed by the same blur / noise / exposure operators "
            "that the attacker uses, flagged at the threshold giving 5\\% false alarms on clean real clips. TPR: full camera-aware attacker "
            "at that threshold. RAT: trained with randomised operators on fakes only; SYM: the same operators on real clips too.}",
            f"\\label{{{label}}}", r"\end{table}", ""]
    return "\n".join(out)


def _tex_table(df: pd.DataFrame, cols: dict, caption: str, label: str) -> str:
    cols = {k: v for k, v in cols.items() if k in df.columns}              # only the columns that exist (e.g. no CLIP probe)
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
                                       "CLIP probe": "UnivFD$^\\dagger$", "NPR probe": "NPR$^\\dagger$",
                                       "CPS-2 one-class": "CPS-2 (one-class)", "CPS-2 supervised": "CPS-2", FUSION: "CPS-2+CLIP"},
                                "Real vs. fake AUC on held-out scenes: clean, recompressed and resolution-control settings. "
                                "$^\\dagger$: probe retrained on the benchmark's dev scenes (the other generic detectors are zero-shot).",
                                "tab:main"))
        att = m[m.setting.map(analysis_is_attack_label)]
        if len(att):
            parts.append(_tex_table(att, ATTACK_COLS, ATTACK_CAPTION.format(split="held-out scenes"), "tab:attacks"))
    if "cps2_cv" in tables and len(tables["cps2_cv"]):
        c = tables["cps2_cv"]
        att = c[c.setting.map(analysis_is_attack_label)]
        if len(att):
            parts.append(_tex_table(att, ATTACK_COLS, ATTACK_CAPTION.format(split="a scene-grouped 5-fold cross-validation over all scenes"),
                                    "tab:attacks-cv"))
    for key, split, label in (("cps2_operators", "held-out scenes", "tab:operators"),
                              ("cps2_operators_cv", "a scene-grouped 5-fold cross-validation over all scenes", "tab:operators-cv")):
        if key in tables and len(tables[key]):
            parts.append(_tex_operators(tables[key], split, label))
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

    tables, err, emb_cols = {}, None, {}
    if len(pf):
        try:
            pf, emb_cols = attach_embeddings(pf, log=lambda m: text.append(m + "\n"))
            text.append(f"Learned backbones available: {', '.join(emb_cols) or 'none'}.\n")
            tables = physics_tables(pf, sw, emb_cols)
        except Exception as e:                                          # noqa: BLE001
            err = f"## CPS-2\n\nSKIPPED: {e!r}\n```\n{traceback.format_exc()[-900:]}\n```\n"
    else:
        err = "## CPS-2\n\nSKIPPED: results/physics_features.csv is missing (run `python -m c44 physics`).\n"
    if err:
        text.append(err)
    titles = {"cps2_main": "1. every detector and training protocol vs. real, test scenes (AUC, higher score = fake)",
              "cps2_operators": "2. the price of a defence: false alarms on REAL clips processed by the attack operators (test scenes)",
              "cps2_families": "5. which cue family carries the signal (clean-trained CPS-2, test scenes)",
              "cps2_transfer": "6. transfer between fake types (test scenes)",
              "cps2_features": "7. single-feature AUC on dev scenes (<0.5: lower in fakes)"}
    for name in ("cps2_main", "cps2_operators"):
        if name in tables:
            section(titles[name], name, lambda n=name: tables[n])
    cv: dict = {}

    def _cv(name):
        def fn():
            if not cv:
                cv.update(cv_tables(pf, emb_cols=emb_cols))
            tables[name] = cv[name]
            return cv[name]
        return fn

    if len(pf):
        section("3. pre-specified 5-fold scene-grouped CV over ALL scenes (same settings, ~3x the sample)", "cps2_cv", _cv("cps2_cv"))
        section("4. the same CV: false alarms on processed real clips", "cps2_operators_cv", _cv("cps2_operators_cv"))
    for name in ("cps2_families", "cps2_transfer", "cps2_features"):
        if name in tables:
            section(titles[name], name, lambda n=name: tables[n])
    section("8. attack calibration (attacked / paired real clip; 1.0 = matched)", "attack_calibration",
            lambda: attack_calibration(pf) if len(pf) else pd.DataFrame())
    section("9. generic detectors and CPS-1 on all scenes", "detectors", lambda: detectors_table(sw))
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
