"""Analysis of the weakness experiments (CPU). Reads results/{physics_features,physics_features_weak,joint_features}.csv and the CLIP / NPR
embeddings, trains on the DEV scenes only, evaluates on the TEST scenes (same hash split as the main analysis), and writes results/weak/*.csv
and results/weak/summary.md.

Detectors      CPS-2 (23 cues) | CPS-3 (23 + 8 joint cues) | CLIP probe | CPS-2 + CLIP | CPS-3 + CLIP
Training       clean                 clean RC / RR / GR only
               sym3                  the paper's SYM: blur / noise / exposure operators on real AND fake clips
               sym-wide              SYM with all 9 operator families (+ composites)
               rat-wide              the same operators on fakes only (RAT)
               lofo-<family>         SYM-wide without every operator kind that contains <family>: the family is UNSEEN
Evaluation     ladder     RC vs fake under the attacker ladder L1..L3 (+ the main run's attacker and CRF35)
               lofo       for every family F: AUC(RC vs fake+F) 'attack', AUC(real+F vs fake+F) 'symmetric' (the operator carries no
                          label), and the false-alarm rate on real+F at a threshold fixed on out-of-fold dev scores
Statistics     scene-cluster BCa bootstrap, paired differences with Holm correction (c44.bootstats)
"""

from __future__ import annotations

import re
import time
import zlib

import numpy as np
import pandas as pd

from . import analysis as an
from . import bootstats as bs
from . import embed, learned, physics, pipeline, weak

PROTOCOL_ORDER = ["clean", "sym3", "sym-wide", "rat-wide"]
OLD_KINDS = ["nz-blur", "nz-noise", "nz-ae", "nz-blur+noise", "nz-all"]
FAM_KIND = {"blur": "nz-blur", "noise": "nz-noise", "ae": "nz-ae", **{f: f"nz-{f}" for f in weak.NEW_FAMS}}
FULL_OLD = "blurcal+noiseadapt+aecal"
ID_RE = an.ID_RE


def parse(df: pd.DataFrame) -> pd.DataFrame:
    """Like analysis._parse, but also accepts the weak variants."""
    m = df.clip_id.str.extract(ID_RE)
    df = df.assign(base=m[0], scene=m[1], var=m[2].fillna(""))
    df = df[df.base.notna()]
    ok = (df["var"] == "") | df["var"].isin(pipeline.VARIANTS) | df["var"].isin(weak.VARIANTS)
    df = df[ok]
    return df.assign(cell=df.base + np.where(df["var"] != "", "@" + df["var"], ""))


def load(res, log=print):
    """(pf, embedding columns): CPS-2 + J features of every clip that has J, with embeddings attached when complete."""
    pf0 = pd.read_csv(res / "physics_features.csv")
    pw = res / "physics_features_weak.csv"
    if pw.exists():
        pf0 = pd.concat([pf0, pd.read_csv(pw)], ignore_index=True)
    pf0 = pf0.drop_duplicates("clip_id", keep="last")
    J = pd.read_csv(res / "joint_features.csv").drop_duplicates("clip_id", keep="last")
    pf = parse(pf0).merge(J[["clip_id"] + weak.JOINT], on="clip_id", how="inner")
    pf = pf[pf[physics.ALL].notna().any(axis=1) & pf[weak.JOINT].notna().any(axis=1)].reset_index(drop=True)
    pf["test"] = pf.scene.map(an.is_test).astype(bool)
    pf, emb_cols = an.attach_embeddings(pf, log=log)
    pf = pf.reset_index(drop=True)
    log(f"[weak:analyse] {len(pf)} clips, {pf.scene.nunique()} scenes ({int(pf[pf.test].scene.nunique())} test); cells: "
        f"{pf.cell.value_counts().to_dict()}; backbones: {list(emb_cols) or 'none'}")
    return pf, emb_cols


def registry(emb_cols: dict) -> dict:
    """name -> (factory, columns the detector reads). Passing only those columns to the workers keeps the jobs small."""
    P2, P3 = list(physics.ALL), list(physics.ALL) + list(weak.JOINT)
    reg = {"CPS-2": (lambda: physics.PhysicsClassifier(), P2 + ["scene"]),
           "CPS-3": (lambda: physics.PhysicsClassifier(features=P3), P3 + ["scene"])}
    if "CLIP" in emb_cols:
        cc = emb_cols["CLIP"]
        reg["CLIP probe"] = (lambda: learned.ProbeClassifier(cc), cc + ["scene"])
        reg["CPS-2 + CLIP"] = (lambda: learned.FusionClassifier({"phys": lambda: physics.PhysicsClassifier(), "clip": lambda: learned.ProbeClassifier(cc)}),
                               P2 + cc + ["scene"])
        reg["CPS-3 + CLIP"] = (lambda: learned.FusionClassifier({"phys": lambda: physics.PhysicsClassifier(features=P3),
                                                                  "clip": lambda: learned.ProbeClassifier(cc)}), P3 + cc + ["scene"])
    return reg


def protocols(pf: pd.DataFrame, lofo: bool = True) -> dict:
    have = set(pf["var"].unique())
    old = [k for k in OLD_KINDS if k in have]
    wide = [k for k in weak.NZ_KINDS if k in have]
    P = {"clean": dict(kinds=[], real_ops=False), "sym3": dict(kinds=old, real_ops=True),
         "sym-wide": dict(kinds=wide, real_ops=True), "rat-wide": dict(kinds=wide, real_ops=False)}
    if lofo:
        for f in weak.ALL_FAMS:
            P[f"lofo-{f}"] = dict(kinds=[k for k in wide if f not in weak.PARTS[k]], real_ops=True)
    return {k: v for k, v in P.items() if k == "clean" or v["kinds"]}


def score(pf: pd.DataFrame, protos: dict, reg: dict, split: str, n_jobs: int | None, log=print) -> dict:
    """(protocol, detector) -> scores over pf. split 'heldout': train on dev, score test. 'dev-cv': scene-grouped 5-fold inside dev
    (out-of-fold scores of the dev scenes, used to fix thresholds before the test scenes are touched)."""
    test = pf.test.to_numpy()
    if split == "heldout":
        folds = [(~test, test)]
    else:
        fold = pf.scene.map(lambda s: zlib.crc32(s.encode()) % 5).to_numpy()
        folds = [(~test & (fold != f), ~test & (fold == f)) for f in range(5)]
    out = {(p, d): np.full(len(pf), np.nan) for p in protos for d in reg}
    jobs, where = [], []
    for tr, te in folds:
        train, tst, idx = pf[tr], pf[te], np.where(te)[0]
        for pname, pr in protos.items():
            d_, y_ = an._train_set(train, False, pr["kinds"], pr["real_ops"])
            if len(np.unique(y_)) < 2:
                continue
            for dn, (make, cols) in reg.items():
                jobs.append((make, d_[cols], y_, tst[cols]))
                where.append(((pname, dn), idx))
    t = time.time()
    log(f"[weak:analyse] fitting {len(jobs)} models ({split}) ...")
    for (key, idx), sc in zip(where, an._fit_many(jobs, n_jobs)):
        out[key][idx] = sc
    log(f"[weak:analyse]   done in {time.time() - t:.0f}s")
    return out


def _contrast(df, neg, pos, B, seed, restrict=True):
    N, P = df[df.cell == neg], df[df.cell == pos]
    if restrict:
        N = N[N.scene.isin(set(P.scene))]
    if len(N) < 8 or len(P) < 8:
        return None
    return bs.Contrast(N, P, B, seed)


def _row(C, s):
    a = C.auc(s)
    if not np.isfinite(a.point):
        return None, None
    lo, hi = a.ci()
    return a, dict(n_real=int(np.isfinite(np.asarray(s)[C.ni]).sum()), n_fake=int(np.isfinite(np.asarray(s)[C.pi]).sum()), auc=a.point, lo=lo, hi=hi)


def run(store=None, B: int = 10000, seed: int = 0, n_jobs: int | None = None, log=print, lofo: bool = True) -> str:
    res = pipeline.results_dir()
    out = res / "weak"
    out.mkdir(exist_ok=True)
    pf, emb_cols = load(res, log)
    reg = registry(emb_cols)
    protos = protocols(pf, lofo)
    test = pf[pf.test]
    cells = set(pf.cell)
    log(f"[weak:analyse] detectors {list(reg)}; protocols {list(protos)}")
    H = score(pf, protos, reg, "heldout", n_jobs, log)
    D = score(pf, protos, reg, "dev-cv", n_jobs, log)
    files, text, stats = [], ["# Weakness experiments\n", f"B = {B} scene-bootstrap resamples; trained on dev scenes, evaluated on {test.scene.nunique()} test scenes.\n"], {}

    def save(df, name, title, note="", cols=None, rnd=3):
        p = out / f"{name}.csv"
        df.to_csv(p, index=False)
        files.append(p)
        show = df if cols is None else df[cols]
        text.append(f"\n## {title}\n\n{note}\n\n```\n{show.round(rnd).to_string(index=False) if len(df) else '(nothing)'}\n```\n")

    # ---- 1. attacker ladder + the main run's settings ----------------------------------------------------------------------
    rows = []
    settings = [("clean", "RC", "{f}", True), ("crf35", "RC@crf35", "{f}@crf35", False), ("main attacker (blur+noise+exposure)", "RC", "{f}@" + FULL_OLD, True)]
    settings += [(f"ladder {k.split('-')[1]}", "RC", "{f}@" + k, True) for k in weak.LADDER]
    for fake in ("GR", "RR"):
        for label, neg, pos, restrict in settings:
            C = _contrast(test, neg, pos.format(f=fake), B, seed, restrict=restrict and label != "clean")
            if C is None:
                continue
            for (p, d), s in H.items():
                a, r = _row(C, s)
                if r:
                    rows.append(dict(fake=fake, setting=label, protocol=p, detector=d, **r))
    lad = pd.DataFrame(rows, columns=["fake", "setting", "protocol", "detector", "n_real", "n_fake", "auc", "lo", "hi"])
    keep = lad[lad.protocol.isin(PROTOCOL_ORDER)]
    save(keep, "ladder", "Detectability versus attacker effort (AUC, RC vs fake)",
         "L1 = main attacker re-implemented with optical-flow motion; L2 adds optics + rolling shutter; L3 adds the ISP chain after the sensor noise "
         "(noise matched post-ISP). Every rung is calibrated to its paired real clip.",
         cols=["fake", "setting", "protocol", "detector", "n_fake", "auc", "lo", "hi"], rnd=3)
    stats["ladder"] = keep

    # ---- 2. leave-one-family-out ------------------------------------------------------------------------------------------------
    thr = {}
    dev_rc = np.where((pf.cell == "RC").to_numpy() & ~pf.test.to_numpy())[0]
    for key, s in D.items():
        v = s[dev_rc]
        v = v[np.isfinite(v)]
        thr[key] = float(np.quantile(v, 0.95, method="higher")) if len(v) >= 20 else np.nan
    rows, frows = [], []
    for fam in weak.ALL_FAMS:
        kind = FAM_KIND[fam]
        if kind not in set(pf["var"]):
            continue
        for fake in ("GR", "RR"):
            Ca = _contrast(test, "RC", f"{fake}@{kind}", B, seed)
            Cs = _contrast(test, f"RC@{kind}", f"{fake}@{kind}", B, seed)
            for (p, d), s in H.items():
                if p.startswith("lofo-") and p != f"lofo-{fam}":
                    continue                                              # a lofo model is only evaluated on its own held-out family
                for ctype, C in (("attack", Ca), ("symmetric", Cs)):
                    if C is not None:
                        a, r = _row(C, s)
                        if r:
                            rows.append(dict(family=fam, fake=fake, contrast=ctype, protocol=p, detector=d, **r))
        # false alarms: real clips that went through the operator, at the clean-real 95th percentile of the same model's out-of-fold dev scores
        x = test[test.cell == f"RC@{kind}"]
        for (p, d), s in H.items():
            if p.startswith("lofo-") and p != f"lofo-{fam}":
                continue
            t = thr.get((p, d), np.nan)
            v = s[x.index]
            v = v[np.isfinite(v)]
            if len(v) and np.isfinite(t):
                k = int((v > t).sum())
                lo, hi = bs.wilson(k, len(v))
                frows.append(dict(family=fam, protocol=p, detector=d, flagged=k, n=len(v), fpr=k / len(v), lo=lo, hi=hi, threshold=t))
    lo_df = pd.DataFrame(rows, columns=["family", "fake", "contrast", "protocol", "detector", "n_real", "n_fake", "auc", "lo", "hi"])
    fp_df = pd.DataFrame(frows, columns=["family", "protocol", "detector", "flagged", "n", "fpr", "lo", "hi", "threshold"])
    save(lo_df, "lofo", "Leave-one-family-out: does training on other operators help against an UNSEEN operator family?",
         "'attack' = real vs fakes that went through family F; 'symmetric' = real+F vs fake+F (the operator carries no label). A lofo-<F> model "
         "has seen every operator kind except those containing F. Rows with protocol sym-wide have seen F (reference).",
         cols=["family", "fake", "contrast", "protocol", "detector", "auc", "lo", "hi"])
    save(fp_df, "lofo_false_alarms", "False alarms on REAL clips processed by family F (threshold = 95th pct of the model's out-of-fold clean dev real scores)",
         "RAT-style training flags processed real clips; symmetric training should not.", cols=["family", "protocol", "detector", "flagged", "n", "fpr", "lo", "hi"])

    # mean over the nine families, with a CI (same scenes in every family so the bootstrap resamples are shared)
    summ, srows = {}, []
    for fake in ("GR", "RR"):
        for ctype, negf in (("attack", lambda k: "RC"), ("symmetric", lambda k: f"RC@{k}")):
            fams = [f for f in weak.ALL_FAMS if FAM_KIND[f] in set(pf["var"])]
            sets = []
            for f in fams:
                k = FAM_KIND[f]
                sets.append(set(test[test.cell == f"{fake}@{k}"].scene) & set(test[test.cell == negf(k)].scene))
            common = set.intersection(*sets) if sets else set()
            if len(common) < 10:
                continue
            cs = {}
            for f in fams:
                k = FAM_KIND[f]
                N, P = test[(test.cell == negf(k)) & test.scene.isin(common)], test[(test.cell == f"{fake}@{k}") & test.scene.isin(common)]
                cs[f] = bs.Contrast(N, P, B, seed)
            for d in reg:
                for p in PROTOCOL_ORDER + ["lofo"]:
                    parts = []
                    for f in fams:
                        key = (f"lofo-{f}" if p == "lofo" else p, d)
                        if key in H:
                            parts.append(cs[f].auc(H[key]))
                    if len(parts) != len(fams):
                        continue
                    tot = parts[0]
                    for q in parts[1:]:
                        tot = tot + q
                    mean = tot / len(parts)
                    lo, hi = mean.ci()
                    summ[(fake, ctype, d, p)] = mean
                    srows.append(dict(fake=fake, contrast=ctype, detector=d, protocol=("lofo (family unseen)" if p == "lofo" else p), families=len(parts),
                                      n_scenes=len(common), mean_auc=mean.point, lo=lo, hi=hi))
    sm = pd.DataFrame(srows, columns=["fake", "contrast", "detector", "protocol", "families", "n_scenes", "mean_auc", "lo", "hi"])
    save(sm, "lofo_mean", "Mean AUC over the nine operator families", "The row 'lofo (family unseen)' is the honest transfer number.",
         cols=["fake", "contrast", "detector", "protocol", "families", "mean_auc", "lo", "hi"])

    # ---- 3. paired comparisons (Holm within family) ------------------------------------------------------------------------------
    prow = []
    for (fake, ctype, d, p), st in summ.items():
        if p == "lofo":
            for ref in ("clean", "sym3"):
                if (fake, ctype, d, ref) in summ:
                    diff = st - summ[(fake, ctype, d, ref)]
                    lo, hi = diff.ci()
                    prow.append(dict(family="LOFO: unseen-family training minus " + ref, fake=fake, contrast=ctype, detector=d, a=st.point,
                                     b=summ[(fake, ctype, d, ref)].point, diff=diff.point, lo=lo, hi=hi, p=diff.p()))
        if p == "sym-wide" and (fake, ctype, d, "sym3") in summ:
            diff = st - summ[(fake, ctype, d, "sym3")]
            lo, hi = diff.ci()
            prow.append(dict(family="seen: sym-wide minus sym3", fake=fake, contrast=ctype, detector=d, a=st.point,
                             b=summ[(fake, ctype, d, "sym3")].point, diff=diff.point, lo=lo, hi=hi, p=diff.p()))
    for fake in ("GR", "RR"):                                             # CPS-3 vs CPS-2 on every ladder / main setting, clean and sym-wide training
        for label, neg, pos, restrict in settings:
            C = _contrast(test, neg, pos.format(f=fake), B, seed, restrict=restrict and label != "clean")
            if C is None:
                continue
            for p in ("clean", "sym-wide"):
                for a_, b_ in (("CPS-3", "CPS-2"), ("CPS-3 + CLIP", "CPS-2 + CLIP")):
                    if (p, a_) in H and (p, b_) in H:
                        sa, sb = C.pair(H[(p, a_)], H[(p, b_)])
                        diff = sa - sb
                        lo, hi = diff.ci()
                        prow.append(dict(family=f"J cues: {a_} minus {b_}", fake=fake, contrast=f"{label} [{p}]", detector=a_, a=sa.point, b=sb.point,
                                         diff=diff.point, lo=lo, hi=hi, p=diff.p()))
    pr = pd.DataFrame(prow, columns=["family", "fake", "contrast", "detector", "a", "b", "diff", "lo", "hi", "p"])
    if len(pr):
        pr["p_holm"] = np.nan
        for _, idx in pr.groupby("family").groups.items():
            pr.loc[idx, "p_holm"] = bs.holm(pr.loc[idx, "p"].to_numpy())
        pr["verdict"] = np.where(pr.p_holm < 0.05, np.where(pr["diff"] > 0, "A better", "B better"), "no significant difference")
    save(pr, "paired", "Paired, scene-clustered AUC differences (A minus B; Holm within each comparison family)",
         cols=["family", "fake", "contrast", "detector", "a", "b", "diff", "lo", "hi", "p", "p_holm", "verdict"] if len(pr) else None, rnd=4)
    if not len(pr):
        pr["p_holm"], pr["verdict"] = [], []

    # ---- 4. plain-language findings ----------------------------------------------------------------------------------------------
    text.append("\n## Findings (generated from the tables above)\n")
    try:
        if len(sm):
            for fake in ("GR", "RR"):
                for d in ([x for x in ("CPS-3 + CLIP", "CPS-2 + CLIP", "CPS-2") if x in reg][:2]):
                    g = lambda p: sm[(sm.fake == fake) & (sm.contrast == "symmetric") & (sm.detector == d) & (sm.protocol == p)]     # noqa: E731
                    vals = {p: (float(g(p).mean_auc.iloc[0]) if len(g(p)) else float("nan")) for p in ("clean", "sym3", "sym-wide", "lofo (family unseen)")}
                    text.append(f"- {fake}, {d}: mean symmetric AUC over 9 families: clean-trained {vals['clean']:.3f}, SYM(blur/noise/exposure) {vals['sym3']:.3f}, "
                                f"SYM-wide (family seen) {vals['sym-wide']:.3f}, SYM-wide with the family held out {vals['lofo (family unseen)']:.3f}.")
        if len(lad):
            for fake in ("GR", "RR"):
                for d in ([x for x in ("CPS-2 + CLIP", "CPS-2") if x in reg]):
                    x = lad[(lad.fake == fake) & (lad.detector == d) & (lad.protocol == "clean")].set_index("setting").auc
                    t = ", ".join(f"{k} {v:.3f}" for k, v in x.items() if k.startswith(("clean", "ladder", "main")))
                    text.append(f"- {fake}, clean-trained {d} along the ladder: {t}.")
        if len(pr):
            j = pr[pr.family.str.startswith("J cues")]
            if len(j):
                text.append(f"- Joint cues: {int((j.p_holm < 0.05).sum())} of {len(j)} CPS-3-vs-CPS-2 comparisons are significant after Holm "
                            f"(best {j['diff'].max():+.3f}, worst {j['diff'].min():+.3f}).")
    except Exception as e:                                                  # noqa: BLE001 - the tables are the result; the prose is a convenience
        text.append(f"(could not write the prose summary: {e!r})")
    (out / "summary.md").write_text("\n".join(text), encoding="utf-8")
    files.append(out / "summary.md")
    _figure(lad, out, files)
    if store is not None:
        for p in files:
            store.stage(p)
        store.flush("c44: weakness analysis")
    return "\n".join(text)


def _figure(lad: pd.DataFrame, out, files) -> None:
    """Detectability versus attacker effort (optional: needs matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = ["clean", "main attacker (blur+noise+exposure)", "ladder L1", "ladder L2", "ladder L3"]
        order = [o for o in order if o in set(lad.setting)]
        det = [d for d in ("CPS-2", "CPS-3", "CLIP probe", "CPS-2 + CLIP") if d in set(lad.detector)]
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        for a, fake in zip(ax, ("GR", "RR")):
            for d in det:
                for p, ls in (("clean", "-"), ("sym-wide", "--")):
                    x = lad[(lad.fake == fake) & (lad.detector == d) & (lad.protocol == p)].set_index("setting").reindex(order)
                    if x.auc.notna().any():
                        a.plot(range(len(order)), x.auc, ls, marker="o", label=f"{d} [{p}]")
            a.axhline(0.5, color="k", lw=0.6)
            a.set_xticks(range(len(order)))
            a.set_xticklabels([o.replace("main attacker (blur+noise+exposure)", "main").replace("ladder ", "") for o in order])
            a.set_title(f"{fake}: AUC vs attacker effort")
        ax[0].set_ylabel("AUC (real vs fake)")
        ax[1].legend(fontsize=6, ncol=2)
        fig.tight_layout()
        p = out / "fig_ladder.png"
        fig.savefig(p, dpi=150)
        files.append(p)
    except Exception as e:                                                    # noqa: BLE001
        print(f"[weak:analyse] figure skipped: {e!r}")
