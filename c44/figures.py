"""Paper figures from the analysis CSVs (needs matplotlib; run locally: `python -m c44 figures --dir <results/analysis>`).

  fig_main.pdf      held-out AUC of every detector, clean and recompressed, generated worlds vs. real-scene renders
  fig_attacks.pdf   CPS-2 under camera-aware attacks, undefended and with attack-aware training (3 defender assumptions)
  fig_families.pdf  which cue family carries the signal
Error bars are the 95% scene-bootstrap intervals of the tables.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_AUC = re.compile(r"([0-9.]+) \[([0-9.]+),([0-9.]+)\]")
ATTACK_NAMES = {"aeadapt": "exposure\njitter", "blurcal": "blur", "noiseadapt": "noise", "blurcal+noiseadapt": "blur\n+ noise",
                "blurcal+noiseadapt+aeadapt": "blur + noise\n+ exposure"}


def parse_auc(s) -> tuple[float, float, float]:
    m = _AUC.match(s) if isinstance(s, str) else None
    return (float(m[1]), float(m[2]), float(m[3])) if m else (np.nan, np.nan, np.nan)


def _bars(ax, x, vals, width, label, color):
    v = np.array([parse_auc(s) for s in vals], dtype=float)
    ax.bar(x, np.nan_to_num(v[:, 0]), width, label=label, color=color,
           yerr=[np.nan_to_num(v[:, 0] - v[:, 1]), np.nan_to_num(v[:, 2] - v[:, 0])], capsize=2, error_kw={"lw": 0.8})


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    return plt


def fig_main(main: pd.DataFrame, path: Path) -> None:
    plt = _style()
    dets = [("univfd", "UnivFD"), ("npr", "NPR"), ("consistency", "Consistency"), ("CPS-1 (one-class)", "CPS-1"),
            ("CPS-2 one-class", "CPS-2 one-class"), ("CPS-2 supervised", "CPS-2")]
    rows = [(f, s) for f in ("GR", "RR") for s in ("clean", "crf28", "crf35")]
    sel = []
    for f, s in rows:
        r = main[(main.setting == s) & main.pair.str.endswith(f"vs {f}@{s}" if s != "clean" else f"vs {f}")]
        if len(r):
            sel.append((f, s, r.iloc[0]))
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    w = 0.8 / len(dets)
    colors = ["#9aa0a6", "#b7bcc2", "#cfd3d7", "#d9a441", "#7aa6c2", "#1f5f8b"]
    x = np.arange(len(sel))
    for k, (col, name) in enumerate(dets):
        _bars(ax, x + (k - (len(dets) - 1) / 2) * w, [r.get(col) for _, _, r in sel], w, name, colors[k])
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    ax.set_xticks(x, [f"{'Generated' if f == 'GR' else 'Real-scene'}\n{'clean' if s == 'clean' else s.upper().replace('CRF', 'CRF ')}" for f, s, _ in sel])
    ax.set_ylabel("AUC (real vs fake)")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_attacks(table: pd.DataFrame, path: Path) -> None:
    plt = _style()
    cols = [("CPS-2 supervised", "no attack training", "#1f5f8b"), ("CPS-2 + aug (unseen components)", "aug: components unseen", "#c0504d"),
            ("CPS-2 + aug (unseen combination)", "aug: new combination", "#d9a441"), ("CPS-2 + aug (attack seen)", "aug: attack seen", "#5b9a68")]
    kinds = [k for k in ATTACK_NAMES if (table.setting == k).any()]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True, constrained_layout=True)
    for ax, f, title in zip(axes, ("GR", "RR"), ("Generated worlds", "Real-scene renders")):
        w = 0.8 / len(cols)
        x = np.arange(len(kinds))
        for k, (col, name, color) in enumerate(cols):
            vals = [table[(table.setting == a) & table.pair.str.endswith(f"vs {f}@{a}")].iloc[0].get(col) for a in kinds]
            _bars(ax, x + (k - (len(cols) - 1) / 2) * w, vals, w, name, color)
        ax.axhline(0.5, color="k", lw=0.6, ls="--")
        ax.set_xticks(x, [ATTACK_NAMES[a] for a in kinds])
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("AUC (real vs attacked fake)")
    axes[0].set_ylim(0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="outside upper center", frameon=False, fontsize=7)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_families(fams: pd.DataFrame, path: Path) -> None:
    plt = _style()
    only = fams[fams.features.str.startswith("only")]
    names = [f.replace("only ", "") for f in only.features]
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    y = np.arange(len(names))
    for k, (col, label, color) in enumerate((("GR", "Generated", "#1f5f8b"), ("RR", "Real-scene", "#d9a441"))):
        v = np.array([parse_auc(s) for s in only[col]], dtype=float)
        ax.barh(y + (k - 0.5) * 0.38, np.nan_to_num(v[:, 0]), 0.38, label=label, color=color,
                xerr=[np.nan_to_num(v[:, 0] - v[:, 1]), np.nan_to_num(v[:, 2] - v[:, 0])], capsize=2, error_kw={"lw": 0.8})
    ax.axvline(0.5, color="k", lw=0.6, ls="--")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("AUC with that family only")
    ax.set_xlim(0.3, 1.0)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def make_figures(analysis_dir, out_dir) -> list[Path]:
    a, o = Path(analysis_dir), Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    made = []
    main = pd.read_csv(a / "cps2_main.csv")
    fig_main(main, o / "fig_main.pdf"); made.append(o / "fig_main.pdf")
    att_src = pd.read_csv(a / "cps2_cv.csv") if (a / "cps2_cv.csv").exists() else main        # the larger cross-validated sample
    fig_attacks(att_src, o / "fig_attacks.pdf"); made.append(o / "fig_attacks.pdf")
    if (a / "cps2_families.csv").exists():
        fig_families(pd.read_csv(a / "cps2_families.csv"), o / "fig_families.pdf"); made.append(o / "fig_families.pdf")
    return made
