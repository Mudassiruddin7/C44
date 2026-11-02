"""Learned detectors on frozen backbone embeddings, and their fusion with the capture-physics classifier (CPS-2).

ProbeClassifier   standardise + L2 logistic regression on an embedding (the UnivFD recipe, retrained on the benchmark's dev scenes).
FusionClassifier  combines member scores. Scores are put on a common scale with the OUT-OF-FOLD scores of the real training
                  clips (scene-grouped folds), because in-sample scores of a high-dimensional probe are over-confident.
                  mode 'mean': sum of z-scores (nothing to fit);  mode 'stack': logistic regression on the z-scores.
"""

from __future__ import annotations

import zlib

import numpy as np

from . import physics


class ProbeClassifier:
    def __init__(self, features: list[str], C: float = 0.05):
        self.features, self.C = list(features), C

    def fit(self, df, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        X = df[self.features].to_numpy(np.float32)
        self.sc = StandardScaler().fit(X)
        self.lr = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced").fit(self.sc.transform(X), np.asarray(y))
        return self

    def score(self, df) -> np.ndarray:
        return self.lr.decision_function(self.sc.transform(df[self.features].to_numpy(np.float32)))


def scene_folds(scenes, k: int = 5, salt: str = "fusion") -> np.ndarray:
    return np.array([zlib.crc32((s + salt).encode()) % k for s in scenes])


class FusionClassifier:
    def __init__(self, members: dict, mode: str = "mean", inner_folds: int = 5):
        """members: name -> zero-argument factory of an object with fit(df, y) / score(df)."""
        self.factories, self.mode, self.k = members, mode, inner_folds

    def fit(self, df, y):
        y = np.asarray(y)
        fold = scene_folds(df.scene.to_numpy(), self.k)
        oof = np.full((len(df), len(self.factories)), np.nan)
        for f in range(self.k):
            tr, te = fold != f, fold == f
            if len(np.unique(y[tr])) < 2 or not te.any():
                continue
            for j, make in enumerate(self.factories.values()):
                oof[te, j] = make().fit(df[tr], y[tr]).score(df[te])
        self.models = [make().fit(df, y) for make in self.factories.values()]
        real = (y == 0) & np.isfinite(oof).all(1)
        self.mu = np.nanmean(oof[real], 0) if real.any() else np.zeros(oof.shape[1])
        self.sd = np.nanstd(oof[real], 0) + 1e-9 if real.any() else np.ones(oof.shape[1])
        if self.mode == "stack":
            from sklearn.linear_model import LogisticRegression
            ok = np.isfinite(oof).all(1)
            self.stack = LogisticRegression(C=1.0, class_weight="balanced").fit((oof[ok] - self.mu) / self.sd, y[ok])
        return self

    def _z(self, df) -> np.ndarray:
        return (np.stack([m.score(df) for m in self.models], 1) - self.mu) / self.sd

    def score(self, df) -> np.ndarray:
        z = self._z(df)
        return z.sum(1) if self.mode == "mean" else self.stack.decision_function(z)


def make_models(emb_cols: dict[str, list[str]], mode: str = "mean") -> dict:
    """The detectors compared in the paper: physics only, one probe per backbone, and the physics + CLIP fusion."""
    out = {"CPS-2": lambda: physics.PhysicsClassifier()}
    for name, cols in emb_cols.items():
        out[f"{name} probe"] = (lambda c=cols: ProbeClassifier(c))
    if "CLIP" in emb_cols:
        out["CPS-2 + CLIP"] = lambda: FusionClassifier(
            {"phys": lambda: physics.PhysicsClassifier(), "clip": lambda: ProbeClassifier(emb_cols["CLIP"])}, mode=mode)
    return out
