"""
model_utils.py
--------------
Model wrapper classes. Kept in their own module so the pickled model in
models/credit_score_model.pkl can be loaded from any script or notebook.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression


class RankEnsemble:
    """Blend of several fitted models: half mean percentile rank, half mean probability."""

    def __init__(self, models, features):
        self.models, self.features = models, features

    def predict_proba(self, X):
        X = X[self.features]
        probs = [m.predict_proba(X)[:, 1] for m in self.models.values()]
        ranks = [rankdata(p) / len(X) for p in probs]
        score = 0.5 * np.mean(ranks, axis=0) + 0.5 * np.mean(probs, axis=0)
        return np.vstack([1 - score, score]).T


class CalibratedModel:
    """
    Platt-scales a fitted model so predicted probabilities match the real
    bad rate. Class weighting during training inflates raw probabilities;
    calibration fixes the scale without changing the ranking (so Gini and
    rank ordering are unchanged).
    """

    def __init__(self, base, features):
        self.base, self.features = base, features
        self.platt = LogisticRegression(C=1e6, max_iter=1000)

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit_calibration(self, oof_prob, y):
        self.platt.fit(self._logit(oof_prob), y)
        return self

    def predict_proba(self, X):
        raw = self.base.predict_proba(X[self.features])[:, 1]
        return self.platt.predict_proba(self._logit(raw))
