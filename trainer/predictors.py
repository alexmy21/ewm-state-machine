# predictors.py — the predictor portfolio behind one interface.
#
# Every predictor is a pure function of the trajectory history (an (t, d)
# numpy array of BSS rows) and returns the predicted next row. The portfolio
# is the reference behaviour extracted from notebooks 10/11.

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def dft_period(X: np.ndarray, tol_frac: float = 0.05) -> Optional[float]:
    """Dominant period of a (t, d) series via its average power spectrum.

    Harmonics can carry more power than the fundamental, so the strongest bin
    is not the period. Instead: take every bin with at least `tol_frac` of the
    total power and return the period implied by their gcd — the fundamental
    of the harmonic set. Returns None when no period is clearly present.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    t = X.shape[0]
    Xc = X - X.mean(axis=0)
    p = np.abs(np.fft.rfft(Xc, axis=0)) ** 2
    p[0] = 0.0
    total = p.sum(axis=1)          # power summed over the d dimensions
    k_hi = t // 2
    if k_hi < 2:
        return None
    total_power = float(total[1:].sum())
    if total_power <= 0:
        return None
    thr = tol_frac * total_power
    k_sig = [k for k in range(2, k_hi + 1) if total[k] >= thr]
    if not k_sig:
        return None
    g = 0
    for k in k_sig:
        g = k if g == 0 else math.gcd(g, k)
    if g < 2:
        return None
    return t / g


class Predictor:
    """Base predictor: `history -> prediction` (a (d,) numpy row)."""

    name = "base"

    def predict(self, history: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PersistencePredictor(Predictor):
    name = "persistence"

    def predict(self, history: np.ndarray) -> np.ndarray:
        h = np.asarray(history, dtype=float)
        return h[-1].copy()


class LinearPredictor(Predictor):
    name = "linear"

    def predict(self, history: np.ndarray) -> np.ndarray:
        h = np.asarray(history, dtype=float)
        if len(h) < 2:
            return h[-1].copy()
        return 2.0 * h[-1] - h[-2]


class DftPeriodicPredictor(Predictor):
    name = "dft-periodic"

    def predict(self, history: np.ndarray) -> np.ndarray:
        h = np.asarray(history, dtype=float)
        t = h.shape[0]
        if t < 8:
            return h[-1].copy()
        P = dft_period(h)
        if P is None:
            return h[-1].copy()
        P = max(2, min(int(round(P)), t - 1))
        idx = [n for n in range(t) if n % P == t % P]
        if idx:
            return h[idx].mean(axis=0)
        return h[-1].copy()


class RidgePredictor(Predictor):
    """Online ridge fit over `[x(t), Δx(t), sin(2πt/P), cos(2πt/P)]`."""

    name = "ridge"

    def __init__(self, lam: float = 1e-3):
        self.lam = lam

    def predict(self, history: np.ndarray) -> np.ndarray:
        h = np.asarray(history, dtype=float)
        if h.ndim == 1:
            h = h[:, None]
        t, d = h.shape
        if t < 4:
            return h[-1].copy()
        P = dft_period(h)
        if P is None:
            P = float(t)
        P = max(2.0, P)

        def feats(i: int) -> np.ndarray:
            x = h[i]
            dm = h[i] - h[i - 1] if i > 0 else np.zeros(d)
            ph = 2.0 * np.pi * i / P
            return np.concatenate([x, dm, [np.sin(ph), np.cos(ph)]])

        X = np.array([feats(i) for i in range(t - 1)])
        Y = h[1:]
        XtX = X.T @ X + self.lam * np.eye(X.shape[1])
        W = np.linalg.solve(XtX, X.T @ Y)
        return feats(t - 1) @ W


def default_portfolio() -> dict[str, Predictor]:
    """The notebook-11 portfolio."""
    portfolio: list[Predictor] = [
        PersistencePredictor(),
        LinearPredictor(),
        DftPeriodicPredictor(),
        RidgePredictor(),
    ]
    return {p.name: p for p in portfolio}
