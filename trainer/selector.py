# selector.py — the meta-controller that learns to choose a predictor.
#
# Two selectors behind one interface:
#
#   EwmaSelector     — EWMA surprise + epsilon-greedy (notebook 11 baseline).
#   LearnedSelector  — a linear scoring policy conditioned on trajectory
#                      features, updated online from full-feedback surprise.
#
# Both implement `choose(features=None)` and `update(surprises, features=None)`,
# so the loop can drive either one unchanged.

from __future__ import annotations

from .protocol import Selection

import numpy as np


def selector_features(hist: np.ndarray, ewma: dict[str, float], period=None) -> np.ndarray:
    """The feature vector a learned selector conditions on.

    [current BSS (d), ΔBSS (d), detected period (1), per-predictor EWMA
    surprise (k), sin/cos phase (2), bias (1)].
    """
    h = np.asarray(hist, dtype=float)
    cur = h[-1]
    dm = cur - h[-2] if len(h) >= 2 else np.zeros_like(cur)
    P = float(period) if period is not None else 0.0
    if P >= 2.0:
        ph = 2.0 * np.pi * len(h) / P
        phase_feats = [np.sin(ph), np.cos(ph)]
    else:
        phase_feats = [0.0, 0.0]
    ew = [float(ewma[n]) for n in ewma]
    return np.concatenate([cur, dm, [P], ew, phase_feats, [1.0]])


class EwmaSelector:
    """EWMA surprise per predictor + epsilon-greedy choice.

    `choose()` is called once per closed-loop step (after the first step, which
    has no prediction). `update(surprises)` is called after the actual state is
    observed, so the next choice uses only past performance.
    """

    def __init__(
        self,
        names,
        eps: float = 0.1,
        decay: float = 0.8,
        warmup: int = 2,
        seed: int = 0,
    ):
        self.names = list(names)
        self.eps = eps
        self.decay = decay
        self.warmup = warmup
        self.ewma = {n: 1.0 for n in self.names}
        self.counts = {n: 0 for n in self.names}
        self.steps = 0
        self.last_choice = None
        self.rng = np.random.default_rng(seed)

    def choose(self, features=None) -> str:
        if self.steps < self.warmup or self.rng.random() < self.eps:
            choice = str(self.rng.choice(self.names))
        else:
            choice = min(self.names, key=lambda n: self.ewma[n])
        self.steps += 1
        self.last_choice = choice
        self.counts[choice] += 1
        return choice

    def update(self, surprises: dict[str, float], features=None) -> None:
        for n in self.names:
            s = float(surprises.get(n, 0.0))
            self.ewma[n] = self.decay * self.ewma[n] + (1.0 - self.decay) * s

    def selection(self) -> Selection:
        return Selection(
            weights=dict(self.ewma),
            choice=self.last_choice or "",
            epsilon=self.eps,
        )


class LearnedSelector:
    """A linear scoring policy conditioned on trajectory features.

    For each predictor `a` a linear model `score_a = w_a · φ(t)` is fit online
    (ridge) to predict the reward `-surprise_a(t)`. Because the loop observes
    the surprise of *every* predictor each step, this is full-feedback online
    learning; the choice is epsilon-greedy over the predicted scores, with a
    warm-up of random choices.
    """

    def __init__(
        self,
        names,
        lam: float = 1e-3,
        eps: float = 0.1,
        warmup: int = 4,
        decay: float = 0.8,
        seed: int = 0,
    ):
        self.names = list(names)
        self.lam = lam
        self.eps = eps
        self.warmup = warmup
        self.decay = decay
        self.ewma = {n: 1.0 for n in self.names}
        self.counts = {n: 0 for n in self.names}
        self.steps = 0
        self.last_choice = None
        self.rng = np.random.default_rng(seed)
        self._X: list[np.ndarray] = []
        self._Y: dict[str, list[float]] = {n: [] for n in self.names}
        self.W: np.ndarray | None = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None

    def _scale(self, x: np.ndarray) -> np.ndarray:
        if self._mu is None:
            return x
        return (x - self._mu) / (self._sd + 1e-9)

    def choose(self, features: np.ndarray) -> str:
        self.steps += 1
        if self.steps <= self.warmup or self.W is None or self.rng.random() < self.eps:
            choice = str(self.rng.choice(self.names))
        else:
            xs = self._scale(np.asarray(features, dtype=float))
            scores = xs @ self.W
            choice = self.names[int(np.argmax(scores))]
        self.last_choice = choice
        self.counts[choice] += 1
        return choice

    def update(self, surprises: dict[str, float], features: np.ndarray) -> None:
        # Recent-performance features for the next step.
        for n in self.names:
            s = float(surprises.get(n, 0.0))
            self.ewma[n] = self.decay * self.ewma[n] + (1.0 - self.decay) * s
        # Accumulate a training pair (features used for the choice -> rewards).
        x = np.asarray(features, dtype=float)
        self._X.append(x)
        for n in self.names:
            self._Y[n].append(-float(surprises.get(n, 0.0)))   # reward = -surprise
        # Refit the scoring model on all history (cheap for the loop sizes).
        X = np.array(self._X)
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0)
        Xs = (X - self._mu) / (self._sd + 1e-9)
        Y = np.array([self._Y[n] for n in self.names]).T
        XtX = Xs.T @ Xs + self.lam * np.eye(Xs.shape[1])
        self.W = np.linalg.solve(XtX, Xs.T @ Y)

    def selection(self) -> Selection:
        return Selection(
            weights=dict(self.ewma),
            choice=self.last_choice or "",
            epsilon=self.eps,
        )
