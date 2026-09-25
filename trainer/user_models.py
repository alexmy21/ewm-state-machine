# user_models.py — per-user typed-head training and registry.
#
# Enterprise support: the ewm-sm apparatus stays shared and vocabulary-agnostic;
# each user gets a small typed-head router trained on their own analytical
# context (their queries, their codebook, their structural trajectory).
#
#   ctx   = UserContext(user_id="acme-finance", queries=..., vocab=...)
#   data  = collect_user_dataset(ctx, adapter, teacher, ewm)      # loops -> logs
#   bundle, metrics = train_user_head(ctx, data.routing_log)       # distill
#   register_user_model(ctx, bundle, metrics)                      # registry
#   router = UserHeadRouter.load("acme-finance")                   # serve
#
# Registry layout (one directory per user):
#
#   ~/.cache/ewm-models/<user_id>/
#   ├── config.json        user id, queries, options, feature spec
#   ├── trajectory.json    open-loop metadata
#   ├── routing_log.jsonl  (state, options, decision) per step — distillation set
#   ├── typed_head.pt      the per-user model
#   └── metrics.json       holdout acc/KL, label histogram, trained-at

from __future__ import annotations

import json
import os
import time
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .adapters import Adapter, DecisionRouter
from .loop import run_jev_loop, run_open_loop
from .protocol import DecisionRecord, ProbeConfig

DEFAULT_MODEL_ROOT = os.path.expanduser("~/.cache/ewm-models")
DEFAULT_OPTION_ORDER = ("llm_a", "llm_b", "llm_c")
# Series available in the structural state (windows + crossings are keyed by these).
CORE_SERIES = ("np", "rp", "dp", "residual", "dim", "spill_first")


@dataclass
class UserContext:
    """One enterprise user's analytical context: id, query bank, codebook."""

    user_id: str
    queries: list[str]
    vocab: Optional[list[str]] = None        # optional per-user codebook tokens
    options: Optional[dict[str, str]] = None  # router option descriptions
    instructions: str = "What kind of answer does this query need?"
    model_root: str = DEFAULT_MODEL_ROOT
    T: int = 16                               # teacher-loop steps for data collection
    cycles: int = 2                           # open-loop cycles
    max_new_tokens: int = 32                  # probe generation cap

    @property
    def work_dir(self) -> str:
        return os.path.join(self.model_root, self.user_id)

    def ensure(self) -> str:
        os.makedirs(self.work_dir, exist_ok=True)
        return self.work_dir


class VocabAdapter(Adapter):
    """Deterministic per-user synthetic probes over a small enterprise codebook.

    Mirrors `SyntheticAdapter`, but every emitted token comes from the user's
    own vocabulary — the enterprise setting: smaller, more consistent streams.
    """

    def __init__(
        self,
        names=("llm_a", "llm_b", "llm_c"),
        vocab: Optional[list[str]] = None,
        seed: int = 0,
        tokens_per_step: int = 24,
    ):
        self.names = tuple(names)
        self.vocab = list(vocab or [f"tid{i}" for i in range(128)])
        self.seed = seed
        self.tokens_per_step = tokens_per_step

    def run(self, cfg: ProbeConfig) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        half = max(1, self.tokens_per_step // 2)
        quarter = max(0, self.tokens_per_step // 4)
        phase = cfg.step % 8
        qhash = zlib.crc32(cfg.query.encode("utf-8")) % len(self.vocab)
        for i, n in enumerate(self.names):
            if cfg.route is not None and n != cfg.route:
                out[n] = []                      # routing mode: only one answers
                continue
            base = [self.vocab[(qhash + phase * 7 + i * 3 + j) % len(self.vocab)]
                    for j in range(half)]
            drift = [self.vocab[(cfg.step + i * 5 + j) % len(self.vocab)]
                     for j in range(quarter)]
            echo = list(cfg.memory[:4])          # memory feedback, as in SyntheticAdapter
            out[n] = base + drift + echo
        return out


# ---------------------------------------------------------------- features

def _structural_features(state: dict, option_order) -> list[float]:
    bss = state.get("bss") or {}
    drn = state.get("drn") or {}
    ring = state.get("ring") or {}
    t = float(state.get("step", 0))
    ph = 2.0 * np.pi * t / 8.0
    feats = [float(bss.get(n, 0.0)) for n in option_order]
    feats += [float(drn.get("dp", 0)), float(drn.get("rp", 0)),
              float(drn.get("np", 0)), float(drn.get("ind1", 0)),
              float(drn.get("ind3", 0))]
    feats += [float(ring.get("residual", 0)), float(ring.get("in_span", False)),
              float(ring.get("dim", 0)), float(ring.get("rotation_count", 0)),
              float(ring.get("spill_first", 0))]
    feats += [float(np.sin(ph)), float(np.cos(ph)), t / 16.0, 1.0]
    return feats


def user_features(state: dict, option_order, query_index: Optional[dict] = None) -> np.ndarray:
    """The per-user head input: structural (17) + windows (MA2/MA5) +
    crossings (x2x4/x3x5 up/down) + optional query one-hot."""
    core = [f"bss_{n}" for n in option_order] + list(CORE_SERIES)
    windows = state.get("windows") or {}
    crossings = state.get("crossings") or {}

    feats = _structural_features(state, option_order)
    for name in core:
        feats += [float(windows.get(f"{name}_ma2", 0.0)),
                  float(windows.get(f"{name}_ma5", 0.0))]
    for name in core:
        for ks, kl in ((2, 4), (3, 5)):
            for d in ("up", "down"):
                feats.append(1.0 if crossings.get(f"{name}_x{ks}x{kl}_{d}", False) else 0.0)
    if query_index is not None:
        onehot = np.zeros(len(query_index), dtype=np.float32)
        q = str(state.get("query", ""))
        if q in query_index:
            onehot[query_index[q]] = 1.0
        feats += onehot.tolist()
    return np.asarray(feats, dtype=np.float32)


def feature_dim(n_options: int, n_queries: int, use_query: bool = True) -> int:
    return (17 + (n_options + len(CORE_SERIES)) * 2 + (n_options + len(CORE_SERIES)) * 4
            + (n_queries if use_query else 0))


def _make_head(in_dim: int, n_out: int, hidden: int = 16, dropout: float = 0.4):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, n_out),
    )


# ---------------------------------------------------------------- pipeline

def collect_user_dataset(
    ctx: UserContext,
    adapter: Adapter,
    teacher: DecisionRouter,
    ewm,
    T: Optional[int] = None,
    cycles: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
):
    """Run the user's open loop + teacher closed loop and persist the logs.

    Returns (open_result, jev_result); the distillation dataset is
    `routing_log.jsonl` in the user's registry directory.
    """
    T = T or ctx.T
    cycles = cycles or ctx.cycles
    mnt = max_new_tokens or ctx.max_new_tokens
    ctx.ensure()

    open_result = run_open_loop(adapter, ewm, ctx.queries, ctx.work_dir,
                                cycles=cycles, max_new_tokens=mnt)
    result = run_jev_loop(
        adapter, teacher, ewm, open_result, ctx.queries, ctx.work_dir,
        T=T, max_new_tokens=mnt, instructions=ctx.instructions,
        options=ctx.options, structural=True, ingest_decision=True,
    )

    with open(os.path.join(ctx.work_dir, "routing_log.jsonl"), "w") as fh:
        for r in result.routing_log:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(ctx.work_dir, "trajectory.json"), "w") as fh:
        json.dump({
            "open_bss_shape": list(open_result.M_ol.shape),
            "period": open_result.period,
            "same_union": open_result.same_union,
            "names": list(adapter.names),
        }, fh, indent=2)
    return open_result, result


def train_user_head(
    ctx: UserContext,
    routing_log: list[dict],
    option_order=DEFAULT_OPTION_ORDER,
    use_query: bool = True,
    seed: int = 0,
    epochs: int = 800,
    hidden: int = 16,
    dropout: float = 0.4,
    lr: float = 2e-3,
    weight_decay: float = 3e-3,
):
    """Distill the teacher's soft labels into the user's typed head.

    Returns (bundle, metrics) where bundle holds the checkpoint + feature spec
    and metrics holds holdout acc/KL + the teacher label histogram.
    """
    import torch

    query_index = {q: i for i, q in enumerate(ctx.queries)} if use_query else None
    X = np.stack([user_features(r["state"], option_order, query_index) for r in routing_log])
    Y = np.stack([[r["decision"]["probabilities"].get(n, 0.0) for n in option_order]
                  for r in routing_log])

    rng = np.random.default_rng(seed)
    y_cls = np.argmax(Y, axis=1)
    tr, te = [], []
    for c in np.unique(y_cls):
        idx_c = np.where(y_cls == c)[0]
        rng.shuffle(idx_c)
        n_tr_c = max(1, int(0.8 * len(idx_c)))
        tr += list(idx_c[:n_tr_c])
        te += list(idx_c[n_tr_c:])
    tr, te = np.array(tr), np.array(te)
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-9
    Xs = (X - mu) / sd

    torch.manual_seed(seed)
    model = _make_head(X.shape[1], len(option_order), hidden=hidden, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr = torch.tensor(Xs[tr]); Ytr = torch.tensor(Y[tr])
    Xte = torch.tensor(Xs[te]); Yte = torch.tensor(Y[te])
    for _ in range(epochs):
        opt.zero_grad()
        logp = torch.log_softmax(model(Xtr), dim=1)
        loss = torch.mean(torch.sum(-Ytr * logp, dim=1))
        loss.backward()
        opt.step()

    with torch.no_grad():
        p_te = torch.softmax(model(Xte), dim=1).numpy()
    acc = float(np.mean(np.argmax(p_te, axis=1) == np.argmax(Y[te], axis=1)))
    kl = float(np.mean(np.sum(Y[te] * (np.log(Y[te] + 1e-9) - np.log(p_te + 1e-9)), axis=1)))

    bundle = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "mu": mu, "sd": sd,
        "option_order": list(option_order),
        "query_index": query_index or {},
        "use_query": use_query,
        "feature_dim": int(X.shape[1]),
        "hidden": hidden,
        "dropout": dropout,
    }
    metrics = {
        "n_labelled": int(len(X)),
        "holdout_acc": acc,
        "holdout_kl": kl,
        "teacher_label_histogram": Y.sum(axis=0).tolist(),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return bundle, metrics


def register_user_model(ctx: UserContext, bundle: dict, metrics: dict) -> str:
    """Write the user's model + config + metrics into the registry directory."""
    import torch

    ctx.ensure()
    config = {
        "user_id": ctx.user_id,
        "queries": ctx.queries,
        "vocab_size": len(ctx.vocab or []),
        "options": ctx.options,
        "instructions": ctx.instructions,
        "option_order": bundle["option_order"],
        "query_index": bundle["query_index"],
        "use_query": bundle["use_query"],
        "feature_dim": bundle["feature_dim"],
        "hidden": bundle["hidden"],
        "dropout": bundle["dropout"],
    }
    torch.save({"state_dict": bundle["state_dict"], "mu": bundle["mu"], "sd": bundle["sd"]},
               os.path.join(ctx.work_dir, "typed_head.pt"))
    with open(os.path.join(ctx.work_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    with open(os.path.join(ctx.work_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    return ctx.work_dir


class UserHeadRouter(DecisionRouter):
    """The per-user typed head served behind the standard DecisionRouter seam.

    `UserHeadRouter.load(user_id)` restores the registry checkpoint; `route()`
    is a single CPU forward pass over the user's structural state (windows and
    crossings included, so run the loop with `structural=True`).
    """

    def __init__(self, user_id: str, model_root: str = DEFAULT_MODEL_ROOT):
        import torch

        work_dir = os.path.join(model_root, user_id)
        with open(os.path.join(work_dir, "config.json")) as fh:
            self.config = json.load(fh)
        ckpt = torch.load(os.path.join(work_dir, "typed_head.pt"), map_location="cpu",
                          weights_only=False)
        self.model = _make_head(self.config["feature_dim"], len(self.config["option_order"]),
                                hidden=self.config["hidden"], dropout=self.config["dropout"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.mu = ckpt["mu"]
        self.sd = ckpt["sd"]
        self.option_order = self.config["option_order"]
        self.query_index = self.config.get("query_index") or {}
        self.user_id = user_id

    @staticmethod
    def load(user_id: str, model_root: str = DEFAULT_MODEL_ROOT) -> "UserHeadRouter":
        return UserHeadRouter(user_id, model_root=model_root)

    def route(self, state, options, instructions, extra_noul=None) -> DecisionRecord:
        import torch

        qi = self.query_index if self.config.get("use_query", True) else None
        x = (user_features(state, self.option_order, qi) - self.mu) / self.sd
        with torch.no_grad():
            logits = self.model(torch.tensor(x).unsqueeze(0))[0]
        p = torch.softmax(logits, dim=0).numpy()
        probs = {n: float(p[i]) for i, n in enumerate(self.option_order)}
        decision = max(probs, key=probs.get)
        return DecisionRecord(
            decision=decision,
            confidence=float(probs[decision]),
            probabilities=probs,
            decision_type="route_query",
            aux={},
            model=f"user-head:{self.user_id}",
            mock=False,
        )


def evaluate_router_on_log(router: DecisionRouter, routing_log: list[dict]) -> dict:
    """Argmax-agreement + mean KL of a router against a routing log's teacher
    labels — the cross-user evaluation tool."""
    import torch

    option_order = tuple(sorted({n for r in routing_log for n in r["decision"]["probabilities"]}))
    aggr, kls = [], []
    for r in routing_log:
        rec = router.route(r["state"], r["options"], r["instructions"])
        p_head = np.array([rec.probabilities.get(n, 0.0) for n in option_order], dtype=float)
        p_teacher = np.array([r["decision"]["probabilities"].get(n, 0.0) for n in option_order], dtype=float)
        aggr.append(int(np.argmax(p_head) == np.argmax(p_teacher)))
        kls.append(float(np.sum(p_teacher * (np.log(p_teacher + 1e-9) - np.log(p_head + 1e-9)))))
    return {"agreement": float(np.mean(aggr)), "mean_kl": float(np.mean(kls)), "n": len(routing_log)}
