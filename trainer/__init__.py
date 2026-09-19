# trainer — the reference implementation of the trainer/controller layer.
#
# This package is the notebook logic extracted into a form a future DSL must
# reproduce. It is intentionally outside ewm-sm: the apparatus stays in the
# Rust crates, this package only reads its JSON and writes probe configs.
#
# See docs/TRAINER.md for the separation map and the three protocol shapes.

from .protocol import ProbeConfig, TrajectoryRecord, Prediction, Selection
from .predictors import (
    dft_period,
    Predictor,
    PersistencePredictor,
    LinearPredictor,
    DftPeriodicPredictor,
    RidgePredictor,
    default_portfolio,
)
from .selector import EwmaSelector, LearnedSelector, selector_features
from .adapters import Adapter, SyntheticAdapter, LlmAdapter, memory_tokens, build_prompt
from .ewm import EwmScene, write_frames, write_pyramid, write_union
from .loop import (
    LoopResult,
    OpenLoopResult,
    run_open_loop,
    run_closed_loop,
)

__all__ = [
    "ProbeConfig",
    "TrajectoryRecord",
    "Prediction",
    "Selection",
    "dft_period",
    "Predictor",
    "PersistencePredictor",
    "LinearPredictor",
    "DftPeriodicPredictor",
    "RidgePredictor",
    "default_portfolio",
    "EwmaSelector",
    "LearnedSelector",
    "selector_features",
    "Adapter",
    "SyntheticAdapter",
    "LlmAdapter",
    "memory_tokens",
    "build_prompt",
    "EwmScene",
    "write_frames",
    "write_pyramid",
    "write_union",
    "LoopResult",
    "OpenLoopResult",
    "run_open_loop",
    "run_closed_loop",
]
