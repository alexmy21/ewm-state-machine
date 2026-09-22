# trainer — the reference implementation of the trainer/controller layer.
#
# This package is the notebook logic extracted into a form a future DSL must
# reproduce. It is intentionally outside ewm-sm: the apparatus stays in the
# Rust crates, this package only reads its JSON and writes probe configs.
#
# See docs/TRAINER.md for the separation map and the three protocol shapes.

from .protocol import DecisionRecord, ProbeConfig, TrajectoryRecord, Prediction, Selection
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
from .adapters import Adapter, SyntheticAdapter, LlmAdapter, JevAdapter, LayaAdapter, BonsaiAdapter, memory_tokens, displacement_tokens, build_prompt
from .ewm import EwmScene, write_frames, write_pyramid, write_union
from .loop import (
    LoopResult,
    OpenLoopResult,
    JevLoopResult,
    run_open_loop,
    run_open_loop_from_streams,
    run_closed_loop,
    run_jev_loop,
    run_jev_loop_from_streams,
)

__all__ = [
    "ProbeConfig",
    "TrajectoryRecord",
    "Prediction",
    "Selection",
    "DecisionRecord",
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
    "JevAdapter",
    "LayaAdapter",
    "BonsaiAdapter",
    "memory_tokens",
    "displacement_tokens",
    "build_prompt",
    "EwmScene",
    "write_frames",
    "write_pyramid",
    "write_union",
    "LoopResult",
    "OpenLoopResult",
    "JevLoopResult",
    "run_open_loop",
    "run_open_loop_from_streams",
    "run_closed_loop",
    "run_jev_loop",
    "run_jev_loop_from_streams",
]
