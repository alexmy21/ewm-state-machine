# protocol.py — the three JSON shapes of the trainer/controller layer.
#
# These are the interfaces from docs/TRAINER.md §3, made explicit as
# dataclasses. No third-party dependencies: `dataclasses.asdict` is the
# serialization.

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional


def _from_dict(cls, d: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ProbeConfig:
    """Controller -> adapter. What one LLM probe is asked to do for one step."""

    step: int
    query: str
    memory: list[str] = field(default_factory=list)
    gate: Optional[str] = None
    temperature: Optional[float] = None
    max_new_tokens: int = 48

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProbeConfig":
        return _from_dict(cls, d)


@dataclass
class TrajectoryRecord:
    """ewm-sm -> trainer. One step of the system trajectory."""

    step: int
    union_key: str
    union_pop: int
    bss: list[float]
    drn: dict[str, int]
    ring: dict[str, Any]
    per_llm: dict[str, Any] = field(default_factory=dict)
    memory_ordered: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrajectoryRecord":
        return _from_dict(cls, d)


@dataclass
class Prediction:
    """Predictor interface: history -> prediction."""

    name: str
    history: list[list[float]]
    prediction: list[float]

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class Selection:
    """Selector interface: performance weights -> choice."""

    weights: dict[str, float]
    choice: str
    epsilon: float

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)
