from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class ForecastContext:
    """The causal context available at one forecast origin."""

    origin_index: int
    origin_time: datetime
    values: Array
    timestamps: tuple[datetime, ...]


@dataclass
class ForecastRecord:
    """A prediction snapshot. It is immutable in meaning after creation."""

    forecast_id: int
    origin_index: int
    origin_time: datetime
    target_indices: tuple[int, ...]
    target_times: tuple[datetime, ...]
    adapter_version: str
    meta_version: str
    base_forecast: Array
    raw_correction: Array
    gate: Array
    final_forecast: Array
    target_values: Array | None = None
    loss_base: float | None = None
    loss_final: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_mature(self) -> bool:
        return self.target_values is not None


@dataclass(frozen=True)
class AdapterDescription:
    version: str
    name: str
    configuration: Mapping[str, Any]
    state_summary: Mapping[str, Any]


@dataclass(frozen=True)
class RevisionProposal:
    action: str
    evidence: tuple[str, ...]
    diagnosis: str
    change: str
    expected_effect: str
    tool_calls: tuple[Mapping[str, Any], ...] = ()


@dataclass
class RevisionRecord:
    revision_id: str
    parent_adapter_version: str
    parent_meta_version: str
    action: str
    observed_problem: str
    diagnosis: str
    proposed_change: str
    expected_effect: str
    actual_effect: str | None = None
    accepted: bool | None = None
    status: str = "proposed"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    ok: bool
    message: str
    changed_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationSummary:
    candidate_id: str
    baseline_loss: float
    candidate_loss: float
    improvement: float
    accepted: bool
    evaluated_records: int
