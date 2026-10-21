from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ts_rsi.foundation.base import BlackBoxForecaster
from ts_rsi.streaming.buffer import CompletedRecordBuffer
from ts_rsi.streaming.pending import PendingForecasts
from ts_rsi.streaming.versioned import RunningVersion
from ts_rsi.types import ForecastContext, ForecastRecord


@dataclass(frozen=True)
class StreamStep:
    current_index: int
    matured: tuple[ForecastRecord, ...]
    predictions: tuple[ForecastRecord, ...]


class CausalStreamRunner:
    """Runs the fixed time-order protocol; revisions are injected by callbacks."""

    def __init__(
        self,
        values: np.ndarray,
        timestamps: tuple,
        lookback: int,
        horizon: int,
        foundation: BlackBoxForecaster,
        versions: list[RunningVersion],
        completed: CompletedRecordBuffer,
        pending: PendingForecasts,
    ) -> None:
        if values.ndim != 2:
            raise ValueError("values must be a T x D matrix")
        if len(timestamps) != values.shape[0]:
            raise ValueError("timestamps and values have different lengths")
        if lookback <= 0 or horizon <= 0:
            raise ValueError("lookback and horizon must be positive")
        if not versions:
            raise ValueError("At least one running version is required")
        self.values = values
        self.timestamps = timestamps
        self.lookback = lookback
        self.horizon = horizon
        self.foundation = foundation
        self.versions = versions
        self.completed = completed
        self.pending = pending
        self._next_forecast_id = 0

    def run(self) -> Iterable[StreamStep]:
        last_origin = self.values.shape[0] - self.horizon - 1
        for current_index in range(self.lookback - 1, self.values.shape[0]):
            matured = self.pending.pop_ready(current_index, self.values)
            if matured:
                self.completed.append(matured)
                for version in self.versions:
                    version_records = [
                        record for record in matured if record.adapter_version == version.version
                    ]
                    if version_records:
                        version.update(version_records)
            predictions: list[ForecastRecord] = []
            origin = current_index
            if origin <= last_origin:
                context_values = self.values[origin - self.lookback + 1 : origin + 1].copy()
                context = ForecastContext(
                    origin_index=origin,
                    origin_time=self.timestamps[origin],
                    values=context_values,
                    timestamps=tuple(self.timestamps[origin - self.lookback + 1 : origin + 1]),
                )
                base_forecast = self.foundation.predict(context, self.horizon)
                for version in self.versions:
                    target_times = tuple(
                        self.timestamps[index]
                        for index in range(origin + 1, origin + 1 + self.horizon)
                    )
                    record = version.predict(
                        context,
                        base_forecast,
                        self._next_forecast_id,
                        target_times,
                    )
                    self._next_forecast_id += 1
                    self.pending.add(record)
                    predictions.append(record)
            yield StreamStep(current_index, tuple(matured), tuple(predictions))
