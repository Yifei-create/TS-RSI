from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ts_rsi.types import ForecastRecord


@dataclass(frozen=True)
class MaturedRecord:
    record: ForecastRecord
    target_values: np.ndarray


class PendingForecasts:
    """Stores forecasts until their complete H-step target is observable."""

    def __init__(self) -> None:
        self._records: dict[int, ForecastRecord] = {}

    def add(self, record: ForecastRecord) -> None:
        if record.forecast_id in self._records:
            raise ValueError(f"Duplicate forecast_id: {record.forecast_id}")
        self._records[record.forecast_id] = record

    def pop_ready(self, current_index: int, values: np.ndarray) -> list[ForecastRecord]:
        ready_ids = sorted(
            forecast_id
            for forecast_id, record in self._records.items()
            if record.target_indices[-1] <= current_index
        )
        matured: list[ForecastRecord] = []
        for forecast_id in ready_ids:
            record = self._records.pop(forecast_id)
            target = values[list(record.target_indices)].copy()
            record.target_values = target
            matured.append(record)
        return matured

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> tuple[ForecastRecord, ...]:
        return tuple(self._records.values())
