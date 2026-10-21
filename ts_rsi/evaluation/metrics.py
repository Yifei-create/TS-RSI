from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ts_rsi.types import ForecastRecord


def _require_target(record: ForecastRecord) -> np.ndarray:
    if record.target_values is None:
        raise ValueError(f"Record {record.forecast_id} has no target_values")
    return record.target_values


def standardized_mse(record: ForecastRecord, scale: np.ndarray, use_final: bool = True) -> float:
    target = _require_target(record)
    prediction = record.final_forecast if use_final else record.base_forecast
    normalized = (target - prediction) / scale
    return float(np.mean(normalized**2))


def standardized_mae(record: ForecastRecord, scale: np.ndarray, use_final: bool = True) -> float:
    target = _require_target(record)
    prediction = record.final_forecast if use_final else record.base_forecast
    normalized = np.abs((target - prediction) / scale)
    return float(np.mean(normalized))


def signed_error(record: ForecastRecord, use_final: bool = True) -> np.ndarray:
    target = _require_target(record)
    prediction = record.final_forecast if use_final else record.base_forecast
    return target - prediction


@dataclass(frozen=True)
class LossPair:
    base: float
    final: float


def loss_pair(record: ForecastRecord, scale: np.ndarray) -> LossPair:
    return LossPair(
        base=standardized_mse(record, scale, use_final=False),
        final=standardized_mse(record, scale, use_final=True),
    )
