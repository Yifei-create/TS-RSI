from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSeriesData:
    name: str
    timestamps: tuple[pd.Timestamp, ...]
    values: np.ndarray
    columns: tuple[str, ...]


@dataclass(frozen=True)
class Standardization:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


def load_csv(path: Path, name: str, timestamp_column: str) -> TimeSeriesData:
    frame = pd.read_csv(path)
    timestamps = tuple(pd.to_datetime(frame[timestamp_column]))
    numeric = frame.drop(columns=[timestamp_column]).apply(pd.to_numeric)
    values = numeric.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"Dataset {path} has no non-empty numeric matrix")
    if not np.isfinite(values).all():
        raise ValueError(f"Dataset {path} contains NaN or infinite values")
    return TimeSeriesData(name, timestamps, values, tuple(numeric.columns))


def fit_standardization(values: np.ndarray, floor: float = 1e-8) -> Standardization:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.maximum(scale, floor)
    return Standardization(mean=mean, scale=scale)


def chronological_split(
    data: TimeSeriesData,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[TimeSeriesData, TimeSeriesData, TimeSeriesData]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must lie in (0, 1)")
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must lie in [0, 1)")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be smaller than 1")
    n = len(data.timestamps)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + validation_ratio))

    def part(start: int, end: int, suffix: str) -> TimeSeriesData:
        return TimeSeriesData(
            name=f"{data.name}_{suffix}",
            timestamps=data.timestamps[start:end],
            values=data.values[start:end],
            columns=data.columns,
        )

    return part(0, train_end, "train"), part(train_end, valid_end, "validation"), part(
        valid_end, n, "test"
    )
