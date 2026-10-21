from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ts_rsi.data.loader import TimeSeriesData
from ts_rsi.types import ForecastContext


@dataclass(frozen=True)
class StreamWindow:
    context: ForecastContext
    target: np.ndarray


def iter_windows(data: TimeSeriesData, lookback: int, horizon: int) -> list[StreamWindow]:
    if lookback <= 0 or horizon <= 0:
        raise ValueError("lookback and horizon must be positive")
    last_origin = len(data.values) - horizon
    windows: list[StreamWindow] = []
    for origin in range(lookback - 1, last_origin):
        context_start = origin - lookback + 1
        target_start = origin + 1
        context = ForecastContext(
            origin_index=origin,
            origin_time=data.timestamps[origin].to_pydatetime(),
            values=data.values[context_start : origin + 1].copy(),
            timestamps=tuple(ts.to_pydatetime() for ts in data.timestamps[context_start : origin + 1]),
        )
        target = data.values[target_start : target_start + horizon].copy()
        windows.append(StreamWindow(context=context, target=target))
    return windows
