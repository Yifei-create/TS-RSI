from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ts_rsi.adapters.base import AdapterAgent
from ts_rsi.types import AdapterDescription, ForecastContext, ForecastRecord


class MeanResidualAdapter(AdapterAgent):
    """Mean residual correction over the latest completed forecast records."""

    def __init__(
        self,
        version: str,
        history_window: int = 128,
        cache_size: int = 2048,
        dimension: int | None = None,
    ) -> None:
        if history_window <= 0:
            raise ValueError("history_window must be positive")
        if cache_size < history_window:
            raise ValueError("cache_size must be >= history_window")
        self._version = version
        self.history_window = history_window
        self.cache_size = cache_size
        self._residuals: deque[np.ndarray] = deque(maxlen=cache_size)
        self._dimension = dimension
        self._updates = 0

    @property
    def version(self) -> str:
        return self._version

    def predict(self, context: ForecastContext, base_forecast: np.ndarray) -> np.ndarray:
        if base_forecast.ndim != 2:
            raise ValueError("base_forecast must have shape H x D")
        dimension = base_forecast.shape[1]
        if self._dimension is None:
            self._dimension = dimension
        if self._dimension != dimension:
            raise ValueError("Adapter dimension changed after initialization")
        if not self._residuals:
            return np.zeros_like(base_forecast)
        recent = list(self._residuals)[-self.history_window :]
        mean_residual = np.mean(np.stack(recent, axis=0), axis=(0, 1))
        return np.repeat(mean_residual[None, :], base_forecast.shape[0], axis=0)

    def update(self, matured_records: list[ForecastRecord]) -> None:
        for record in matured_records:
            if record.target_values is None:
                raise ValueError("Cannot update adapter with an immature record")
            residual = record.target_values - record.base_forecast
            self._residuals.append(residual.copy())
            self._updates += 1

    def describe(self) -> AdapterDescription:
        return AdapterDescription(
            version=self.version,
            name=self.__class__.__name__,
            configuration=self.configuration(),
            state_summary={
                "stored_records": len(self._residuals),
                "updates": self._updates,
                "dimension": self._dimension,
            },
        )

    def clone(self, version: str) -> "MeanResidualAdapter":
        cloned = MeanResidualAdapter(
            version=version,
            history_window=self.history_window,
            cache_size=self.cache_size,
            dimension=self._dimension,
        )
        cloned._residuals.extend(residual.copy() for residual in self._residuals)
        cloned._updates = self._updates
        return cloned

    def configuration(self) -> dict[str, Any]:
        return {
            "type": "mean_residual",
            "history_window": self.history_window,
            "cache_size": self.cache_size,
        }
