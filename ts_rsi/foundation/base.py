from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ts_rsi.types import ForecastContext


class BlackBoxForecaster(ABC):
    """Only exposes predictions; adapters must not access model internals."""

    @abstractmethod
    def predict(self, context: ForecastContext, horizon: int) -> np.ndarray:
        """Return an H x D forecast for the supplied causal context."""


class CallableForecaster(BlackBoxForecaster):
    def __init__(self, predict_fn: object) -> None:
        self._predict_fn = predict_fn

    def predict(self, context: ForecastContext, horizon: int) -> np.ndarray:
        prediction = self._predict_fn(context.values, horizon)  # type: ignore[operator]
        result = np.asarray(prediction, dtype=np.float64)
        if result.shape != (horizon, context.values.shape[1]):
            raise ValueError(
                f"Foundation model returned {result.shape}, expected {(horizon, context.values.shape[1])}"
            )
        if not np.isfinite(result).all():
            raise ValueError("Foundation model returned NaN or infinite values")
        return result
