from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ts_rsi.types import AdapterDescription, ForecastContext, ForecastRecord


class AdapterAgent(ABC):
    """Executable local adapter; no LLM call occurs in this interface."""

    @property
    @abstractmethod
    def version(self) -> str:
        ...

    @abstractmethod
    def predict(self, context: ForecastContext, base_forecast: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def update(self, matured_records: list[ForecastRecord]) -> None:
        ...

    @abstractmethod
    def describe(self) -> AdapterDescription:
        ...

    @abstractmethod
    def clone(self, version: str) -> "AdapterAgent":
        ...

    @abstractmethod
    def configuration(self) -> dict[str, Any]:
        ...

    def record_features(self, context: ForecastContext, base_forecast: np.ndarray) -> np.ndarray | None:
        """Optional causal features of the current situation, stored on the record.

        Adapters that correct by retrieving similar past situations must expose the
        features they used at predict time so update() can pair them with the realized
        residual later. Default: no features.
        """
        return None
