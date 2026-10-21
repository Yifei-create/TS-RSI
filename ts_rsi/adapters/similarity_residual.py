from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ts_rsi.adapters.base import AdapterAgent
from ts_rsi.types import AdapterDescription, ForecastContext, ForecastRecord


def situation_features(context_values: np.ndarray, base_forecast: np.ndarray) -> np.ndarray:
    """Causal description of the current situation, one block per channel.

    Per channel: context last value, recent mean, recent std, context slope,
    base-forecast mean, base-forecast slope. Every channel block is normalized by
    its own context std so distances are comparable across channels and datasets.
    """
    values = np.asarray(context_values, dtype=np.float64)
    base = np.asarray(base_forecast, dtype=np.float64)
    scale = values.std(axis=0)
    scale = np.maximum(scale, 1e-8)
    normalized = values / scale
    base_normalized = base / scale
    recent = normalized[-24:]
    count = normalized.shape[0]
    time_axis = np.arange(count, dtype=np.float64)
    time_axis = time_axis - time_axis.mean()
    slope_context = (normalized * time_axis[:, None]).sum(axis=0) / (time_axis**2).sum()
    horizon_axis = np.arange(base.shape[0], dtype=np.float64)
    horizon_axis = horizon_axis - horizon_axis.mean()
    slope_base = (base_normalized * horizon_axis[:, None]).sum(axis=0) / (horizon_axis**2).sum()
    features = np.concatenate(
        [
            normalized[-1],
            recent.mean(axis=0),
            recent.std(axis=0),
            slope_context,
            base_normalized.mean(axis=0),
            slope_base,
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class SimilarityResidualAdapter(AdapterAgent):
    """Correct by retrieving past situations similar to the current one.

    Instead of extrapolating the global mean residual, the correction is the
    similarity-weighted average of residuals observed under situations whose
    causal feature vector is close to the current one.
    """

    def __init__(
        self,
        version: str,
        k: int = 16,
        cache_size: int = 2048,
        shrink: float = 1.0,
        dimension: int | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        if cache_size < k:
            raise ValueError("cache_size must be >= k")
        if not 0.0 < shrink <= 1.0:
            raise ValueError("shrink must lie in (0, 1]")
        self._version = version
        self.k = k
        self.cache_size = cache_size
        self.shrink = shrink
        self._dimension = dimension
        self._features: deque[np.ndarray] = deque(maxlen=cache_size)
        self._residuals: deque[np.ndarray] = deque(maxlen=cache_size)
        self._updates = 0

    @property
    def version(self) -> str:
        return self._version

    def record_features(self, context: ForecastContext, base_forecast: np.ndarray) -> np.ndarray:
        return situation_features(context.values, base_forecast)

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
        current = situation_features(context.values, base_forecast)
        past = np.stack(list(self._features), axis=0)
        distances = np.linalg.norm(past - current[None, :], axis=1)
        k = min(self.k, past.shape[0])
        nearest = np.argpartition(distances, k - 1)[:k]
        nearest_distances = distances[nearest]
        bandwidth = np.median(nearest_distances) + 1e-8
        weights = np.exp(-0.5 * (nearest_distances / bandwidth) ** 2)
        weights = weights / weights.sum()
        residuals = np.stack([list(self._residuals)[i] for i in nearest], axis=0)
        correction = np.tensordot(weights, residuals, axes=(0, 0)) * self.shrink
        return np.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)

    def update(self, matured_records: list[ForecastRecord]) -> None:
        for record in matured_records:
            if record.target_values is None:
                raise ValueError("Cannot update adapter with an immature record")
            features = record.metadata.get("features")
            if features is None:
                raise ValueError("Matured record is missing situation features")
            residual = record.target_values - record.base_forecast
            self._features.append(np.asarray(features, dtype=np.float64))
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

    def clone(self, version: str) -> "SimilarityResidualAdapter":
        cloned = SimilarityResidualAdapter(
            version=version,
            k=self.k,
            cache_size=self.cache_size,
            shrink=self.shrink,
            dimension=self._dimension,
        )
        cloned._features.extend(item.copy() for item in self._features)
        cloned._residuals.extend(item.copy() for item in self._residuals)
        cloned._updates = self._updates
        return cloned

    def configuration(self) -> dict[str, Any]:
        return {
            "type": "similarity_residual",
            "k": self.k,
            "cache_size": self.cache_size,
            "shrink": self.shrink,
        }
