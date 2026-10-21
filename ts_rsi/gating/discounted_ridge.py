from __future__ import annotations

import numpy as np


class DiscountedRidgeGate:
    """Per-variable exponentially discounted ridge gate in [0, 1]."""

    def __init__(
        self,
        dimension: int,
        half_life: float,
        ridge_lambda: float,
        scale: np.ndarray,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        if ridge_lambda < 0:
            raise ValueError("ridge_lambda must be non-negative")
        scale_array = np.asarray(scale, dtype=np.float64)
        if scale_array.shape != (dimension,):
            raise ValueError("scale must have shape (dimension,)")
        if np.any(scale_array <= 0) or not np.isfinite(scale_array).all():
            raise ValueError("scale must be finite and strictly positive")
        self.dimension = dimension
        self.half_life = half_life
        self.ridge_lambda = ridge_lambda
        self.scale = scale_array.copy()
        self._rho = 2.0 ** (-1.0 / half_life)
        self._s = np.zeros(dimension, dtype=np.float64)
        self._q = np.zeros(dimension, dtype=np.float64)
        self._z = 0.0

    def current(self) -> np.ndarray:
        if self._z == 0.0:
            return np.zeros(self.dimension, dtype=np.float64)
        gate = self._s / (self._q + self.ridge_lambda * self._z)
        return np.clip(gate, 0.0, 1.0)

    def update(self, error: np.ndarray, correction: np.ndarray) -> np.ndarray:
        error_array = np.asarray(error, dtype=np.float64)
        correction_array = np.asarray(correction, dtype=np.float64)
        if error_array.shape != correction_array.shape or error_array.ndim != 2:
            raise ValueError("error and correction must have the same H x D shape")
        if error_array.shape[1] != self.dimension:
            raise ValueError("gate dimension does not match error")
        normalized_error = error_array / self.scale
        normalized_correction = correction_array / self.scale
        self._s = self._rho * self._s + np.mean(
            normalized_error * normalized_correction, axis=0
        )
        self._q = self._rho * self._q + np.mean(normalized_correction**2, axis=0)
        self._z = self._rho * self._z + 1.0
        return self.current()

    def clone(self) -> "DiscountedRidgeGate":
        cloned = DiscountedRidgeGate(
            dimension=self.dimension,
            half_life=self.half_life,
            ridge_lambda=self.ridge_lambda,
            scale=self.scale,
        )
        cloned._s = self._s.copy()
        cloned._q = self._q.copy()
        cloned._z = self._z
        return cloned

    def state_summary(self) -> dict[str, object]:
        return {
            "gate": self.current().tolist(),
            "half_life": self.half_life,
            "ridge_lambda": self.ridge_lambda,
            "effective_records": self._z,
        }
