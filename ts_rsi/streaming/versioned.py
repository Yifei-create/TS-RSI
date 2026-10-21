from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ts_rsi.adapters.base import AdapterAgent
from ts_rsi.gating.discounted_ridge import DiscountedRidgeGate
from ts_rsi.types import ForecastContext, ForecastRecord


@dataclass
class RunningVersion:
    adapter: AdapterAgent
    gate: DiscountedRidgeGate
    meta_version: str
    active: bool = True

    @property
    def version(self) -> str:
        return self.adapter.version

    def predict(
        self,
        context: ForecastContext,
        base_forecast: np.ndarray,
        forecast_id: int,
        target_times: tuple,
    ) -> ForecastRecord:
        correction = self.adapter.predict(context, base_forecast)
        gate = self.gate.current()
        final = base_forecast + correction * gate[None, :]
        target_indices = tuple(range(context.origin_index + 1, context.origin_index + 1 + base_forecast.shape[0]))
        if len(target_times) != base_forecast.shape[0]:
            raise ValueError("target_times must have one entry per forecast horizon")
        return ForecastRecord(
            forecast_id=forecast_id,
            origin_index=context.origin_index,
            origin_time=context.origin_time,
            target_indices=target_indices,
            target_times=target_times,
            adapter_version=self.adapter.version,
            meta_version=self.meta_version,
            base_forecast=base_forecast.copy(),
            raw_correction=correction.copy(),
            gate=gate.copy(),
            final_forecast=final.copy(),
        )

    def update(self, matured: list[ForecastRecord]) -> None:
        self.adapter.update(matured)
        for record in matured:
            if record.target_values is None:
                raise ValueError("Matured record is missing target_values")
            error = record.target_values - record.base_forecast
            self.gate.update(error=error, correction=record.raw_correction)
