from __future__ import annotations

from pathlib import Path

import numpy as np

from ts_rsi.foundation.base import BlackBoxForecaster
from ts_rsi.foundation.registry import resolve_foundation_model
from ts_rsi.types import ForecastContext


def _validate_forecast(result: np.ndarray, horizon: int, dimension: int) -> np.ndarray:
    forecast = np.asarray(result, dtype=np.float64)
    if forecast.shape != (horizon, dimension):
        raise ValueError(f"Frozen model returned {forecast.shape}; expected {(horizon, dimension)}")
    if not np.isfinite(forecast).all():
        raise ValueError("Frozen model returned NaN or infinite values")
    return forecast


class FrozenChronos2(BlackBoxForecaster):
    """Local Chronos-2 checkpoint wrapper. The loaded pipeline is inference-only."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        import torch
        from chronos import Chronos2Pipeline

        kwargs: dict[str, object] = {}
        if device.startswith("cuda"):
            kwargs = {"device_map": device, "torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True}
        self.pipeline = Chronos2Pipeline.from_pretrained(str(checkpoint), **kwargs)
        self.device = device

    def predict(self, context: ForecastContext, horizon: int) -> np.ndarray:
        import torch

        inputs = [{"target": context.values.astype(np.float32).T}]
        with torch.no_grad():
            quantiles, _ = self.pipeline.predict_quantiles(
                inputs=inputs,
                prediction_length=horizon,
                batch_size=1,
                quantile_levels=[0.5],
                cross_learning=False,
            )
        values = quantiles[0].detach().cpu().numpy()
        if values.ndim == 3:
            values = values[:, :, 0].T
        elif values.ndim == 2:
            values = values.T
        return _validate_forecast(values, horizon, context.values.shape[1])


class FrozenMoirai2(BlackBoxForecaster):
    """Local Moirai-2 checkpoint wrapper. Parameters are never updated."""

    def __init__(self, checkpoint: str | Path, horizon: int, context_length: int, dimension: int, device: str = "cpu") -> None:
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

        self.model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(str(checkpoint)),
            prediction_length=horizon,
            context_length=context_length,
            target_dim=dimension,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self.device = device

    def predict(self, context: ForecastContext, horizon: int) -> np.ndarray:
        values = context.values[-self.model.context_length :].astype(np.float32)
        forecast = self.model.predict([values])
        result = forecast.detach().cpu().numpy() if hasattr(forecast, "detach") else np.asarray(forecast)
        return _validate_forecast(np.squeeze(result, axis=0), horizon, context.values.shape[1])


class FrozenTimesFM25(BlackBoxForecaster):
    """Local TimesFM 2.5 checkpoint wrapper using the ORCA call convention."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        from timesfm import configs
        from timesfm.timesfm_2p5 import timesfm_2p5_torch

        self.configs = configs
        self.model = timesfm_2p5_torch.TimesFM_2p5_200M_torch()
        self.model.load_checkpoint(path=str(checkpoint), torch_compile=False)
        self.device = device
        self.model.compile(
            forecast_config=configs.ForecastConfig(
                max_context=15360,
                max_horizon=1024,
                infer_is_positive=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
                force_flip_invariance=True,
                return_backcast=False,
                normalize_inputs=True,
                per_core_batch_size=1,
            )
        )

    def predict(self, context: ForecastContext, horizon: int) -> np.ndarray:
        inputs = [context.values[:, column].astype(np.float32) for column in range(context.values.shape[1])]
        _, forecast = self.model.forecast(horizon=horizon, inputs=inputs)
        values = np.asarray(forecast)
        if values.ndim == 3:
            values = values[:, :, 0]
        return _validate_forecast(values.T, horizon, context.values.shape[1])


def load_frozen_foundation_model(
    name: str,
    checkpoint: str | Path,
    horizon: int,
    context_length: int,
    dimension: int,
    device: str = "cpu",
) -> BlackBoxForecaster:
    key = resolve_foundation_model(name).name
    if key == "chronos-2":
        return FrozenChronos2(checkpoint, device)
    if key == "moirai-2":
        return FrozenMoirai2(checkpoint, horizon, context_length, dimension, device)
    if key == "timesfm-2.5":
        return FrozenTimesFM25(checkpoint, device)
    raise ValueError(f"No local loader registered for {key}")
