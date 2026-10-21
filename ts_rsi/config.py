from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    timestamp_column: str
    target_columns: tuple[str, ...] | None
    train_ratio: float
    validation_ratio: float


@dataclass(frozen=True)
class StreamConfig:
    lookback: int
    horizon: int
    review_batch_records: int
    adapter_history_window: int
    adapter_cache_size: int
    gate_half_life: float
    gate_lambda: float
    candidate_warmup_records: int
    candidate_evaluation_records: int
    max_active_candidates: int


@dataclass(frozen=True)
class FoundationConfig:
    name: str
    checkpoint: Path
    device: str


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    model: str
    api_key_env: str
    max_calls_per_session: int
    timeout_seconds: float
    input_price_per_mtok: float | None = None
    output_price_per_mtok: float | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    dataset: DatasetConfig
    foundation: FoundationConfig
    stream: StreamConfig
    api: ApiConfig


def load_config(path: Path) -> ExperimentConfig:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    dataset_raw = raw["dataset"]
    stream_raw = raw["stream"]
    foundation_raw = raw["foundation"]
    api_raw = raw["api"]
    target_columns = dataset_raw.get("target_columns")
    return ExperimentConfig(
        seed=int(raw["seed"]),
        dataset=DatasetConfig(
            name=str(dataset_raw["name"]),
            path=Path(dataset_raw["path"]),
            timestamp_column=str(dataset_raw["timestamp_column"]),
            target_columns=tuple(target_columns) if target_columns else None,
            train_ratio=float(dataset_raw["train_ratio"]),
            validation_ratio=float(dataset_raw["validation_ratio"]),
        ),
        foundation=FoundationConfig(
            name=str(foundation_raw["name"]),
            checkpoint=Path(foundation_raw["checkpoint"]),
            device=str(foundation_raw["device"]),
        ),
        stream=StreamConfig(
            lookback=int(stream_raw["lookback"]),
            horizon=int(stream_raw["horizon"]),
            review_batch_records=int(stream_raw["review_batch_records"]),
            adapter_history_window=int(stream_raw["adapter_history_window"]),
            adapter_cache_size=int(stream_raw["adapter_cache_size"]),
            gate_half_life=float(stream_raw["gate_half_life"]),
            gate_lambda=float(stream_raw["gate_lambda"]),
            candidate_warmup_records=int(stream_raw["candidate_warmup_records"]),
            candidate_evaluation_records=int(stream_raw["candidate_evaluation_records"]),
            max_active_candidates=int(stream_raw["max_active_candidates"]),
        ),
        api=ApiConfig(
            base_url=str(api_raw["base_url"]),
            model=str(api_raw["model"]),
            api_key_env=str(api_raw["api_key_env"]),
            max_calls_per_session=int(api_raw["max_calls_per_session"]),
            timeout_seconds=float(api_raw["timeout_seconds"]),
            input_price_per_mtok=(
                float(api_raw["input_price_per_mtok"])
                if api_raw.get("input_price_per_mtok") is not None
                else None
            ),
            output_price_per_mtok=(
                float(api_raw["output_price_per_mtok"])
                if api_raw.get("output_price_per_mtok") is not None
                else None
            ),
        ),
    )
