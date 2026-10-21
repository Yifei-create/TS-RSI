from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ts_rsi.evaluation.metrics import signed_error, standardized_mse
from ts_rsi.types import AdapterDescription, ForecastRecord, RevisionRecord


@dataclass(frozen=True)
class ReviewReport:
    report_id: str
    mature_records: int
    base_mse: float
    final_mse: float
    mean_signed_error: list[float]
    mean_signed_error_by_horizon: list[float]
    gate_mean: list[float]
    correction_abs_mean: list[float]
    adapter: dict[str, Any]
    recent_revisions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_review_report(
    report_id: str,
    records: list[ForecastRecord],
    scale: np.ndarray,
    adapter: AdapterDescription,
    revisions: list[RevisionRecord],
) -> ReviewReport:
    if not records:
        raise ValueError("At least one mature record is required")
    base_losses = [standardized_mse(record, scale, use_final=False) for record in records]
    final_losses = [standardized_mse(record, scale, use_final=True) for record in records]
    errors = np.stack([signed_error(record) for record in records])
    gates = np.stack([record.gate for record in records])
    corrections = np.stack([np.abs(record.raw_correction) for record in records])
    recent = [asdict(item) for item in revisions[-10:]]
    return ReviewReport(
        report_id=report_id,
        mature_records=len(records),
        base_mse=float(np.mean(base_losses)),
        final_mse=float(np.mean(final_losses)),
        mean_signed_error=np.mean(errors, axis=(0, 1)).tolist(),
        mean_signed_error_by_horizon=np.mean(errors, axis=(0, 2)).tolist(),
        gate_mean=np.mean(gates, axis=0).tolist(),
        correction_abs_mean=np.mean(corrections, axis=(0, 1)).tolist(),
        adapter={"version": adapter.version, "name": adapter.name, "configuration": dict(adapter.configuration), "state_summary": dict(adapter.state_summary)},
        recent_revisions=recent,
    )
