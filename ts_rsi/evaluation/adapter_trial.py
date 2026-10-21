from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ts_rsi.evaluation.metrics import standardized_mse
from ts_rsi.types import EvaluationSummary, ForecastRecord


@dataclass
class CandidateTrial:
    candidate_id: str
    baseline_version: str
    candidate_version: str
    warmup_records: int
    evaluation_records: int
    records: list[tuple[ForecastRecord, ForecastRecord]]

    def add_pair(self, baseline: ForecastRecord, candidate: ForecastRecord) -> None:
        if baseline.origin_index != candidate.origin_index:
            raise ValueError("Baseline and candidate records must share the origin")
        self.records.append((baseline, candidate))

    def ready(self) -> bool:
        return len(self.records) >= self.warmup_records + self.evaluation_records

    def summarize(self, scale: np.ndarray) -> EvaluationSummary:
        if not self.ready():
            raise ValueError("Candidate trial is not ready for evaluation")
        evaluation = self.records[self.warmup_records :]
        baseline_loss = float(np.mean([standardized_mse(old, scale) for old, _ in evaluation]))
        candidate_loss = float(np.mean([standardized_mse(candidate, scale) for _, candidate in evaluation]))
        improvement = baseline_loss - candidate_loss
        return EvaluationSummary(
            candidate_id=self.candidate_id,
            baseline_loss=baseline_loss,
            candidate_loss=candidate_loss,
            improvement=improvement,
            accepted=improvement > 0.0,
            evaluated_records=len(evaluation),
        )
