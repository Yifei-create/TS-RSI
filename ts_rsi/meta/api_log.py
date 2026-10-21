from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from ts_rsi.meta.api_client import ChatResponse


@dataclass(frozen=True)
class ApiPrices:
    """Optional per-million-token prices used only for cost estimation in the log."""

    input_per_mtok: float | None = None
    output_per_mtok: float | None = None


class ApiCallLogger:
    """Append-only JSONL log of LLM API calls; tool executions are not logged here."""

    def __init__(self, path: Path, prices: ApiPrices | None = None) -> None:
        self.path = path
        self.prices = prices or ApiPrices()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.call_count = 0

    def log(
        self,
        *,
        session_id: str,
        category: str,
        model: str,
        response: ChatResponse | None,
        latency_seconds: float,
        error: str | None = None,
    ) -> None:
        self.call_count += 1
        cost: float | None = None
        if (
            response is not None
            and response.input_tokens is not None
            and response.output_tokens is not None
            and self.prices.input_per_mtok is not None
            and self.prices.output_per_mtok is not None
        ):
            cost = (
                response.input_tokens * self.prices.input_per_mtok
                + response.output_tokens * self.prices.output_per_mtok
            ) / 1_000_000
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "session_id": session_id,
            "category": category,
            "model": model,
            "input_tokens": response.input_tokens if response else None,
            "output_tokens": response.output_tokens if response else None,
            "latency_seconds": round(latency_seconds, 3),
            "estimated_cost": cost,
            "error": error,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
