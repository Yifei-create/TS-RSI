from __future__ import annotations

from collections import deque

from ts_rsi.types import ForecastRecord


class CompletedRecordBuffer:
    def __init__(self, max_records: int) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self._records: deque[ForecastRecord] = deque(maxlen=max_records)

    def append(self, records: list[ForecastRecord]) -> None:
        for record in records:
            if not record.is_mature:
                raise ValueError("Only mature records may enter the completed buffer")
            self._records.append(record)

    def latest(self, count: int | None = None) -> list[ForecastRecord]:
        values = list(self._records)
        return values if count is None else values[-count:]

    def __len__(self) -> int:
        return len(self._records)
