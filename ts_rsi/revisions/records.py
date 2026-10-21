from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ts_rsi.types import RevisionRecord


class RevisionLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[RevisionRecord] = []

    def append(self, record: RevisionRecord) -> None:
        self.records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def recent(self, count: int = 10) -> list[RevisionRecord]:
        return self.records[-count:]
