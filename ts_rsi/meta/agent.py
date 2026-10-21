from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

from ts_rsi.meta.api_client import ChatResponse, OpenAICompatibleClient
from ts_rsi.meta.prompts import SYSTEM_PROMPT, build_repair_prompt, build_review_prompt
from ts_rsi.types import AdapterDescription, RevisionProposal, RevisionRecord


def parse_proposal(content: str) -> RevisionProposal:
    raw = json.loads(content)
    action = str(raw["action"])
    if action not in {"WAIT", "REVISE_ADAPTER", "REVISE_META"}:
        raise ValueError(f"Unsupported MetaAgent action: {action}")
    return RevisionProposal(
        action=action,
        evidence=tuple(str(item) for item in raw.get("evidence", [])),
        diagnosis=str(raw.get("diagnosis", "")),
        change=str(raw.get("change", "")),
        expected_effect=str(raw.get("expected_effect", "")),
        tool_calls=tuple(raw.get("tool_calls", [])),
    )


class MetaAgent(ABC):
    @property
    @abstractmethod
    def version(self) -> str:
        ...

    @abstractmethod
    def review(self, report: dict[str, Any], history: list[RevisionRecord]) -> RevisionProposal:
        ...


class WaitingMetaAgent(MetaAgent):
    def __init__(self, version: str = "meta-v0") -> None:
        self._version = version

    @property
    def version(self) -> str:
        return self._version

    def review(self, report: dict[str, Any], history: list[RevisionRecord]) -> RevisionProposal:
        return RevisionProposal("WAIT", (), "API disabled", "", "", ())


class ApiMetaAgent(MetaAgent):
    def __init__(self, version: str, client: OpenAICompatibleClient, max_calls: int = 4) -> None:
        self._version = version
        self.client = client
        self.max_calls = max_calls
        self.calls = 0
        self.last_response: ChatResponse | None = None

    @property
    def version(self) -> str:
        return self._version

    def begin_session(self) -> None:
        """Reset the per-session call budget before a new review/repair session."""
        self.calls = 0

    def _call(self, user_prompt: str) -> RevisionProposal:
        if self.calls >= self.max_calls:
            raise RuntimeError("MetaAgent session call budget exhausted")
        self.calls += 1
        response = self.client.complete(SYSTEM_PROMPT, user_prompt)
        self.last_response = response
        return parse_proposal(response.content)

    def review(self, report: dict[str, Any], history: list[RevisionRecord]) -> RevisionProposal:
        history_dict = [asdict(record) for record in history]
        return self._call(build_review_prompt(report, history_dict))

    def repair(self, feedback: str) -> RevisionProposal:
        return self._call(build_repair_prompt(feedback))
