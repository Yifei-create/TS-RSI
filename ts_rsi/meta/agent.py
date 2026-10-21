from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

from ts_rsi.meta.api_client import ChatResponse, OpenAICompatibleClient
from ts_rsi.meta.prompts import build_repair_prompt, build_review_prompt
from ts_rsi.types import RevisionProposal, RevisionRecord


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


def meta_version_of(system_prompt: str) -> str:
    return "meta-" + hashlib.sha1(system_prompt.encode("utf-8")).hexdigest()[:8]


class MetaAgent(ABC):
    @property
    @abstractmethod
    def version(self) -> str:
        ...

    @abstractmethod
    def review(self, report: dict[str, Any], history: list[RevisionRecord]) -> RevisionProposal:
        ...


class ApiMetaAgent(MetaAgent):
    """MetaAgent whose behavior is defined by its (editable) prompt files.

    The system prompt text is the agent's identity: revising it produces a new
    version. Every review and repair call goes through the configured LLM API.
    """

    def __init__(
        self,
        client: OpenAICompatibleClient,
        system_prompt: str,
        max_calls: int = 4,
    ) -> None:
        self._version = meta_version_of(system_prompt)
        self.client = client
        self.system_prompt = system_prompt
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
        response = self.client.complete(self.system_prompt, user_prompt)
        self.last_response = response
        return parse_proposal(response.content)

    def review(self, report: dict[str, Any], history: list[RevisionRecord]) -> RevisionProposal:
        history_dict = [asdict(record) for record in history]
        return self._call(build_review_prompt(report, history_dict))

    def repair(self, feedback: str) -> RevisionProposal:
        return self._call(build_repair_prompt(feedback))
