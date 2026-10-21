from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ChatResponse:
    content: str
    input_tokens: int | None
    output_tokens: int | None
    raw: dict[str, object]


class OpenAICompatibleClient:
    """Small dependency-free client for OpenAI-compatible chat-completions APIs."""

    def __init__(self, base_url: str, model: str, api_key: str, timeout_seconds: float = 120.0) -> None:
        if not model:
            raise ValueError("model must be configured before MetaAgent calls")
        if not api_key:
            raise ValueError("api_key must be provided before MetaAgent calls")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(cls, base_url: str, model: str, api_key_env: str, timeout_seconds: float) -> "OpenAICompatibleClient":
        return cls(base_url, model, os.environ[api_key_env], timeout_seconds)

    def complete(self, system_prompt: str, user_prompt: str) -> ChatResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
        message = raw["choices"][0]["message"]["content"]
        usage = raw.get("usage", {})
        return ChatResponse(
            content=str(message),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            raw=raw,
        )
