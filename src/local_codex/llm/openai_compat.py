from __future__ import annotations

from local_codex.llm.base import LLMClient, Message
from local_codex.llm.http import HTTPClientError, post_json


class OpenAICompatibleClient(LLMClient):
    def __init__(self, endpoint: str, model: str, api_key: str | None = None):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key

    def chat(self, messages: list[Message], temperature: float = 0.1) -> str:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }

        data = post_json(self.endpoint, payload, headers=headers)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise HTTPClientError(
                f"Unexpected OpenAI-compatible response shape: {data}"
            ) from exc
