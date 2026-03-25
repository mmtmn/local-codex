from __future__ import annotations

from local_codex.llm.base import LLMClient, Message
from local_codex.llm.http import HTTPClientError, post_json


class OllamaClient(LLMClient):
    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint
        self.model = model

    def chat(self, messages: list[Message], temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": {"temperature": temperature},
        }

        data = post_json(self.endpoint, payload)
        try:
            return data["message"]["content"]
        except KeyError as exc:
            raise HTTPClientError(f"Unexpected Ollama response shape: {data}") from exc
