from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class Message:
    role: str
    content: str


class LLMClient(Protocol):
    def chat(self, messages: list[Message], temperature: float = 0.1) -> str:
        ...
