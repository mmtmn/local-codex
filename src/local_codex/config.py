from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    provider: str = "ollama"
    model: str = "qwen3:8b"
    endpoint: str = "http://127.0.0.1:11434/api/chat"
    api_key: str | None = None
    workspace: Path = Path.cwd()
    max_steps: int = 12
    shell_timeout_seconds: int = 45
    auto_approve: bool = False
    temperature: float = 0.1

    @property
    def workspace_resolved(self) -> Path:
        return self.workspace.resolve()
