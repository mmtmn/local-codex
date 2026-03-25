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
    memory_file: Path | None = None
    plugins_dir: Path | None = None
    max_delegation_depth: int = 2

    @property
    def workspace_resolved(self) -> Path:
        return self.workspace.resolve()

    @property
    def memory_file_resolved(self) -> Path:
        if self.memory_file is None:
            return (self.workspace_resolved / ".local_codex" / "memory.json").resolve()
        if self.memory_file.is_absolute():
            return self.memory_file.resolve()
        return (self.workspace_resolved / self.memory_file).resolve()

    @property
    def plugins_dir_resolved(self) -> Path:
        if self.plugins_dir is None:
            return (self.workspace_resolved / ".local_codex" / "plugins").resolve()
        if self.plugins_dir.is_absolute():
            return self.plugins_dir.resolve()
        return (self.workspace_resolved / self.plugins_dir).resolve()
