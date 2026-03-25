from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_codex.llm.base import LLMClient, Message
from local_codex.tools import ToolExecutor

TOOL_RESULT_PROMPT = (
    "Tool execution result:\n"
    "{result}\n\n"
    "Continue. If the task is complete, respond with a final message."
)

SYSTEM_PROMPT_TEMPLATE = """You are LocalCodex, a terminal coding assistant.

You can use tools to inspect and modify files, and run shell commands.
Only use tools when needed.

Workspace root:
{workspace_root}

Available tools JSON:
{tool_specs}

Response rules:
1) Always respond with exactly one JSON object and no surrounding markdown.
2) Use one of these shapes:
   - {{"type": "tool", "tool": "<tool_name>", "args": {{...}}, "reason": "short reason"}}
   - {{"type": "final", "message": "final user-facing response"}}
3) If a tool errors, adapt and continue.
4) Prefer minimal, safe changes that satisfy the request.
"""


@dataclass(slots=True)
class Action:
    type: str
    payload: dict[str, Any]


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolExecutor,
        workspace_root: str,
        max_steps: int = 12,
        temperature: float = 0.1,
        on_tool_event: Callable[[str], None] | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.temperature = temperature
        self.on_tool_event = on_tool_event

        self._system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            workspace_root=workspace_root,
            tool_specs=tools.render_specs_for_prompt(),
        )
        self._messages: list[Message] = [Message(role="system", content=self._system_prompt)]
        self.tool_history: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._messages = [Message(role="system", content=self._system_prompt)]
        self.tool_history = []

    def run(self, user_prompt: str) -> str:
        self._messages.append(Message(role="user", content=user_prompt))

        for _ in range(self.max_steps):
            model_response = self.llm.chat(self._messages, temperature=self.temperature)
            action = self._parse_action(model_response)

            if action is None:
                self._messages.append(Message(role="assistant", content=model_response))
                return model_response.strip()

            self._messages.append(
                Message(role="assistant", content=json.dumps(action.payload, ensure_ascii=True))
            )

            if action.type == "final":
                message = str(action.payload.get("message", "")).strip()
                return message or "I do not have a final message yet."

            if action.type == "tool":
                tool_name = str(action.payload.get("tool", "")).strip()
                args = action.payload.get("args")
                if not isinstance(args, dict):
                    args = {}

                if self.on_tool_event is not None:
                    self.on_tool_event(f"tool {tool_name} {json.dumps(args, ensure_ascii=True)}")

                result = self.tools.execute(tool_name, args)
                self.tool_history.append(
                    {
                        "tool": tool_name,
                        "args": json.loads(json.dumps(args, ensure_ascii=True)),
                        "result": result,
                    }
                )

                if self.on_tool_event is not None:
                    self.on_tool_event(f"result {result[:240].replace(chr(10), ' ')}")

                self._messages.append(
                    Message(role="user", content=TOOL_RESULT_PROMPT.format(result=result))
                )
                continue

            self._messages.append(
                Message(
                    role="user",
                    content=(
                        "Invalid action type. Respond with JSON using"
                        " either type='tool' or type='final'."
                    ),
                )
            )

        return (
            "I hit the max tool step limit before completing the task. "
            "Please retry with a narrower request or higher --max-steps."
        )

    def session_payload(self) -> dict[str, Any]:
        messages = [{"role": message.role, "content": message.content} for message in self._messages]
        return {
            "version": 1,
            "messages": messages,
            "tool_history": self.tool_history,
        }

    def save_session(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.session_payload()
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return target

    def load_session(self, path: str | Path) -> Path:
        source = Path(path).expanduser()
        raw = source.read_text(encoding="utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("session payload must be a JSON object")

        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError("session payload missing messages list")

        restored_messages: list[Message] = []
        for entry in raw_messages:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role")
            content = entry.get("content")
            if isinstance(role, str) and isinstance(content, str):
                restored_messages.append(Message(role=role, content=content))

        # Keep the current system prompt so sessions survive prompt/schema evolution.
        non_system_messages = [message for message in restored_messages if message.role != "system"]
        self._messages = [Message(role="system", content=self._system_prompt), *non_system_messages]

        restored_tool_history: list[dict[str, Any]] = []
        raw_tool_history = payload.get("tool_history")
        if isinstance(raw_tool_history, list):
            for entry in raw_tool_history:
                if not isinstance(entry, dict):
                    continue
                tool = entry.get("tool")
                result = entry.get("result")
                args = entry.get("args")
                if not isinstance(tool, str) or not isinstance(result, str):
                    continue
                if not isinstance(args, dict):
                    args = {}
                restored_tool_history.append({"tool": tool, "args": args, "result": result})
        self.tool_history = restored_tool_history
        return source

    @staticmethod
    def _parse_action(text: str) -> Action | None:
        obj = Agent._extract_json_object(text)
        if not isinstance(obj, dict):
            return None

        action_type = obj.get("type")
        if not isinstance(action_type, str):
            return None

        return Action(type=action_type, payload=obj)

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        stripped = text.strip()

        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for idx, char in enumerate(stripped):
            if char != "{":
                continue
            candidate = stripped[idx:]
            try:
                parsed, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None
