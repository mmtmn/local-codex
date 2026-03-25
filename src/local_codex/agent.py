from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
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

    def reset(self) -> None:
        self._messages = [Message(role="system", content=self._system_prompt)]

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
