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

PARALLEL_RESULT_PROMPT = (
    "Parallel tool execution result:\n"
    "{result}\n\n"
    "Continue. If the task is complete, respond with a final message."
)

DELEGATE_RESULT_PROMPT = (
    "Delegated subtask result:\n"
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
   - {{"type": "parallel_tools", "calls": [{{"tool": "<tool>", "args": {{...}}}}], "max_workers": 4, "reason": "short reason"}}
   - {{"type": "delegate", "prompt": "subtask prompt", "max_steps": 6, "reason": "short reason"}}
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
        delegation_depth: int = 0,
        max_delegation_depth: int = 2,
    ):
        self.llm = llm
        self.tools = tools
        self.workspace_root = workspace_root
        self.max_steps = max_steps
        self.temperature = temperature
        self.on_tool_event = on_tool_event
        self.delegation_depth = delegation_depth
        self.max_delegation_depth = max_delegation_depth

        self._system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            workspace_root=workspace_root,
            tool_specs=tools.render_specs_for_prompt(),
        )
        self._messages: list[Message] = [Message(role="system", content=self._system_prompt)]
        self.tool_history: list[dict[str, Any]] = []
        self.delegate_history: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._messages = [Message(role="system", content=self._system_prompt)]
        self.tool_history = []
        self.delegate_history = []

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
                result = self._handle_tool_action(action.payload)
                self._messages.append(Message(role="user", content=TOOL_RESULT_PROMPT.format(result=result)))
                continue

            if action.type == "parallel_tools":
                result = self._handle_parallel_action(action.payload)
                self._messages.append(
                    Message(role="user", content=PARALLEL_RESULT_PROMPT.format(result=result))
                )
                continue

            if action.type == "delegate":
                result = self._handle_delegate_action(action.payload)
                self._messages.append(
                    Message(role="user", content=DELEGATE_RESULT_PROMPT.format(result=result))
                )
                continue

            self._messages.append(
                Message(
                    role="user",
                    content=(
                        "Invalid action type. Respond with JSON using type='tool', "
                        "type='parallel_tools', type='delegate', or type='final'."
                    ),
                )
            )

        return (
            "I hit the max tool step limit before completing the task. "
            "Please retry with a narrower request or higher --max-steps."
        )

    def _handle_tool_action(self, payload: dict[str, Any]) -> str:
        tool_name = str(payload.get("tool", "")).strip()
        args = payload.get("args")
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

        return result

    def _handle_parallel_action(self, payload: dict[str, Any]) -> str:
        calls = payload.get("calls")
        max_workers = int(payload.get("max_workers", 4))

        args = {
            "calls": calls,
            "max_workers": max_workers,
        }

        if self.on_tool_event is not None:
            self.on_tool_event(
                f"parallel_tools {json.dumps(args, ensure_ascii=True)[:240]}"
            )

        result = self.tools.execute("parallel_tools", args)
        self.tool_history.append(
            {
                "tool": "parallel_tools",
                "args": json.loads(json.dumps(args, ensure_ascii=True)),
                "result": result,
            }
        )

        if self.on_tool_event is not None:
            self.on_tool_event(f"result {result[:240].replace(chr(10), ' ')}")

        return result

    def _handle_delegate_action(self, payload: dict[str, Any]) -> str:
        prompt = str(payload.get("prompt", "")).strip()
        if prompt == "":
            return "ERROR: delegate action requires non-empty prompt"

        if self.delegation_depth >= self.max_delegation_depth:
            return (
                "ERROR: delegation depth limit reached "
                f"({self.max_delegation_depth}). Continue without delegating."
            )

        child_max_steps = int(payload.get("max_steps", max(2, self.max_steps // 2)))
        child_max_steps = max(1, min(child_max_steps, self.max_steps))

        def child_event(message: str) -> None:
            if self.on_tool_event is not None:
                self.on_tool_event(f"delegate[{self.delegation_depth + 1}] {message}")

        child_agent = Agent(
            llm=self.llm,
            tools=self.tools,
            workspace_root=self.workspace_root,
            max_steps=child_max_steps,
            temperature=self.temperature,
            on_tool_event=child_event if self.on_tool_event is not None else None,
            delegation_depth=self.delegation_depth + 1,
            max_delegation_depth=self.max_delegation_depth,
        )

        result = child_agent.run(prompt)
        self.delegate_history.append(
            {
                "prompt": prompt,
                "max_steps": child_max_steps,
                "result": result,
            }
        )

        if self.on_tool_event is not None:
            self.on_tool_event(f"delegate_result {result[:240].replace(chr(10), ' ')}")

        return result

    def session_payload(self) -> dict[str, Any]:
        messages = [{"role": message.role, "content": message.content} for message in self._messages]
        return {
            "version": 2,
            "messages": messages,
            "tool_history": self.tool_history,
            "delegate_history": self.delegate_history,
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

        self.tool_history = self._restore_history(payload.get("tool_history"))
        self.delegate_history = self._restore_history(payload.get("delegate_history"))
        return source

    @staticmethod
    def _restore_history(raw_value: Any) -> list[dict[str, Any]]:
        restored: list[dict[str, Any]] = []
        if not isinstance(raw_value, list):
            return restored

        for entry in raw_value:
            if not isinstance(entry, dict):
                continue
            restored.append(json.loads(json.dumps(entry, ensure_ascii=True)))
        return restored

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
