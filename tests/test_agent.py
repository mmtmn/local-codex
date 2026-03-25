from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_codex.agent import Agent
from local_codex.tools import ToolExecutor


class StubLLM:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.index = 0

    def chat(self, messages, temperature: float = 0.1) -> str:  # noqa: ANN001
        if self.index >= len(self.responses):
            return '{"type": "final", "message": "stub done"}'
        value = self.responses[self.index]
        self.index += 1
        return value


class AgentParsingTest(unittest.TestCase):
    def test_parse_simple_json_action(self) -> None:
        action = Agent._parse_action('{"type": "final", "message": "ok"}')
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.type, "final")

    def test_parse_fenced_json_action(self) -> None:
        response = "```json\n{\"type\": \"tool\", \"tool\": \"list_files\", \"args\": {}}\n```"
        action = Agent._parse_action(response)
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.type, "tool")
        self.assertEqual(action.payload.get("tool"), "list_files")

    def test_extract_embedded_json_object(self) -> None:
        response = "I will do this now. {\"type\":\"final\",\"message\":\"done\"}"
        action = Agent._parse_action(response)
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.payload.get("message"), "done")

    def test_parse_invalid_response_returns_none(self) -> None:
        action = Agent._parse_action("not json")
        self.assertIsNone(action)

    def test_session_round_trip_restores_messages_and_tool_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            tools = ToolExecutor(workspace_root=workspace, auto_approve=True)
            llm = StubLLM(
                [
                    '{"type":"tool","tool":"list_files","args":{"path":".","max_depth":1}}',
                    '{"type":"final","message":"done"}',
                ]
            )

            agent = Agent(llm=llm, tools=tools, workspace_root=str(workspace), max_steps=4)
            final_message = agent.run("List files then finish")
            self.assertEqual(final_message, "done")
            self.assertEqual(len(agent.tool_history), 1)

            session_file = workspace / "session.json"
            saved = agent.save_session(session_file)
            self.assertTrue(saved.exists())

            new_agent = Agent(
                llm=StubLLM(['{"type":"final","message":"ok"}']),
                tools=tools,
                workspace_root=str(workspace),
                max_steps=4,
            )
            loaded = new_agent.load_session(session_file)
            self.assertEqual(loaded, session_file)

            payload = new_agent.session_payload()
            self.assertGreater(len(payload["messages"]), 1)
            self.assertEqual(len(payload["tool_history"]), 1)


if __name__ == "__main__":
    unittest.main()
