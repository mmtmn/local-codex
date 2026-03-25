from __future__ import annotations

import argparse
import os
from pathlib import Path

from local_codex.agent import Agent
from local_codex.config import Settings
from local_codex.llm.base import LLMClient
from local_codex.llm.ollama import OllamaClient
from local_codex.llm.openai_compat import OpenAICompatibleClient
from local_codex.tools import ToolExecutor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-codex",
        description="Local coding assistant CLI for open-source LLMs",
    )
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--workspace", default=str(Path.cwd()))
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--shell-timeout", type=int, default=45)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument(
        "--memory-file",
        default=None,
        help="Optional JSON file for persistent memory (default .local_codex/memory.json)",
    )
    parser.add_argument(
        "--plugins-dir",
        default=None,
        help="Optional directory for plugin tool JSON specs (default .local_codex/plugins)",
    )
    parser.add_argument(
        "--max-delegation-depth",
        type=int,
        default=2,
        help="Maximum depth for delegate sub-agents",
    )
    parser.add_argument(
        "--session-file",
        default=None,
        help="Optional JSON file to persist conversation and tool history",
    )
    parser.add_argument(
        "--fresh-session",
        action="store_true",
        help="Ignore any existing --session-file and start a new session",
    )
    parser.add_argument("--prompt", default=None, help="Run one prompt non-interactively")
    return parser


def make_settings(args: argparse.Namespace) -> Settings:
    endpoint = args.endpoint
    if endpoint is None:
        if args.provider == "ollama":
            endpoint = "http://127.0.0.1:11434/api/chat"
        else:
            endpoint = "http://127.0.0.1:8000/v1/chat/completions"

    api_key = args.api_key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    return Settings(
        provider=args.provider,
        model=args.model,
        endpoint=endpoint,
        api_key=api_key,
        workspace=Path(args.workspace),
        max_steps=max(1, args.max_steps),
        shell_timeout_seconds=max(1, args.shell_timeout),
        auto_approve=bool(args.auto_approve),
        temperature=args.temperature,
        memory_file=Path(args.memory_file).expanduser() if args.memory_file else None,
        plugins_dir=Path(args.plugins_dir).expanduser() if args.plugins_dir else None,
        max_delegation_depth=max(0, args.max_delegation_depth),
    )


def make_client(settings: Settings) -> LLMClient:
    if settings.provider == "ollama":
        return OllamaClient(endpoint=settings.endpoint, model=settings.model)

    return OpenAICompatibleClient(
        endpoint=settings.endpoint,
        model=settings.model,
        api_key=settings.api_key,
    )


def run_once(agent: Agent, prompt: str, session_file: Path | None = None) -> int:
    response = agent.run(prompt)
    print(response)
    if session_file is not None:
        agent.save_session(session_file)
    return 0


def repl(agent: Agent, session_file: Path | None = None) -> int:
    print("local-codex interactive mode")
    print("Commands: /exit, /quit, /reset, /save [path], /load [path], /reload_plugins")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not user_input:
            continue

        if user_input.startswith("/"):
            command, _, rest = user_input.partition(" ")
            argument = rest.strip()

            if command in {"/exit", "/quit"}:
                return 0

            if command == "/reset":
                agent.reset()
                print("Conversation reset.")
                continue

            if command == "/save":
                target = Path(argument).expanduser() if argument else session_file
                if target is None:
                    print("assistant> error: no session path provided")
                    continue
                try:
                    saved_to = agent.save_session(target)
                except Exception as exc:  # noqa: BLE001 - CLI should keep running
                    print(f"assistant> error: {exc}")
                    continue
                print(f"assistant> session saved: {saved_to}")
                continue

            if command == "/load":
                target = Path(argument).expanduser() if argument else session_file
                if target is None:
                    print("assistant> error: no session path provided")
                    continue
                try:
                    loaded_from = agent.load_session(target)
                except Exception as exc:  # noqa: BLE001 - CLI should keep running
                    print(f"assistant> error: {exc}")
                    continue
                print(f"assistant> session loaded: {loaded_from}")
                continue

            if command == "/reload_plugins":
                result = agent.tools.execute("reload_plugins", {})
                print(f"assistant> {result}")
                continue

            print(f"assistant> error: unknown command {command}")
            continue

        try:
            answer = agent.run(user_input)
        except Exception as exc:  # noqa: BLE001 - CLI should keep running
            print(f"assistant> error: {exc}")
            continue

        print(f"assistant> {answer}")
        if session_file is not None:
            try:
                agent.save_session(session_file)
            except Exception as exc:  # noqa: BLE001 - CLI should keep running
                print(f"assistant> warning: failed to save session: {exc}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = make_settings(args)

    tools = ToolExecutor(
        workspace_root=settings.workspace_resolved,
        auto_approve=settings.auto_approve,
        shell_timeout_seconds=settings.shell_timeout_seconds,
        memory_file=settings.memory_file_resolved,
        plugins_dir=settings.plugins_dir_resolved,
    )
    client = make_client(settings)

    def on_tool_event(message: str) -> None:
        print(f"[tool] {message}")

    agent = Agent(
        llm=client,
        tools=tools,
        workspace_root=str(settings.workspace_resolved),
        max_steps=settings.max_steps,
        temperature=settings.temperature,
        on_tool_event=on_tool_event,
        max_delegation_depth=settings.max_delegation_depth,
    )

    session_file: Path | None = None
    if args.session_file:
        session_file = Path(args.session_file).expanduser()
        if session_file.exists() and not args.fresh_session:
            try:
                agent.load_session(session_file)
                print(f"Loaded session: {session_file}")
            except Exception as exc:  # noqa: BLE001 - startup should continue
                print(f"Warning: failed to load session {session_file}: {exc}")
        elif args.fresh_session and session_file.exists():
            print(f"Starting fresh session. Ignoring existing file: {session_file}")

    if args.prompt:
        raise SystemExit(run_once(agent, args.prompt, session_file=session_file))

    raise SystemExit(repl(agent, session_file=session_file))


if __name__ == "__main__":
    main()
