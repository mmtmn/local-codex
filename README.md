# local-cli

`local-cli` is a Rust, open-source coding orchestrator for local/open LLM backends.

It is designed as a practical local alternative focused on useful core workflows:
- multi-model routing by role (`text`, `planner`, `coder`, `image`)
- named agents and delegated subtasks
- git-aware coding tools
- persistent memory and session save/load
- structured code editing tools
- plugin-style external tools
- auto mode by default (no approval prompts)

No skills, MCP integration, or external sandbox layer are built in.

## Features

### Orchestration
- JSON action loop with tool calls, parallel tool calls, and delegation.
- Named agents (`/agent new`, `/agent use`, `/agent run`) for task decomposition.
- Delegation depth control with `--max-delegation-depth`.

### Tooling
- File tools: `list_files`, `read_file`, `write_file`, `replace_in_file`.
- Structured edits: `structured_patch` operations.
- Symbol-aware Python edits: `python_symbol_overview`, `replace_python_symbol`.
- Shell execution: `run_shell`.
- Calculator launcher: `open_calculator` (plus `run_shell` shortcut for `calc`/`calculator`).
- Git tools: `git_status`, `git_diff`, `git_log`, `git_commit_plan`, `git_commit`, `apply_patch`.
- Persistent memory: `memory_get`, `memory_set`, `memory_delete`, `memory_search`.
- Parallel execution: `parallel_tools`.
- Plugin tools loaded from `.local_codex/plugins/*.json`.

### Model Routing
Choose models per role at CLI startup:
- `--text-model`
- `--planner-model`
- `--coder-model`
- `--image-model`

If planner/coder/image are omitted, they fall back to `--text-model`.

## Quickstart

### 1) Prerequisites
- Rust toolchain (`cargo`)
- A local or remote chat-completions backend
  - Ollama example: Qwen model installed locally

### 2) Run with Ollama + Qwen

```bash
cargo run --bin local -- \
  --provider ollama \
  --text-model qwen3:8b \
  --endpoint http://127.0.0.1:11434/api/chat
```

If your model tag differs, use your installed tag.

Install once so you can just type `local`:

```bash
cargo install --path .
local
```

### 3) One-shot prompt mode

```bash
cargo run --bin local -- \
  --provider ollama \
  --text-model qwen3:8b \
  --prompt "List files and summarize this repository"
```

## Interactive Commands

- `/exit`, `/quit`
- `/reset`
- `/save [path]`
- `/load [path]`
- `/reload_plugins`
- `/models` or `/routing`
- `/agents`
- `/agent list`
- `/agent new <name>`
- `/agent use <name>`
- `/agent run <name> <prompt>`
- `/agent reset <name>`

## Session Persistence

Use `--session-file` to load/save state automatically:

```bash
cargo run --bin local -- \
  --provider ollama \
  --text-model qwen3:8b \
  --session-file .local_codex/session.json
```

Start fresh while keeping the same file path:

```bash
cargo run --bin local -- \
  --provider ollama \
  --text-model qwen3:8b \
  --session-file .local_codex/session.json \
  --fresh-session
```

## Approvals

Auto mode is enabled by default (no permission prompts).
If you want prompts for shell/plugin/commit/calculator actions, run with:

```bash
cargo run --bin local -- \
  --provider ollama \
  --text-model qwen3:8b \
  --ask-permissions
```

(`--auto-mode` / `--auto-approve` are still accepted and force auto mode on.)

## OpenAI-Compatible Endpoints

```bash
cargo run --bin local -- \
  --provider openai \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --text-model Qwen/Qwen2.5-Coder-7B-Instruct
```

If needed:

```bash
export OPENAI_API_KEY=your_key
```

## Plugin Tool Specs

Place JSON files in `.local_codex/plugins/` (or override with `--plugins-dir`).

Example:

```json
{
  "name": "echo",
  "description": "Echo a value",
  "command": ["python3", "-c", "import json,sys; p=json.load(sys.stdin); print(p.get('value',''))"],
  "args_schema": {"value": "value to echo"}
}
```

This creates `plugin.echo`.

## Tests

```bash
cargo test
```
