# local-codex

`local-codex` is a lightweight open-source coding assistant CLI designed to run with local LLMs (starting with Qwen via Ollama).

## What this MVP does

- Runs an interactive terminal coding assistant loop.
- Uses a local model through Ollama (`/api/chat`) or any OpenAI-compatible endpoint.
- Lets the model use core coding tools:
  - `list_files`
  - `read_file`
  - `write_file`
  - `replace_in_file`
  - `structured_patch`
  - `python_symbol_overview`
  - `replace_python_symbol`
  - `run_shell`
  - `git_status`
  - `git_diff`
  - `git_log`
  - `git_commit_plan`
  - `git_commit`
  - `apply_patch`
  - `memory_get` / `memory_set` / `memory_delete` / `memory_search`
  - `parallel_tools`
- Supports plugin-style tools loaded from `.local_codex/plugins/*.json`.
- Supports task decomposition through delegated sub-agents (`delegate` action type).
- Restricts file operations and patch targets to your workspace root.
- Supports persistent sessions and memory.

## Quickstart

1. Make sure Python 3.11+ is installed.
2. Make sure Ollama is running and your Qwen model is available.

Example (model name may vary based on your install):

```bash
ollama list
# then use the exact model name below
```

3. Create a virtual environment and install this package in editable mode:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

4. Start the assistant:

```bash
local-codex --provider ollama --model qwen3:8b
```

5. Type a request, for example:

```text
Create a hello world script and run it.
```

## Non-interactive mode

```bash
local-codex --provider ollama --model qwen3:8b --prompt "List the files in this project"
```

## Session persistence

Use `--session-file` to auto-load and auto-save a session:

```bash
local-codex --provider ollama --model qwen3:8b --session-file .local_codex_session.json
```

Start fresh while keeping the same session path:

```bash
local-codex --provider ollama --model qwen3:8b --session-file .local_codex_session.json --fresh-session
```

In interactive mode:

- `/save [path]` saves the current session.
- `/load [path]` loads a session.
- `/reset` clears current in-memory conversation/history.
- `/reload_plugins` reloads plugin tools from disk.

## Persistent memory

By default memory is stored at `.local_codex/memory.json`. Override it:

```bash
local-codex \
  --provider ollama \
  --model qwen3:8b \
  --memory-file .local_codex/memory.json
```

The model can use memory tools to keep long-lived project context.

## Plugin tools

By default plugin specs are loaded from `.local_codex/plugins/*.json`. Override:

```bash
local-codex \
  --provider ollama \
  --model qwen3:8b \
  --plugins-dir .local_codex/plugins
```

Minimal plugin spec example:

```json
{
  "name": "echo",
  "description": "Echo a value from tool args",
  "command": ["python3", "-c", "import json,sys; p=json.load(sys.stdin); print(p.get('value',''))"],
  "args_schema": { "value": "value to print" }
}
```

This exposes a new tool named `plugin.echo`.

## Multi-Agent And Parallelism

- The model can return `{\"type\":\"parallel_tools\", ...}` to run multiple tools concurrently.
- The model can return `{\"type\":\"delegate\", ...}` to spin a bounded sub-agent for a subtask.
- Control max delegate depth with `--max-delegation-depth` (default `2`).

## OpenAI-compatible local endpoints

You can also target local servers that expose OpenAI-compatible chat completions:

```bash
local-codex \
  --provider openai \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen/Qwen2.5-Coder-7B-Instruct
```

If your server requires an API key, set:

```bash
export OPENAI_API_KEY=your_key
```

## Safety notes

- `run_shell` executes commands in your workspace directory.
- By default, shell commands require approval at runtime.
- Use `--auto-approve` to allow all shell commands without prompts.
- `apply_patch` validates patch paths and rejects unsafe targets (for example `../` escapes).
- Plugin commands also require approval unless `--auto-approve` is enabled.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
