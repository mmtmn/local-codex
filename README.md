# local-codex

`local-codex` is a lightweight open-source coding assistant CLI designed to run with local LLMs (starting with Qwen via Ollama).

## What this MVP does

- Runs an interactive terminal coding assistant loop.
- Uses a local model through Ollama (`/api/chat`) or any OpenAI-compatible endpoint.
- Lets the model use core coding tools:
  - `list_files`
  - `read_file`
  - `write_file`
  - `run_shell`
- Restricts file operations to your workspace root.

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

## Roadmap ideas

- Persistent memory and task sessions.
- Git-aware workflows (diffs, commit planning, patch mode).
- Smarter code editing tools (structured patch tool, symbol-aware edits).
- Plugin-style tool system.
- Multi-agent decomposition and parallel tool execution.
