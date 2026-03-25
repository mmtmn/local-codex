from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    args: dict[str, str]


@dataclass(slots=True)
class PluginSpec:
    name: str
    description: str
    command: list[str]
    args_schema: dict[str, str]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class ToolExecutor:
    def __init__(
        self,
        workspace_root: Path,
        auto_approve: bool = False,
        shell_timeout_seconds: int = 45,
        memory_file: Path | None = None,
        plugins_dir: Path | None = None,
    ):
        self.workspace_root = workspace_root.resolve()
        self.auto_approve = auto_approve
        self.shell_timeout_seconds = shell_timeout_seconds

        raw_memory_file = memory_file or (self.workspace_root / ".local_codex" / "memory.json")
        if not raw_memory_file.is_absolute():
            raw_memory_file = self.workspace_root / raw_memory_file
        self.memory_file = raw_memory_file.resolve()

        raw_plugins_dir = plugins_dir or (self.workspace_root / ".local_codex" / "plugins")
        if not raw_plugins_dir.is_absolute():
            raw_plugins_dir = self.workspace_root / raw_plugins_dir
        self.plugins_dir = raw_plugins_dir.resolve()

        self.plugins: dict[str, PluginSpec] = self._load_plugins()

    def list_specs(self) -> list[ToolSpec]:
        specs = [
            ToolSpec(
                name="list_files",
                description="Recursively list files and directories under a path.",
                args={
                    "path": "string path relative to workspace, default '.'",
                    "max_depth": "integer depth limit, default 3",
                },
            ),
            ToolSpec(
                name="read_file",
                description="Read a text file with line numbers.",
                args={
                    "path": "string path relative to workspace",
                    "start_line": "1-based start line, default 1",
                    "end_line": "1-based end line inclusive, default 200",
                },
            ),
            ToolSpec(
                name="write_file",
                description="Write text content to a file (overwrites existing).",
                args={
                    "path": "string path relative to workspace",
                    "content": "full file content as string",
                },
            ),
            ToolSpec(
                name="replace_in_file",
                description="Replace text in a file with exact string matching.",
                args={
                    "path": "string path relative to workspace",
                    "old": "text to replace",
                    "new": "replacement text",
                    "count": "optional max replacements, default all",
                },
            ),
            ToolSpec(
                name="structured_patch",
                description="Apply structured edit operations to files.",
                args={
                    "operations": "list of operations: replace_text, replace_lines, append_text",
                },
            ),
            ToolSpec(
                name="python_symbol_overview",
                description="List top-level Python symbols (functions/classes) in a file.",
                args={"path": "python file path relative to workspace"},
            ),
            ToolSpec(
                name="replace_python_symbol",
                description="Replace one top-level Python function/class by symbol name.",
                args={
                    "path": "python file path relative to workspace",
                    "symbol": "function or class name",
                    "symbol_type": "optional 'function' or 'class'",
                    "code": "replacement code block",
                },
            ),
            ToolSpec(
                name="run_shell",
                description="Run a shell command in the workspace root.",
                args={
                    "command": "shell command string",
                },
            ),
            ToolSpec(
                name="git_status",
                description="Show git status (short + branch) for the workspace repository.",
                args={},
            ),
            ToolSpec(
                name="git_diff",
                description="Show git diff for workspace, optionally for a specific path.",
                args={
                    "path": "optional path relative to workspace",
                    "staged": "optional boolean, default false",
                    "unified": "optional context lines, default 3",
                },
            ),
            ToolSpec(
                name="git_log",
                description="Show recent commit log lines.",
                args={"count": "optional number of commits, default 10"},
            ),
            ToolSpec(
                name="git_commit_plan",
                description="Summarize pending changes and propose a commit message.",
                args={},
            ),
            ToolSpec(
                name="git_commit",
                description="Create a git commit in workspace repository.",
                args={
                    "message": "commit message",
                    "add_all": "optional boolean, default false",
                },
            ),
            ToolSpec(
                name="apply_patch",
                description="Apply a unified diff patch using `git apply`.",
                args={
                    "patch": "unified diff text",
                },
            ),
            ToolSpec(
                name="memory_get",
                description="Read persistent memory key/value entries.",
                args={"key": "optional key; omit to return full memory"},
            ),
            ToolSpec(
                name="memory_set",
                description="Set a persistent memory key/value entry.",
                args={"key": "memory key", "value": "memory value"},
            ),
            ToolSpec(
                name="memory_delete",
                description="Delete a persistent memory key.",
                args={"key": "memory key"},
            ),
            ToolSpec(
                name="memory_search",
                description="Search persistent memory by text query.",
                args={"query": "search string", "limit": "optional max results, default 10"},
            ),
            ToolSpec(
                name="parallel_tools",
                description="Execute multiple tool calls concurrently and return all results.",
                args={
                    "calls": "list of tool calls: [{tool: str, args: dict}]",
                    "max_workers": "optional worker count, default 4",
                },
            ),
            ToolSpec(
                name="reload_plugins",
                description="Reload plugin tools from the plugins directory.",
                args={},
            ),
        ]

        for plugin_name in sorted(self.plugins):
            plugin = self.plugins[plugin_name]
            specs.append(
                ToolSpec(
                    name=f"plugin.{plugin.name}",
                    description=plugin.description,
                    args=plugin.args_schema,
                )
            )

        return specs

    def render_specs_for_prompt(self) -> str:
        specs = []
        for spec in self.list_specs():
            specs.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "args": spec.args,
                }
            )
        return json.dumps(specs, indent=2, ensure_ascii=True)

    def execute(self, tool_name: str, args: dict[str, Any]) -> str:
        try:
            if tool_name.startswith("plugin."):
                return self._run_plugin_tool(tool_name.removeprefix("plugin."), args)

            if tool_name == "list_files":
                return self._list_files(
                    path=str(args.get("path", ".")),
                    max_depth=int(args.get("max_depth", 3)),
                )
            if tool_name == "read_file":
                return self._read_file(
                    path=str(args["path"]),
                    start_line=int(args.get("start_line", 1)),
                    end_line=int(args.get("end_line", 200)),
                )
            if tool_name == "write_file":
                return self._write_file(
                    path=str(args["path"]),
                    content=str(args.get("content", "")),
                )
            if tool_name == "replace_in_file":
                return self._replace_in_file(
                    path=str(args["path"]),
                    old=str(args["old"]),
                    new=str(args.get("new", "")),
                    count=args.get("count"),
                )
            if tool_name == "structured_patch":
                return self._structured_patch(args.get("operations"))
            if tool_name == "python_symbol_overview":
                return self._python_symbol_overview(path=str(args["path"]))
            if tool_name == "replace_python_symbol":
                return self._replace_python_symbol(
                    path=str(args["path"]),
                    symbol=str(args["symbol"]),
                    code=str(args["code"]),
                    symbol_type=str(args.get("symbol_type", "")) or None,
                )
            if tool_name == "run_shell":
                return self._run_shell(command=str(args["command"]))

            if tool_name == "git_status":
                return self._git_status()
            if tool_name == "git_diff":
                path_value = args.get("path")
                path = None if path_value is None else str(path_value)
                return self._git_diff(
                    path=path,
                    staged=bool(args.get("staged", False)),
                    unified=int(args.get("unified", 3)),
                )
            if tool_name == "git_log":
                return self._git_log(count=int(args.get("count", 10)))
            if tool_name == "git_commit_plan":
                return self._git_commit_plan()
            if tool_name == "git_commit":
                return self._git_commit(
                    message=str(args["message"]),
                    add_all=bool(args.get("add_all", False)),
                )
            if tool_name == "apply_patch":
                return self._apply_patch(patch=str(args["patch"]))

            if tool_name == "memory_get":
                key = args.get("key")
                return self._memory_get(key=None if key is None else str(key))
            if tool_name == "memory_set":
                return self._memory_set(key=str(args["key"]), value=args.get("value"))
            if tool_name == "memory_delete":
                return self._memory_delete(key=str(args["key"]))
            if tool_name == "memory_search":
                return self._memory_search(
                    query=str(args["query"]),
                    limit=int(args.get("limit", 10)),
                )

            if tool_name == "parallel_tools":
                return self._parallel_tools(
                    calls=args.get("calls"),
                    max_workers=int(args.get("max_workers", 4)),
                )
            if tool_name == "reload_plugins":
                return self._reload_plugins()

            return f"ERROR: unknown tool '{tool_name}'"
        except KeyError as exc:
            return f"ERROR: missing argument {exc} for tool '{tool_name}'"
        except Exception as exc:  # noqa: BLE001 - tool errors are returned to model
            return f"ERROR: {exc}"

    def _resolve(self, path: str) -> Path:
        target = (self.workspace_root / path).resolve()
        if not _is_relative_to(target, self.workspace_root):
            raise ValueError("path escapes workspace root")
        return target

    def _list_files(self, path: str = ".", max_depth: int = 3) -> str:
        base = self._resolve(path)
        if not base.exists():
            return f"ERROR: path does not exist: {path}"

        max_depth = max(0, min(max_depth, 8))
        lines: list[str] = []
        max_entries = 1500

        def walk(node: Path, depth_left: int) -> None:
            nonlocal lines
            if len(lines) >= max_entries:
                return

            if node.is_file():
                rel = node.relative_to(self.workspace_root)
                lines.append(str(rel))
                return

            rel = node.relative_to(self.workspace_root)
            label = "." if str(rel) == "." else f"{rel}/"
            lines.append(label)

            if depth_left == 0:
                return

            try:
                children = sorted(node.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except PermissionError:
                lines.append(f"{label} [permission denied]")
                return

            for child in children:
                if len(lines) >= max_entries:
                    break
                walk(child, depth_left - 1)

        walk(base, max_depth)

        if len(lines) >= max_entries:
            lines.append("... output truncated ...")

        return "\n".join(lines)

    def _read_file(self, path: str, start_line: int = 1, end_line: int = 200) -> str:
        target = self._resolve(path)
        if not target.exists() or not target.is_file():
            return f"ERROR: file not found: {path}"

        if start_line < 1:
            start_line = 1
        if end_line < start_line:
            end_line = start_line
        end_line = min(end_line, start_line + 999)

        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        start_idx = start_line - 1
        end_idx = min(end_line, len(lines))

        numbered: list[str] = []
        for i in range(start_idx, end_idx):
            numbered.append(f"{i + 1:5d}: {lines[i]}")

        header = f"FILE: {path} (lines {start_line}-{end_idx} of {len(lines)})"
        if not numbered:
            return f"{header}\n<no content in selected range>"
        return f"{header}\n" + "\n".join(numbered)

    def _write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"WROTE: {path} ({len(content)} bytes)"

    def _replace_in_file(self, path: str, old: str, new: str, count: Any = None) -> str:
        if old == "":
            return "ERROR: old text cannot be empty"

        target = self._resolve(path)
        if not target.exists() or not target.is_file():
            return f"ERROR: file not found: {path}"

        text = target.read_text(encoding="utf-8", errors="replace")
        occurrences = text.count(old)
        if occurrences == 0:
            return f"NO_CHANGE: '{old}' not found in {path}"

        replace_count = -1
        if count is not None:
            replace_count = int(count)
            if replace_count < 0:
                replace_count = -1

        if replace_count == -1:
            updated = text.replace(old, new)
            applied = occurrences
        else:
            updated = text.replace(old, new, replace_count)
            applied = min(occurrences, replace_count)

        target.write_text(updated, encoding="utf-8")
        return f"REPLACED: {path} occurrences={occurrences} applied={applied}"

    def _structured_patch(self, operations: Any) -> str:
        if not isinstance(operations, list) or not operations:
            return "ERROR: operations must be a non-empty list"

        results: list[str] = []
        for index, operation in enumerate(operations):
            if not isinstance(operation, dict):
                results.append(f"op#{index}: ERROR operation must be an object")
                continue

            op_type = str(operation.get("type", "")).strip()
            if op_type == "replace_text":
                result = self._replace_in_file(
                    path=str(operation["path"]),
                    old=str(operation["old"]),
                    new=str(operation.get("new", "")),
                    count=operation.get("count"),
                )
                results.append(f"op#{index} replace_text: {result}")
                continue

            if op_type == "replace_lines":
                result = self._replace_lines(
                    path=str(operation["path"]),
                    start_line=int(operation["start_line"]),
                    end_line=int(operation["end_line"]),
                    new_text=str(operation.get("new_text", "")),
                )
                results.append(f"op#{index} replace_lines: {result}")
                continue

            if op_type == "append_text":
                result = self._append_text(
                    path=str(operation["path"]),
                    text=str(operation.get("text", "")),
                )
                results.append(f"op#{index} append_text: {result}")
                continue

            results.append(f"op#{index}: ERROR unknown operation type '{op_type}'")

        return "\n".join(results)

    def _replace_lines(self, path: str, start_line: int, end_line: int, new_text: str) -> str:
        target = self._resolve(path)
        if not target.exists() or not target.is_file():
            return f"ERROR: file not found: {path}"
        if start_line < 1 or end_line < start_line:
            return "ERROR: invalid line range"

        source = target.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()
        had_newline = source.endswith("\n")

        if start_line > len(lines) + 1:
            return f"ERROR: start_line out of range for {path}"

        start_idx = start_line - 1
        end_idx = min(end_line, len(lines))
        replacement_lines = new_text.splitlines()
        lines[start_idx:end_idx] = replacement_lines

        updated = "\n".join(lines)
        if had_newline:
            updated += "\n"

        target.write_text(updated, encoding="utf-8")
        return f"WROTE: {path} lines {start_line}-{end_line}"

    def _append_text(self, path: str, text: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(text)
        return f"APPENDED: {path} ({len(text)} bytes)"

    def _python_symbol_overview(self, path: str) -> str:
        target = self._resolve(path)
        if not target.exists() or not target.is_file():
            return f"ERROR: file not found: {path}"
        if target.suffix != ".py":
            return "ERROR: python_symbol_overview only supports .py files"

        source = target.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return f"ERROR: could not parse Python file: {exc}"

        symbols: list[dict[str, Any]] = []
        for node in tree.body:
            symbol_type = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol_type = "function"
            elif isinstance(node, ast.ClassDef):
                symbol_type = "class"

            if symbol_type is None:
                continue

            symbols.append(
                {
                    "name": node.name,
                    "type": symbol_type,
                    "start_line": int(getattr(node, "lineno", 0)),
                    "end_line": int(getattr(node, "end_lineno", getattr(node, "lineno", 0))),
                }
            )

        payload = {
            "path": path,
            "symbols": symbols,
        }
        return json.dumps(payload, indent=2, ensure_ascii=True)

    def _replace_python_symbol(
        self,
        path: str,
        symbol: str,
        code: str,
        symbol_type: str | None = None,
    ) -> str:
        target = self._resolve(path)
        if not target.exists() or not target.is_file():
            return f"ERROR: file not found: {path}"
        if target.suffix != ".py":
            return "ERROR: replace_python_symbol only supports .py files"

        source = target.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return f"ERROR: could not parse Python file: {exc}"

        wanted_type = symbol_type.strip().lower() if symbol_type else ""
        selected_node = None
        for node in tree.body:
            is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            is_class = isinstance(node, ast.ClassDef)
            if not (is_function or is_class):
                continue
            if node.name != symbol:
                continue
            if wanted_type == "function" and not is_function:
                continue
            if wanted_type == "class" and not is_class:
                continue
            selected_node = node
            break

        if selected_node is None:
            return f"ERROR: symbol not found: {symbol}"

        start_line = int(getattr(selected_node, "lineno", 0))
        end_line = int(getattr(selected_node, "end_lineno", start_line))
        if start_line < 1 or end_line < start_line:
            return f"ERROR: invalid symbol line range for {symbol}"

        original_lines = source.splitlines()
        had_newline = source.endswith("\n")
        replacement_lines = code.splitlines()

        original_lines[start_line - 1 : end_line] = replacement_lines
        updated = "\n".join(original_lines)
        if had_newline:
            updated += "\n"

        target.write_text(updated, encoding="utf-8")
        return f"REPLACED_SYMBOL: {symbol} in {path} (lines {start_line}-{end_line})"

    def _run_shell(self, command: str) -> str:
        if not self.auto_approve:
            answer = input(f"Approve shell command? [y/N]\n$ {command}\n> ").strip().lower()
            if answer not in {"y", "yes"}:
                return "DENIED: shell command not approved by user"

        completed = subprocess.run(
            command,
            cwd=self.workspace_root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.shell_timeout_seconds,
        )

        return self._format_process_output(completed.returncode, completed.stdout, completed.stderr)

    def _git_status(self) -> str:
        git_error = self._require_git_repo()
        if git_error is not None:
            return git_error

        completed = subprocess.run(
            ["git", "-C", str(self.workspace_root), "status", "--short", "--branch"],
            capture_output=True,
            text=True,
            timeout=self.shell_timeout_seconds,
        )
        return self._format_process_output(completed.returncode, completed.stdout, completed.stderr)

    def _git_diff(self, path: str | None = None, staged: bool = False, unified: int = 3) -> str:
        git_error = self._require_git_repo()
        if git_error is not None:
            return git_error

        safe_unified = max(0, min(unified, 20))
        command = [
            "git",
            "-C",
            str(self.workspace_root),
            "diff",
            "--no-color",
            f"--unified={safe_unified}",
        ]
        if staged:
            command.append("--staged")

        if path:
            resolved = self._resolve(path)
            rel = resolved.relative_to(self.workspace_root)
            command.extend(["--", rel.as_posix()])

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.shell_timeout_seconds,
        )
        if completed.returncode == 0 and not completed.stdout.strip():
            return "NO_DIFF: no changes for selected scope"
        return self._format_process_output(completed.returncode, completed.stdout, completed.stderr)

    def _git_log(self, count: int = 10) -> str:
        git_error = self._require_git_repo()
        if git_error is not None:
            return git_error

        safe_count = max(1, min(count, 100))
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(self.workspace_root),
                "log",
                "--oneline",
                "--decorate",
                "-n",
                str(safe_count),
            ],
            capture_output=True,
            text=True,
            timeout=self.shell_timeout_seconds,
        )
        return self._format_process_output(completed.returncode, completed.stdout, completed.stderr)

    def _git_commit_plan(self) -> str:
        git_error = self._require_git_repo()
        if git_error is not None:
            return git_error

        completed = subprocess.run(
            ["git", "-C", str(self.workspace_root), "status", "--porcelain=1"],
            capture_output=True,
            text=True,
            timeout=self.shell_timeout_seconds,
        )
        if completed.returncode != 0:
            return self._format_process_output(completed.returncode, completed.stdout, completed.stderr)

        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []

        for raw_line in completed.stdout.splitlines():
            if len(raw_line) < 4:
                continue
            status = raw_line[:2]
            path = raw_line[3:]
            if status == "??":
                untracked.append(path)
                continue
            if status[0] not in {" ", "?"}:
                staged.append(path)
            if status[1] != " ":
                unstaged.append(path)

        changed_paths = sorted({*staged, *unstaged, *untracked})
        message = self._suggest_commit_message(changed_paths)

        payload = {
            "staged": sorted(staged),
            "unstaged": sorted(unstaged),
            "untracked": sorted(untracked),
            "changed_paths": changed_paths,
            "suggested_commit_message": message,
        }
        return json.dumps(payload, indent=2, ensure_ascii=True)

    def _suggest_commit_message(self, changed_paths: list[str]) -> str:
        if not changed_paths:
            return "chore: no pending changes"

        scopes = sorted({path.split("/", 1)[0] for path in changed_paths})
        scope_summary = ", ".join(scopes[:3])
        if len(scopes) > 3:
            scope_summary += ", ..."

        if any(path.startswith("tests/") for path in changed_paths):
            prefix = "test"
        elif any(path.startswith("src/") for path in changed_paths):
            prefix = "feat"
        else:
            prefix = "chore"

        return f"{prefix}: update {len(changed_paths)} files ({scope_summary})"

    def _git_commit(self, message: str, add_all: bool = False) -> str:
        if not message.strip():
            return "ERROR: commit message cannot be empty"

        git_error = self._require_git_repo()
        if git_error is not None:
            return git_error

        if add_all:
            add_result = subprocess.run(
                ["git", "-C", str(self.workspace_root), "add", "-A"],
                capture_output=True,
                text=True,
                timeout=self.shell_timeout_seconds,
            )
            if add_result.returncode != 0:
                return (
                    "ERROR: git add failed\n"
                    + self._format_process_output(
                        add_result.returncode,
                        add_result.stdout,
                        add_result.stderr,
                    )
                )

        if not self.auto_approve:
            answer = input(f"Approve git commit? [y/N]\nmessage: {message}\n> ").strip().lower()
            if answer not in {"y", "yes"}:
                return "DENIED: git commit not approved by user"

        completed = subprocess.run(
            ["git", "-C", str(self.workspace_root), "commit", "-m", message],
            capture_output=True,
            text=True,
            timeout=self.shell_timeout_seconds,
        )
        return self._format_process_output(completed.returncode, completed.stdout, completed.stderr)

    def _apply_patch(self, patch: str) -> str:
        if not patch.strip():
            return "ERROR: patch is empty"

        git_error = self._require_git_repo()
        if git_error is not None:
            return git_error

        invalid_path = self._find_unsafe_patch_path(patch)
        if invalid_path is not None:
            return f"ERROR: unsafe patch path '{invalid_path}'"

        completed = subprocess.run(
            ["git", "-C", str(self.workspace_root), "apply", "--whitespace=nowarn", "-"],
            input=patch,
            capture_output=True,
            text=True,
            timeout=self.shell_timeout_seconds,
        )
        if completed.returncode != 0:
            return (
                "ERROR: git apply failed\n"
                + self._format_process_output(completed.returncode, completed.stdout, completed.stderr)
            )

        status = subprocess.run(
            ["git", "-C", str(self.workspace_root), "status", "--short"],
            capture_output=True,
            text=True,
            timeout=self.shell_timeout_seconds,
        )
        git_status = status.stdout.strip() or "<clean>"
        if len(git_status) > 4000:
            git_status = git_status[:4000] + "\n... status truncated ..."
        return f"PATCH APPLIED\nGIT_STATUS:\n{git_status}"

    def _memory_get(self, key: str | None = None) -> str:
        data = self._load_memory()
        if key is None:
            return json.dumps(data, indent=2, ensure_ascii=True)
        if key not in data:
            return f"NOT_FOUND: memory key '{key}'"
        return json.dumps({"key": key, "value": data[key]}, indent=2, ensure_ascii=True)

    def _memory_set(self, key: str, value: Any) -> str:
        if key.strip() == "":
            return "ERROR: memory key cannot be empty"
        data = self._load_memory()
        data[key] = value
        self._save_memory(data)
        return f"MEMORY_SET: {key}"

    def _memory_delete(self, key: str) -> str:
        data = self._load_memory()
        if key not in data:
            return f"NOT_FOUND: memory key '{key}'"
        del data[key]
        self._save_memory(data)
        return f"MEMORY_DELETED: {key}"

    def _memory_search(self, query: str, limit: int = 10) -> str:
        if query.strip() == "":
            return "ERROR: query cannot be empty"

        data = self._load_memory()
        safe_limit = max(1, min(limit, 100))
        lowered = query.lower()

        matches: list[dict[str, Any]] = []
        for key, value in data.items():
            haystack = f"{key}\n{json.dumps(value, ensure_ascii=True)}".lower()
            if lowered in haystack:
                matches.append({"key": key, "value": value})
            if len(matches) >= safe_limit:
                break

        return json.dumps(matches, indent=2, ensure_ascii=True)

    def _load_memory(self) -> dict[str, Any]:
        if not self.memory_file.exists():
            return {}

        raw = self.memory_file.read_text(encoding="utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        return parsed

    def _save_memory(self, data: dict[str, Any]) -> None:
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")

    def _parallel_tools(self, calls: Any, max_workers: int = 4) -> str:
        if not isinstance(calls, list) or not calls:
            return "ERROR: calls must be a non-empty list"

        normalized: list[tuple[int, str, dict[str, Any]]] = []
        for index, call in enumerate(calls):
            if not isinstance(call, dict):
                return f"ERROR: calls[{index}] must be an object"
            tool = call.get("tool")
            if not isinstance(tool, str) or not tool.strip():
                return f"ERROR: calls[{index}].tool must be a non-empty string"
            if tool == "parallel_tools":
                return "ERROR: parallel_tools cannot call itself"

            raw_args = call.get("args", {})
            if not isinstance(raw_args, dict):
                return f"ERROR: calls[{index}].args must be an object"
            normalized.append((index, tool, raw_args))

        worker_count = max(1, min(max_workers, 8))

        results: list[dict[str, Any]] = [
            {"index": index, "tool": tool, "args": args, "result": ""}
            for index, tool, args in normalized
        ]

        def run_call(tool_name: str, tool_args: dict[str, Any]) -> str:
            return self.execute(tool_name, tool_args)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {}
            for index, tool_name, tool_args in normalized:
                future = executor.submit(run_call, tool_name, tool_args)
                futures[future] = index

            for future, index in futures.items():
                try:
                    result_value = future.result()
                except Exception as exc:  # noqa: BLE001
                    result_value = f"ERROR: parallel call failed: {exc}"
                results[index]["result"] = result_value

        return json.dumps({"parallel_results": results}, indent=2, ensure_ascii=True)

    def _reload_plugins(self) -> str:
        self.plugins = self._load_plugins()
        names = sorted(self.plugins)
        return json.dumps({"loaded_plugins": names}, ensure_ascii=True)

    def _load_plugins(self) -> dict[str, PluginSpec]:
        plugins: dict[str, PluginSpec] = {}
        if not self.plugins_dir.exists() or not self.plugins_dir.is_dir():
            return plugins

        for path in sorted(self.plugins_dir.glob("*.json")):
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
                payload = json.loads(raw)
            except (OSError, json.JSONDecodeError):
                continue

            if not isinstance(payload, dict):
                continue

            name = payload.get("name")
            description = payload.get("description")
            command = payload.get("command")
            args_schema = payload.get("args_schema", {})

            if not isinstance(name, str) or not self._is_valid_plugin_name(name):
                continue
            if not isinstance(description, str) or not description.strip():
                continue
            if not isinstance(command, list) or not command or not all(
                isinstance(part, str) and part != "" for part in command
            ):
                continue
            if not isinstance(args_schema, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in args_schema.items()
            ):
                continue

            plugins[name] = PluginSpec(
                name=name,
                description=description.strip(),
                command=command,
                args_schema=args_schema,
            )

        return plugins

    @staticmethod
    def _is_valid_plugin_name(name: str) -> bool:
        return re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", name) is not None

    def _run_plugin_tool(self, plugin_name: str, args: dict[str, Any]) -> str:
        plugin = self.plugins.get(plugin_name)
        if plugin is None:
            return f"ERROR: plugin tool not found: plugin.{plugin_name}"

        if not self.auto_approve:
            answer = input(
                "Approve plugin command? [y/N]\n"
                f"plugin: {plugin_name}\n"
                f"command: {' '.join(plugin.command)}\n> "
            ).strip().lower()
            if answer not in {"y", "yes"}:
                return "DENIED: plugin command not approved by user"

        env = os.environ.copy()
        env["LOCAL_CODEX_WORKSPACE"] = str(self.workspace_root)
        completed = subprocess.run(
            plugin.command,
            cwd=self.workspace_root,
            input=json.dumps(args, ensure_ascii=True),
            capture_output=True,
            text=True,
            timeout=self.shell_timeout_seconds,
            env=env,
        )

        return (
            f"PLUGIN: {plugin_name}\n"
            + self._format_process_output(completed.returncode, completed.stdout, completed.stderr)
        )

    def _require_git_repo(self) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(self.workspace_root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=self.shell_timeout_seconds,
        )
        if completed.returncode != 0 or completed.stdout.strip() != "true":
            return "ERROR: workspace is not a git repository"
        return None

    @staticmethod
    def _normalize_patch_path(raw: str) -> str | None:
        value = raw.strip().strip('"').strip("'")
        if not value:
            return None
        if value == "/dev/null":
            return None
        if value.startswith("a/") or value.startswith("b/"):
            value = value[2:]
        if value == "dev/null":
            return None
        return value

    def _find_unsafe_patch_path(self, patch: str) -> str | None:
        for line in patch.splitlines():
            candidate: str | None = None
            if line.startswith("diff --git "):
                parts = line.split()
                if len(parts) >= 4:
                    for raw in (parts[2], parts[3]):
                        normalized = self._normalize_patch_path(raw)
                        if normalized and self._is_unsafe_patch_path(normalized):
                            return normalized
                continue
            if line.startswith("--- ") or line.startswith("+++ "):
                candidate = line[4:].strip().split("\t", 1)[0]
            elif line.startswith("rename from "):
                candidate = line[len("rename from ") :].strip()
            elif line.startswith("rename to "):
                candidate = line[len("rename to ") :].strip()

            normalized = self._normalize_patch_path(candidate or "")
            if normalized and self._is_unsafe_patch_path(normalized):
                return normalized
        return None

    def _is_unsafe_patch_path(self, path_value: str) -> bool:
        if "\\" in path_value:
            return True

        path = PurePosixPath(path_value)
        if path.is_absolute():
            return True
        if any(part == ".." for part in path.parts):
            return True
        if path.parts and ":" in path.parts[0]:
            return True

        resolved = (self.workspace_root / Path(*path.parts)).resolve()
        return not _is_relative_to(resolved, self.workspace_root)

    @staticmethod
    def _format_process_output(returncode: int, stdout: str, stderr: str) -> str:
        clean_stdout = stdout.strip()
        clean_stderr = stderr.strip()
        if len(clean_stdout) > 12000:
            clean_stdout = clean_stdout[:12000] + "\n... stdout truncated ..."
        if len(clean_stderr) > 12000:
            clean_stderr = clean_stderr[:12000] + "\n... stderr truncated ..."
        return (
            f"EXIT_CODE: {returncode}\n"
            f"STDOUT:\n{clean_stdout or '<empty>'}\n"
            f"STDERR:\n{clean_stderr or '<empty>'}"
        )
