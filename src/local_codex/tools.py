from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    args: dict[str, str]


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
    ):
        self.workspace_root = workspace_root.resolve()
        self.auto_approve = auto_approve
        self.shell_timeout_seconds = shell_timeout_seconds

    def list_specs(self) -> list[ToolSpec]:
        return [
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
                name="apply_patch",
                description="Apply a unified diff patch using `git apply`.",
                args={
                    "patch": "unified diff text",
                },
            ),
        ]

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
        return json.dumps(specs, indent=2)

    def execute(self, tool_name: str, args: dict[str, Any]) -> str:
        try:
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
            if tool_name == "apply_patch":
                return self._apply_patch(patch=str(args["patch"]))

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

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if len(stdout) > 12000:
            stdout = stdout[:12000] + "\n... stdout truncated ..."
        if len(stderr) > 12000:
            stderr = stderr[:12000] + "\n... stderr truncated ..."

        return (
            f"EXIT_CODE: {completed.returncode}\n"
            f"STDOUT:\n{stdout or '<empty>'}\n"
            f"STDERR:\n{stderr or '<empty>'}"
        )

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
