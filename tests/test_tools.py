from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from local_codex.tools import ToolExecutor


def run_checked(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


class ToolExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)

        run_checked(["git", "init", "-b", "main"], cwd=self.workspace)
        run_checked(["git", "config", "user.email", "test@example.com"], cwd=self.workspace)
        run_checked(["git", "config", "user.name", "Test User"], cwd=self.workspace)

        (self.workspace / "hello.txt").write_text("one\n", encoding="utf-8")
        run_checked(["git", "add", "hello.txt"], cwd=self.workspace)
        run_checked(["git", "commit", "-m", "init"], cwd=self.workspace)

        plugins_dir = self.workspace / ".local_codex" / "plugins"
        plugins_dir.mkdir(parents=True, exist_ok=True)
        plugin_spec = {
            "name": "echo",
            "description": "Echo input value",
            "command": [
                sys.executable,
                "-c",
                (
                    "import json,sys; "
                    "payload=json.load(sys.stdin); "
                    "print('ECHO:' + str(payload.get(\"value\", \"\")))"
                ),
            ],
            "args_schema": {"value": "text to echo"},
        }
        (plugins_dir / "echo.json").write_text(json.dumps(plugin_spec), encoding="utf-8")

        self.executor = ToolExecutor(
            workspace_root=self.workspace,
            auto_approve=True,
            shell_timeout_seconds=30,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_read_file_blocks_workspace_escape(self) -> None:
        result = self.executor.execute("read_file", {"path": "../secret.txt"})
        self.assertIn("path escapes workspace root", result)

    def test_write_file_blocks_workspace_escape(self) -> None:
        result = self.executor.execute("write_file", {"path": "../secret.txt", "content": "x"})
        self.assertIn("path escapes workspace root", result)

    def test_list_files_blocks_workspace_escape(self) -> None:
        result = self.executor.execute("list_files", {"path": ".."})
        self.assertIn("path escapes workspace root", result)

    def test_apply_patch_rejects_unsafe_path(self) -> None:
        patch = "\n".join(
            [
                "diff --git a/../evil.txt b/../evil.txt",
                "--- a/../evil.txt",
                "+++ b/../evil.txt",
                "@@ -0,0 +1 @@",
                "+pwnd",
            ]
        )
        result = self.executor.execute("apply_patch", {"patch": patch})
        self.assertIn("unsafe patch path", result)

    def test_git_tools_and_patch_workflow(self) -> None:
        patch = "\n".join(
            [
                "diff --git a/hello.txt b/hello.txt",
                "--- a/hello.txt",
                "+++ b/hello.txt",
                "@@ -1 +1 @@",
                "-one",
                "+two",
                "",
            ]
        )

        apply_result = self.executor.execute("apply_patch", {"patch": patch})
        self.assertIn("PATCH APPLIED", apply_result)

        file_value = (self.workspace / "hello.txt").read_text(encoding="utf-8")
        self.assertEqual(file_value, "two\n")

        diff_result = self.executor.execute("git_diff", {"path": "hello.txt"})
        self.assertIn("+two", diff_result)

        status_result = self.executor.execute("git_status", {})
        self.assertIn("hello.txt", status_result)

        plan_result = self.executor.execute("git_commit_plan", {})
        self.assertIn("suggested_commit_message", plan_result)

        commit_result = self.executor.execute(
            "git_commit",
            {"message": "test: update hello", "add_all": True},
        )
        self.assertIn("EXIT_CODE: 0", commit_result)

    def test_memory_lifecycle(self) -> None:
        set_result = self.executor.execute(
            "memory_set",
            {"key": "project_goal", "value": "build local codex"},
        )
        self.assertIn("MEMORY_SET", set_result)

        get_result = self.executor.execute("memory_get", {"key": "project_goal"})
        self.assertIn("build local codex", get_result)

        search_result = self.executor.execute("memory_search", {"query": "codex"})
        self.assertIn("project_goal", search_result)

        delete_result = self.executor.execute("memory_delete", {"key": "project_goal"})
        self.assertIn("MEMORY_DELETED", delete_result)

    def test_structured_patch_and_symbol_aware_edits(self) -> None:
        source_file = self.workspace / "sample.py"
        source_file.write_text(
            "def greet():\n    return 'hi'\n\nclass A:\n    pass\n",
            encoding="utf-8",
        )

        overview = self.executor.execute("python_symbol_overview", {"path": "sample.py"})
        self.assertIn('"name": "greet"', overview)
        self.assertIn('"name": "A"', overview)

        replace_symbol = self.executor.execute(
            "replace_python_symbol",
            {
                "path": "sample.py",
                "symbol": "greet",
                "symbol_type": "function",
                "code": "def greet():\n    return 'hello'",
            },
        )
        self.assertIn("REPLACED_SYMBOL", replace_symbol)
        self.assertIn("return 'hello'", source_file.read_text(encoding="utf-8"))

        structured = self.executor.execute(
            "structured_patch",
            {
                "operations": [
                    {
                        "type": "replace_text",
                        "path": "sample.py",
                        "old": "class A",
                        "new": "class B",
                    },
                    {
                        "type": "append_text",
                        "path": "sample.py",
                        "text": "\n# end\n",
                    },
                ]
            },
        )
        self.assertIn("replace_text", structured)
        updated = source_file.read_text(encoding="utf-8")
        self.assertIn("class B", updated)
        self.assertIn("# end", updated)

    def test_plugin_and_parallel_tools(self) -> None:
        specs = self.executor.list_specs()
        tool_names = {spec.name for spec in specs}
        self.assertIn("plugin.echo", tool_names)

        plugin_result = self.executor.execute("plugin.echo", {"value": "ok"})
        self.assertIn("ECHO:ok", plugin_result)

        parallel_result = self.executor.execute(
            "parallel_tools",
            {
                "calls": [
                    {"tool": "read_file", "args": {"path": "hello.txt"}},
                    {"tool": "memory_set", "args": {"key": "k", "value": "v"}},
                ],
                "max_workers": 2,
            },
        )
        self.assertIn("parallel_results", parallel_result)
        self.assertIn('"tool": "read_file"', parallel_result)


if __name__ == "__main__":
    unittest.main()
