from __future__ import annotations

import subprocess
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


if __name__ == "__main__":
    unittest.main()
