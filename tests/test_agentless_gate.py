"""
The agentless regression gate must see every test, not stop at the first red one.

`agentless.validate` keeps a candidate when its pass/fail counts are no worse
than the unpatched baseline. Those counts only mean something if the whole
selection runs. With pytest's `-x`, a repository that already has one failing
test stops there on both runs, so a candidate that breaks a *later* test
produces identical counts and is kept. The harness passes its own command into
the gate, so this test drives the gate with exactly that command.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import eval.harness as harness
from agentless.repair import PatchCandidate, RepairResult
from agentless.validate import validate
from sandbox.workspace import CommandResult

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

# Whatever run_instance_agentless hands the gate.
GATE_COMMAND = getattr(harness, "AGENTLESS_VALIDATE_COMMAND", harness.REGRESSION_TEST_COMMAND)


class LocalShell:
    task_id = "gate-test"

    def __init__(self, root: Path):
        self.root = root
        self._posix_root = str(root).replace("\\", "/")

    def _map(self, text: str) -> str:
        return text.replace("/repo", self._posix_root)

    def write_file(self, path: str, content: str) -> None:
        Path(self._map(path)).write_text(content, encoding="utf-8")

    def read_file(self, path: str) -> str:
        return Path(self._map(path)).read_text(encoding="utf-8")

    def list_files(self, directory: str = "/repo") -> list[str]:
        base = Path(self._map(directory))
        if not base.is_absolute():
            base = self.root / base
        if not base.is_dir():
            return []
        return [str(p) for p in base.rglob("*") if p.is_file()]

    def run(self, command: str, timeout: int = 600, workdir: str = "/repo") -> CommandResult:
        cmd = self._map(command).replace(
            "python -m pytest", f'"{sys.executable}" -m pytest -p no:cacheprovider'
        )
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        r = subprocess.run(
            cmd, shell=True, cwd=self.root, capture_output=True, text=True, timeout=timeout, env=env
        )
        return CommandResult(command=command, stdout=r.stdout, stderr=r.stderr, exit_code=r.returncode)


def test_candidate_that_breaks_a_later_test_is_rejected(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg" / "tests").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "tests" / "__init__.py").write_text("")
    original = "VALUE = 1\nOTHER = 1\n"
    (root / "pkg" / "mod.py").write_text(original)
    (root / "pkg" / "tests" / "test_mod.py").write_text(
        "from pkg.mod import OTHER, VALUE\n\n"
        "def test_0():\n    assert VALUE == 1\n\n"
        "def test_a_preexisting_failure():\n    assert False\n\n"
        "def test_b():\n    assert OTHER == 1\n"
    )
    ws = LocalShell(root)
    broken = PatchCandidate(
        file_path="/repo/pkg/mod.py",
        original_content=original,
        patched_content="VALUE = 1\nOTHER = 2\n",  # breaks test_b, after the red test
        explanation="",
        sample_index=0,
    )
    result = validate(ws, RepairResult(candidates=[broken]), test_command=GATE_COMMAND)
    assert result.resolved is False, result.best_validation.test_output
