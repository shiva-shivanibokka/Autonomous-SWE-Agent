"""
In-harness grading must decide "resolved" by exact test id, over every id.

Each test builds a tiny repository, applies an instance-shaped test patch via
`eval.harness.grade`, and runs real pytest in a subprocess. The three cases are
the three ways the previous grader could be wrong:

  * cap        -- only the first 20 ids were run, so breaking the 21st+
                  PASS_TO_PASS test still scored as resolved;
  * substring  -- bare names were selected with `-k`, a substring match, so an
                  unlisted test whose name merely contains the id was run and
                  its pre-existing failure scored a correct patch as unresolved;
  * missing    -- the same substring match let a test that does not exist be
                  "satisfied" by a differently named test that passed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from eval.harness import grade
from sandbox.workspace import CommandResult

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


class LocalShell:
    """Just enough of the Workspace contract for grade(), on a temp dir."""

    def __init__(self, root: Path):
        self.root = root
        self._posix_root = str(root).replace("\\", "/")

    def _map(self, text: str) -> str:
        return text.replace("/repo", self._posix_root)

    def write_file(self, path: str, content: str) -> None:
        Path(self._map(path)).write_text(content, encoding="utf-8", newline="\n")

    def run(self, command: str, timeout: int = 600, workdir: str = "/repo") -> CommandResult:
        cmd = self._map(command).replace(
            "python -m pytest", f'"{sys.executable}" -m pytest -p no:cacheprovider'
        )
        r = subprocess.run(
            cmd, shell=True, cwd=self.root, capture_output=True, text=True, timeout=timeout
        )
        return CommandResult(command=command, stdout=r.stdout, stderr=r.stderr, exit_code=r.returncode)


def _new_file_patch(path: str, body: str) -> str:
    lines = body.splitlines()
    out = [
        f"diff --git a/{path} b/{path}",
        "new file mode 100644",
        "--- /dev/null",
        f"+++ b/{path}",
        f"@@ -0,0 +1,{len(lines)} @@",
        *[f"+{line}" for line in lines],
    ]
    return "\n".join(out) + "\n"


def _repo(tmp_path: Path) -> LocalShell:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "tests").mkdir()
    (root / "tests" / "__init__.py").write_text("")
    (root / "mod.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"],
        cwd=root,
        check=True,
    )
    return LocalShell(root)


def _instance(test_body: str, f2p: list[str], p2p: list[str]) -> dict:
    import json

    return {
        "instance_id": "demo__demo-1",
        "test_patch": _new_file_patch("tests/test_mod.py", test_body),
        "FAIL_TO_PASS": json.dumps(f2p),
        "PASS_TO_PASS": json.dumps(p2p),
    }


def test_every_pass_to_pass_id_is_graded_not_just_the_first_20(tmp_path):
    ws = _repo(tmp_path)
    body = "from mod import VALUE\n\n"
    body += "def test_fixed():\n    assert VALUE == 1\n\n"
    for i in range(25):
        # The agent's patch broke test 24 - the 26th id overall.
        cond = "VALUE == 2" if i == 24 else "True"
        body += f"def test_keep_{i}():\n    assert {cond}\n\n"
    f2p = ["tests/test_mod.py::test_fixed"]
    p2p = [f"tests/test_mod.py::test_keep_{i}" for i in range(25)]
    assert grade(ws, _instance(body, f2p, p2p), diff="x") is False


def test_bare_name_does_not_select_tests_that_merely_contain_it(tmp_path):
    ws = _repo(tmp_path)
    body = (
        "from mod import VALUE\n\n"
        "def test_value():\n    assert VALUE == 1\n\n"
        # Pre-existing failure, not listed in FAIL_TO_PASS or PASS_TO_PASS.
        "def test_value_legacy():\n    assert False\n"
    )
    assert grade(ws, _instance(body, ["test_value"], []), diff="x") is True


def test_missing_test_is_not_satisfied_by_a_similarly_named_one(tmp_path):
    ws = _repo(tmp_path)
    body = "def test_gone_helper():\n    assert True\n"
    assert grade(ws, _instance(body, ["test_gone"], []), diff="x") is False


def test_outcome_parsing_and_label_matching():
    from eval.harness import _id_passed, parse_pytest_outcomes

    out = (
        "PASSED tests/test_a.py::TestX::test_one\n"
        "FAILED tests/test_a.py::TestY::test_one - AssertionError: boom\n"
        "PASSED tests/test_a.py::test_two[1-2]\n"
    )
    outcomes = parse_pytest_outcomes(out)
    assert outcomes["tests/test_a.py::TestY::test_one"] == "FAILED"
    assert _id_passed("tests/test_a.py::test_two[1-2]", outcomes)
    # A django label pins the class, so the failing TestY twin does not count.
    assert _id_passed("test_one (app.tests.TestX)", outcomes)
    assert not _id_passed("test_one (app.tests.TestY)", outcomes)
    # A bare name must hold for every node it names.
    assert not _id_passed("test_one", outcomes)
