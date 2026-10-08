"""
The workspace must not let the agent reach the future fix.

SWE-bench checks out the commit just before the fix. If the clone keeps the
`origin` remote and its refs (or tags, or reflog, or unreachable objects), then
`git log --all -p` hands the agent the maintainers' patch. These tests build a
throwaway upstream with a "base" commit and a later "fix" commit, clone at the
base through `clone_repo`, and assert the fix is not reachable by any git
command an agent could run.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from sandbox.workspace import clone_repo

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

FUTURE_MARKER = "THE_FUTURE_FIX_7f3a"


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def upstream(tmp_path: Path) -> tuple[Path, str, str]:
    """An upstream repo: base commit, then a fix commit (tagged) after it."""
    src = tmp_path / "upstream"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    # Let clients fetch an exact SHA and use partial clone, as GitHub does.
    _git(src, "config", "uploadpack.allowAnySHA1InWant", "true")
    _git(src, "config", "uploadpack.allowFilter", "true")
    (src / "mod.py").write_text("def f():\n    return 1\n")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "base")
    base = _git(src, "rev-parse", "HEAD").stdout.strip()
    (src / "mod.py").write_text(f"def f():\n    return 2  # {FUTURE_MARKER}\n")
    _git(src, "commit", "-q", "-am", f"fix: {FUTURE_MARKER}")
    fix = _git(src, "rev-parse", "HEAD").stdout.strip()
    _git(src, "tag", "v-after-fix")
    _git(src, "branch", "release")  # a second ref pointing at the future
    return src, base, fix


def _assert_no_future(dest: Path, base: str, fix: str) -> None:
    assert _git(dest, "rev-parse", "HEAD").stdout.strip() == base
    # The fix commit object must not exist at all, reachable or not.
    assert _git(dest, "cat-file", "-e", f"{fix}^{{commit}}").returncode != 0
    log_all = _git(dest, "log", "--all", "--format=%H %s").stdout
    assert fix not in log_all and FUTURE_MARKER not in log_all
    assert FUTURE_MARKER not in _git(dest, "log", "--all", "-p").stdout
    assert _git(dest, "remote").stdout.strip() == ""
    refs = _git(dest, "for-each-ref", "--format=%(objectname)").stdout
    assert fix not in refs
    assert FUTURE_MARKER not in _git(dest, "fsck", "--lost-found", "--no-reflogs").stdout
    assert (dest / "mod.py").read_text().strip().endswith("return 1")


def test_clone_at_base_cannot_reach_future_fix(upstream, tmp_path):
    src, base, fix = upstream
    dest = tmp_path / "clone"
    clone_repo(src.as_uri(), base, str(dest))
    assert (dest / ".git" / "shallow").exists()  # took the single-commit path
    _assert_no_future(dest, base, fix)


def test_fallback_path_cannot_reach_future_fix(upstream, tmp_path, monkeypatch):
    """Servers that refuse fetch-by-SHA take the full-history fallback; that
    path must also end with the future gone. Protocol v2 servers accept any
    reachable SHA, so force protocol v0 to make the refusal real."""
    src, base, fix = upstream
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "protocol.version")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "0")
    _git(src, "config", "uploadpack.allowAnySHA1InWant", "false")
    _git(src, "config", "uploadpack.allowReachableSHA1InWant", "false")
    _git(src, "config", "uploadpack.allowTipSHA1InWant", "false")
    dest = tmp_path / "clone"
    clone_repo(src.as_uri(), base, str(dest))
    assert not (dest / ".git" / "shallow").exists()  # took the full-history path
    _assert_no_future(dest, base, fix)
