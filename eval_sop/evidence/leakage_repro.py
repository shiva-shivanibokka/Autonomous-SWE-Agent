"""
Reproduce (or rule out) the history leak in sandbox.workspace.clone_repo.

Builds a local upstream (a base commit, then a "future fix" commit with a tag),
clones it at the base with clone_repo, and prints what an agent could see with
plain git. Output before and after the fix is committed next to this file.

    python eval_sop/evidence/leakage_repro.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sandbox.workspace import clone_repo  # noqa: E402


def git(cwd: Path, *args: str) -> str:
    cmd = ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "up"
        src.mkdir()
        git(src, "init", "-q", "-b", "main")
        git(src, "config", "uploadpack.allowAnySHA1InWant", "true")
        git(src, "config", "uploadpack.allowFilter", "true")
        (src / "mod.py").write_text("return 1\n")
        git(src, "add", "-A")
        git(src, "commit", "-q", "-m", "base")
        base = git(src, "rev-parse", "HEAD").strip()
        (src / "mod.py").write_text("return 2  # FUTURE FIX\n")
        git(src, "commit", "-q", "-am", "the future fix")
        git(src, "tag", "v-after-fix")

        dest = Path(tmp) / "clone"
        clone_repo(src.as_uri(), base, str(dest))
        print("HEAD           :", git(dest, "rev-parse", "--short", "HEAD").strip(), "(base)")
        print("git remote -v  :", git(dest, "remote", "-v").strip() or "(none)")
        print("git log --all  :")
        print(git(dest, "log", "--all", "--oneline", "--decorate"))
        shown = git(dest, "show", "origin/main:mod.py").strip()
        print("git show origin/main:mod.py ->", shown or "(unavailable)")


if __name__ == "__main__":
    main()
