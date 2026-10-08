"""
Seeded agentless sampling must still draw N different samples.

The repair phase sends the same prompt N times and relies on temperature for
diversity. Once the LLM config carries a seed (for reproducible runs), sending
that same seed with the same prompt makes every temperature-1 sample identical,
silently turning "N candidates" into one. Each sample needs its own seed,
derived deterministically from the run's seed.
"""

from __future__ import annotations

import importlib

repair_mod = importlib.import_module("agentless.repair")  # the package re-exports a function named `repair`
from agent.llm import LLMConfig, LLMResponse  # noqa: E402
from agentless.localize import LocalizationResult  # noqa: E402


class _FileOnly:
    task_id = "seed-test"

    def read_file(self, path: str) -> str:
        return "def f():\n    return 1\n"


def _capture(monkeypatch) -> list:
    seen = []

    def fake_complete(cfg, messages, **kwargs):
        seen.append((cfg.seed, kwargs.get("temperature")))
        text = '{"explanation": "x", "search": "return 1", "replace": "return 2"}'
        return LLMResponse(text, [], 1, 1, 0.0, "stop")

    monkeypatch.setattr(repair_mod, "complete", fake_complete)
    return seen


def _loc() -> LocalizationResult:
    return LocalizationResult(
        suspect_files=["/repo/m.py"],
        suspect_locations=[{"file": "/repo/m.py", "function_name": "f", "class_name": None}],
        repo_map="",
    )


def test_each_sample_gets_its_own_seed(monkeypatch):
    seen = _capture(monkeypatch)
    llm = LLMConfig("ollama", "m", "", seed=3)
    repair_mod.repair(_FileOnly(), "issue", _loc(), llm, num_samples=4)
    seeds = [s for s, _ in seen]
    assert len(seeds) == 4
    assert len(set(seeds)) == 4, seeds
    # Deterministic: the same run seed gives the same per-sample seeds.
    seen.clear()
    repair_mod.repair(_FileOnly(), "issue", _loc(), llm, num_samples=4)
    assert [s for s, _ in seen] == seeds


def test_unseeded_config_is_left_alone(monkeypatch):
    seen = _capture(monkeypatch)
    repair_mod.repair(_FileOnly(), "issue", _loc(), LLMConfig("openai", "m", "k"), num_samples=3)
    assert [s for s, _ in seen] == [None, None, None]
