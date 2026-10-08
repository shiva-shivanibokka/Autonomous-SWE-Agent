"""
Tests for the BYOK provider registry and the LiteLLM adapter layer.

These cover the pure plumbing (no network, no litellm/docker needed): the
registry shape the frontend depends on, the LiteLLM model-string builder, the
tool-schema converter, and the assistant-message rebuild used by the agent loop.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from agent.llm import LLMConfig, LLMResponse, ToolCall, assistant_message, to_openai_tools
from agent.providers import PROVIDERS, key_env_for, litellm_model, providers_payload


class TestRegistry:
    def test_expected_providers_present(self):
        assert set(PROVIDERS) == {"anthropic", "openai", "google", "groq"}

    def test_every_provider_has_models(self):
        for p in PROVIDERS.values():
            assert p.models, f"{p.key} has no models"
            assert p.key_env and p.key_url

    def test_model_ids_unique_within_provider(self):
        for p in PROVIDERS.values():
            ids = [m.id for m in p.models]
            assert len(ids) == len(set(ids)), f"duplicate model id in {p.key}"


class TestLitellmModel:
    def test_prefixes_by_provider(self):
        assert litellm_model("anthropic", "claude-opus-4-8") == "anthropic/claude-opus-4-8"
        # Google routes through the "gemini" prefix, not "google".
        assert litellm_model("google", "gemini-3.1-pro") == "gemini/gemini-3.1-pro"
        assert litellm_model("groq", "llama-3.3-70b-versatile") == "groq/llama-3.3-70b-versatile"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            litellm_model("cohere", "whatever")

    def test_key_env_for(self):
        assert key_env_for("openai") == "OPENAI_API_KEY"


class TestProvidersPayload:
    def test_shape_matches_frontend_contract(self):
        payload = providers_payload()
        assert isinstance(payload, list) and payload
        for entry in payload:
            assert set(entry) == {"key", "label", "keyUrl", "models"}
            for m in entry["models"]:
                assert set(m) == {"id", "label"}


class TestToOpenAITools:
    def test_converts_anthropic_style_schema(self):
        anthropic_style = [
            {
                "name": "bash",
                "description": "run a command",
                "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
            }
        ]
        out = to_openai_tools(anthropic_style)
        assert out[0]["type"] == "function"
        fn = out[0]["function"]
        assert fn["name"] == "bash"
        assert fn["description"] == "run a command"
        assert fn["parameters"]["properties"]["command"]["type"] == "string"

    def test_missing_input_schema_defaults_to_empty_object(self):
        out = to_openai_tools([{"name": "noop"}])
        assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


class TestAssistantMessage:
    def test_plain_text_has_no_tool_calls(self):
        resp = LLMResponse("done", [], 1, 1, 0.0, "stop")
        msg = assistant_message(resp)
        assert msg == {"role": "assistant", "content": "done"}

    def test_tool_calls_serialized_as_json_arguments(self):
        resp = LLMResponse(
            "",
            [ToolCall(id="t1", name="bash", arguments={"command": "ls"})],
            1,
            1,
            0.0,
            "tool_calls",
        )
        msg = assistant_message(resp)
        tc = msg["tool_calls"][0]
        assert tc["id"] == "t1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "bash"
        # arguments must round-trip as a JSON string (LiteLLM/OpenAI contract)
        assert json.loads(tc["function"]["arguments"]) == {"command": "ls"}


class TestLLMConfig:
    def test_is_hashable_and_frozen(self):
        cfg = LLMConfig(provider="openai", model="gpt-5.6-terra", api_key="sk-x")
        assert {cfg}  # hashable
        with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
            cfg.api_key = "changed"  # frozen


class TestLocalProviderAndSeed:
    """Plumbing added for the free, reproducible eval (eval_sop/)."""

    def _fake_litellm(self, captured):
        from types import SimpleNamespace

        def completion(**kwargs):
            captured.update(kwargs)
            msg = SimpleNamespace(content="ok", tool_calls=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
            )

        return SimpleNamespace(completion=completion, completion_cost=lambda **_: 0.0)

    def test_ollama_is_local_only(self):
        from agent.providers import LOCAL_PROVIDERS

        assert "ollama" not in PROVIDERS  # never offered in the hosted BYOK UI
        assert "ollama" in LOCAL_PROVIDERS
        assert litellm_model("ollama", "qwen2.5:7b") == "ollama_chat/qwen2.5:7b"

    def test_seed_api_base_and_options_are_forwarded(self, monkeypatch):
        import agent.llm as llm

        captured: dict = {}
        monkeypatch.setattr(llm, "_get_litellm", lambda: self._fake_litellm(captured))
        cfg = LLMConfig(
            "ollama", "qwen2.5:7b", "", seed=7, api_base="http://x:1", options=(("num_ctx", 8192),)
        )
        llm.complete(cfg, [{"role": "user", "content": "hi"}])
        assert captured["seed"] == 7
        assert captured["api_base"] == "http://x:1"
        assert captured["num_ctx"] == 8192

    def test_defaults_unchanged_for_byok_providers(self, monkeypatch):
        import agent.llm as llm

        captured: dict = {}
        monkeypatch.setattr(llm, "_get_litellm", lambda: self._fake_litellm(captured))
        llm.complete(LLMConfig("openai", "m", "k"), [{"role": "user", "content": "hi"}])
        assert "seed" not in captured and "api_base" not in captured
        with pytest.raises(llm.LLMError):
            llm.complete(LLMConfig("openai", "m", ""), [{"role": "user", "content": "hi"}])
