"""Model-family request parameters and the verify-model fallback (no network: fake OpenAI client)."""
import json
from types import SimpleNamespace

import pytest

from pipeline.config import Settings
from pipeline.llm import LLM


class FakeCompletions:
    def __init__(self, fail_models=()):
        self.fail_models, self.requests = set(fail_models), []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if kwargs["model"] in self.fail_models:
            raise RuntimeError("model_not_found")
        msg = SimpleNamespace(content=json.dumps({"ok": True}))
        usage = SimpleNamespace(prompt_tokens=900, completion_tokens=120)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)


def make_llm(verify_model="gpt-6-luna", fail_models=(), **settings_kw):
    settings = Settings(supabase_url="x", supabase_key="x", openai_api_key="k", reasoning_effort="minimal",
                        verify_model=verify_model, **settings_kw)
    llm = LLM.__new__(LLM)
    import threading  # the real constructor needs an API key and the SDK; wire the pieces by hand
    llm.settings, llm._lock, llm.calls, llm.tokens = settings, threading.Lock(), {}, {"input": 0, "output": 0}
    llm.tokens_by_model, llm._warned_fallback = {}, False
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(fail_models)))
    return llm


def call(llm, model):
    return llm.structured(kind="verify", model=model, system="s", user="u", schema={}, schema_name="x", max_output_tokens=300)


def test_gpt6_gets_low_effort_headroom_and_no_temperature():
    llm = make_llm()
    call(llm, "gpt-6-luna")
    req = llm.client.chat.completions.requests[-1]
    assert req["reasoning_effort"] == "low" and "temperature" not in req and req["max_completion_tokens"] == 2300


def test_gpt5_keeps_minimal_effort_and_small_cap():
    llm = make_llm(verify_model="gpt-5-mini")
    call(llm, "gpt-5-mini")
    req = llm.client.chat.completions.requests[-1]
    assert req["reasoning_effort"] == "minimal" and "temperature" not in req and req["max_completion_tokens"] == 300


def test_gpt6_minimal_is_mapped_to_low_and_none_is_passed_through():
    assert make_llm(verify_reasoning_effort="minimal").reasoning_effort_for("gpt-6-luna") == "low"
    assert make_llm(verify_reasoning_effort="none").reasoning_effort_for("gpt-6-luna") == "none"


def test_verify_model_failure_falls_back_to_gpt5_mini(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    llm = make_llm(fail_models={"gpt-6-luna"})
    assert call(llm, "gpt-6-luna") == {"ok": True}
    assert llm.client.chat.completions.requests[-1]["model"] == "gpt-5-mini"
    assert "gpt-5-mini: 1 calls" in llm.usage_summary()


def test_enrich_model_failure_does_not_fall_back(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    llm = make_llm(fail_models={"gpt-5-nano"})
    with pytest.raises(RuntimeError):
        llm.structured(kind="enrich", model="gpt-5-nano", system="s", user="u", schema={}, schema_name="x")
