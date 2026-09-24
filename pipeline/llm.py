"""OpenAI wrapper: structured-output chat calls and embeddings, with retries and call counting."""
from __future__ import annotations

import json
import threading
import time
from typing import Any

from .config import Settings


class LLM:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        from openai import OpenAI  # imported lazily so tests can run without the SDK

        self.client = OpenAI(api_key=settings.openai_api_key, max_retries=3, timeout=60)
        self.settings = settings
        self._lock = threading.Lock()
        self.calls: dict[str, int] = {}
        self.tokens: dict[str, int] = {"input": 0, "output": 0}
        self.tokens_by_model: dict[str, dict[str, int]] = {}
        self._warned_fallback = False

    def _count(self, kind: str, usage: Any, model: str | None = None) -> None:
        with self._lock:
            self.calls[kind] = self.calls.get(kind, 0) + 1
            if usage is not None:
                tin = getattr(usage, "prompt_tokens", 0) or 0
                tout = getattr(usage, "completion_tokens", 0) or 0
                self.tokens["input"] += tin
                self.tokens["output"] += tout
                if model:
                    per = self.tokens_by_model.setdefault(model, {"calls": 0, "input": 0, "output": 0})
                    per["calls"] += 1
                    per["input"] += tin
                    per["output"] += tout

    def reasoning_effort_for(self, model: str) -> str | None:
        """gpt-5 family: the global setting ('minimal' by default). gpt-6 family: OPENAI_VERIFY_REASONING_EFFORT,
        default 'low'. gpt-6 has no 'minimal' and defaults to 'medium' if nothing is sent, which bills far more
        reasoning tokens as output -- so an effort is always sent."""
        if model.startswith("gpt-5"):
            return self.settings.reasoning_effort
        if model.startswith("gpt-6"):
            effort = self.settings.verify_reasoning_effort or "low"
            return "low" if effort == "minimal" else effort
        return None

    def structured(self, *, kind: str, model: str, system: str, user: str, schema: dict, schema_name: str,
                   max_output_tokens: int = 1200) -> dict:
        """Structured-output call. If the verify-tier model fails after retries, the configured fallback
        (OPENAI_VERIFY_FALLBACK_MODEL, default gpt-5-mini) is tried once so a new model cannot stall the feed."""
        try:
            return self._structured(kind=kind, model=model, system=system, user=user, schema=schema,
                                    schema_name=schema_name, max_output_tokens=max_output_tokens)
        except Exception as exc:
            fallback = self.settings.verify_fallback_model
            if not fallback or model == fallback or model != self.settings.verify_model:
                raise
            if not self._warned_fallback:
                print(f"[llm] {model} failed ({str(exc)[:160]}); falling back to {fallback}")
                self._warned_fallback = True
            return self._structured(kind=kind, model=fallback, system=system, user=user, schema=schema,
                                    schema_name=schema_name, max_output_tokens=max_output_tokens)

    def _structured(self, *, kind: str, model: str, system: str, user: str, schema: dict, schema_name: str,
                    max_output_tokens: int = 1200) -> dict:
        effort = self.reasoning_effort_for(model)
        if effort and effort not in ("minimal", "none"):
            # reasoning tokens count toward max_completion_tokens; headroom is free (billed on actual usage)
            max_output_tokens += 2000
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
            "max_completion_tokens": max_output_tokens,
        }
        if effort:
            kwargs["reasoning_effort"] = effort
        elif not model.startswith(("gpt-5", "gpt-6")):
            kwargs["temperature"] = 0
        last: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self._count(kind, getattr(resp, "usage", None), model)
                choice = resp.choices[0]
                content = choice.message.content or ""
                if choice.finish_reason == "length":
                    raise RuntimeError("response truncated (max_completion_tokens)")
                return json.loads(content)
            except json.JSONDecodeError as exc:
                last = exc
            except Exception as exc:  # rate limits etc. (SDK already retried transport errors)
                last = exc
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"{kind} failed: {last}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            batch = [t[:8000] for t in texts[i : i + 64]]
            resp = self.client.embeddings.create(model=self.settings.embed_model, input=batch, dimensions=self.settings.embed_dims)
            self._count("embed", getattr(resp, "usage", None))
            out.extend([d.embedding for d in sorted(resp.data, key=lambda d: d.index)])
        return out

    def usage_summary(self) -> str:
        calls = ", ".join(f"{k}={v}" for k, v in sorted(self.calls.items())) or "none"
        per_model = "; ".join(f"{m}: {t['calls']} calls, in={t['input']} out={t['output']}"
                              for m, t in sorted(self.tokens_by_model.items()))
        return f"llm calls: {calls}; tokens in={self.tokens['input']} out={self.tokens['output']}" + (
            f" | by model: {per_model}" if per_model else "")
