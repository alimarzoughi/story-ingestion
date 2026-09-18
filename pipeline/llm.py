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

    def _count(self, kind: str, usage: Any) -> None:
        with self._lock:
            self.calls[kind] = self.calls.get(kind, 0) + 1
            if usage is not None:
                self.tokens["input"] += getattr(usage, "prompt_tokens", 0) or 0
                self.tokens["output"] += getattr(usage, "completion_tokens", 0) or 0

    def structured(self, *, kind: str, model: str, system: str, user: str, schema: dict, schema_name: str,
                   max_output_tokens: int = 1200) -> dict:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
            "max_completion_tokens": max_output_tokens,
        }
        if model.startswith("gpt-5") and self.settings.reasoning_effort:
            kwargs["reasoning_effort"] = self.settings.reasoning_effort
        if not model.startswith("gpt-5"):
            kwargs["temperature"] = 0
        last: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self._count(kind, getattr(resp, "usage", None))
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
        return f"llm calls: {calls}; tokens in={self.tokens['input']} out={self.tokens['output']}"
