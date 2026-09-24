"""Environment / settings. Reads .env (tolerant of CRLF and quotes) then os.environ."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_env(env_path: str | Path = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip().rstrip("\r")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass
class Settings:
    supabase_url: str
    supabase_key: str
    openai_api_key: str | None
    enrich_model: str = "gpt-5-nano"
    verify_model: str = "gpt-5-mini"
    embed_model: str = "text-embedding-3-small"
    embed_dims: int = 768
    reasoning_effort: str | None = None          # only sent for gpt-5* models
    prompt_version: str = "2026-09-17.2"
    # ingest
    fetch_concurrency: int = 8
    feed_max_age_hours: int = 72                 # ignore feed entries older than this
    http_timeout: int = 20
    fetch_impersonate: str = "chrome"            # curl_cffi target; "none" to use plain requests
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    # enrich
    enrich_concurrency: int = 4
    enrich_batch: int = 200
    body_chars_for_llm: int = 4000
    # cluster
    attach_threshold: float = 0.83
    verify_threshold: float = 0.66
    candidate_window_hours: int = 96
    event_date_max_gap_days: int = 3
    assign_batch: int = 300
    auto_attach_window_hours: int = 12           # later arrivals always go through the verifier (follow-up check)
    # merge (duplicate stories)
    merge_min_similarity: float = 0.86
    merge_lookback_hours: int = 48
    merge_max_checks: int = 30
    merge_max_event_gap_days: int = 1
    # pairs
    pairs_lookback_hours: int = 48
    near_duplicate_title_ratio: float = 0.90
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, require_db: bool = True) -> "Settings":
        load_env()
        url = _env("SUPABASE_URL") or ""
        key = _env("SUPABASE_SECRET_KEY") or _env("SUPABASE_SERVICE_ROLE_KEY") or _env("SUPABASE_KEY") or ""
        if require_db and not (url and key):
            raise RuntimeError("SUPABASE_URL and SUPABASE_SECRET_KEY must be set")
        enrich_model = _env("OPENAI_ENRICH_MODEL", "gpt-5-nano")
        effort = _env("OPENAI_REASONING_EFFORT")
        if effort is None and enrich_model.startswith("gpt-5"):
            effort = "minimal"
        return cls(
            supabase_url=url.rstrip("/"),
            supabase_key=key,
            openai_api_key=_env("OPENAI_API_KEY"),
            enrich_model=enrich_model,
            verify_model=_env("OPENAI_VERIFY_MODEL", "gpt-5-mini"),
            embed_model=_env("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            embed_dims=int(_env("OPENAI_EMBED_DIMS", "768")),
            reasoning_effort=effort,
            fetch_concurrency=int(_env("FETCH_CONCURRENCY", "8")),
            feed_max_age_hours=int(_env("FEED_MAX_AGE_HOURS", "72")),
            fetch_impersonate=_env("FETCH_IMPERSONATE", "chrome"),
            enrich_concurrency=int(_env("ENRICH_CONCURRENCY", "4")),
            enrich_batch=int(_env("ENRICH_BATCH", "200")),
            attach_threshold=float(_env("ATTACH_THRESHOLD", "0.83")),
            verify_threshold=float(_env("VERIFY_THRESHOLD", "0.66")),
            candidate_window_hours=int(_env("CANDIDATE_WINDOW_HOURS", "96")),
            auto_attach_window_hours=int(_env("AUTO_ATTACH_WINDOW_HOURS", "12")),
            merge_min_similarity=float(_env("MERGE_MIN_SIMILARITY", "0.86")),
        )
