"""Source registry: sources.yaml is the source of truth; `sync` mirrors it into the `sources` table."""
from __future__ import annotations

from pathlib import Path

import yaml

from .db import Db

SIDE_BY_LEANING = {
    "LEFT": "left",
    "LEAN_LEFT": "left",
    "CENTER": "center",
    "LEAN_RIGHT": "right",
    "RIGHT": "right",
}
VALID_LEANINGS = set(SIDE_BY_LEANING)


def load_sources(path: str | Path = "sources.yaml") -> list[dict]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    sources = []
    seen = set()
    for entry in raw:
        sid = entry["id"]
        if sid in seen:
            raise ValueError(f"duplicate source id {sid}")
        seen.add(sid)
        if entry["leaning"] not in VALID_LEANINGS:
            raise ValueError(f"{sid}: invalid leaning {entry['leaning']}")
        sources.append(
            {
                "id": sid,
                "outlet": entry["outlet"],
                "feed_url": entry["feed_url"],
                "homepage": entry.get("homepage"),
                "leaning": entry["leaning"],
                "section": entry.get("section"),
                "extractor": entry.get("extractor", "generic"),
                "enabled": bool(entry.get("enabled", True)),
                "max_per_run": int(entry.get("max_per_run", 40)),
                "notes": entry.get("notes"),
                "verified": bool(entry.get("verified", False)),
                "alt_feed_urls": list(entry.get("alt_feed_urls") or []),
            }
        )
    return sources


def sync_sources(db: Db, sources: list[dict]) -> int:
    rows = [{k: v for k, v in s.items() if k not in ("verified", "alt_feed_urls")} for s in sources]
    db.insert("sources", rows, on_conflict="id", returning=False)
    # disable anything that disappeared from the yaml
    ids = [s["id"] for s in sources]
    existing = db.select("sources", select="id")
    stale = [row["id"] for row in existing if row["id"] not in ids]
    for sid in stale:
        db.update("sources", {"id": f"eq.{sid}"}, {"enabled": False})
    return len(rows)


def side_for(leaning: str | None) -> str | None:
    return SIDE_BY_LEANING.get(leaning or "")
