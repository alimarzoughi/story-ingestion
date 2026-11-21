"""One-off maintenance commands (safe to re-run): `fix-text` and `reenrich`."""
from __future__ import annotations

from .db import Db
from .textclean import clean_text

_TEXT_COLUMNS = {
    "articles_v3": ("title", "headline", "subheadline", "snippet"),
    "stories": ("title", "summary"),
}


def fix_text(db: Db, *, dry_run: bool = False, page_size: int = 500) -> dict:
    """Decode HTML entities left in already-ingested titles/snippets and story titles.

    Walks each table in id order (keyset pagination) and patches only rows whose text changes.
    """
    stats: dict[str, int] = {}
    for table, columns in _TEXT_COLUMNS.items():
        scanned = fixed = 0
        last_id = None
        while True:
            params = {"select": "id," + ",".join(columns), "order": "id.asc", "limit": str(page_size)}
            if last_id is not None:
                params["id"] = f"gt.{last_id}"
            rows = db.select(table, **params)
            if not rows:
                break
            for row in rows:
                patch = {}
                for col in columns:
                    value = row.get(col)
                    if value and "&" in value:
                        cleaned = clean_text(value)
                        if cleaned and cleaned != value:
                            patch[col] = cleaned
                if patch:
                    fixed += 1
                    if not dry_run:
                        db.update(table, {"id": f"eq.{row['id']}"}, patch)
            scanned += len(rows)
            last_id = rows[-1]["id"]
            if len(rows) < page_size:
                break
        stats[f"{table}_scanned"] = scanned
        stats[f"{table}_fixed"] = fixed
    mode = " (dry run)" if dry_run else ""
    print("[fix-text]" + mode + " " + " ".join(f"{k}={v}" for k, v in stats.items()))
    return stats


def reenrich(db: Db, llm, settings, *, days: int = 5, limit: int | None = None, workers: int = 12,
             dry_run: bool = False, page_size: int = 200) -> dict:
    """Re-run enrichment under the current prompt for recent articles (the 2026-11-21 switch to main-point summaries).

    Resumable: only articles enriched under an older prompt_version are picked up, so re-running continues where an
    interrupted run stopped. Story membership is not touched; afterwards every open story's summary is re-anchored on
    its founding article's new summary and its centroid recomputed from the new embeddings, so new articles are
    compared with the same kind of text. `recheck-stories` then re-verifies the members.
    """
    from datetime import datetime, timedelta, timezone

    from .cluster import founding_article
    from .enrich import ARTICLE_SELECT, enrich_rows

    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    stats = {"enriched": 0, "failed": 0, "text_mismatch": 0, "stories_reanchored": 0, "stories_refreshed": 0}
    failed: set[str] = set()  # failed rows keep their old version and come back at the front of the next page
    todo = limit or (20 if dry_run else None)
    while todo is None or todo > 0:
        batch = page_size if todo is None else min(page_size, todo)
        rows = db.select("articles_v3", select=ARTICLE_SELECT, enrichment_status="eq.done", published_at=f"gte.{since}",
                         **{"enrichment->>prompt_version": f"neq.{settings.prompt_version}"},
                         order="published_at.asc", limit=str(batch))
        if not rows:
            break
        out = enrich_rows(db, llm, settings, rows, dry_run=dry_run, workers=workers, mark_failed=False)
        stats["enriched"] += out["done"]
        failed.update(out["failed_ids"])
        stats["failed"] = len(failed)
        stats["text_mismatch"] += out["text_mismatch"]
        if todo is not None:
            todo -= len(rows)
        if dry_run or out["done"] == 0:
            break  # a dry run writes nothing (the same rows would come back); no progress means the API is failing
    if dry_run:
        print(f"[reenrich] dry run: sample shown above ({stats['text_mismatch']} with text that doesn't match the "
              "headline); nothing written")
        return stats

    for story in db.select_all("stories", select="id,summary,event_date", status="eq.open"):
        members = db.select("articles_v3", select="id,published_at,event_summary,event_date,article_type,is_wire_copy",
                            story_id=f"eq.{story['id']}")
        founder = founding_article(members)
        if founder and founder["event_summary"] != story.get("summary"):
            db.update("stories", {"id": f"eq.{story['id']}"}, {
                "summary": founder["event_summary"], "event_date": founder.get("event_date") or story.get("event_date")})
            stats["stories_reanchored"] += 1
        db.rpc("refresh_story_stats", {"target": story["id"]})
        stats["stories_refreshed"] += 1
    print("[reenrich] " + " ".join(f"{k}={v}" for k, v in stats.items()) + f" | {llm.usage_summary()}")
    return stats
