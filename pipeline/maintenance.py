"""One-off maintenance commands (safe to re-run; each only writes rows that actually change)."""
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
