"""Framing tags ("what it puts first" + one-line framing) for the See Details panel.

The main enrichment call sometimes comes back with empty framing (mostly roundups and "latest update" pieces).
`ensure_framing` re-asks for just those two fields; `backfill_framing` does the same for recent articles that were
enriched before this existed. A framing-only call is small (title + first ~1,500 characters): about $0.00006 on
gpt-5-nano.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

FRAMING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "framing_axes": {"type": "array", "items": {"type": "string"},
                         "description": "2-4 short phrases (2-4 words each) naming what the piece puts first, e.g. 'cost to taxpayers', 'harm to families', 'legal risk'."},
        "framing_summary": {"type": "string",
                            "description": "One sentence: what this article foregrounds and whom it blames or credits. For a neutral update, say what it leads with."},
    },
    "required": ["framing_axes", "framing_summary"],
}

FRAMING_SYSTEM = """You describe how a news article frames its story. Every article has a framing, even a neutral
update or a roundup: say what it leads with and emphasizes. Never answer with an empty list or an empty sentence.
framing_axes: 2-4 short phrases. framing_summary: one sentence, at most 35 words. Return only JSON."""

FRAMING_BODY_CHARS = 1500


def needs_framing(stance: dict | None) -> bool:
    stance = stance or {}
    axes = [a for a in (stance.get("framing_axes") or []) if isinstance(a, str) and a.strip()]
    summary = stance.get("framing_summary")
    return not axes or not (isinstance(summary, str) and summary.strip())


def _framing_prompt(article: dict) -> str:
    body = (article.get("body_text") or "")[:FRAMING_BODY_CHARS]
    return (
        f"Outlet: {article.get('outlet')}\n"
        f"Headline: {article.get('headline') or article.get('title')}\n"
        f"Page title: {article.get('title')}\n"
        f"Subheadline: {article.get('subheadline') or 'n/a'}\n\n"
        f"Article text (truncated):\n{body or 'n/a'}"
    )


def _clean(out: dict) -> tuple[list[str], str | None]:
    axes = [" ".join(a.split()) for a in (out.get("framing_axes") or []) if isinstance(a, str) and a.strip()][:4]
    summary = out.get("framing_summary")
    summary = " ".join(summary.split()) if isinstance(summary, str) and summary.strip() else None
    return axes, summary


def ask_framing(llm, settings, article: dict) -> tuple[list[str], str | None]:
    out = llm.structured(kind="framing", model=settings.enrich_model, system=FRAMING_SYSTEM,
                         user=_framing_prompt(article), schema=FRAMING_SCHEMA, schema_name="article_framing",
                         max_output_tokens=300)
    return _clean(out)


def merge_framing(stance: dict | None, axes: list[str], summary: str | None) -> dict:
    """Fill only what's missing; keep whatever the main call already got right."""
    merged = dict(stance or {})
    if not [a for a in (merged.get("framing_axes") or []) if isinstance(a, str) and a.strip()] and axes:
        merged["framing_axes"] = axes
    current = merged.get("framing_summary")
    if not (isinstance(current, str) and current.strip()) and summary:
        merged["framing_summary"] = summary
    return merged


def ensure_framing(llm, settings, article: dict, out: dict) -> dict:
    """Called on each fresh enrichment result: one retry for empty framing. Never fails the enrichment."""
    stance = out.get("stance") or {}
    if not needs_framing(stance):
        return out
    try:
        axes, summary = ask_framing(llm, settings, article)
    except Exception as exc:  # framing is display-only; the article is still usable without it
        print(f"[enrich] framing retry failed {article.get('id')}: {str(exc)[:120]}")
        return out
    return {**out, "stance": merge_framing(stance, axes, summary)}


def backfill_framing(db, llm, settings, *, days: int = 7, limit: int = 3000, dry_run: bool = False) -> dict:
    """Re-ask framing for articles in stories from the last `days` days that have none."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    candidates: list[dict] = []
    page, offset = 1000, 0
    while True:
        rows = db.select("articles_v3", select="id,stance", story_id="not.is.null", enrichment_status="eq.done",
                         published_at=f"gte.{since}", order="published_at.desc",
                         limit=str(page), offset=str(offset))
        candidates.extend(r for r in rows if needs_framing(r.get("stance")))
        if len(rows) < page:
            break
        offset += page
    candidates = candidates[:limit]
    stats = {"missing": len(candidates), "filled": 0, "failed": 0}
    print(f"[framing] {len(candidates)} articles from the last {days} days have no framing")
    if dry_run or not candidates:
        return stats

    def work(row: dict) -> tuple[dict, dict | None]:
        full = db.select("articles_v3", select="id,outlet,title,headline,subheadline,body_text", id=f"eq.{row['id']}")
        if not full:
            return row, None
        try:
            axes, summary = ask_framing(llm, settings, full[0])
        except Exception as exc:
            print(f"[framing] failed {row['id']}: {str(exc)[:120]}")
            return row, None
        if not axes and not summary:
            return row, None
        return row, merge_framing(row.get("stance"), axes, summary)

    with ThreadPoolExecutor(max_workers=settings.enrich_concurrency) as pool:
        for row, stance in pool.map(work, candidates):
            if stance is None:
                stats["failed"] += 1
                continue
            db.update("articles_v3", {"id": f"eq.{row['id']}"}, {"stance": stance})
            stats["filled"] += 1
    print(f"[framing] filled={stats['filled']} failed={stats['failed']} | {llm.usage_summary()}")
    return stats
