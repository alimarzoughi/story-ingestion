"""Stage 3b: fold duplicate stories together.

Assign only ever compares a new article with existing stories, so two stories about the same development
can form in parallel (e.g. two outlets' first reports processed before either story existed) and then both keep
growing. Each run: find open stories whose centroids are very close, require compatible event dates, ask the
verifier (same main-point rule as assign), and merge the smaller into the larger.

The merge itself runs in SQL (`merge_stories`, schema_patch_002.sql) so moving articles, recounting and retiring
the absorbed story's pair happen in one transaction.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from .config import Settings
from .db import Db
from .llm import LLM

MERGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "same_development": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["same_development", "confidence", "reason"],
}

MERGE_SYSTEM = """You decide whether two story clusters from a news comparison app are the SAME story and should be merged.
A story is ONE specific development: a single concrete event, action or announcement (e.g. 'Trump bans CNN, MS NOW and Politico from the White House'), never a broad topic or ongoing saga ('Trump vs the press').
Each cluster is given as a title, a summary and sample headlines. They are the SAME story when their main development is the same: coverage of it, reactions to it and analysis of it all belong together.
They are DIFFERENT when one is a follow-up development of the other (a lawsuit over an action, a court ruling, a retaliation, a resignation, a vote on a response), even though they share the same people and one recaps the other.
If unsure, answer false. Return only JSON."""

STORY_SELECT = "id,title,summary,event_date,first_seen,last_seen,article_count,status"


def split_between(db: Db, a_id: str, b_id: str) -> bool:
    """True if either story holds an article that recheck/assign split off from the other one. Such a pair is a
    story and its follow-up by an earlier decision; merging would silently undo that split."""
    for here, other in ((a_id, b_id), (b_id, a_id)):
        if db.select("articles_v3", select="id", story_id=f"eq.{here}", **{"assignment->>split_from": f"eq.{other}"}, limit="1"):
            return True
    return False


def _date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def dates_compatible(a: dict, b: dict, max_gap_days: int) -> bool:
    da, db_ = _date(a.get("event_date")), _date(b.get("event_date"))
    if da is None or db_ is None:
        return True
    return abs((da - db_).days) <= max_gap_days


def survivor_of(a: dict, b: dict) -> tuple[dict, dict]:
    """(survivor, absorbed): the larger story survives; ties go to the older one."""
    key = lambda s: (int(s.get("article_count") or 0), -datetime.fromisoformat(s["first_seen"].replace("Z", "+00:00")).timestamp())
    return (a, b) if key(a) >= key(b) else (b, a)


def _describe(db: Db, story: dict) -> str:
    heads = db.select("articles_v3", select="outlet,title", story_id=f"eq.{story['id']}", order="published_at.asc", limit="6")
    lines = "\n".join(f"    - {h['outlet']}: {h['title']}" for h in heads)
    return (f"title: {story['title']}\n  summary: {story['summary']}\n  event_date: {story.get('event_date')}, "
            f"first reported: {story.get('first_seen')}\n  sample headlines:\n{lines}")


def run_merge(db: Db, llm: LLM | None, settings: Settings) -> dict:
    stats = {"candidates": 0, "checked": 0, "merged": 0, "moved": 0, "rejected_date": 0, "rejected_split": 0}
    if llm is None:
        print("[merge] skipped: no LLM (merges are never made without verification)")
        return stats
    since = (datetime.now(timezone.utc) - timedelta(hours=settings.merge_lookback_hours)).isoformat()
    rows = db.rpc("story_merge_candidates", {"min_similarity": settings.merge_min_similarity, "updated_since": since,
                                             "max_pairs": settings.merge_max_checks * 3}) or []
    seen: set[frozenset] = set()
    pairs = []
    for row in sorted(rows, key=lambda r: -float(r["similarity"])):
        key = frozenset((row["a_id"], row["b_id"]))
        if key not in seen and row["a_id"] != row["b_id"]:
            seen.add(key)
            pairs.append(row)
    stats["candidates"] = len(pairs)

    absorbed_into: dict[str, str] = {}
    for row in pairs:
        if stats["checked"] >= settings.merge_max_checks:
            break
        a_id, b_id = row["a_id"], row["b_id"]
        if a_id in absorbed_into or b_id in absorbed_into:
            continue  # one side was already folded into another story this run
        found = {s["id"]: s for s in db.select("stories", select=STORY_SELECT, id=f"in.({a_id},{b_id})")}
        a, b = found.get(a_id), found.get(b_id)
        if not a or not b or a.get("status") != "open" or b.get("status") != "open":
            continue
        if not dates_compatible(a, b, settings.merge_max_event_gap_days):
            stats["rejected_date"] += 1
            continue
        if split_between(db, a_id, b_id):
            stats["rejected_split"] += 1
            continue
        user = f"STORY A\n  {_describe(db, a)}\n\nSTORY B\n  {_describe(db, b)}\n\nAre A and B the same specific development?"
        try:
            out = llm.structured(kind="merge", model=settings.verify_model, system=MERGE_SYSTEM, user=user,
                                 schema=MERGE_SCHEMA, schema_name="story_merge", max_output_tokens=300)
        except Exception as exc:
            print(f"[merge] verifier failed for {a_id}/{b_id}: {str(exc)[:160]}")
            continue
        stats["checked"] += 1
        if not (out.get("same_development") and float(out.get("confidence") or 0) >= 0.7):
            continue
        keep, drop = survivor_of(a, b)
        moved = db.rpc("merge_stories", {"survivor": keep["id"], "absorbed": drop["id"]}) or 0
        absorbed_into[drop["id"]] = keep["id"]
        stats["merged"] += 1
        stats["moved"] += int(moved)
        print(f"[merge] '{drop['title'][:70]}' ({drop.get('article_count')}) -> '{keep['title'][:70]}' ({keep.get('article_count')}) sim={float(row['similarity']):.3f}")
    print(f"[merge] candidates={stats['candidates']} checked={stats['checked']} merged={stats['merged']} "
          f"moved={stats['moved']} rejected_date={stats['rejected_date']} rejected_split={stats['rejected_split']}")
    return stats
