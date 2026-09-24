"""Stage 4: materialise the best LEFT/RIGHT pair per story into story_pairs."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any

from .config import Settings
from .db import Db
from .llm import LLM
from .textclean import is_non_article_title

OPPOSING = {("supports", "opposes"), ("opposes", "supports"), ("critical_of_left", "critical_of_right"), ("critical_of_right", "critical_of_left")}

TITLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "story_title": {"type": "string", "description": "Neutral headline for the Development text only, <= 12 words, no outlet voice."},
        "headline_contrast": {"type": "string", "description": "One sentence (<= 30 words) stating what the left-side piece foregrounds versus what the right-side piece foregrounds. Refer to them as 'Left' and 'Right'."},
    },
    "required": ["story_title", "headline_contrast"],
}
TITLE_SYSTEM = (
    "You write neutral story titles and one-line framing contrasts for a news comparison app. Be concrete and even-handed. "
    "The story title must describe ONLY the development stated under 'Development' (e.g. 'Trump bans three outlets from "
    "White House'). The two pieces are given only for the framing contrast: never put reactions to the development or "
    "later follow-ups from their headlines into the title. Return only JSON."
)

# never shown as a side's headline (still counted as story members)
NOT_DISPLAYABLE_TYPES = {"roundup"}


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower()).strip()


def near_duplicate(a: dict, b: dict, ratio: float) -> bool:
    ta, tb = _norm_title(a.get("title", "")), _norm_title(b.get("title", ""))
    if ta and tb and SequenceMatcher(None, ta, tb).ratio() >= ratio:
        return True
    sa, sb = (a.get("snippet") or "")[:200], (b.get("snippet") or "")[:200]
    return bool(sa and sb and SequenceMatcher(None, sa, sb).ratio() >= 0.9)


def _dir(article: dict) -> tuple[str | None, float]:
    stance = article.get("stance") or {}
    return stance.get("direction"), float(stance.get("confidence") or 0)


def _axes(article: dict) -> set[str]:
    return {a.strip().lower() for a in ((article.get("stance") or {}).get("framing_axes") or []) if a}


def displayable(article: dict) -> bool:
    """Can this member be shown as a headline? Excludes wire copy, roundups/newsletters, show pages and transcripts."""
    return (not article.get("is_wire_copy")
            and article.get("article_type") not in NOT_DISPLAYABLE_TYPES
            and not is_non_article_title(article.get("title")))


def score_pair(left: dict, right: dict, current_outlets: set[str] | None = None) -> tuple[float, str, dict]:
    score, detail = 0.0, {}
    (dl, cl), (dr, cr) = _dir(left), _dir(right)
    kind = "same_story"
    if dl and dr and (dl, dr) in OPPOSING and cl >= 0.5 and cr >= 0.5:
        score += 3.0
        kind = "opposing_stance"
        detail["opposing"] = [dl, dr]
    al, ar = _axes(left), _axes(right)
    if al and ar and not (al & ar):
        score += 1.5
        detail["framing_disjoint"] = True
        if kind == "same_story":
            kind = "framing_contrast"
    if left.get("article_type") == right.get("article_type"):
        score += 1.0
    elif {left.get("article_type"), right.get("article_type")} == {"news", "opinion"}:
        score -= 1.0
    if "live" in (left.get("article_type"), right.get("article_type")):
        score -= 1.5
    if left.get("image_url") and right.get("image_url"):
        score += 1.0
    if left.get("snippet") and right.get("snippet"):
        score += 0.5
    hours = abs((_ts(left["published_at"]) - _ts(right["published_at"])).total_seconds()) / 3600
    score -= 0.02 * hours
    detail["hours_apart"] = round(hours, 1)
    if current_outlets and not ({left["outlet"], right["outlet"]} & current_outlets):
        score += 0.5
    return round(score, 3), kind, detail


def title_and_contrast(llm: LLM, settings: Settings, story: dict, left: dict, right: dict) -> tuple[str | None, str | None]:
    """(story title, headline contrast). The story's summary is its anchor (the founding article's event summary)
    and is never rewritten here: a summary written from later pair articles drifts toward follow-ups, and every
    later attach / recheck / merge decision is measured against it."""
    user = (
        f"Development: {story['summary']}\n\n"
        f"LEFT-side piece ({left['outlet']}): title: {left['title']}\n  framing: {(left.get('stance') or {}).get('framing_summary')}\n"
        f"RIGHT-side piece ({right['outlet']}): title: {right['title']}\n  framing: {(right.get('stance') or {}).get('framing_summary')}\n"
    )
    out = llm.structured(kind="title", model=settings.verify_model, system=TITLE_SYSTEM, user=user,
                         schema=TITLE_SCHEMA, schema_name="story_title_and_contrast", max_output_tokens=300)
    title = (out.get("story_title") or "").strip()[:200] or None
    return title, out.get("headline_contrast")


def _ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def best_pair(members: list[dict], settings: Settings, current: dict | None) -> tuple[dict, dict, float, str, dict] | None:
    lefts = [m for m in members if m.get("side") == "left" and displayable(m)]
    rights = [m for m in members if m.get("side") == "right" and displayable(m)]
    if not lefts or not rights:
        return None
    current_outlets = None
    if current:
        current_outlets = {current["left_outlet"], current["right_outlet"]}
    best = None
    for l in lefts:
        for r in rights:
            if l["outlet"] == r["outlet"]:
                continue
            if near_duplicate(l, r, settings.near_duplicate_title_ratio):
                continue
            s, kind, detail = score_pair(l, r, current_outlets)
            if best is None or s > best[2]:
                best = (l, r, s, kind, detail)
    return best


MEMBER_SELECT = "id,outlet,source_id,title,headline,snippet,image_url,published_at,article_type,stance,side,is_wire_copy,loaded_terms,event_summary"


def run_pairs(db: Db, llm: LLM | None, settings: Settings, *, lookback_hours: int | None = None, story_id: str | None = None) -> dict:
    stats = {"stories": 0, "created": 0, "replaced": 0, "kept": 0, "titled": 0, "no_pair": 0}
    since = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours or settings.pairs_lookback_hours)).isoformat()
    params: dict[str, Any] = {"select": "id,title,summary,title_source,left_count,right_count", "left_count": "gte.1", "right_count": "gte.1", "status": "eq.open"}
    if story_id:
        params["id"] = f"eq.{story_id}"
    else:
        params["updated_at"] = f"gte.{since}"
    stories = db.select("stories", **params)
    stats["stories"] = len(stories)

    for story in stories:
        members = db.select("articles_v3", select=MEMBER_SELECT, story_id=f"eq.{story['id']}")
        existing = db.select("story_pairs", select="id,left_article_id,right_article_id,divergence_score,featured_at",
                             story_id=f"eq.{story['id']}", is_current="eq.true")
        current = None
        if existing:
            by_id = {m["id"]: m for m in members}
            l, r = by_id.get(existing[0]["left_article_id"]), by_id.get(existing[0]["right_article_id"])
            if l and r:
                current = {**existing[0], "left_outlet": l["outlet"], "right_outlet": r["outlet"]}
        choice = best_pair(members, settings, current)
        if choice is None:
            if existing:  # the old pair no longer stands (members detached or reclassified)
                db.update("story_pairs", {"story_id": f"eq.{story['id']}", "is_current": "eq.true"}, {"is_current": False})
                stats["retired"] = stats.get("retired", 0) + 1
            stats["no_pair"] += 1
            continue
        left, right, score, kind, detail = choice
        featured_at = max(left["published_at"], right["published_at"])
        if current:
            same = {current["left_article_id"], current["right_article_id"]} == {left["id"], right["id"]}
            newer = _ts(featured_at) - _ts(current["featured_at"]) > timedelta(hours=12)
            better = score >= float(current["divergence_score"]) + 1.0
            if same or not (better or (newer and score >= float(current["divergence_score"]) - 0.5)):
                stats["kept"] += 1
                if llm is not None and story.get("title_source") != "llm":
                    # story was (re)anchored: give it a proper title even though its pair stays
                    by_id = {m["id"]: m for m in members}
                    try:
                        title, _ = title_and_contrast(llm, settings, story, by_id[current["left_article_id"]],
                                                      by_id[current["right_article_id"]])
                        if title:
                            db.update("stories", {"id": f"eq.{story['id']}"}, {"title": title, "title_source": "llm"})
                            stats["titled"] += 1
                    except Exception as exc:
                        print(f"[pairs] title call failed for {story['id']}: {str(exc)[:160]}")
                continue

        contrast, title_patch = None, {}
        if llm is not None:
            try:
                title, contrast = title_and_contrast(llm, settings, story, left, right)
                if story.get("title_source") != "llm" and title:
                    title_patch = {"title": title, "title_source": "llm"}  # never the summary: it is the anchor
                    stats["titled"] += 1
            except Exception as exc:
                print(f"[pairs] title/contrast call failed for {story['id']}: {str(exc)[:160]}")

        if existing:  # includes a stale current pair whose articles are no longer members
            db.update("story_pairs", {"story_id": f"eq.{story['id']}", "is_current": "eq.true"}, {"is_current": False})
        db.insert("story_pairs", {
            "story_id": story["id"], "left_article_id": left["id"], "right_article_id": right["id"],
            "pair_kind": kind, "divergence_score": score, "headline_contrast": contrast, "featured_at": featured_at,
            "is_current": True, "debug": detail,
        }, returning=False, on_conflict="story_id,left_article_id,right_article_id")
        if title_patch:
            db.update("stories", {"id": f"eq.{story['id']}"}, title_patch)
        stats["replaced" if existing else "created"] += 1

    print(f"[pairs] stories={stats['stories']} created={stats['created']} replaced={stats['replaced']} kept={stats['kept']} no_pair={stats['no_pair']} titled={stats['titled']}")
    return stats
