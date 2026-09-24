"""Stage 3: assign each enriched article to a story (vector retrieval -> rules -> gray-zone verifier)."""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .config import Settings
from .db import Db, pg_to_vec, vec_to_pg
from .llm import LLM
from .sources import side_for

CREATES_STORIES = {"news", "live", "explainer"}  # opinion/analysis/roundup may only attach

VERIFY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "same_story_id": {"type": ["string", "null"], "description": "The candidate id whose story is the SAME specific development as the article, or null if none."},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["same_story_id", "confidence", "reason"],
}

# Bumped whenever the attach rule changes; stored on every assignment so `recheck-stories` can skip
# articles already decided under the current rule.
ASSIGN_RULE = "2026-09-23.main-point"

VERIFY_SYSTEM = """You decide which story, if any, a news article belongs to.
A story is ONE specific development: a single concrete event, action or announcement (e.g. 'Trump bans CNN, MS NOW and Politico from the White House', 'Russian missile strike on Kyiv on Sept 23'). It is never a broad topic or ongoing saga ('Trump vs the press', 'War in Ukraine').

The article belongs to a candidate story only if the article's MAIN POINT (what its headline and opening are about) is that story's development.
SAME story: other outlets reporting the development; reactions, statements, criticism or defences of it; analysis and opinion about it; the action simply being carried out (e.g. reporters turned away once a ban takes effect); new details about the same development.
DIFFERENT story (a follow-up): the article's main point is a NEW development that grows out of the original, e.g. the banned outlets filing a lawsuit, a court ruling, a retaliation, a resignation, a vote on a response, an investigation opening. This holds even when the article recaps the original development at length.
Examples: 'Trump bans three outlets' vs 'Outlets sue Trump over the ban' -> DIFFERENT. 'Trump bans three outlets' vs 'Fox hosts criticise the ban' -> SAME. 'Israel strikes Tehran on April 3' vs 'Iran retaliates on April 4' -> DIFFERENT. An opinion column about a Senate vote -> SAME as the vote.
If a candidate is itself the follow-up development the article is about, choose that candidate.
If no candidate fits, or you are unsure, answer null. Return only JSON."""


def _parse_ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def jaccard(a: list[str], b: list[str]) -> float:
    sa = {x.lower() for x in a or []}
    sb = {x.lower() for x in b or []}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def score_candidate(article: dict, cand: dict) -> tuple[float, dict]:
    sim = float(cand["similarity"])
    ent = jaccard(article.get("key_entities") or [], cand.get("key_entities") or [])
    days_gap = abs((_parse_ts(article["published_at"]) - _parse_ts(cand["last_seen"])).total_seconds()) / 86400
    score = sim + 0.05 * ent - 0.02 * days_gap
    return score, {"sim": round(sim, 4), "entity_jaccard": round(ent, 3), "days_gap": round(days_gap, 2), "score": round(score, 4)}


def date_compatible(article: dict, cand: dict, max_gap_days: int) -> bool:
    a, c = _parse_date(article.get("event_date")), _parse_date(cand.get("event_date"))
    if a is None or c is None:
        return True
    return abs((a - c).days) <= max_gap_days


def needs_verification(article: dict, story: dict, settings: Settings) -> str | None:
    """Why a high-scoring match must still go through the verifier, or None if it can attach directly.

    Follow-up developments (a lawsuit over a ban, a retaliation after a strike) share nearly all their entities
    with the original story, so similarity alone cannot tell them apart. They arrive later and usually carry a
    later event date, so only early, same-event-date coverage is trusted without a verifier call.
    """
    first_seen = story.get("first_seen")
    if not first_seen:
        return "no_first_seen"
    if _parse_ts(article["published_at"]) - _parse_ts(first_seen) > timedelta(hours=settings.auto_attach_window_hours):
        return "late"
    a, c = _parse_date(article.get("event_date")), _parse_date(story.get("event_date"))
    if a and c and a > c:
        return "later_event_date"
    return None


def decide(article: dict, candidates: list[dict], settings: Settings) -> tuple[str, dict | None, list[dict]]:
    """Returns (decision, chosen_candidate, scored) where decision in {'attach','verify','create'}."""
    scored: list[dict] = []
    for cand in candidates:
        if not date_compatible(article, cand, settings.event_date_max_gap_days):
            scored.append({"id": cand["id"], "rejected": "event_date_gap"})
            continue
        s, detail = score_candidate(article, cand)
        scored.append({"id": cand["id"], "title": cand.get("title"), **detail})
    live = [x for x in scored if "score" in x]
    live.sort(key=lambda x: x["score"], reverse=True)
    if not live:
        return "create", None, scored
    best = live[0]
    if best["score"] >= settings.attach_threshold:
        chosen = next(c for c in candidates if c["id"] == best["id"])
        reason = needs_verification(article, chosen, settings)
        if reason is None:
            return "attach", chosen, scored
        best["verify_reason"] = reason
        return "verify", None, scored
    if best["score"] >= settings.verify_threshold:
        return "verify", None, scored
    return "create", None, scored


def verify_candidates(llm: LLM, settings: Settings, article: dict, candidates: list[dict]) -> tuple[dict | None, dict]:
    """Ask the verifier which candidate story (if any) has the article's main point as its development.
    Returns (chosen candidate or None, raw verifier output)."""
    by_id = {c["id"]: c for c in candidates}
    cand_text = "\n".join(
        f"- id={c['id']}: {c.get('title')} | {c.get('summary')} (event_date={c.get('event_date')}, first reported={c.get('first_seen') or 'n/a'})"
        for c in candidates
    )
    user = (
        f"ARTICLE\nOutlet: {article.get('outlet')}\nPublished: {article.get('published_at')}\nHeadline: {article.get('title')}\n"
        f"Neutral summary of its main development: {article.get('event_summary')}\nEvent date: {article.get('event_date')}\n"
        f"Entities: {', '.join(article.get('key_entities') or [])}\n\n"
        f"CANDIDATE STORIES\n{cand_text}\n\nWhich candidate story, if any, is this article's main point?"
    )
    out = llm.structured(kind="verify", model=settings.verify_model, system=VERIFY_SYSTEM, user=user,
                         schema=VERIFY_SCHEMA, schema_name="story_verification", max_output_tokens=300)
    sid = out.get("same_story_id")
    if sid in by_id and float(out.get("confidence") or 0) >= 0.6:
        return by_id[sid], out
    return None, out


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def merged_centroid(old: list[float] | None, n_old: int, new: list[float]) -> list[float]:
    if not old or n_old <= 0:
        return _normalize(new)
    merged = [(o * n_old + x) / (n_old + 1) for o, x in zip(old, new)]
    return _normalize(merged)


class Assigner:
    def __init__(self, db: Db, llm: LLM | None, settings: Settings, source_leaning: dict[str, str]) -> None:
        self.db, self.llm, self.settings = db, llm, settings
        self.source_leaning = source_leaning
        self.stats = {"processed": 0, "attached": 0, "created": 0, "verified": 0, "verifier_attached": 0, "orphaned": 0, "failed": 0}

    def side_of(self, article: dict) -> str | None:
        label, conf = article.get("leaning_label"), float(article.get("leaning_conf") or 0)
        if label in ("LEFT", "RIGHT") and conf >= 0.6:
            return label.lower()
        return side_for(self.source_leaning.get(article["source_id"]))

    def candidates_for(self, article: dict, embedding: list[float]) -> list[dict]:
        since = (_parse_ts(article["published_at"]) - timedelta(hours=self.settings.candidate_window_hours)).isoformat()
        rows = self.db.rpc("match_stories", {"query": vec_to_pg(embedding), "since": since, "k": 10}) or []
        return rows

    def verify(self, article: dict, scored: list[dict], candidates: list[dict]) -> dict | None:
        if self.llm is None:
            return None
        top_ids = [x["id"] for x in scored if "score" in x][:3]
        by_id = {c["id"]: c for c in candidates}
        chosen, out = verify_candidates(self.llm, self.settings, article, [by_id[i] for i in top_ids])
        self.stats["verified"] += 1
        if chosen is not None:
            self.stats["verifier_attached"] += 1
            return {**chosen, "_verifier": out}
        return {"_verifier": out}  # no match

    def attach(self, article: dict, story: dict, embedding: list[float], method: str, scored: list[dict], extra: dict | None = None) -> None:
        full = self.db.select("stories", select="id,centroid,article_count,left_count,right_count,center_count,key_entities,event_date,last_seen,first_seen", id=f"eq.{story['id']}")[0]
        n = int(full["article_count"])
        centroid = merged_centroid(pg_to_vec(full.get("centroid")), n, embedding)
        side = self.side_of(article)
        entities = list(dict.fromkeys((full.get("key_entities") or []) + (article.get("key_entities") or [])))[:24]
        published = article["published_at"]
        patch = {
            "centroid": vec_to_pg(centroid),
            "article_count": n + 1,
            "left_count": int(full["left_count"]) + (side == "left"),
            "right_count": int(full["right_count"]) + (side == "right"),
            "center_count": int(full["center_count"]) + (side == "center"),
            "key_entities": entities,
            "last_seen": max(full["last_seen"], published),
            "first_seen": min(full["first_seen"], published),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if not full.get("event_date") and article.get("event_date"):
            patch["event_date"] = article["event_date"]
        self.db.update("stories", {"id": f"eq.{story['id']}"}, patch)
        self.db.update("articles_v3", {"id": f"eq.{article['id']}"}, {
            "story_id": story["id"], "side": side, "assigned_at": datetime.now(timezone.utc).isoformat(),
            "assignment": {"method": method, "rule": ASSIGN_RULE, "candidates": scored[:5], "thresholds": [self.settings.attach_threshold, self.settings.verify_threshold], **(extra or {})},
        })

    def create(self, article: dict, embedding: list[float], scored: list[dict], extra: dict | None = None) -> None:
        side = self.side_of(article)
        title = (article.get("event_summary") or article["title"]).split(". ")[0][:140]
        story = self.db.insert("stories", {
            "title": title,
            "summary": article.get("event_summary") or article["title"],
            "title_source": "auto",
            "category": article.get("category"),
            "event_date": article.get("event_date"),
            "first_seen": article["published_at"],
            "last_seen": article["published_at"],
            "status": "open",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "centroid": vec_to_pg(_normalize(embedding)),
            "key_entities": (article.get("key_entities") or [])[:24],
            "article_count": 1,
            "left_count": int(side == "left"),
            "right_count": int(side == "right"),
            "center_count": int(side == "center"),
        })[0]
        self.db.update("articles_v3", {"id": f"eq.{article['id']}"}, {
            "story_id": story["id"], "side": side, "assigned_at": datetime.now(timezone.utc).isoformat(),
            "assignment": {"method": "create", "rule": ASSIGN_RULE, "candidates": scored[:5], "thresholds": [self.settings.attach_threshold, self.settings.verify_threshold], **(extra or {})},
        })

    def orphan(self, article: dict, scored: list[dict], reason: str) -> None:
        self.db.update("articles_v3", {"id": f"eq.{article['id']}"}, {
            "assigned_at": datetime.now(timezone.utc).isoformat(), "side": self.side_of(article),
            "assignment": {"method": "orphan", "reason": reason, "candidates": scored[:5]},
        })
        self.stats["orphaned"] += 1

    def process(self, article: dict) -> None:
        embedding = pg_to_vec(article.get("embedding"))
        if not embedding:
            self.orphan(article, [], "no embedding")
            return
        candidates = self.candidates_for(article, embedding)
        decision, chosen, scored = decide(article, candidates, self.settings)
        extra: dict[str, Any] = {}
        if decision == "verify":
            result = self.verify(article, scored, candidates)
            extra["verifier"] = (result or {}).get("_verifier")
            if result and "id" in result:
                self.attach(article, result, embedding, "verifier", scored, extra)
                self.stats["attached"] += 1
                return
            decision = "create"
        if decision == "attach" and chosen:
            self.attach(article, chosen, embedding, "threshold", scored, extra)
            self.stats["attached"] += 1
            return
        if article.get("article_type") in CREATES_STORIES and not article.get("is_wire_copy"):
            self.create(article, embedding, scored, extra)
            self.stats["created"] += 1
        else:
            self.orphan(article, scored, f"{article.get('article_type')}{' wire' if article.get('is_wire_copy') else ''} cannot create a story")


ASSIGN_SELECT = "id,source_id,outlet,title,published_at,event_summary,event_date,key_entities,category,article_type,leaning_label,leaning_conf,is_wire_copy,embedding"


def run_assign(db: Db, llm: LLM | None, settings: Settings, source_leaning: dict[str, str], *, limit: int | None = None) -> dict:
    limit = limit or settings.assign_batch
    rows = db.select("articles_v3", select=ASSIGN_SELECT, enrichment_status="eq.done", story_id="is.null",
                     assigned_at="is.null", order="published_at.asc", limit=str(limit))
    # retry orphans (opinion/analysis whose story did not exist yet) for 48h, at most once per hour
    since = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    retry_before = (datetime.now(timezone.utc) - timedelta(minutes=55)).isoformat()
    orphans = db.select("articles_v3", select=ASSIGN_SELECT, enrichment_status="eq.done", story_id="is.null",
                        **{"assignment->>method": "eq.orphan", "published_at": f"gte.{since}", "assigned_at": f"lte.{retry_before}"},
                        order="published_at.asc", limit="100")
    assigner = Assigner(db, llm, settings, source_leaning)
    assigner.stats["orphan_retries"] = len(orphans)
    for article in rows + orphans:
        try:
            assigner.process(article)
        except Exception as exc:
            assigner.stats["failed"] += 1
            print(f"[assign] failed {article['id']}: {str(exc)[:200]}")
        assigner.stats["processed"] += 1
    s = assigner.stats
    print(f"[assign] processed={s['processed']} attached={s['attached']} (via verifier {s['verifier_attached']}/{s['verified']}) created={s['created']} orphaned={s['orphaned']} orphan_retries={s['orphan_retries']} failed={s['failed']}")
    closed = db.rpc("close_stale_stories", {"older_than": "5 days"})
    if closed:
        print(f"[assign] closed {closed} stale stories")
    return s


# ---------------------------------------------------------------------------
# recheck: re-apply the current attach rule to articles attached under an older rule
# ---------------------------------------------------------------------------
RECHECK_SELECT = ASSIGN_SELECT + ",story_id,assignment"


def recheck_suspects(members: list[dict], story: dict, settings: Settings) -> list[dict]:
    """Members that the current rule would have sent to the verifier but that were never verified under it."""
    out = []
    for m in members:
        assignment = m.get("assignment") or {}
        if assignment.get("method") == "create" or assignment.get("rule") == ASSIGN_RULE:
            continue
        if needs_verification(m, story, settings) is None:
            continue
        out.append(m)
    return out


def run_recheck(db: Db, llm: LLM, settings: Settings, *, dry_run: bool = False, max_checks: int = 2000) -> dict:
    """Detach members whose main point is a follow-up development, so the next assign pass files them correctly.

    Only touches articles attached under an older rule and only those the current rule would have verified
    (published > auto_attach_window_hours after the story began, or with a later event date).
    Detached articles get story_id = NULL / assigned_at = NULL and are picked up by run_assign.
    """
    from concurrent.futures import ThreadPoolExecutor

    stats = {"stories": 0, "suspects": 0, "kept": 0, "detached": 0, "failed": 0}
    stories = db.select("stories", select="id,title,summary,event_date,first_seen,last_seen,article_count",
                        status="eq.open", article_count="gte.2")
    work: list[tuple[dict, dict]] = []
    for story in stories:
        members = db.select("articles_v3", select=RECHECK_SELECT, story_id=f"eq.{story['id']}")
        suspects = recheck_suspects(members, story, settings)
        if suspects:
            stats["stories"] += 1
            work.extend((m, story) for m in suspects)
    work = work[:max_checks]
    stats["suspects"] = len(work)
    print(f"[recheck] {stats['suspects']} articles in {stats['stories']} stories need re-verification under rule {ASSIGN_RULE}")

    def check(item: tuple[dict, dict]) -> tuple[dict, dict, dict | None, dict]:
        article, story = item
        chosen, out = verify_candidates(llm, settings, article, [story])
        return article, story, chosen, out

    touched: set[str] = set()
    samples: list[str] = []
    with ThreadPoolExecutor(max_workers=settings.enrich_concurrency) as pool:
        for future in [pool.submit(check, item) for item in work]:
            try:
                article, story, chosen, out = future.result()
            except Exception as exc:
                stats["failed"] += 1
                print(f"[recheck] verify failed: {str(exc)[:160]}")
                continue
            if chosen is not None:
                stats["kept"] += 1
                if not dry_run:
                    db.update("articles_v3", {"id": f"eq.{article['id']}"},
                              {"assignment": {**(article.get("assignment") or {}), "rule": ASSIGN_RULE, "rechecked": True}})
                continue
            stats["detached"] += 1
            if len(samples) < 40:
                samples.append(f"  - [{story['title'][:60]}] -> {article.get('outlet')}: {article.get('title', '')[:90]}")
            if not dry_run:
                db.update("articles_v3", {"id": f"eq.{article['id']}"}, {
                    "story_id": None, "assigned_at": None, "side": None,
                    "assignment": {"method": "detached", "rule": ASSIGN_RULE, "from_story": story["id"],
                                   "verifier": out, "previous": article.get("assignment")},
                })
                touched.add(story["id"])
    if not dry_run:
        for story_id in touched:
            db.rpc("refresh_story_stats", {"target": story_id})
    mode = " (dry run: nothing written)" if dry_run else ""
    print(f"[recheck]{mode} kept={stats['kept']} detached={stats['detached']} failed={stats['failed']}")
    if samples:
        print("[recheck] detached (story -> article):" if not dry_run else "[recheck] would detach (story -> article):")
        print("\n".join(samples))
    return stats
