"""Stage 3: assign each enriched article to a story (vector retrieval -> rules -> gray-zone verifier)."""
from __future__ import annotations

import math
import re
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
        "candidate_id": {"type": ["string", "null"], "description": "The candidate story the article is closest to: the one whose development it is about, or grows out of. Null only if it relates to none of them."},
        "relation": {"type": "string", "enum": ["same_development", "follow_up", "unrelated"],
                     "description": "How the article's main point relates to that candidate's development."},
        "confidence": {"type": "number", "description": "0-1: how confident you are in the relation you chose."},
        "reason": {"type": "string"},
    },
    "required": ["candidate_id", "relation", "confidence", "reason"],
}

# Bumped whenever the attach rule changes; stored on every assignment so `recheck-stories` can skip
# articles already decided under the current rule.
ASSIGN_RULE = "2026-09-25c.split-guard"

VERIFY_SYSTEM = """You decide which story, if any, a news article belongs to.
A story is ONE specific development: a single concrete event, action or announcement (e.g. 'Trump bans CNN, MS NOW and Politico from the White House', 'Russian missile strike on Kyiv on Sept 23'). It is never a broad topic or ongoing saga ('Trump vs the press', 'War in Ukraine').

The article belongs to a candidate story only if the article's MAIN POINT (what its headline and opening are about) is that story's development.
SAME story: other outlets reporting the development; reactions, statements, criticism or defences of it; analysis and opinion about it, including op-eds arguing for or against it; the action simply being carried out (e.g. reporters turned away once a ban takes effect); new details about the same development. Commentary and reaction pieces (how a speech, ad, poll or decision was received, what critics or supporters said about it) belong to that development's story even when their angle or tone is new.
DIFFERENT story (a follow-up): the article's main point is a NEW development that grows out of the original, e.g. the banned outlets filing a lawsuit, a court ruling, a retaliation, a resignation, a vote on a response, an investigation opening. This holds even when the article recaps the original development at length.
Examples: 'Trump bans three outlets' vs 'Outlets sue Trump over the ban' -> DIFFERENT. 'Trump bans three outlets' vs 'Fox hosts criticise the ban' -> SAME. 'Israel strikes Tehran on April 3' vs 'Iran retaliates on April 4' -> DIFFERENT. An opinion column about a Senate vote -> SAME as the vote.
If a candidate is itself the follow-up development the article is about, choose that candidate with relation same_development.

Answer with the closest candidate and one relation:
- same_development: the article's main point is that candidate's development (including any reaction, commentary or analysis of it).
- follow_up: the article's main point is clearly a NEW action or event that grows out of that candidate's development.
- unrelated: the article is about something else entirely.
Choose follow_up only for a genuinely new action or event, never for a new angle, tone or reaction. When torn between same_development and follow_up, choose same_development. Give an honest confidence. Return only JSON."""


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


def verify_candidates(llm: LLM, settings: Settings, article: dict, candidates: list[dict], *,
                      keep_id: str | None = None) -> tuple[dict | None, dict]:
    """Ask the verifier which candidate story the article's main point belongs to.

    Returns (chosen candidate or None, raw verifier output). Two modes:
    - keep_id=None (gray zone, weak similarity): attach only on an affirmative same_development answer.
    - keep_id=<story id> (strong similarity, or an existing member being rechecked): SAME BY DEFAULT. The article
      stays with keep_id unless the verifier is at least `split_min_confidence` sure it is a follow-up or unrelated.
      Lumping a follow-up in with the original is a smaller error than splitting coverage of the original off
      (a split-off opinion piece has no story to live in and vanishes from the feed).
    """
    by_id = {c["id"]: c for c in candidates}
    cand_text = "\n".join(
        f"- id={c['id']}: {c.get('title')} | {c.get('summary')} (event_date={c.get('event_date')}, first reported={c.get('first_seen') or 'n/a'})"
        for c in candidates
    )
    user = (
        f"ARTICLE\nOutlet: {article.get('outlet')}\nPublished: {article.get('published_at')}\nHeadline: {article.get('title')}\n"
        f"Neutral summary of its main development: {article.get('event_summary')}\nEvent date: {article.get('event_date')}\n"
        f"Entities: {', '.join(article.get('key_entities') or [])}\n\n"
        f"CANDIDATE STORIES\n{cand_text}\n\nWhich candidate is this article closest to, and how does its main point relate to it?"
    )
    out = llm.structured(kind="verify", model=settings.verify_model, system=VERIFY_SYSTEM, user=user,
                         schema=VERIFY_SCHEMA, schema_name="story_verification", max_output_tokens=300)
    relation, cid = out.get("relation"), out.get("candidate_id")
    try:
        confidence = float(out.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    if relation == "same_development" and cid in by_id and (keep_id is not None or confidence >= 0.6):
        return by_id[cid], out
    if keep_id is not None:
        if relation in ("follow_up", "unrelated") and confidence >= settings.split_min_confidence:
            return None, out
        return by_id[keep_id], out  # not confident it is a new development: it stays
    return None, out


_ABBREVIATIONS = {"rep", "sen", "gov", "gen", "dr", "mr", "mrs", "ms", "st", "jr", "sr", "vs", "no", "lt", "col",
                  "sgt", "mt", "ft", "u.s", "u.k", "u.n", "d.c", "inc", "corp", "co", "ltd", "jan", "feb", "aug", "sept", "oct", "nov", "dec"}


def placeholder_title(text: str, limit: int = 120) -> str:
    """Title for a new story until the pairs stage writes an LLM title. First sentence, but never cut at an
    abbreviation ('Rep. Maria Salazar...', 'Colts vs. Jets') -- the old '. ' split produced titles like 'Rep'."""
    text = " ".join((text or "").split())
    first = text
    for match in re.finditer(r"\.\s+(?=[A-Z0-9])", text):
        word = text[: match.start()].rsplit(" ", 1)[-1].lower()
        if word in _ABBREVIATIONS or len(word) == 1 or match.start() < 20:
            continue  # "Rep.", "vs.", an initial like "J.", or too early to be a whole sentence
        first = text[: match.start()]
        break
    if len(first) <= limit:
        return first
    return first[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "\u2026"


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
            best = max((x for x in scored if "score" in x), key=lambda x: x["score"], default=None)
            if best and best.get("verify_reason"):  # strong match, no verifier available: same by default
                return next(c for c in candidates if c["id"] == best["id"])
            return None
        live = sorted((x for x in scored if "score" in x), key=lambda x: x["score"], reverse=True)[:3]
        by_id = {c["id"]: c for c in candidates}
        # a strong match that was only sent here because it arrived late / carries a later event date
        keep_id = live[0]["id"] if live and live[0].get("verify_reason") else None
        chosen, out = verify_candidates(self.llm, self.settings, article, [by_id[x["id"]] for x in live], keep_id=keep_id)
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
        title = placeholder_title(article.get("event_summary") or article["title"])
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
            "assignment": {"method": "orphan", "reason": reason, "candidates": scored[:5], **split_memory(article)},
        })
        self.stats["orphaned"] += 1

    def process(self, article: dict) -> None:
        embedding = pg_to_vec(article.get("embedding"))
        if not embedding:
            self.orphan(article, [], "no embedding")
            return
        candidates = self.candidates_for(article, embedding)
        decision, chosen, scored = decide(article, candidates, self.settings)
        extra: dict[str, Any] = split_memory(article)
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


def split_memory(article: dict) -> dict:
    """Remember which story an article was split off from (`split_from`), across re-assignment and orphan retries,
    so the merge stage never folds the follow-up story back into the story it was split from."""
    a = article.get("assignment") or {}
    origin = a.get("split_from") or (a.get("from_story") if a.get("method") == "detached" else None)
    return {"split_from": origin} if origin else {}


def run_assign(db: Db, llm: LLM | None, settings: Settings, source_leaning: dict[str, str], *, limit: int | None = None) -> dict:
    limit = limit or settings.assign_batch
    rows = db.select("articles_v3", select=ASSIGN_SELECT + ",assignment", enrichment_status="eq.done", story_id="is.null",
                     assigned_at="is.null", order="published_at.asc", limit=str(limit))
    # retry orphans (opinion/analysis whose story did not exist yet) for 48h, at most once per hour
    since = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    retry_before = (datetime.now(timezone.utc) - timedelta(minutes=55)).isoformat()
    orphans = db.select("articles_v3", select=ASSIGN_SELECT + ",assignment", enrichment_status="eq.done", story_id="is.null",
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


def founding_article(members: list[dict]) -> dict | None:
    """The story's first actual report: the earliest non-wire member of a type that can start a story
    (news / live / explainer), else the earliest member with a summary. Deliberately NOT "the member whose
    assignment says create": a merge carries the absorbed story's creator along (e.g. a lawsuit report), and
    that must never become the anchor of the story it was merged into."""
    with_summary = [m for m in members if m.get("event_summary")]
    reports = [m for m in with_summary if m.get("article_type") in CREATES_STORIES and not m.get("is_wire_copy")]
    pool = reports or with_summary
    return min(pool, key=lambda m: m["published_at"]) if pool else None


def recheck_suspects(members: list[dict], story: dict, settings: Settings, anchor_id: str | None = None) -> list[dict]:
    """Members that the current rule would have sent to the verifier but that were never verified under it.
    Only the founding article is exempt: a story absorbed by a merge brings its own creator, which must be checked."""
    out = []
    for m in members:
        assignment = m.get("assignment") or {}
        if assignment.get("rule") == ASSIGN_RULE:
            continue
        if m["id"] == anchor_id or (anchor_id is None and assignment.get("method") == "create"):
            continue
        if needs_verification(m, story, settings) is None:
            continue
        out.append(m)
    return out


def run_recheck(db: Db, llm: LLM, settings: Settings, *, dry_run: bool = False, max_checks: int = 2000,
                report_path: str | None = None) -> dict:
    """Detach members whose main point is a follow-up development, so the next assign pass files them correctly.

    Only touches articles attached under an older rule and only those the current rule would have verified
    (published > auto_attach_window_hours after the story began, or with a later event date).
    Detached articles get story_id = NULL / assigned_at = NULL and are picked up by run_assign.
    """
    from concurrent.futures import ThreadPoolExecutor

    stats = {"stories": 0, "suspects": 0, "kept": 0, "detached": 0, "failed": 0}
    stats["reanchored"] = 0
    stories = db.select("stories", select="id,title,summary,title_source,event_date,first_seen,last_seen,article_count",
                        status="eq.open", article_count="gte.2")
    work: list[tuple[dict, dict]] = []
    reanchored: list[str] = []
    for story in stories:
        members = db.select("articles_v3", select=RECHECK_SELECT, story_id=f"eq.{story['id']}")
        founder = founding_article(members)
        # A summary rewritten from later pair articles can drift to include a follow-up ("...; media sue over
        # access"), which then makes the follow-up look like the same development. Re-anchor it on the founder.
        if founder and story.get("title_source") == "llm" and founder["event_summary"] != story.get("summary"):
            old_title = story["title"]
            story = {**story, "summary": founder["event_summary"], "title": placeholder_title(founder["event_summary"]),
                     "title_source": "auto", "_old_title": old_title}
            stats["reanchored"] += 1
            if len(reanchored) < 15:
                reanchored.append(f"  - {old_title[:70]}  ->  {story['title'][:70]}")
            if not dry_run:
                db.update("stories", {"id": f"eq.{story['id']}"}, {
                    "summary": story["summary"], "title": story["title"], "title_source": "auto",
                    "updated_at": datetime.now(timezone.utc).isoformat()})
        suspects = recheck_suspects(members, story, settings, anchor_id=founder["id"] if founder else None)
        if suspects:
            stats["stories"] += 1
            work.extend((m, story) for m in suspects)
    work = work[:max_checks]
    stats["suspects"] = len(work)
    print(f"[recheck] re-anchored {stats['reanchored']} story summaries on their founding article"
          f"{' (in memory only)' if dry_run else ''}; e.g. old title -> anchored placeholder:")
    if reanchored:
        print("\n".join(reanchored))
    print(f"[recheck] {stats['suspects']} articles in {stats['stories']} stories need re-verification under rule {ASSIGN_RULE}")

    def check(item: tuple[dict, dict]) -> tuple[dict, dict, dict | None, dict]:
        article, story = item
        chosen, out = verify_candidates(llm, settings, article, [story], keep_id=story["id"])
        return article, story, chosen, out

    touched: set[str] = set()
    samples: list[str] = []
    report_rows: list[list[str]] = []
    candidates_to_split: list[tuple[float, str]] = []  # every follow_up/unrelated answer, whatever its confidence
    with ThreadPoolExecutor(max_workers=settings.enrich_concurrency) as pool:
        for future in [pool.submit(check, item) for item in work]:
            try:
                article, story, chosen, out = future.result()
            except Exception as exc:
                stats["failed"] += 1
                print(f"[recheck] verify failed: {str(exc)[:160]}")
                continue
            relation = out.get("relation")
            try:
                confidence = float(out.get("confidence") or 0)
            except (TypeError, ValueError):
                confidence = 0.0
            report_rows.append([story["id"], story.get("_old_title") or story["title"], story["title"],
                                article["id"], article.get("outlet") or "", article.get("published_at") or "",
                                article.get("title") or "", relation or "", f"{confidence:.2f}",
                                "kept" if chosen is not None else "detached", (out.get("reason") or "").replace("\t", " ")])
            if relation in ("follow_up", "unrelated"):
                candidates_to_split.append((confidence, f"[{relation} {confidence:.2f}] [{story['title'][:50]}] -> "
                                                        f"{article.get('outlet')}: {article.get('title', '')[:80]}"))
            if chosen is not None:
                stats["kept"] += 1
                if not dry_run:
                    db.update("articles_v3", {"id": f"eq.{article['id']}"},
                              {"assignment": {**(article.get("assignment") or {}), "rule": ASSIGN_RULE, "rechecked": True}})
                continue
            stats["detached"] += 1
            if len(samples) < 40:
                samples.append(f"  - [{relation} {confidence:.2f}] [{story['title'][:55]}] -> {article.get('outlet')}: {article.get('title', '')[:85]}")
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
    if report_path:
        import csv
        with open(report_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["story_id", "story_title_before", "story_anchor", "article_id", "outlet", "published_at",
                        "headline", "relation", "confidence", "decision", "reason"])
            w.writerows(sorted(report_rows, key=lambda r: (r[1], r[5])))
        print(f"[recheck] wrote {len(report_rows)} decisions to {report_path} (verify model: {settings.verify_model})")
    if candidates_to_split:
        # How many would be split at other thresholds: pick SPLIT_MIN_CONFIDENCE from real answers, not a guess.
        print(f"[recheck] verifier said follow_up/unrelated for {len(candidates_to_split)} articles; "
              f"split at threshold (current = {settings.split_min_confidence}):")
        for t in (0.5, 0.6, 0.66, 0.7, 0.75, 0.8, 0.85, 0.9):
            n = sum(1 for c, _ in candidates_to_split if c >= t)
            print(f"    >= {t:.2f}: {n:>4}")
        border = sorted((x for x in candidates_to_split if 0.5 <= x[0] < 0.85), key=lambda x: x[0])
        if border:
            print("[recheck] borderline answers (0.50-0.84), lowest first:")
            print("\n".join("  - " + line for _, line in border[:40]))
    if samples:
        print("[recheck] detached (story -> article):" if not dry_run else "[recheck] would detach (story -> article):")
        print("\n".join(samples))
    return stats
