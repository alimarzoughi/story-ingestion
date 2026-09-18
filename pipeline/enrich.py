"""Stage 2: one structured LLM call per article + embedding of the neutral event summary."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from .config import Settings
from .db import Db, vec_to_pg
from .llm import LLM

CATEGORIES = ["politics", "world", "economy", "business", "technology", "health", "science", "climate",
              "crime", "courts", "immigration", "education", "culture", "sports", "other"]
ARTICLE_TYPES = ["news", "analysis", "opinion", "roundup", "live", "explainer", "other"]
LEANINGS = ["LEFT", "CENTER", "RIGHT"]
DIRECTIONS = ["supports", "opposes", "critical_of_left", "critical_of_right", "neutral", "mixed"]
WIRE_EVIDENCE_RE = re.compile(r"\b(associated press|\(ap\)|\bap\b|reuters|afp|agence france|bloomberg)\b", re.I)

ENRICH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "article_type": {"type": "string", "enum": ARTICLE_TYPES},
        "event_summary": {"type": "string", "description": "1-2 neutral sentences (max 45 words) stating the concrete development: who did what. Mention place or date only when they are essential to identifying the event. No evaluative adjectives, no outlet voice, no filler like 'the incident occurred'."},
        "event_date": {"type": ["string", "null"], "description": "YYYY-MM-DD of the central development, or null if none is identifiable."},
        "event_date_confidence": {"type": "number"},
        "key_entities": {"type": "array", "items": {"type": "string"}, "description": "3-6 canonical full names of the people, organisations, places or works CENTRAL to the story, most central first (e.g. 'Donald Trump' not 'Trump'; 'Maria Elvira Salazar'). Exclude generic institutions and countries unless they are the actor (no 'United States', 'Congress', 'White House', 'DHS'), exclude news outlets, adjectives and pronouns."},
        "leaning": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "label": {"type": "string", "enum": LEANINGS},
                "confidence": {"type": "number"},
            },
            "required": ["label", "confidence"],
        },
        "stance": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "contested_question": {"type": ["string", "null"], "description": "The real-world question people disagree about that this piece bears on, phrased neutrally as a yes/no question about the world (e.g. 'Should Republicans break with Trump on immigration enforcement?'). NEVER a question about the article itself ('Does the article...'). Null if the piece is not about a contested matter."},
                "direction": {"type": ["string", "null"], "enum": DIRECTIONS + [None]},
                "confidence": {"type": "number"},
                "framing_axes": {"type": "array", "items": {"type": "string"}, "description": "2-4 short phrases naming what the piece foregrounds (e.g. 'cost to taxpayers', 'harm to families', 'legal risk')."},
                "framing_summary": {"type": "string", "description": "One sentence: what this article foregrounds and whom it blames or credits."},
            },
            "required": ["contested_question", "direction", "confidence", "framing_axes", "framing_summary"],
        },
        "loaded_terms": {"type": "array", "items": {"type": "string"}, "description": "Charged or evaluative words/phrases in the headline (verbatim), empty if none."},
        "is_wire_copy": {"type": "boolean", "description": "True ONLY if this text is a syndicated wire story (AP, Reuters, AFP, Bloomberg) republished by the outlet, as shown by the byline or dateline. Quoting or citing a wire service or another outlet does NOT make it a wire copy."},
        "wire_copy_evidence": {"type": ["string", "null"], "description": "If is_wire_copy is true, the verbatim byline or dateline that shows it (e.g. 'By Associated Press', 'WASHINGTON (AP) —'). Otherwise null."},
        "evidence": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "headline_terms": {"type": "array", "items": {"type": "string"}},
                "date_strings": {"type": "array", "items": {"type": "string"}, "description": "Verbatim date/time expressions found in the text that support event_date."},
            },
            "required": ["headline_terms", "date_strings"],
        },
    },
    "required": ["category", "article_type", "event_summary", "event_date", "event_date_confidence", "key_entities",
                 "leaning", "stance", "loaded_terms", "is_wire_copy", "wire_copy_evidence", "evidence"],
}

SYSTEM_PROMPT = """You are a careful news analyst producing structured metadata for a story-matching system.
Rules:
- event_summary must be neutral and outlet-agnostic: state the concrete development (who did what) so that two outlets covering the same development would produce near-identical summaries. Max 45 words. Include place/date only when essential to identify the event; never pad with 'the incident occurred in...'. No opinion, adjectives, or speculation. If the piece is opinion/analysis, summarise the underlying news development it reacts to, not the author's argument.
- event_date is the date of the central development, resolved from the publish timestamp when the text says 'Tuesday', 'yesterday', etc. Null if the piece has no single central development.
- key_entities: 3-6 canonical full names, most central first. No generic institutions or countries unless they are the actor, no outlets.
- leaning classifies the article's own framing (word choice, emphasis, who is blamed or credited), not the topic and not the outlet. Straight, balanced reporting is CENTER. Use confidence honestly; low confidence is fine.
- stance: contested_question is the real-world yes/no question the piece bears on (never a question about the article). direction is the piece's position on it; 'neutral' with low confidence is the right answer for straight reporting. 'critical_of_left'/'critical_of_right' are for pieces whose main move is attacking one side rather than arguing a position.
- is_wire_copy is about authorship, not sourcing: only true when the byline/dateline shows the text itself is AP/Reuters/AFP/Bloomberg copy.
- Never guess. Prefer null and low confidence over invention.
Return only JSON matching the schema."""


def build_user_prompt(article: dict, settings: Settings) -> str:
    body = (article.get("body_text") or "")[: settings.body_chars_for_llm]
    return (
        f"Outlet: {article.get('outlet')}\n"
        f"Published at (UTC): {article.get('published_at')}\n"
        f"Page title: {article.get('title')}\n"
        f"Feed headline: {article.get('headline') or 'n/a'}\n"
        f"Subheadline/description: {article.get('subheadline') or 'n/a'}\n"
        f"Section: {(article.get('evidence') or {}).get('section') or 'n/a'}\n"
        f"Authors: {', '.join(article.get('authors') or []) or 'n/a'}\n\n"
        f"Article text (truncated):\n{body}"
    )


def embedding_text(enriched: dict) -> str:
    entities = ", ".join(enriched.get("key_entities") or [])
    return f"{enriched.get('event_summary', '')} | Entities: {entities}"


def _clamp(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _valid_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def enrichment_to_patch(article: dict, out: dict, settings: Settings, embedding: list[float] | None) -> dict:
    stance = out.get("stance") or {}
    stance_patch = {
        "contested_question": stance.get("contested_question"),
        "direction": stance.get("direction") if stance.get("direction") in DIRECTIONS else None,
        "confidence": _clamp(stance.get("confidence")),
        "framing_axes": [s for s in (stance.get("framing_axes") or []) if isinstance(s, str)][:4],
        "framing_summary": stance.get("framing_summary"),
    }
    evidence = dict(article.get("evidence") or {})
    evidence.update({
        "headline_terms": (out.get("evidence") or {}).get("headline_terms") or [],
        "date_strings": (out.get("evidence") or {}).get("date_strings") or [],
        "event_date_confidence": _clamp(out.get("event_date_confidence")),
        "body_chars": article.get("body_chars"),
    })
    wire_hint = bool((article.get("evidence") or {}).get("wire_hint"))
    llm_wire = bool(out.get("is_wire_copy")) and bool(WIRE_EVIDENCE_RE.search(str(out.get("wire_copy_evidence") or "")))
    evidence["wire_copy_evidence"] = out.get("wire_copy_evidence")
    return {
        "enrichment_status": "done",
        "category": out.get("category") if out.get("category") in CATEGORIES else "other",
        "article_type": out.get("article_type") if out.get("article_type") in ARTICLE_TYPES else "other",
        "event_summary": (out.get("event_summary") or "").strip() or None,
        "event_date": _valid_date(out.get("event_date")),
        "key_entities": [e.strip() for e in (out.get("key_entities") or []) if isinstance(e, str) and e.strip()][:8],
        "leaning_label": (out.get("leaning") or {}).get("label") if (out.get("leaning") or {}).get("label") in LEANINGS else None,
        "leaning_conf": _clamp((out.get("leaning") or {}).get("confidence")),
        "stance": stance_patch,
        "loaded_terms": [t for t in (out.get("loaded_terms") or []) if isinstance(t, str)][:8],
        "is_wire_copy": llm_wire or wire_hint,
        "evidence": evidence,
        "enrichment": {"model": settings.enrich_model, "prompt_version": settings.prompt_version,
                       "at": datetime.now(timezone.utc).isoformat(), "raw": out},
        "embedding": vec_to_pg(embedding) if embedding else None,
    }


ARTICLE_SELECT = "id,outlet,title,headline,subheadline,body_text,body_chars,published_at,authors,evidence"


def run_enrich(db: Db, llm: LLM, settings: Settings, *, limit: int | None = None, dry_run: bool = False) -> dict:
    limit = limit or settings.enrich_batch
    pending = db.select("articles_v3", select=ARTICLE_SELECT, enrichment_status="eq.pending",
                        order="published_at.desc", limit=str(limit))
    stats = {"pending": len(pending), "done": 0, "failed": 0}
    if not pending:
        print("[enrich] nothing pending")
        return stats

    def work(article: dict) -> tuple[dict, dict | None, str | None]:
        try:
            out = llm.structured(kind="enrich", model=settings.enrich_model, system=SYSTEM_PROMPT,
                                 user=build_user_prompt(article, settings), schema=ENRICH_SCHEMA, schema_name="article_enrichment")
            if not out.get("event_summary"):
                return article, None, "empty event_summary"
            return article, out, None
        except Exception as exc:
            return article, None, str(exc)

    results: list[tuple[dict, dict | None, str | None]] = []
    with ThreadPoolExecutor(max_workers=settings.enrich_concurrency) as pool:
        results = list(pool.map(work, pending))

    ok = [(a, o) for a, o, err in results if o is not None]
    texts = [embedding_text(o) for _, o in ok]
    embeddings: list[list[float] | None] = [None] * len(ok)
    if texts:
        try:
            embeddings = list(llm.embed(texts))  # type: ignore[assignment]
        except Exception as exc:
            print(f"[enrich] embedding batch failed: {exc}")

    for (article, out), emb in zip(ok, embeddings):
        patch = enrichment_to_patch(article, out, settings, emb)
        if emb is None:
            patch["enrichment_status"] = "failed"
            patch["enrichment"]["error"] = "embedding failed"
        if dry_run:
            print(json.dumps({k: patch[k] for k in ("category", "article_type", "event_summary", "event_date", "key_entities", "leaning_label", "leaning_conf", "stance", "is_wire_copy")}, indent=1)[:1500])
            continue
        db.update("articles_v3", {"id": f"eq.{article['id']}"}, patch)
        stats["done"] += 1 if patch["enrichment_status"] == "done" else 0
        stats["failed"] += 1 if patch["enrichment_status"] != "done" else 0

    for article, out, err in results:
        if out is None:
            stats["failed"] += 1
            print(f"[enrich] failed {article['id']}: {err[:200] if err else 'unknown'}")
            if not dry_run:
                db.update("articles_v3", {"id": f"eq.{article['id']}"},
                          {"enrichment_status": "failed", "enrichment": {"error": (err or "")[:500], "model": settings.enrich_model,
                                                                             "at": datetime.now(timezone.utc).isoformat()}})
    print(f"[enrich] pending={stats['pending']} done={stats['done']} failed={stats['failed']} | {llm.usage_summary()}")
    return stats
