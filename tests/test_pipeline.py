from datetime import datetime, timedelta, timezone

import pytest

from pipeline.cluster import date_compatible, decide, jaccard, merged_centroid, score_candidate
from pipeline.config import Settings
from pipeline.enrich import ENRICH_SCHEMA, enrichment_to_patch
from pipeline.extract import extract_page, make_snippet, parse_datetime
from pipeline.ingest import build_row
from pipeline.pairs import best_pair, near_duplicate, score_pair
from pipeline.sources import load_sources, side_for
from pipeline.urls import canonicalize, url_hash


SETTINGS = Settings(supabase_url="x", supabase_key="x", openai_api_key=None)


# ---------------------------------------------------------------- urls
def test_canonicalize_strips_tracking_and_amp():
    a = canonicalize("https://www.foxnews.com/politics/some-story?utm_source=rss&utm_medium=feed#frag")
    b = canonicalize("https://foxnews.com/politics/some-story/amp/")
    assert a == b == "https://foxnews.com/politics/some-story"
    assert url_hash(a) == url_hash(b)


def test_canonicalize_keeps_meaningful_query():
    assert canonicalize("https://x.com/a?id=42&utm_campaign=z") == "https://x.com/a?id=42"


# ---------------------------------------------------------------- sources
def test_sources_yaml_loads_and_sides():
    sources = load_sources("sources.yaml")
    assert len(sources) >= 16
    assert side_for("LEAN_LEFT") == "left" and side_for("RIGHT") == "right" and side_for("CENTER") == "center"
    lefts = [s for s in sources if side_for(s["leaning"]) == "left" and s["enabled"]]
    rights = [s for s in sources if side_for(s["leaning"]) == "right" and s["enabled"]]
    assert len(lefts) >= 4 and len(rights) >= 4


# ---------------------------------------------------------------- extract
HTML = """<html><head><title>Senate passes bill | Outlet</title>
<meta property="og:image" content="https://cdn.x.com/i.jpg">
<meta property="article:published_time" content="2026-09-17T14:03:00Z">
<script type="application/ld+json">{"@type":"NewsArticle","headline":"Senate passes bill","datePublished":"2026-09-17T14:03:00+00:00","author":[{"@type":"Person","name":"Jane Doe"}],"articleSection":"Politics"}</script>
</head><body><article><h1>Senate passes bill</h1>
<p>WASHINGTON (AP) — The Senate voted 52-48 on Thursday to pass a spending bill that funds the government through December, sending it to the House.</p>
<p>Lawmakers from both parties said the measure was a compromise, though several members criticised provisions they said were added at the last minute without debate.</p>
<p>The House is expected to take up the measure next week, according to aides familiar with the schedule who spoke on condition of anonymity.</p>
</article></body></html>"""


def test_extract_page_fields():
    page = extract_page(HTML, "https://www.outlet.com/a")
    assert page["title"] == "Senate passes bill"
    assert page["published_at"] == datetime(2026, 9, 17, 14, 3, tzinfo=timezone.utc)
    assert page["authors"] == ["Jane Doe"]
    assert page["section"] == "Politics"
    assert page["image"] == "https://cdn.x.com/i.jpg"
    assert page["wire_hint"] is True
    assert "52-48" in page["text"]
    assert not page["text"].startswith("Senate passes bill")


def test_snippet_and_dates():
    assert make_snippet("Short.") is None
    s = make_snippet("First sentence is here and long enough. Second sentence follows it. " * 10)
    assert s and len(s) <= 520
    assert parse_datetime("Thu, 17 Sep 2026 10:00:00 GMT").hour == 10
    assert parse_datetime("2026-09-17T10:00:00Z").tzinfo is not None


def test_build_row_marks_short_bodies_skipped():
    source = {"id": "s", "outlet": "O"}
    entry = {"link": "https://www.o.com/x?utm_source=rss", "title": "Feed title", "published_at": None, "summary": None, "image_url": None, "author": None}
    page = extract_page(HTML, "https://www.o.com/x")
    row = build_row(source, entry, page, 1000)
    assert row["canonical_url"] == "https://o.com/x"
    assert row["headline"] == "Feed title"
    assert row["is_wire_copy"] is True
    assert row["enrichment_status"] == "pending"
    page["text"] = "tiny"
    assert build_row(source, entry, page, 1000)["enrichment_status"] == "skipped"


def test_build_row_stale_guard_prefers_the_fresher_of_page_and_feed():
    """A page-level date is often the ORIGINAL publication date of a republished story.
    It must not silently drop an entry the feed says is fresh (the CNN zero-insert bug)."""
    now = datetime.now(timezone.utc)
    source = {"id": "s", "outlet": "O"}
    page = extract_page(HTML, "https://www.o.com/x")

    # page date stale, feed date fresh -> keep, using the feed date
    page_old = {**page, "published_at": now - timedelta(days=30)}
    entry_fresh = {"link": "https://www.o.com/x", "title": "T", "published_at": now - timedelta(hours=2),
                   "summary": None, "image_url": None, "author": None}
    row = build_row(source, entry_fresh, page_old, 1000, max_age_hours=72)
    assert not row.get("_stale")
    assert row["evidence"]["date_source"] == "feed_override"
    assert row["published_at"] == (now - timedelta(hours=2)).isoformat()

    # both stale -> dropped
    entry_old = {**entry_fresh, "published_at": now - timedelta(days=20)}
    assert build_row(source, entry_old, page_old, 1000, max_age_hours=72)["_stale"] is True

    # both fresh -> page date wins (more precise than the feed's)
    page_fresh = {**page, "published_at": now - timedelta(hours=5)}
    row = build_row(source, entry_fresh, page_fresh, 1000, max_age_hours=72)
    assert row["evidence"]["date_source"] == "page"
    assert row["published_at"] == (now - timedelta(hours=5)).isoformat()


# ---------------------------------------------------------------- enrich schema
def test_enrich_schema_is_strict_compatible():
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or (isinstance(node.get("type"), list) and "object" in node["type"]):
                assert node.get("additionalProperties") is False
                assert set(node["required"]) == set(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(ENRICH_SCHEMA)


def test_enrichment_to_patch_sanitises():
    article = {"id": "a", "body_chars": 1200, "evidence": {"wire_hint": False, "section": "Politics"}}
    out = {"category": "politics", "article_type": "news", "event_summary": "The Senate passed a bill.", "event_date": "2026-13-40",
           "event_date_confidence": 2, "key_entities": ["U.S. Senate", " ", 5], "leaning": {"label": "CENTER", "confidence": 0.9},
           "stance": {"contested_question": None, "direction": "bogus", "confidence": 0.2, "framing_axes": ["cost"], "framing_summary": "x"},
           "loaded_terms": [], "is_wire_copy": False, "evidence": {"headline_terms": [], "date_strings": ["Thursday"]}}
    patch = enrichment_to_patch(article, out, SETTINGS, [0.1] * 3)
    assert patch["event_date"] is None
    assert patch["key_entities"] == ["U.S. Senate"]
    assert patch["stance"]["direction"] is None
    assert patch["evidence"]["event_date_confidence"] == 1.0
    assert patch["embedding"].startswith("[0.1")


def test_clean_entities_drops_outlet_names():
    from pipeline.enrich import clean_entities
    out = clean_entities(["María Elvira Salazar", "Fox News Digital", " Hispanic voters ", "Fox News Poll", "Donald Trump", "Donald Trump", "x"], "Fox News")
    assert out == ["María Elvira Salazar", "Hispanic voters", "Donald Trump"]


def test_llm_wire_flag_needs_evidence():
    article = {"id": "a", "body_chars": 1200, "evidence": {"wire_hint": False}}
    base = {"category": "politics", "article_type": "news", "event_summary": "x", "event_date": None, "event_date_confidence": 0,
            "key_entities": [], "leaning": {"label": "CENTER", "confidence": 0.5},
            "stance": {"contested_question": None, "direction": None, "confidence": 0, "framing_axes": [], "framing_summary": ""},
            "loaded_terms": [], "evidence": {"headline_terms": [], "date_strings": []}}
    no_ev = {**base, "is_wire_copy": True, "wire_copy_evidence": "quotes an Axios interview"}
    assert enrichment_to_patch(article, no_ev, SETTINGS, [0.1]).get("is_wire_copy") is False
    with_ev = {**base, "is_wire_copy": True, "wire_copy_evidence": "WASHINGTON (AP) —"}
    assert enrichment_to_patch(article, with_ev, SETTINGS, [0.1]).get("is_wire_copy") is True
    hint = {**base, "is_wire_copy": False, "wire_copy_evidence": None}
    assert enrichment_to_patch({**article, "evidence": {"wire_hint": True}}, hint, SETTINGS, [0.1]).get("is_wire_copy") is True


# ---------------------------------------------------------------- cluster
def _art(**kw):
    base = {"published_at": "2026-09-17T12:00:00+00:00", "event_date": "2026-09-17", "key_entities": ["U.S. Senate", "Chuck Schumer"], "article_type": "news", "is_wire_copy": False}
    return {**base, **kw}


def _cand(sim, **kw):
    base = {"id": "c1", "similarity": sim, "last_seen": "2026-09-17T10:00:00+00:00", "event_date": "2026-09-17", "key_entities": ["U.S. Senate"], "title": "t", "summary": "s"}
    return {**base, **kw}


def test_three_zone_decision():
    assert decide(_art(), [_cand(0.95)], SETTINGS)[0] == "attach"
    assert decide(_art(), [_cand(0.75)], SETTINGS)[0] == "verify"
    assert decide(_art(), [_cand(0.50)], SETTINGS)[0] == "create"
    assert decide(_art(), [], SETTINGS)[0] == "create"


def test_event_date_gap_rejects_candidate():
    cand = _cand(0.95, event_date="2026-09-10")
    assert not date_compatible(_art(), cand, 3)
    decision, chosen, scored = decide(_art(), [cand], SETTINGS)
    assert decision == "create" and scored[0]["rejected"] == "event_date_gap"


def test_scoring_components():
    score, detail = score_candidate(_art(), _cand(0.9))
    assert detail["entity_jaccard"] == 0.5
    assert 0.9 < score < 0.93
    assert jaccard([], ["x"]) == 0.0


def test_centroid_running_mean_is_normalised():
    c = merged_centroid([1.0, 0.0], 1, [0.0, 1.0])
    assert abs((c[0] ** 2 + c[1] ** 2) - 1.0) < 1e-9
    assert abs(c[0] - c[1]) < 1e-9


# ---------------------------------------------------------------- pairs
def _member(side, outlet, direction=None, conf=0.9, axes=(), atype="news", title="Title", **kw):
    return {"id": f"{side}-{outlet}", "side": side, "outlet": outlet, "title": title, "snippet": "snippet text " + outlet,
            "image_url": "i", "published_at": "2026-09-17T12:00:00+00:00", "article_type": atype, "is_wire_copy": False,
            "stance": {"direction": direction, "confidence": conf, "framing_axes": list(axes)}, **kw}


def test_opposing_stance_outranks_framing_only():
    l1 = _member("left", "HuffPost", "supports", axes=["harm to families"], title="Bill would hurt families, advocates say")
    l2 = _member("left", "Vox", None, axes=["harm to families"], title="What the spending bill means for families")
    r = _member("right", "Fox News", "opposes", axes=["cost to taxpayers"], title="GOP slams bill's price tag")
    choice = best_pair([l1, l2, r], SETTINGS, None)
    assert choice[0]["outlet"] == "HuffPost" and choice[3] == "opposing_stance"
    s, kind, _ = score_pair(l2, r)
    assert kind == "framing_contrast"


def test_wire_copies_and_duplicates_never_pair():
    l = _member("left", "HuffPost", title="Senate passes spending bill 52-48")
    r = _member("right", "Fox News", title="Senate passes spending bill 52-48")
    assert near_duplicate(l, r, 0.9)
    assert best_pair([l, r], SETTINGS, None) is None
    r2 = _member("right", "Fox News", title="GOP hails bill as win", is_wire_copy=True)
    assert best_pair([l, r2], SETTINGS, None) is None
    r3 = _member("right", "Fox News", title="GOP hails bill as win", snippet="different")
    assert best_pair([l, r3], SETTINGS, None) is not None


def test_news_vs_opinion_penalised():
    l = _member("left", "HuffPost", atype="news", title="A")
    r_news = _member("right", "Fox News", atype="news", title="B")
    r_op = _member("right", "Daily Caller", atype="opinion", title="C")
    assert score_pair(l, r_news)[0] > score_pair(l, r_op)[0]
