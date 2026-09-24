"""Story specificity (main-point rule), duplicate merging, display filtering and text hygiene."""
from pipeline.cluster import ASSIGN_RULE, decide, needs_verification, recheck_suspects, run_assign, run_recheck
from pipeline.config import Settings
from pipeline.extract import extract_page
from pipeline.ingest import build_row
from pipeline.maintenance import fix_text
from pipeline.merge import dates_compatible, run_merge, survivor_of
from pipeline.pairs import best_pair, displayable, run_pairs
from pipeline.textclean import clean_text, is_non_article_title
from tests.test_flow_fakedb import FakeDb, article

SETTINGS = Settings(supabase_url="x", supabase_key="x", openai_api_key=None)
BAN = "Trump bans CNN, MS NOW and Politico from the White House"


class FakeLLM:
    """Answers verifier / merge calls with a rule; records every prompt."""

    def __init__(self, rule):
        self.rule, self.calls = rule, []

    def structured(self, *, kind, user, **kw):
        self.calls.append((kind, user))
        return self.rule(kind, user)


# ---------------------------------------------------------------- text hygiene
def test_clean_text_decodes_double_encoded_entities():
    assert clean_text("Trump Bans CNN from White House over &#8216;FAKE NEWS&#8217;") == "Trump Bans CNN from White House over ‘FAKE NEWS’"
    assert clean_text("Waltz: Trump &amp;#8216;spot on&amp;#8217;") == "Waltz: Trump ‘spot on’"
    assert clean_text("  a \n  b ") == "a b" and clean_text("") is None and clean_text(None) is None


def test_non_article_titles():
    assert is_non_article_title("9/18: CBS Evening News")
    assert is_non_article_title("9/19/26: Saturday Morning")
    assert is_non_article_title("Transcript: Rep. Maria Elvira Salazar on Face the Nation")
    assert not is_non_article_title("Trump says CNN, MS NOW, Politico banned from White House")
    assert not is_non_article_title("Live updates: Trump boots 3 media outlets from White House coverage")
    assert not is_non_article_title("5 takeaways from the 9/11 hearing")


def test_build_row_cleans_titles_and_drops_show_pages():
    html = "<html><head><title>x</title></head><body><article><p>" + ("Body text. " * 60) + "</p></article></body></html>"
    source = {"id": "bb", "outlet": "Breitbart"}
    entry = {"link": "https://b.com/a", "title": "Trump Bans CNN over &#8216;FAKE NEWS&#8217;", "published_at": None,
             "summary": None, "image_url": None, "author": None}
    page = extract_page(html, "https://b.com/a")
    page["title"] = "Trump Bans CNN over &#8216;FAKE NEWS&#8217;"
    page["published_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    row = build_row(source, entry, page, 1000)
    assert row["title"] == "Trump Bans CNN over ‘FAKE NEWS’" and row["headline"] is None
    page["title"] = "9/18: CBS Evening News"
    assert build_row(source, entry, page, 1000) == {"_non_article": True}


# ---------------------------------------------------------------- the main-point rule
def _art(**kw):
    return {"published_at": "2026-09-18T15:00:00+00:00", "event_date": "2026-09-18", "key_entities": ["Donald Trump", "CNN"],
            "article_type": "news", "is_wire_copy": False, **kw}


def _story(**kw):
    return {"id": "s1", "similarity": 0.95, "first_seen": "2026-09-18T10:00:00+00:00", "last_seen": "2026-09-18T14:00:00+00:00",
            "event_date": "2026-09-18", "key_entities": ["Donald Trump", "CNN"], "title": BAN, "summary": BAN, **kw}


def test_early_same_day_coverage_attaches_without_verifier():
    assert needs_verification(_art(), _story(), SETTINGS) is None
    assert decide(_art(), [_story()], SETTINGS)[0] == "attach"


def test_late_or_later_dated_articles_must_be_verified():
    late = _art(published_at="2026-09-21T15:00:00+00:00", event_date="2026-09-18")
    assert needs_verification(late, _story(), SETTINGS) == "late"
    followup = _art(event_date="2026-09-19")
    assert needs_verification(followup, _story(), SETTINGS) == "later_event_date"
    assert needs_verification(_art(), _story(first_seen=None), SETTINGS) == "no_first_seen"
    decision, _, scored = decide(late, [_story()], SETTINGS)
    assert decision == "verify" and scored[0]["verify_reason"] == "late"


def _ban_db():
    db = FakeDb()
    leaning = {"hp": "LEFT", "fox": "RIGHT", "dw": "RIGHT"}
    db.tables["articles_v3"] = db.insert("articles_v3", [
        article("HuffPost", "hp", BAN, [1, 0, 0], None, published="2026-09-18T10:00:00+00:00", event_date="2026-09-18"),
        article("Fox News", "fox", "Trump bars three outlets from White House", [1, 0.01, 0], None,
                published="2026-09-18T11:00:00+00:00", event_date="2026-09-18"),
    ])
    run_assign(db, None, SETTINGS, leaning)
    assert len(db.tables["stories"]) == 1
    return db, leaning


def test_followup_lawsuit_becomes_its_own_story_but_reactions_stay():
    db, leaning = _ban_db()
    # both arrive days later with near-identical embeddings (same people, same ban) -> both go to the verifier
    db.insert("articles_v3", [
        article("Daily Wire", "dw", "Waltz says Trump was spot on banning CNN", [1, 0.02, 0], None,
                published="2026-09-20T12:00:00+00:00", event_date="2026-09-18"),
        article("HuffPost", "hp", "CNN, MS NOW, Politico announce lawsuit challenging Trump ban", [1, 0.03, 0], None,
                published="2026-09-21T12:00:00+00:00", event_date="2026-09-21"),
    ])
    ban_id = db.tables["stories"][0]["id"]

    def rule(kind, user):
        headline = user.split("Headline: ")[1].split("\n")[0]
        return {"same_story_id": None if "lawsuit" in headline else ban_id, "confidence": 0.9, "reason": "test"}

    llm = FakeLLM(rule)
    stats = run_assign(db, llm, SETTINGS, leaning)
    assert stats["verified"] == 2 and stats["verifier_attached"] == 1 and stats["created"] == 1
    by_title = {a["title"]: a for a in db.tables["articles_v3"]}
    assert by_title["Waltz says Trump was spot on banning CNN"]["story_id"] == ban_id
    lawsuit = by_title["CNN, MS NOW, Politico announce lawsuit challenging Trump ban"]
    assert lawsuit["story_id"] != ban_id and lawsuit["assignment"]["rule"] == ASSIGN_RULE
    assert "MAIN POINT" in __import__("pipeline.cluster", fromlist=["x"]).VERIFY_SYSTEM


def test_recheck_detaches_followups_attached_under_the_old_rule():
    db, leaning = _ban_db()
    ban = db.tables["stories"][0]
    old_rule = {"method": "threshold"}  # attached before the main-point rule existed
    db.insert("articles_v3", [
        article("Daily Wire", "dw", "Waltz says Trump was spot on banning CNN", [1, 0.02, 0], None,
                published="2026-09-20T12:00:00+00:00", story_id=ban["id"], side="right", assignment=old_rule, assigned_at="t"),
        article("HuffPost", "hp", "Outlets sue Trump over White House ban", [1, 0.03, 0], None,
                published="2026-09-21T12:00:00+00:00", event_date="2026-09-21", story_id=ban["id"], side="left",
                assignment=old_rule, assigned_at="t"),
    ])
    members = [a for a in db.tables["articles_v3"] if a.get("story_id") == ban["id"]]
    assert {m["title"] for m in recheck_suspects(members, ban, SETTINGS)} == {
        "Waltz says Trump was spot on banning CNN", "Outlets sue Trump over White House ban"}

    llm = FakeLLM(lambda kind, user: {"same_story_id": None if "sue" in user.split("Headline: ")[1].split("\n")[0] else ban["id"],
                                      "confidence": 0.9, "reason": "t"})
    ban["article_count"] = 4
    dry = run_recheck(db, llm, SETTINGS, dry_run=True)
    assert dry["detached"] == 1 and all(a.get("story_id") == ban["id"] for a in members)  # dry run writes nothing

    stats = run_recheck(db, llm, SETTINGS)
    assert stats == {**stats, "kept": 1, "detached": 1}
    sued = next(a for a in db.tables["articles_v3"] if a["title"].startswith("Outlets sue"))
    assert sued["story_id"] is None and sued["assigned_at"] is None and sued["assignment"]["from_story"] == ban["id"]
    kept = next(a for a in db.tables["articles_v3"] if a["title"].startswith("Waltz"))
    assert kept["assignment"]["rule"] == ASSIGN_RULE
    assert ("refresh_story_stats", {"target": ban["id"]}) in db.rpc_calls
    assert ban["article_count"] == 3
    # a second recheck has nothing left to do
    assert run_recheck(db, llm, SETTINGS)["suspects"] == 0


# ---------------------------------------------------------------- merge
def _two_story_db(event_b="2026-09-18"):
    db = FakeDb()
    db.tables["stories"] = [
        {"id": "big", "title": BAN, "summary": BAN, "event_date": "2026-09-18", "first_seen": "2026-09-18T10:00:00+00:00",
         "last_seen": "2026-09-19T10:00:00+00:00", "article_count": 40, "status": "open"},
        {"id": "dup", "title": "President bans several outlets", "summary": "same", "event_date": event_b,
         "first_seen": "2026-09-18T11:00:00+00:00", "last_seen": "2026-09-19T10:00:00+00:00", "article_count": 12, "status": "open"},
    ]
    db.merge_candidates = [{"a_id": "dup", "b_id": "big", "similarity": 0.97}, {"a_id": "big", "b_id": "dup", "similarity": 0.97}]
    return db


def test_merge_folds_smaller_duplicate_into_larger():
    db = _two_story_db()
    llm = FakeLLM(lambda kind, user: {"same_development": True, "confidence": 0.9, "reason": "t"})
    stats = run_merge(db, llm, SETTINGS)
    assert stats["merged"] == 1 and stats["checked"] == 1  # the reversed duplicate row is not checked twice
    assert ("merge_stories", {"survivor": "big", "absorbed": "dup"}) in db.rpc_calls


def test_merge_never_joins_followups_or_runs_unverified():
    db = _two_story_db(event_b="2026-09-21")  # 3 days apart: a follow-up, not a duplicate
    llm = FakeLLM(lambda kind, user: {"same_development": True, "confidence": 0.9, "reason": "t"})
    assert run_merge(db, llm, SETTINGS)["rejected_date"] == 1 and not llm.calls
    db = _two_story_db()
    assert run_merge(db, FakeLLM(lambda k, u: {"same_development": False, "confidence": 0.9, "reason": "t"}), SETTINGS)["merged"] == 0
    assert run_merge(_two_story_db(), None, SETTINGS)["merged"] == 0
    assert dates_compatible({"event_date": None}, {"event_date": "2026-09-18"}, 1)
    assert survivor_of(db.tables["stories"][1], db.tables["stories"][0])[0]["id"] == "big"


# ---------------------------------------------------------------- display filtering in pairs
def _member(side, outlet, title, atype="news", **kw):
    return {"id": f"{side}-{outlet}-{title[:5]}", "side": side, "outlet": outlet, "title": title, "snippet": "s " + title,
            "image_url": "i", "published_at": "2026-09-18T12:00:00+00:00", "article_type": atype, "is_wire_copy": False,
            "stance": {"direction": None, "confidence": 0, "framing_axes": []}, **kw}


def test_show_pages_roundups_never_lead_and_live_blogs_are_last_resort():
    assert not displayable(_member("left", "CBS", "9/18: CBS Evening News"))
    assert not displayable(_member("left", "The Hill", "Trump bans journalists; how many Senate seats in play?", atype="roundup"))
    r = _member("right", "Fox News", "Trump bars three outlets")
    assert best_pair([_member("left", "CBS", "9/18: CBS Evening News"), r], SETTINGS, None) is None
    live = _member("left", "The Hill", "Live updates: press ban and more", atype="live")
    news = _member("left", "NPR", "Trump says he is banning three outlets")
    assert best_pair([live, news, r], SETTINGS, None)[0]["outlet"] == "NPR"


def test_stale_current_pair_is_retired_when_its_articles_leave():
    db = FakeDb()
    db.tables["stories"] = [{"id": "s", "title": BAN, "summary": BAN, "title_source": "llm", "left_count": 1, "right_count": 1,
                             "status": "open", "updated_at": "2099-01-01T00:00:00+00:00"}]
    db.tables["articles_v3"] = [
        {**_member("left", "HuffPost", BAN), "story_id": "s"},
        {**_member("right", "Fox News", "Trump bars three outlets"), "story_id": "s"},
    ]
    db.tables["story_pairs"] = [{"id": "p0", "story_id": "s", "left_article_id": "gone-1", "right_article_id": "gone-2",
                                 "divergence_score": 9.0, "featured_at": "2026-09-18T12:00:00+00:00", "is_current": True}]
    run_pairs(db, None, SETTINGS)
    current = [p for p in db.tables["story_pairs"] if p["is_current"]]
    assert len(current) == 1 and current[0]["id"] != "p0"


# ---------------------------------------------------------------- fix-text backfill
def test_fix_text_backfill_only_touches_rows_that_change():
    db = FakeDb()
    db.tables["articles_v3"] = [
        {"id": "a1", "title": "Trump Bans &#8216;Fake News&#8217;", "headline": None, "subheadline": "x &amp; y", "snippet": None},
        {"id": "a2", "title": "AT&T earnings beat", "headline": None, "subheadline": None, "snippet": None},
        {"id": "a3", "title": "Plain title", "headline": None, "subheadline": None, "snippet": None},
    ]
    db.tables["stories"] = [{"id": "s1", "title": "Barrasso: ban doesn&#8217;t violate Constitution", "summary": "ok"}]
    assert fix_text(db, dry_run=True)["articles_v3_fixed"] == 1 and db.tables["articles_v3"][0]["title"].startswith("Trump Bans &")
    stats = fix_text(db, page_size=2)
    assert stats == {"articles_v3_scanned": 3, "articles_v3_fixed": 1, "stories_scanned": 1, "stories_fixed": 1}
    assert db.tables["articles_v3"][0]["title"] == "Trump Bans ‘Fake News’" and db.tables["articles_v3"][0]["subheadline"] == "x & y"
    assert db.tables["articles_v3"][1]["title"] == "AT&T earnings beat"
    assert db.tables["stories"][0]["title"] == "Barrasso: ban doesn’t violate Constitution"


def test_placeholder_titles_do_not_stop_at_abbreviations():
    from pipeline.cluster import placeholder_title
    assert placeholder_title("Rep. Maria Elvira Salazar aired a Miami ad critiquing Trump. It ran Sunday.") == \
        "Rep. Maria Elvira Salazar aired a Miami ad critiquing Trump"
    assert placeholder_title("Indianapolis Colts vs. Kansas City Chiefs ended 24-20. Swift attended.").startswith("Indianapolis Colts vs. Kansas")
    assert placeholder_title("The U.S. and Denmark agreed to military rights in Greenland. Talks continue.").endswith("Greenland")
    assert len(placeholder_title("word " * 100)) <= 121
