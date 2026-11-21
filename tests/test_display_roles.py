"""Nov 2026 Nvidia audit: main-point summaries, display roles, eligible-only lead pairs, merge guards, reenrich."""
import csv

import pytest

from pipeline.cluster import ASSIGN_RULE, Assigner, needs_verification, role_of, run_assign, verify_candidates
from pipeline.config import Settings
from pipeline.enrich import SYSTEM_PROMPT, enrichment_to_patch, plausible_event_date, run_enrich
from pipeline.maintenance import reenrich
from pipeline.merge import run_merge, sample_headlines
from pipeline.pairs import best_pair, lead_pool, run_pairs
from tests.test_flow_fakedb import FakeDb, article
from tests.test_story_rules import BAN, FakeLLM, _ban_db, _member, _story, _two_story_db, recheck, visible

SETTINGS = Settings(supabase_url="x", supabase_key="x", openai_api_key=None)
RESIGNS = "Nvidia CEO Jensen Huang resigns; Colette Kress named interim CEO"


# ---------------------------------------------------------------- enrichment
def test_event_dates_outside_the_articles_own_week_are_dropped():
    published = "2026-11-17T21:00:00+00:00"
    assert plausible_event_date("2026-11-17", published) == "2026-11-17"
    assert plausible_event_date("2026-11-12", published) == "2026-11-12"   # within a week: kept (prompt handles it)
    assert plausible_event_date("2025-05-01", published) is None          # recapped background (Computex photos)
    assert plausible_event_date("2026-12-10", published) is None          # scheduled hearing
    assert plausible_event_date("not a date", published) is None


def test_scraped_text_from_another_page_is_flagged_and_its_snippet_replaced():
    art = {"outlet": "The Washington Post", "published_at": "2026-11-13T22:02:00+00:00", "evidence": {},
           "subheadline": "An inspector general report also found insufficient oversight of his office."}
    out = {"event_summary": "Brian Monahan leaves his post as attending physician to Congress.", "event_date": "2026-11-13",
           "text_matches_headline": False, "key_entities": ["Brian Monahan"], "category": "politics", "article_type": "news"}
    patch = enrichment_to_patch(art, out, SETTINGS, [0.1])
    assert patch["evidence"]["text_matches_headline"] is False and patch["snippet"] == art["subheadline"]
    ok = enrichment_to_patch(art, {**out, "text_matches_headline": True}, SETTINGS, [0.1])
    assert ok["evidence"]["text_matches_headline"] is True and "snippet" not in ok
    assert "main point" in SYSTEM_PROMPT and "background" in SYSTEM_PROMPT


def test_enrichment_takes_the_oldest_pending_articles_first():
    db = FakeDb()
    db.tables["articles_v3"] = [{"id": f"a{i}", "enrichment_status": "pending", "published_at": f"2026-11-1{i}T00:00:00+00:00"}
                                for i in range(5)]
    seen = []

    class Llm:
        def structured(self, **kw):
            seen.append(kw["user"])
            raise RuntimeError("stop")

        def usage_summary(self):
            return ""

    run_enrich(db, Llm(), Settings(supabase_url="x", supabase_key="x", openai_api_key="k", enrich_batch=2, enrich_concurrency=1))
    assert len(seen) == 2 and "2026-11-10" in seen[0] and "2026-11-11" in seen[1]


# ---------------------------------------------------------------- verifier roles
def test_verifier_sees_the_card_headline_opening_and_story_contents():
    story = _story(id="s1", title=RESIGNS, summary="Huang resigns.")
    llm = FakeLLM(lambda kind, user: {"candidate_id": "s1", "relation": "same_development", "confidence": 0.9,
                                      "role": "related", "reason": "t"})
    art = {**_art_nv(), "title": "Page title", "headline": "Nvidia scandal hits Trump's political vulnerabilities",
           "snippet": "The scandal connects to voter concerns over AI."}
    chosen, _ = verify_candidates(llm, SETTINGS, art, [story], keep_id="s1",
                                  context={"s1": {"founder": "Jensen Huang resigns as Nvidia CEO", "recent": ["Who is Colette Kress?"]}})
    user = llm.calls[0][1]
    assert "Headline: Nvidia scandal hits Trump's political vulnerabilities" in user
    assert "Opening: The scandal connects" in user and "first report's headline: Jensen Huang resigns" in user
    assert "Who is Colette Kress?" in user and chosen["_role"] == "related"


def test_kept_only_by_same_by_default_is_hidden():
    story = _story(id="s1")
    unsure = FakeLLM(lambda kind, user: {"candidate_id": "s1", "relation": "follow_up", "confidence": 0.6,
                                         "role": "direct_report", "reason": "t"})
    chosen, _ = verify_candidates(unsure, SETTINGS, _art_nv(), [story], keep_id="s1")
    assert chosen["id"] == "s1" and chosen["_role"] == "related"
    bad_role = FakeLLM(lambda kind, user: {"candidate_id": "s1", "relation": "same_development", "confidence": 0.9,
                                           "role": "headline", "reason": "t"})
    assert verify_candidates(bad_role, SETTINGS, _art_nv(), [story], keep_id="s1")[0]["_role"] == "direct_report"


def _art_nv(**kw):
    return {"published_at": "2026-11-13T15:00:00+00:00", "event_date": "2026-11-13", "key_entities": ["Jensen Huang"],
            "article_type": "news", "is_wire_copy": False, "title": "t", "event_summary": "s", **kw}


def test_assign_records_a_role_for_every_filed_article():
    db, leaning = _ban_db()
    founder, early = sorted((a for a in db.tables["articles_v3"]), key=lambda a: a["published_at"])
    assert founder["assignment"]["role"] == early["assignment"]["role"] == "direct_report"  # create / no-check attach
    db.insert("articles_v3", [article("Daily Wire", "dw", "Waltz says Trump was spot on banning CNN", [1, 0.02, 0], None,
                                      published="2026-09-20T12:00:00+00:00", event_date="2026-09-18")])
    ban_id = db.tables["stories"][0]["id"]
    llm = FakeLLM(lambda kind, user: {"candidate_id": ban_id, "relation": "same_development", "confidence": 0.9,
                                      "role": "direct_reaction", "reason": "t"})
    run_assign(db, llm, SETTINGS, leaning)
    waltz = next(a for a in db.tables["articles_v3"] if a["outlet"] == "Daily Wire")
    assert waltz["story_id"] == ban_id and waltz["assignment"]["role"] == "direct_reaction"
    assert waltz["assignment"]["rule"] == ASSIGN_RULE and "latest headlines" in llm.calls[0][1]


def test_an_article_published_before_the_story_began_is_checked():
    art = _art_nv(published_at="2026-11-09T11:00:00+00:00", event_date=None)
    story = _story(first_seen="2026-11-10T17:00:00+00:00", event_date="2026-11-10")
    assert needs_verification(art, story, SETTINGS) == "late"


# ---------------------------------------------------------------- what can lead
def test_only_direct_reports_lead_when_both_sides_have_one():
    def m(side, outlet, title, role, **kw):
        return _member(side, outlet, title, assignment={"role": role}, **kw)
    members = [
        m("left", "The Washington Post", "Nvidia scandal hits Trump's political vulnerabilities", "related",
          stance={"direction": "opposes", "confidence": 0.9, "framing_axes": ["politics"]}),
        m("right", "Washington Free Beacon", "To Nvidia and Beyond: China's infiltration", "related",
          stance={"direction": "supports", "confidence": 0.9, "framing_axes": ["security"]}),
        m("left", "CBS News", "Nvidia CEO Jensen Huang resigns", "direct_report"),
        m("right", "New York Post", "Jensen Huang quits as Nvidia CEO", "direct_report"),
        m("right", "Townhall", "Trump wants Tim Cook to take over Nvidia", "direct_reaction"),
    ]
    left, right, *_ = best_pair(members, SETTINGS, None)
    assert {left["outlet"], right["outlet"]} == {"CBS News", "New York Post"}
    # no direct report on the right: a reaction may lead; related pieces never do
    no_report = [x for x in members if x["outlet"] != "New York Post"]
    assert best_pair(no_report, SETTINGS, None)[1]["outlet"] == "Townhall"
    assert lead_pool([x for x in members if role_of(x) == "related"]) == ([], [])


def test_a_current_pair_that_became_related_loses_its_slot_at_once():
    db, leaning = _ban_db()
    ban = db.tables["stories"][0]
    run_pairs(db, None, SETTINGS)
    pair = db.tables["story_pairs"][0]
    lead_ids = {pair["left_article_id"], pair["right_article_id"]}
    db.insert("articles_v3", [
        article("NPR", "hp", "Trump bars CNN, MS NOW and Politico", [1, 0.01, 0], None, story_id=ban["id"], side="left",
                published="2026-09-18T12:00:00+00:00", assignment={"role": "direct_report"}),
        article("Breitbart", "fox", "White House bans three outlets", [1, 0.01, 0], None, story_id=ban["id"], side="right",
                published="2026-09-18T12:30:00+00:00", assignment={"role": "direct_report"}),
    ])
    for a in db.tables["articles_v3"]:
        if a["id"] in lead_ids:
            a["assignment"] = {**a["assignment"], "role": "related"}
    ban["updated_at"] = "2099-01-01T00:00:00+00:00"
    stats = run_pairs(db, None, SETTINGS)
    current = [p for p in db.tables["story_pairs"] if p["is_current"]]
    assert stats["replaced"] == 1 and len(current) == 1
    assert {current[0]["left_article_id"], current[0]["right_article_id"]}.isdisjoint(lead_ids)


# ---------------------------------------------------------------- merge
def test_merge_never_folds_back_a_story_founded_as_a_follow_up():
    db = _two_story_db()
    db.tables["articles_v3"] = [{"id": "earnings", "story_id": "dup", "assignment": {
        "method": "create", "verifier": {"candidate_id": "big", "relation": "follow_up", "confidence": 0.83}}}]
    llm = FakeLLM(lambda kind, user: {"same_development": True, "confidence": 0.95, "reason": "t"})
    stats = run_merge(db, llm, SETTINGS)
    assert stats["merged"] == 0 and stats["rejected_split"] == 1 and not llm.calls


def test_merge_shows_the_whole_story_and_saves_its_verdict():
    db = _two_story_db()
    stats = run_merge(db, FakeLLM(lambda kind, user: {"same_development": True, "confidence": 0.9, "reason": "dup"}), SETTINGS)
    assert stats["merged"] == 1
    absorbed = next(s for s in db.tables["stories"] if s["status"] == "merged")
    assert absorbed["merge_check"]["verifier"]["reason"] == "dup" and absorbed["merge_check"]["survivor"]
    members = [{"id": f"m{i}", "published_at": f"2026-11-{10 + i}T00:00:00+00:00", "title": f"h{i}",
                "assignment": {"candidates": [{"id": "s", "sim": 0.9 - (0.1 if i == 4 else 0)}]}} for i in range(10)]
    picked = [m["id"] for m in sample_headlines("s", members)]
    assert picked[:3] == ["m0", "m1", "m2"] and picked[3:6] == ["m7", "m8", "m9"] and "m4" in picked


# ---------------------------------------------------------------- recheck + reenrich
def test_recheck_checks_every_member_and_records_roles(tmp_path):
    db, leaning = _ban_db()
    ban = db.tables["stories"][0]
    visible(db, ban["id"])
    for a in db.tables["articles_v3"]:  # filed under an older rule, including the early no-check attach
        a["assignment"] = {**a["assignment"], "rule": "2026-09-25d.no-return"}
    db.insert("articles_v3", [article("Vox", "hp", "Why the press ban matters", [1, 0.02, 0], None, atype="analysis",
                                      published="2026-09-18T13:00:00+00:00", story_id=ban["id"], side="left",
                                      assignment={"method": "threshold"}, assigned_at="t")])
    ban["article_count"] = 3

    def rule(kind, user):
        headline = user.split("Headline: ")[1].split("\n")[0]
        assert headline not in user.split("CANDIDATE STORIES")[1]  # a member never vouches for itself
        return {"candidate_id": ban["id"], "relation": "same_development", "confidence": 0.9, "reason": "t",
                "role": "related" if "matters" in headline else "direct_report"}

    path = tmp_path / "r.tsv"
    stats = recheck(db, FakeLLM(rule), report_path=str(path))
    assert stats["suspects"] == 2 and stats["roles"]["related"] == 1  # early attach checked too; founder exempt
    vox = next(a for a in db.tables["articles_v3"] if a["outlet"] == "Vox")
    assert vox["assignment"]["role"] == "related" and vox["assignment"]["rule"] == ASSIGN_RULE
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    assert {r["role"] for r in rows} == {"related", "direct_report"}


def test_reenrich_rewrites_old_summaries_reanchors_stories_and_never_wipes_a_failure():
    db, leaning = _ban_db()
    ban = db.tables["stories"][0]
    for a in db.tables["articles_v3"]:
        a.update(enrichment={"prompt_version": "2026-09-17.2"}, evidence={}, subheadline=None)
    founder = min(db.tables["articles_v3"], key=lambda a: a["published_at"])
    other = max(db.tables["articles_v3"], key=lambda a: a["published_at"])

    class Llm:
        def structured(self, *, kind, user, **kw):
            if "bars three outlets" in user:
                raise RuntimeError("api down")
            return {"event_summary": "Trump bans CNN, MS NOW and Politico from White House events.", "event_date": "2026-09-18",
                    "text_matches_headline": True, "key_entities": ["Donald Trump"], "category": "politics",
                    "article_type": "news", "stance": {"framing_axes": ["press"], "framing_summary": "f"}}

        def embed(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

        def usage_summary(self):
            return ""

    settings = Settings(supabase_url="x", supabase_key="x", openai_api_key="k", enrich_concurrency=1)
    stats = reenrich(db, Llm(), settings, days=100000, workers=1)
    assert stats["enriched"] == 1 and stats["failed"] == 1 and stats["stories_reanchored"] == 1
    assert founder["enrichment"]["prompt_version"] == settings.prompt_version
    assert ban["summary"] == "Trump bans CNN, MS NOW and Politico from White House events."
    assert other["enrichment_status"] == "done" and other["enrichment"]["prompt_version"] == "2026-09-17.2"
    assert ("refresh_story_stats", {"target": ban["id"]}) in db.rpc_calls


def test_run_and_merge_refuse_to_start_without_patch_006():
    from pipeline.cli import require_patch_006

    class NoPatchDb(FakeDb):
        def select(self, table, **params):
            if params.get("select") == "merge_check":
                raise RuntimeError('GET stories -> 400: {"code":"42703","message":"column stories.merge_check does not exist"}')
            return super().select(table, **params)

    with pytest.raises(SystemExit, match="schema_patch_006.sql"):
        require_patch_006(NoPatchDb())
    require_patch_006(FakeDb())


def test_select_all_pages_past_supabases_1000_row_cap():
    from pipeline.db import Db

    class Paged(Db):
        def __init__(self):
            self.calls = []

        def select(self, table, **params):
            self.calls.append(params)
            start = int(params["offset"])
            return [{"id": i} for i in range(start, min(start + int(params["limit"]), 2500))]

    db = Paged()
    assert len(db.select_all("stories", status="eq.open")) == 2500
    assert [c["offset"] for c in db.calls] == ["0", "1000", "2000"] and db.calls[0]["order"] == "id.asc"
