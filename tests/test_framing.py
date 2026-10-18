"""Framing retry / backfill (See Details always has 'what it puts first' + a framing line)."""
from pipeline.config import Settings
from pipeline.framing import backfill_framing, ensure_framing, merge_framing, needs_framing

SETTINGS = Settings(supabase_url="x", supabase_key="x", openai_api_key=None)


class FramingLLM:
    def __init__(self, answer=None, error=None):
        self.calls, self.answer, self.error = [], answer, error

    def structured(self, *, kind, user, **kw):
        self.calls.append(kind)
        if self.error:
            raise RuntimeError(self.error)
        return self.answer

    def usage_summary(self):
        return "usage"


def test_needs_framing():
    assert needs_framing(None)
    assert needs_framing({"framing_axes": [], "framing_summary": "x"})
    assert needs_framing({"framing_axes": ["a"], "framing_summary": "  "})
    assert needs_framing({"framing_axes": [" "], "framing_summary": "x"})
    assert not needs_framing({"framing_axes": ["cost"], "framing_summary": "Leads with cost."})


def test_merge_keeps_existing_and_fills_missing():
    merged = merge_framing({"framing_axes": ["cost"], "framing_summary": "", "direction": "neutral"}, ["x"], "New line.")
    assert merged == {"framing_axes": ["cost"], "framing_summary": "New line.", "direction": "neutral"}


def test_ensure_framing_retries_only_when_empty():
    llm = FramingLLM({"framing_axes": ["latest developments", " war timeline "], "framing_summary": " Leads with the newest moves. "})
    full = {"stance": {"framing_axes": ["a"], "framing_summary": "b"}}
    assert ensure_framing(llm, SETTINGS, {"id": 1}, full) is full and llm.calls == []
    out = ensure_framing(llm, SETTINGS, {"id": 1}, {"event_summary": "e", "stance": {"framing_axes": [], "framing_summary": ""}})
    assert out["stance"]["framing_axes"] == ["latest developments", "war timeline"]
    assert out["stance"]["framing_summary"] == "Leads with the newest moves."
    assert llm.calls == ["framing"]


def test_ensure_framing_never_fails_enrichment():
    out = {"event_summary": "e", "stance": {}}
    assert ensure_framing(FramingLLM(error="boom"), SETTINGS, {"id": 1}, out) is out


class BackfillDb:
    def __init__(self, rows):
        self.rows, self.updates = rows, []

    def select(self, table, **params):
        if "id" in params:
            rid = params["id"].removeprefix("eq.")
            return [dict(r, title="t", outlet="o", body_text="b") for r in self.rows if r["id"] == rid]
        offset, limit = int(params.get("offset", 0)), int(params["limit"])
        return self.rows[offset:offset + limit]

    def update(self, table, filters, payload):
        self.updates.append((filters["id"], payload))


def test_backfill_fills_only_missing():
    rows = [{"id": "a", "stance": {"framing_axes": ["x"], "framing_summary": "y"}},
            {"id": "b", "stance": {"framing_axes": [], "framing_summary": None, "direction": "neutral"}},
            {"id": "c", "stance": None}]
    db = BackfillDb(rows)
    llm = FramingLLM({"framing_axes": ["p"], "framing_summary": "q"})
    assert backfill_framing(db, llm, SETTINGS, dry_run=True)["missing"] == 2 and not db.updates
    stats = backfill_framing(db, llm, SETTINGS)
    assert stats == {"missing": 2, "filled": 2, "failed": 0}
    assert dict(db.updates)["eq.b"] == {"stance": {"framing_axes": ["p"], "framing_summary": "q", "direction": "neutral"}}
