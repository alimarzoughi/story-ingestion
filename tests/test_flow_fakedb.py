"""End-to-end smoke test of assign -> pairs against an in-memory fake of the PostgREST client (no network, no LLM)."""
import math
import uuid

from pipeline.cluster import run_assign
from pipeline.config import Settings
from pipeline.db import pg_to_vec, vec_to_pg
from pipeline.pairs import run_pairs

SETTINGS = Settings(supabase_url="x", supabase_key="x", openai_api_key=None)


class FakeDb:
    def __init__(self):
        self.tables = {"articles_v3": [], "stories": [], "story_pairs": [], "sources": []}

    @staticmethod
    def _match(row, key, cond):
        op, _, val = cond.partition(".")
        v = row.get(key)
        if op == "eq":
            return str(v) == val or (isinstance(v, bool) and str(v).lower() == val)
        if op == "is":
            return (v is None) if val == "null" else (v is not None)
        if op == "gte":
            return v is not None and str(v) >= val if not isinstance(v, (int, float)) else v >= float(val)
        raise NotImplementedError(op)

    def select(self, table, **params):
        rows = self.tables[table]
        for key, cond in params.items():
            if key in ("select", "order", "limit"):
                continue
            rows = [r for r in rows if self._match(r, key, cond)]
        if "order" in params:
            col, _, direction = params["order"].partition(".")
            rows = sorted(rows, key=lambda r: r.get(col) or "", reverse=direction == "desc")
        if "limit" in params:
            rows = rows[: int(params["limit"])]
        return [dict(r) for r in rows]

    def insert(self, table, rows, **kw):
        rows = rows if isinstance(rows, list) else [rows]
        out = []
        for row in rows:
            row = {"id": str(uuid.uuid4()), **row}
            self.tables[table].append(row)
            out.append(dict(row))
        return out

    def update(self, table, filters, payload, **kw):
        for row in self.tables[table]:
            if all(self._match(row, k, c) for k, c in filters.items()):
                row.update(payload)
        return []

    def rpc(self, name, payload):
        if name == "close_stale_stories":
            return 0
        q = pg_to_vec(payload["query"])
        out = []
        for s in self.tables["stories"]:
            if s.get("status", "open") != "open":
                continue
            c = pg_to_vec(s["centroid"])
            sim = sum(a * b for a, b in zip(q, c))
            out.append({"id": s["id"], "similarity": sim, "last_seen": s["last_seen"], "event_date": s.get("event_date"),
                        "key_entities": s.get("key_entities", []), "title": s["title"], "summary": s["summary"], "article_count": s["article_count"]})
        out.sort(key=lambda r: -r["similarity"])
        return out[: payload["k"]]


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def article(outlet, source_id, title, emb, direction, atype="news", published="2026-09-17T12:00:00+00:00", **kw):
    return {"source_id": source_id, "outlet": outlet, "title": title, "published_at": published, "event_summary": title,
            "event_date": "2026-09-17", "key_entities": ["U.S. Senate"], "category": "politics", "article_type": atype,
            "leaning_label": None, "leaning_conf": 0, "is_wire_copy": False, "embedding": vec_to_pg(unit(emb)),
            "enrichment_status": "done", "story_id": None, "assigned_at": None, "snippet": "snippet " + title, "image_url": "i",
            "stance": {"direction": direction, "confidence": 0.9, "framing_axes": []}, "loaded_terms": [], **kw}


def test_assign_then_pairs_end_to_end():
    db = FakeDb()
    db.tables["sources"] = [{"id": "hp", "leaning": "LEFT"}, {"id": "fox", "leaning": "RIGHT"}, {"id": "vox", "leaning": "LEFT"}]
    leaning = {"hp": "LEFT", "fox": "RIGHT", "vox": "LEFT"}
    # same story: two near-identical embeddings; different story: orthogonal
    db.tables["articles_v3"] = db.insert("articles_v3", [
        article("HuffPost", "hp", "Senate passes bill, families hurt", [1, 0.05, 0], "supports"),
        article("Fox News", "fox", "Senate passes costly bill", [1, 0.0, 0.05], "opposes", published="2026-09-17T13:00:00+00:00"),
        article("Vox", "vox", "Why the Senate bill matters", [1, 0.03, 0.02], None, atype="opinion", published="2026-09-17T14:00:00+00:00"),
        article("Fox News", "fox", "Hurricane hits Florida", [0, 1, 0], None),
        article("HuffPost", "hp", "Wire copy of Senate vote", [1, 0.02, 0.01], None, is_wire_copy=True),
    ])
    stats = run_assign(db, None, SETTINGS, leaning)
    assert stats["created"] == 2 and stats["attached"] == 3 and stats["orphaned"] == 0
    senate = [s for s in db.tables["stories"] if "Senate" in s["title"]][0]
    assert senate["article_count"] == 4 and senate["left_count"] == 3 and senate["right_count"] == 1
    assert abs(sum(x * x for x in pg_to_vec(senate["centroid"])) - 1) < 1e-6

    pstats = run_pairs(db, None, SETTINGS)
    assert pstats["created"] == 1 and pstats["stories"] == 1
    pair = db.tables["story_pairs"][0]
    left = next(a for a in db.tables["articles_v3"] if a["id"] == pair["left_article_id"])
    assert left["outlet"] == "HuffPost" and pair["pair_kind"] == "opposing_stance" and pair["is_current"]

    # second run: nothing changes, pair is kept
    assert run_pairs(db, None, SETTINGS)["kept"] == 1
