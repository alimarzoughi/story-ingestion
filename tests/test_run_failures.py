"""Regressions for the 2026-09-26 (native crash in ingest) and 2026-09-27 (NUL byte) run failures."""
import threading
from concurrent.futures import ThreadPoolExecutor

from pipeline.config import Settings
from pipeline.db import Db, strip_nul
from pipeline.ingest import Fetcher


def test_strip_nul_nested():
    payload = {"body": "a\x00b", "evidence": {"k\x00": ["x\x00", 1, None]}, "n": 3}
    assert strip_nul(payload) == {"body": "ab", "evidence": {"k": ["x", 1, None]}, "n": 3}


def test_db_request_strips_nul(monkeypatch):
    db = Db("https://example.supabase.co", "sb_secret_x")
    sent = {}

    class Resp:
        status_code = 204
        content = b""

    def fake_request(method, url, **kw):
        sent.update(kw)
        return Resp()

    monkeypatch.setattr(db.session, "request", fake_request)
    db.update("articles_v3", {"id": "eq.1"}, {"event_summary": "x\x00y"})
    assert sent["json"] == {"event_summary": "xy"}


def test_fetcher_session_per_thread():
    settings = Settings.from_env(require_db=False)
    fetcher = Fetcher(settings)
    main = fetcher.session
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = set(pool.map(lambda _: (threading.get_ident(), id(fetcher.session)), range(16)))
    sessions = {sid for _, sid in ids}
    assert main is fetcher.session
    assert id(main) not in sessions
    assert len(sessions) == len({tid for tid, _ in ids})


class _RpcDb:
    def __init__(self, result=None, error=None):
        self.calls, self.result, self.error = [], result, error

    def rpc(self, name, payload):
        self.calls.append(name)
        if self.error:
            raise RuntimeError(self.error)
        return self.result


def test_refresh_feed_calls_rpc():
    from pipeline.cli import refresh_feed
    db = _RpcDb(result=412)
    assert refresh_feed(db) == 412
    assert db.calls == ["refresh_feed_stories"]


def test_refresh_feed_missing_patch_fails_loudly():
    import pytest
    from pipeline.cli import refresh_feed
    db = _RpcDb(error='POST rpc/refresh_feed_stories -> 404: {"code":"PGRST202"}')
    with pytest.raises(SystemExit, match="schema_patch_003"):
        refresh_feed(db)
