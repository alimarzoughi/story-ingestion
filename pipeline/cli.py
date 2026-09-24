"""Command-line entry point.

  python -m pipeline.cli verify-feeds [--source ID] [--out ../docs/source-audit.md]   (no DB/LLM needed)
  python -m pipeline.cli sync-sources
  python -m pipeline.cli ingest  [--source ID] [--dry-run] [--include-unverified]   (default: verified feeds only)
  python -m pipeline.cli enrich  [--limit N] [--dry-run]
  python -m pipeline.cli assign  [--limit N]
  python -m pipeline.cli pairs   [--lookback-hours N] [--story ID]
  python -m pipeline.cli merge                                             (fold duplicate stories)
  python -m pipeline.cli run     (sync-sources -> ingest -> enrich -> assign -> merge -> pairs)
  python -m pipeline.cli status

One-off maintenance:
  python -m pipeline.cli fix-text        [--dry-run]   decode HTML entities left in stored titles/snippets
  python -m pipeline.cli recheck-stories [--dry-run]   re-apply the follow-up rule to existing stories, then
                                                       re-assign detached articles, merge duplicates, rebuild pairs
"""
from __future__ import annotations

import argparse
import sys
import time

from .config import Settings, load_env
from .sources import load_sources


def _db(settings: Settings):
    from .db import Db
    return Db(settings.supabase_url, settings.supabase_key)


def _llm(settings: Settings, required: bool = True):
    from .llm import LLM
    if not settings.openai_api_key:
        if required:
            raise SystemExit("OPENAI_API_KEY is required for this command")
        print("[warn] OPENAI_API_KEY not set: verifier and title generation disabled")
        return None
    return LLM(settings)


def cmd_verify_feeds(args) -> int:
    settings = Settings.from_env(require_db=False)
    from .audit import run_audit
    rows = run_audit(settings, load_sources(args.sources), args.out, only=args.source)
    return 0 if any(r["status"] == "ok" for r in rows) else 1


def cmd_sync_sources(args) -> int:
    settings = Settings.from_env()
    from .sources import sync_sources
    n = sync_sources(_db(settings), load_sources(args.sources))
    print(f"[sources] synced {n} feeds")
    return 0


def cmd_ingest(args) -> int:
    settings = Settings.from_env(require_db=not args.dry_run)
    from .ingest import run_ingest
    db = None if args.dry_run else _db(settings)
    results = run_ingest(db, settings, load_sources(args.sources), dry_run=args.dry_run, only=args.source, include_unverified=args.include_unverified)
    if args.dry_run:
        import json
        for r in results:
            for s in r.get("sample", []):
                print("   ", json.dumps(s)[:220])
    total = sum(r["inserted"] for r in results)
    print(f"[ingest] total inserted={total}")
    return 0


def cmd_enrich(args) -> int:
    settings = Settings.from_env()
    from .enrich import run_enrich
    run_enrich(_db(settings), _llm(settings), settings, limit=args.limit, dry_run=args.dry_run)
    return 0


def _source_leaning(db) -> dict[str, str]:
    return {row["id"]: row["leaning"] for row in db.select("sources", select="id,leaning")}


def cmd_assign(args) -> int:
    settings = Settings.from_env()
    from .cluster import run_assign
    db = _db(settings)
    run_assign(db, _llm(settings, required=False), settings, _source_leaning(db), limit=args.limit)
    return 0


def cmd_pairs(args) -> int:
    settings = Settings.from_env()
    from .pairs import run_pairs
    run_pairs(_db(settings), _llm(settings, required=False), settings, lookback_hours=args.lookback_hours, story_id=args.story)
    return 0


_NIL_UUID = "00000000-0000-0000-0000-000000000000"


def require_patch_002(db) -> None:
    """Fail fast, before anything is written, if schema_patch_002.sql has not been applied in Supabase.

    Calls refresh_story_stats on a UUID that matches no story (a no-op). PostgREST answers 404 / PGRST202
    when the function does not exist.
    """
    try:
        db.rpc("refresh_story_stats", {"target": _NIL_UUID})
    except RuntimeError as exc:
        if "PGRST202" in str(exc) or "-> 404" in str(exc):
            raise SystemExit(
                "schema_patch_002.sql has not been applied: the database has no refresh_story_stats() function.\n"
                "Paste story-based-ingestion/schema_patch_002.sql into the Supabase SQL editor and run it, then retry."
            ) from exc
        raise


def cmd_merge(args) -> int:
    settings = Settings.from_env()
    from .merge import run_merge
    db = _db(settings)
    require_patch_002(db)
    run_merge(db, _llm(settings), settings)
    return 0


def cmd_fix_text(args) -> int:
    settings = Settings.from_env()
    from .maintenance import fix_text
    fix_text(_db(settings), dry_run=args.dry_run)
    return 0


def cmd_recheck(args) -> int:
    settings = Settings.from_env()
    from .cluster import run_assign, run_recheck
    from .merge import run_merge
    from .pairs import run_pairs
    db, llm = _db(settings), _llm(settings)
    if not args.dry_run:
        require_patch_002(db)  # never detach articles if the stats/merge functions are missing
    stats = run_recheck(db, llm, settings, dry_run=args.dry_run, report_path=args.report)
    if args.dry_run:
        print(f"[recheck] dry run finished | {llm.usage_summary()}")
        return 0
    # detached articles are unassigned again: file them under the current rule, then clean up.
    # Always run: an earlier interrupted recheck may have left detached articles waiting.
    run_assign(db, llm, settings, _source_leaning(db), limit=max(settings.assign_batch, stats["detached"] + 100))
    run_merge(db, llm, settings)
    run_pairs(db, llm, settings, lookback_hours=24 * 7)
    print(f"[recheck] finished | {llm.usage_summary()}")
    return 0


def cmd_run(args) -> int:
    settings = Settings.from_env()
    from .cluster import run_assign
    from .enrich import run_enrich
    from .ingest import run_ingest
    from .merge import run_merge
    from .pairs import run_pairs
    from .sources import sync_sources

    started = time.time()
    db = _db(settings)
    llm = _llm(settings)
    sources = load_sources(args.sources)
    sync_sources(db, sources)
    run_ingest(db, settings, sources)
    run_enrich(db, llm, settings)
    run_assign(db, llm, settings, _source_leaning(db))
    try:
        run_merge(db, llm, settings)
    except Exception as exc:  # a merge failure must not block the feed update
        print(f"[run] merge failed: {str(exc)[:200]}")
    run_pairs(db, llm, settings)
    try:
        trimmed = db.rpc("trim_old_bodies", {"older_than": "30 days"})
        if trimmed:
            print(f"[run] trimmed body_text on {trimmed} articles older than 30 days")
    except Exception as exc:
        print(f"[run] trim_old_bodies failed: {str(exc)[:120]}")
    print(f"[run] finished in {time.time() - started:.0f}s | {llm.usage_summary()}")
    return 0


def cmd_status(args) -> int:
    settings = Settings.from_env()
    db = _db(settings)
    print("articles_v3 total:      ", db.count("articles_v3"))
    for status in ("pending", "done", "failed", "skipped"):
        print(f"  enrichment {status:<8}", db.count("articles_v3", enrichment_status=f"eq.{status}"))
    print("  unassigned (done):   ", db.count("articles_v3", enrichment_status="eq.done", story_id="is.null", assigned_at="is.null"))
    print("stories open:           ", db.count("stories", status="eq.open"))
    print("stories with both sides:", db.count("stories", left_count="gte.1", right_count="gte.1"))
    print("current pairs:          ", db.count("story_pairs", is_current="eq.true"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", default="sources.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("verify-feeds"); p.add_argument("--source"); p.add_argument("--out", default="../docs/source-audit.md"); p.set_defaults(fn=cmd_verify_feeds)
    p = sub.add_parser("sync-sources"); p.set_defaults(fn=cmd_sync_sources)
    p = sub.add_parser("ingest"); p.add_argument("--source"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--include-unverified", action="store_true"); p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("enrich"); p.add_argument("--limit", type=int); p.add_argument("--dry-run", action="store_true"); p.set_defaults(fn=cmd_enrich)
    p = sub.add_parser("assign"); p.add_argument("--limit", type=int); p.set_defaults(fn=cmd_assign)
    p = sub.add_parser("pairs"); p.add_argument("--lookback-hours", type=int); p.add_argument("--story"); p.set_defaults(fn=cmd_pairs)
    p = sub.add_parser("merge"); p.set_defaults(fn=cmd_merge)
    p = sub.add_parser("fix-text"); p.add_argument("--dry-run", action="store_true"); p.set_defaults(fn=cmd_fix_text)
    p = sub.add_parser("recheck-stories"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--report", help="write every verifier decision to this UTF-8 TSV file"); p.set_defaults(fn=cmd_recheck)
    p = sub.add_parser("run"); p.set_defaults(fn=cmd_run)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
