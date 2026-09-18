"""Stage 1: feeds -> articles_v3 (raw, enrichment_status = pending)."""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import requests

from .config import Settings
from .db import Db
from .extract import extract_page, make_snippet, parse_datetime
from .urls import canonicalize, url_hash


PAGE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}
FEED_HEADERS = {
    # some feed endpoints (HuffPost) answer 406 to a browser Accept header
    "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
    "Accept-Language": "en-US,en;q=0.9",
}


def _make_session(settings: Settings):
    """curl_cffi (Chrome TLS fingerprint) when available and enabled; plain requests otherwise.
    The TLS fingerprint is what gets past Akamai/Cloudflare bot checks (The Hill, Axios, Washington Times)."""
    impersonate = settings.fetch_impersonate
    if impersonate and impersonate != "none":
        try:
            from curl_cffi import requests as cffi_requests  # type: ignore

            session = cffi_requests.Session(impersonate=impersonate)
            session.headers.update({"Accept-Encoding": "gzip, deflate"})
            return session, True
        except Exception as exc:  # not installed / unsupported platform
            print(f"[fetch] curl_cffi unavailable ({exc}); falling back to requests")
    session = requests.Session()
    session.headers.update({"User-Agent": settings.user_agent, "Accept-Encoding": "gzip, deflate"})
    return session, False


class Fetcher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session, self.impersonating = _make_session(settings)

    def get(self, url: str, retries: int = 3, *, kind: str = "page"):
        headers = FEED_HEADERS if kind == "feed" else PAGE_HEADERS
        last: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self.session.get(url, headers=headers, timeout=self.settings.http_timeout, allow_redirects=True)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                return resp
            except RuntimeError as exc:
                last = exc
                break
            except Exception as exc:  # requests / curl_cffi transport errors
                last = exc
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"GET {url} failed: {last}")


def _entry_published(entry: Any) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        value = entry.get(key)
        if value:
            try:
                return datetime(*value[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                pass
    for key in ("published", "updated", "dc_date"):
        dt = parse_datetime(entry.get(key))
        if dt:
            return dt
    return None


def _entry_image(entry: Any) -> str | None:
    for key in ("media_content", "media_thumbnail"):
        items = entry.get(key) or []
        for item in items:
            url = item.get("url") if isinstance(item, dict) else None
            if url:
                return url
    for link in entry.get("links") or []:
        if str(link.get("type", "")).startswith("image/") and link.get("href"):
            return link["href"]
    enclosure = entry.get("enclosures") or []
    for item in enclosure:
        if str(item.get("type", "")).startswith("image/") and item.get("href"):
            return item["href"]
    return None


_XML_ILLEGAL_RE = re.compile(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_BARE_AMP_RE = re.compile(rb"&(?!(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);)")


def _sanitize_xml(raw: bytes) -> bytes:
    cleaned = _XML_ILLEGAL_RE.sub(b"", raw)
    return _BARE_AMP_RE.sub(b"&amp;", cleaned)


def parse_feed(fetcher: Fetcher, feed_url: str, alt_urls: list[str] | None = None) -> list[dict[str, Any]]:
    """Parse a feed; on failure try sanitised bytes, then each alternate URL. Sets parse_feed.used_url."""
    errors: list[str] = []
    for url in [feed_url, *(alt_urls or [])]:
        try:
            resp = fetcher.get(url, kind="feed")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            continue
        parsed = feedparser.parse(resp.content)
        if parsed.bozo and not parsed.entries:
            parsed = feedparser.parse(_sanitize_xml(resp.content))
        if parsed.entries:
            parse_feed.used_url = url  # type: ignore[attr-defined]
            return _entries_from(parsed)
        errors.append(f"{url}: {getattr(parsed, 'bozo_exception', None) or 'no entries'} (content-type {resp.headers.get('Content-Type', '?')[:40]})")
    raise RuntimeError("feed parse error: " + " | ".join(errors))


def _entries_from(parsed) -> list[dict[str, Any]]:
    entries = []
    for entry in parsed.entries:
        link = entry.get("link") or (entry.get("links") or [{}])[0].get("href")
        if not link:
            continue
        entries.append({
            "link": link,
            "title": (entry.get("title") or "").strip() or None,
            "summary": (entry.get("summary") or "").strip() or None,
            "published_at": _entry_published(entry),
            "image_url": _entry_image(entry),
            "author": (entry.get("author") or "").strip() or None,
        })
    return entries


def build_row(source: dict, entry: dict, page: dict, html_len: int, max_age_hours: int | None = None) -> dict[str, Any] | None:
    canonical = canonicalize(entry["link"])
    published = page.get("published_at") or entry.get("published_at")
    if not published:
        return None
    if max_age_hours and published < datetime.now(timezone.utc) - timedelta(hours=max_age_hours):
        return {"_stale": True, "published_at": published.isoformat()}
    title = page.get("title") or entry.get("title")
    if not title:
        return None
    text = page.get("text") or ""
    feed_title = entry.get("title")
    headline = feed_title if feed_title and feed_title.strip() != title.strip() else None
    authors = page.get("authors") or ([entry["author"]] if entry.get("author") else [])
    return {
        "source_id": source["id"],
        "outlet": source["outlet"],
        "url": entry["link"],
        "canonical_url": canonical,
        "url_hash": url_hash(canonical),
        "title": title[:500],
        "headline": headline[:500] if headline else None,
        "subheadline": (page.get("description") or entry.get("summary") or None),
        "body_text": text[:20000] or None,
        "body_chars": len(text),
        "snippet": make_snippet(text) or make_snippet(entry.get("summary") or ""),
        "image_url": page.get("image") or entry.get("image_url"),
        "authors": authors,
        "published_at": published.isoformat(),
        "enrichment_status": "pending" if len(text) >= 300 else "skipped",
        "evidence": {
            "section": page.get("section"),
            "wire_hint": page.get("wire_hint"),
            "jsonld_type": page.get("jsonld_type"),
            "html_bytes": html_len,
            "feed_published_at": entry["published_at"].isoformat() if entry.get("published_at") else None,
        },
        "is_wire_copy": bool(page.get("wire_hint")),
    }


def ingest_source(db: Db | None, fetcher: Fetcher, settings: Settings, source: dict, *, dry_run: bool = False) -> dict[str, Any]:
    stats = {"source": source["id"], "feed_entries": 0, "fresh": 0, "new": 0, "inserted": 0, "skipped_short": 0, "failed": 0, "errors": []}
    try:
        entries = parse_feed(fetcher, source["feed_url"], source.get("alt_feed_urls"))
    except Exception as exc:
        stats["errors"].append(f"feed: {exc}")
        return stats
    stats["feed_entries"] = len(entries)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.feed_max_age_hours)
    fresh = [e for e in entries if (e["published_at"] is None or e["published_at"] >= cutoff)]
    fresh = fresh[: source.get("max_per_run", 40)]
    stats["fresh"] = len(fresh)

    by_canonical: dict[str, dict] = {}
    for entry in fresh:
        by_canonical.setdefault(canonicalize(entry["link"]), entry)
    if db is not None and by_canonical:
        existing = db.select_in("articles_v3", "canonical_url", list(by_canonical), select="canonical_url")
        for row in existing:
            by_canonical.pop(row["canonical_url"], None)
    stats["new"] = len(by_canonical)
    if not by_canonical:
        return stats

    def work(entry: dict) -> dict | None:
        resp = fetcher.get(entry["link"])
        final_url = str(resp.url) if resp.url else entry["link"]
        page = extract_page(resp.text, final_url)
        # follow the final URL after redirects for canonical purposes
        entry = {**entry, "link": final_url}
        return build_row(source, entry, page, len(resp.content), max_age_hours=settings.feed_max_age_hours)

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=settings.fetch_concurrency) as pool:
        futures = {pool.submit(work, entry): entry for entry in by_canonical.values()}
        for future, entry in futures.items():
            try:
                row = future.result()
            except Exception as exc:
                stats["failed"] += 1
                stats["errors"].append(f"{entry['link']}: {exc}")
                continue
            if row is None:
                stats["failed"] += 1
                stats["errors"].append(f"{entry['link']}: missing title or date")
                continue
            if row.get("_stale"):
                stats["stale"] = stats.get("stale", 0) + 1
                continue
            if row["enrichment_status"] == "skipped":
                stats["skipped_short"] += 1
            rows.append(row)

    # dedupe by canonical inside the batch (redirects can collapse two links)
    unique: dict[str, dict] = {}
    for row in rows:
        unique.setdefault(row["canonical_url"], row)
    rows = list(unique.values())
    if dry_run or db is None:
        stats["inserted"] = len(rows)
        stats["sample"] = [{k: r[k] for k in ("title", "canonical_url", "published_at", "body_chars", "image_url")} for r in rows[:3]]
        return stats
    if rows:
        db.insert("articles_v3", rows, returning=False, on_conflict="canonical_url", ignore_duplicates=True)
        stats["inserted"] = len(rows)
    return stats


def run_ingest(db: Db | None, settings: Settings, sources: list[dict], *, dry_run: bool = False, only: str | None = None,
               include_unverified: bool = False) -> list[dict]:
    if db is not None:
        from .sources import sync_sources  # articles_v3.source_id is a FK: the registry must exist first
        sync_sources(db, sources)
    fetcher = Fetcher(settings)
    if fetcher.impersonating:
        print(f"[fetch] using curl_cffi impersonate={settings.fetch_impersonate}")
    results = []
    for source in sources:
        if not source.get("enabled", True):
            continue
        if not include_unverified and not source.get("verified") and not only:
            continue
        if only and source["id"] != only and source["outlet"].lower() != only.lower():
            continue
        started = time.time()
        stats = ingest_source(db, fetcher, settings, source, dry_run=dry_run)
        stats["seconds"] = round(time.time() - started, 1)
        results.append(stats)
        err = f" errors={len(stats['errors'])}" if stats["errors"] else ""
        print(f"[ingest] {source['id']:<32} feed={stats['feed_entries']:<3} fresh={stats['fresh']:<3} new={stats['new']:<3} inserted={stats['inserted']:<3} short={stats['skipped_short']:<2} failed={stats['failed']:<2} {stats['seconds']}s{err}")
        for e in stats["errors"][:3]:
            print(f"          ! {e[:160]}")
    return results
