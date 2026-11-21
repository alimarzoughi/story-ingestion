"""Text hygiene shared by ingest, pairs and the maintenance commands.

- clean_text: undo HTML entity encoding. Some feeds (Breitbart, The Federalist, The Hill) double-encode
  curly quotes, so a title arrives as "&#8216;FAKE NEWS&#8217;" even after feedparser decodes once.
- is_non_article_title: pages that are not articles and must never be shown as a side's headline
  (TV show episode pages, transcripts). Mirrored in schema_patch_002.sql (feed_stories view) --
  keep the two patterns in sync.
- body_matches_page: catches a scraped body that belongs to another article or is site navigation (enrich).
"""
from __future__ import annotations

import html
import re

_WS_RE = re.compile(r"\s+")

# "9/18: CBS Evening News", "9/19/26: Saturday Morning"
_SHOW_EPISODE_RE = re.compile(r"^\s*\d{1,2}/\d{1,2}(?:/\d{2,4})?\s*:")
# "Transcript: Rep. ... on Face the Nation", "Full transcript of ..."
_TRANSCRIPT_RE = re.compile(r"^\s*(?:full\s+)?transcripts?\b", re.I)


def clean_text(value: str | None) -> str | None:
    """Decode HTML entities (repeatedly, for double-encoded feeds) and collapse whitespace."""
    if value is None:
        return None
    text = value
    for _ in range(3):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    text = _WS_RE.sub(" ", text.replace(" ", " ")).strip()
    return text or None


def is_non_article_title(title: str | None) -> bool:
    if not title:
        return False
    return bool(_SHOW_EPISODE_RE.match(title) or _TRANSCRIPT_RE.match(title))


_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "this that with from after over into amid have will what when about their they them says said could would than "
    "more just your been were being which while where there these those here also only some most much very make makes "
    "made takes take news report reports".split())


def _keys(text: str | None) -> set[str]:
    """Content words cut to 5 letters, so 'resigns' matches 'resignation' and 'Nvidia's' matches 'Nvidia'."""
    words = _WORD_RE.findall((text or "").lower().replace("\u2019", "").replace("'", ""))
    return {w[:5] for w in words if len(w) >= 4 and w not in _STOPWORDS}


def body_matches_page(title: str | None, text: str | None, description: str | None = None) -> bool:
    """Is the scraped body about this page? On paywalled pages (Washington Post) trafilatura can return another
    article's text from a "latest news" module, and CBS video pages return site navigation. The body counts as this
    page's when its opening shares a quarter of the headline's words or, for punny headlines, of the description's.
    Tested on 6,634 articles (2026-11-21): 107 flagged, mostly those two outlets. (Asking gpt-5-nano the same
    question flagged 42% of articles, nearly all wrongly.)"""
    head = _keys(title)
    if len(head) < 3 or not text:
        return True  # too little to judge
    body = _keys(text[:1500])
    if len(head & body) / len(head) >= 0.25:
        return True
    desc = _keys(description)
    return len(desc) >= 3 and len(desc & body) / len(desc) >= 0.25
