"""Text hygiene shared by ingest, pairs and the maintenance commands.

- clean_text: undo HTML entity encoding. Some feeds (Breitbart, The Federalist, The Hill) double-encode
  curly quotes, so a title arrives as "&#8216;FAKE NEWS&#8217;" even after feedparser decodes once.
- is_non_article_title: pages that are not articles and must never be shown as a side's headline
  (TV show episode pages, transcripts). Mirrored in schema_patch_002.sql (feed_stories view) --
  keep the two patterns in sync.
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
