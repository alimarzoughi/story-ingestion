"""Article page extraction: trafilatura for body/metadata, plus precise timestamps and wire-copy hints."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import trafilatura
from bs4 import BeautifulSoup

WIRE_DATELINE_RE = re.compile(r"^\s*(?:[A-Z][A-Za-z.'’\- ]{1,40},?\s*)?\((?:AP|Reuters|AFP|Bloomberg)\)\s*[—–-]", re.M)
WIRE_AUTHOR_RE = re.compile(r"\b(associated press|reuters|agence france-presse|afp|bloomberg news)\b", re.I)
WIRE_BODY_HINT_RE = re.compile(r"\b(?:AP|Reuters)\b.{0,40}\bcontributed\b|©\s*20\d\d\s*(?:The Associated Press|Reuters)", re.I)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iter_jsonld(soup: BeautifulSoup):
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text() or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, dict):
                yield item
                if "@graph" in item:
                    stack.append(item["@graph"])


def extract_page(html: str, url: str) -> dict[str, Any]:
    """Return a dict with title, description, text, image, authors, published_at (datetime|None),
    section, wire_hint, jsonld_type."""
    soup = BeautifulSoup(html, "html.parser")

    published_at = None
    authors: list[str] = []
    section = None
    image = None
    jsonld_type = None
    description = None
    headline = None

    for item in _iter_jsonld(soup):
        item_type = item.get("@type")
        types = item_type if isinstance(item_type, list) else [item_type]
        if not any(t in ("NewsArticle", "Article", "ReportageNewsArticle", "OpinionNewsArticle", "AnalysisNewsArticle") for t in types if t):
            continue
        jsonld_type = types[0]
        published_at = published_at or parse_datetime(item.get("datePublished"))
        headline = headline or item.get("headline")
        description = description or item.get("description")
        section = section or item.get("articleSection") if isinstance(item.get("articleSection"), str) else section
        raw_authors = item.get("author") or []
        if isinstance(raw_authors, dict):
            raw_authors = [raw_authors]
        for author in raw_authors:
            name = author.get("name") if isinstance(author, dict) else author
            if isinstance(name, str) and name.strip():
                authors.append(name.strip())
        img = item.get("image")
        if isinstance(img, dict):
            image = image or img.get("url")
        elif isinstance(img, list) and img:
            first = img[0]
            image = image or (first.get("url") if isinstance(first, dict) else first)
        elif isinstance(img, str):
            image = image or img
        if published_at:
            break

    def meta(*selectors: tuple[str, str]) -> str | None:
        for attr, value in selectors:
            node = soup.find("meta", attrs={attr: value})
            if node and node.get("content"):
                return node["content"].strip()
        return None

    published_at = published_at or parse_datetime(
        meta(("property", "article:published_time"), ("name", "article:published_time"),
             ("property", "og:published_time"), ("name", "pubdate"), ("name", "publish-date"),
             ("name", "date"), ("itemprop", "datePublished"), ("name", "parsely-pub-date"))
    )
    image = image or meta(("property", "og:image"), ("name", "twitter:image"))
    description = description or meta(("property", "og:description"), ("name", "description"))
    section = section or meta(("property", "article:section"), ("name", "section"), ("name", "parsely-section"))
    if not authors:
        meta_author = meta(("name", "author"), ("property", "article:author"), ("name", "parsely-author"))
        if meta_author and not meta_author.startswith("http"):
            authors = [a.strip() for a in re.split(r",|\band\b", meta_author) if a.strip()]

    doc = trafilatura.bare_extraction(
        html, url=url, with_metadata=True, include_comments=False, include_tables=False, favor_precision=True
    )
    if doc is None or len((doc.as_dict().get("text") or "")) < 300:
        doc = trafilatura.bare_extraction(
            html, url=url, with_metadata=True, include_comments=False, include_tables=False, favor_recall=True
        ) or doc
    text = ""
    title = headline
    if doc is not None:
        d = doc.as_dict()
        text = (d.get("text") or "").strip()
        title = title or d.get("title")
        description = description or d.get("description")
        image = image or d.get("image")
        if not authors and d.get("author"):
            authors = [a.strip() for a in str(d["author"]).split(";") if a.strip()]
        if not published_at and d.get("date"):
            published_at = parse_datetime(d["date"] + "T00:00:00+00:00")
    if len(text) < 300:
        fallback = _paragraph_fallback(soup)
        if len(fallback) > len(text):
            text = fallback
    if not title:
        og_title = meta(("property", "og:title"))
        title = og_title or (soup.title.get_text(strip=True) if soup.title else None)

    # trafilatura often repeats the title as the first line of text
    if title and text.startswith(title.strip()):
        text = text[len(title.strip()):].lstrip("\n ")
    text = re.sub(r"\n{3,}", "\n\n", text)

    wire_hint = bool(
        WIRE_DATELINE_RE.search(text[:400])
        or any(WIRE_AUTHOR_RE.search(a) for a in authors)
        or WIRE_BODY_HINT_RE.search(text[-600:])
    )

    return {
        "title": (title or "").strip() or None,
        "description": (description or "").strip() or None,
        "text": text,
        "image": image,
        "authors": authors[:6],
        "published_at": published_at,
        "section": section,
        "wire_hint": wire_hint,
        "jsonld_type": jsonld_type,
    }


def _paragraph_fallback(soup: BeautifulSoup) -> str:
    """Last resort: join <p> text from the most paragraph-dense container."""
    best, best_len = "", 0
    for container in soup.select("article, main, [itemprop=articleBody], .article-body, .entry-content, .article__content, .post-content, body"):
        paras = [p.get_text(" ", strip=True) for p in container.find_all("p")]
        paras = [p for p in paras if len(p) >= 40]
        joined = "\n\n".join(paras)
        if len(joined) > best_len:
            best, best_len = joined, len(joined)
        if best_len > 1500 and container.name != "body":
            break
    return best


def make_snippet(text: str, target: int = 400, max_chars: int = 520) -> str | None:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) < 40:
        return None
    sentences = re.findall(r".+?(?:[.!?]+(?:\s|$)|$)", text)
    out, length = [], 0
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        if out and length + len(s) + 1 > max_chars:
            break
        out.append(s)
        length += len(s) + 1
        if length >= target:
            break
    snippet = " ".join(out) or text[:max_chars]
    return snippet if len(snippet) <= max_chars else snippet[: max_chars - 3].rstrip() + "..."
