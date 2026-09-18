"""Canonical URL normalisation for de-duplication."""
from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PARAM_PREFIXES = ("utm_", "mc_", "pk_", "piwik_", "hsa_", "vero_")
TRACKING_PARAMS = {
    "fbclid", "gclid", "dclid", "msclkid", "yclid", "twclid", "igshid", "mkt_tok",
    "ref", "ref_src", "referrer", "source", "src", "cmp", "cmpid", "campaign", "ito", "ncid",
    "intcmp", "ocid", "at_medium", "at_campaign", "at_custom1", "at_custom2", "at_custom3",
    "at_custom4", "at_bbc_team", "at_ptr_name", "at_link_origin", "at_link_type",
    "at_link_id", "at_format", "s", "sh", "smid", "smtyp", "partner", "output",
    "__twitter_impression", "cid", "_hsenc", "_hsmi", "guccounter", "guce_referrer",
    "guce_referrer_sig", "rss", "feed", "format",
}
AMP_SUFFIX_RE = re.compile(r"/amp/?$")


def canonicalize(url: str) -> str:
    url = url.strip()
    parts = urlsplit(url)
    scheme = "https"
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("amp."):
        host = host[4:]
    path = parts.path or "/"
    path = AMP_SUFFIX_RE.sub("/", path)
    path = re.sub(r"/{2,}", "/", path)
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    kept = []
    for key, value in parse_qsl(parts.query, keep_blank_values=False):
        lowered = key.lower()
        if lowered in TRACKING_PARAMS or lowered.startswith(TRACKING_PARAM_PREFIXES):
            continue
        kept.append((key, value))
    kept.sort()
    query = urlencode(kept)
    return urlunsplit((scheme, host, path, query, ""))


def url_hash(canonical_url: str) -> str:
    return hashlib.sha1(canonical_url.encode("utf-8")).hexdigest()
