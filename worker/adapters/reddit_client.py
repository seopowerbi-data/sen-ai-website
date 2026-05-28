"""Reddit URL parsing helpers - v1 contexte-only mode.

Why we do NOT fetch Reddit directly :
  - Reddit closed their Data API for commercial use in 2023 (smoke confirmed
    May 2026 : Hetzner IP → HTTP 403 on the public *.json endpoint with any
    User-Agent ; the block is IP-based, not UA-based).
  - The official commercial tier starts at $12k/mo enterprise, free tier is
    "non-commercial use only" - we don't qualify.
  - The pattern from Sprint 7 #17 applies : sen-ai is the SaaS that does
    NOT scrape. We rely on data we already legitimately have - the LLM
    citation snippets stored in scan_llm_results.citations[].contexte.

This file therefore exposes only URL parsing helpers. The actual data
flow for Reddit threads lives in worker/handlers/audit_reddit_threads.py
which mines the existing LLM citations - it does not call this module's
network code (there isn't any).

If Reddit ever re-opens commercial Data API access, the OAuth-based
fetcher lives in git history (commit f21bc32). Restore from there, swap
the URL source in the handler, and the rest of the pipeline (regex
brand mentions, Haiku sentiment, leverage score) keeps working unchanged.
"""
from __future__ import annotations

import re


_REDDIT_URL_RE = re.compile(
    r"^https?://(?:www\.|old\.|new\.|np\.|m\.)?reddit\.com/(?:r/[^/]+/comments/[a-z0-9]+|comments/[a-z0-9]+)",
    re.IGNORECASE,
)

_SUBREDDIT_RE = re.compile(
    r"^https?://(?:www\.|old\.|new\.|np\.|m\.)?reddit\.com/r/([^/]+)/",
    re.IGNORECASE,
)


def is_reddit_url(url: str) -> bool:
    """Return True if the URL looks like a Reddit thread permalink."""
    if not url:
        return False
    return bool(_REDDIT_URL_RE.match(url.strip()))


def canonical_url(url: str) -> str:
    """Normalize a Reddit thread URL :
      - drop query string + fragment
      - rewrite host variants (old. / new. / np. / m. / no-prefix) to www.
      - strip trailing slash
    `.../comments/abc?ref=foo` and `old.reddit.com/.../comments/abc/`
    collapse into the same key.
    """
    url = url.split("#", 1)[0].split("?", 1)[0]
    url = re.sub(
        r"^https?://(?:www\.|old\.|new\.|np\.|m\.)?reddit\.com",
        "https://www.reddit.com",
        url,
        flags=re.IGNORECASE,
    )
    if url.endswith("/"):
        url = url[:-1]
    return url


def parse_subreddit(url: str) -> str | None:
    """Extract the subreddit name from a Reddit thread URL.
    Returns 'SkincareAddiction' for
    'https://www.reddit.com/r/SkincareAddiction/comments/abc/title' ;
    None for URLs without a /r/<name>/ segment (rare but exists - direct
    /comments/<id> permalinks)."""
    if not url:
        return None
    m = _SUBREDDIT_RE.match(url.strip())
    return m.group(1) if m else None


_SLUG_RE = re.compile(
    r"/comments/[a-z0-9]+/([a-z0-9_]+)",
    re.IGNORECASE,
)


def parse_title_slug(url: str) -> str:
    """Extract the post title slug from a Reddit URL and humanize it.

    Reddit URLs encode the thread title as an underscore-separated slug
    after the comment ID. Example :
      /r/EuroSkincare/comments/1ckumod/opinions_on_ducray_keracnyl_gel_moussant
      →  "opinions on ducray keracnyl gel moussant"

    This is critical for brand-mention detection in v1 contexte-only
    mode : the LLM citation snippets are often just `[Source: reddit.com]`
    with no body text, but the URL slug almost always contains the
    thread's title verbatim. Many threads name the brand directly in
    their title. Returns "" when the URL has no slug (direct permalink).
    """
    if not url:
        return ""
    m = _SLUG_RE.search(url)
    if not m:
        return ""
    return m.group(1).replace("_", " ").replace("-", " ").strip()
