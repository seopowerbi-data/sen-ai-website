"""Plain HTML fetcher for the Sprint 5 GEO audit.

Why a dedicated fetcher instead of reusing the sitemap-pipeline scraper :
  - The sitemap pipeline (handlers/fetch_brand_pages.py) is brand-scoped,
    embedded-vector oriented, and stores body_excerpt truncated. We need the
    full HTML so the Princeton heuristics can count <a> tags, <p> structure,
    statistics, quotes, etc. - truncated text loses most of that.
  - The sitemap pipeline isn't always populated for a brand (the user can
    audit a scan even if the sitemap crawl hasn't run yet).

This adapter is intentionally minimal :
  - httpx with a polite User-Agent + timeout
  - HTTP redirect following (up to 5)
  - Returns the full HTML string + final URL + status code

Failures are non-fatal : the caller gets {status, error} and persists those
so the UI can show "couldn't reach this page" without re-attempting on every
refresh.
"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = (
    "sen-ai/1.0 (+https://sen-ai.fr; contact@sen-ai.fr) "
    "GEO content audit (Princeton 7-pattern heuristic, KDD 24)"
)
TIMEOUT = 15.0
MAX_REDIRECTS = 5
# Hard cap on the HTML we accept. A 5 MB blog post is already 10x what we
# need ; bigger payloads are almost always tracking junk or product galleries.
MAX_BODY_BYTES = 5_000_000


def fetch_page(url: str) -> dict:
    """Fetch a page's HTML. Returns :
        {status: int|None, final_url: str, html: str|None, error: str|None}

    A non-200 status is NOT an error - the caller may still audit a 4xx if
    the body is meaningful (some CDNs return 403 with content). We only flip
    `error` for network failures and explicitly non-HTML responses.
    """
    out: dict = {"status": None, "final_url": url, "html": None, "error": None}
    try:
        with httpx.Client(
            timeout=TIMEOUT,
            follow_redirects=True,
            max_redirects=MAX_REDIRECTS,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "fr,en;q=0.9",
            },
        ) as c:
            r = c.get(url)
            out["status"] = r.status_code
            out["final_url"] = str(r.url)
            # Treat anti-bot / blocked responses as fetch errors so the UI can
            # surface "blocked by site" instead of a misleading 0-score audit
            # built from a Cloudflare "Just a moment..." challenge page.
            # 401/403/429 are the standard bot-block triplet. 503 with html
            # body is typically a Cloudflare interstitial.
            if r.status_code in (401, 403, 429, 503):
                out["error"] = f"blocked_http_{r.status_code}"
                return out
            ctype = (r.headers.get("Content-Type") or "").lower()
            if "html" not in ctype and "xml" not in ctype:
                out["error"] = f"non_html_content_type:{ctype[:80]}"
                return out
            body = r.content[:MAX_BODY_BYTES]
            # Use response.text on the trimmed body so the encoding heuristic
            # in httpx still applies (some pages mis-declare encoding).
            try:
                out["html"] = body.decode(r.encoding or "utf-8", errors="replace")
            except (LookupError, UnicodeDecodeError):
                out["html"] = body.decode("utf-8", errors="replace")
            return out
    except httpx.TimeoutException:
        out["error"] = "timeout"
    except httpx.HTTPError as e:
        out["error"] = f"http_error:{str(e)[:120]}"
    except Exception as e:  # noqa: BLE001
        out["error"] = f"exception:{type(e).__name__}:{str(e)[:120]}"
    return out
