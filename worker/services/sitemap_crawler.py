"""Sitemap discovery + parsing for Phase D sitemap-index.

Pure functions (no DB). Pulls the list of URLs for a brand's domain from
its sitemap.xml — with index recursion (one level), junk filter, and a
hard 5000-URL cap.

Day 1 deliverable. Day 2 will add page-meta fetch + body extraction;
Day 3 will compute embeddings.

The discovery chain tries, in order :
  1. https://{domain}/sitemap.xml
  2. https://{domain}/sitemap_index.xml
  3. <link rel="sitemap" href="..."> in https://{domain}/

If the located file is a <sitemapindex>, every <sitemap><loc> entry is
fetched (one level deep — we don't recurse infinitely to avoid bombs)
and the union of <urlset> URLs is returned.

The return shape is `list[tuple[url: str, lastmod: datetime | None]]`.
`lastmod` is parsed from the <lastmod> child of each <url> entry; ISO-8601
dates with or without time component are both accepted (UTC). Unparseable
or absent lastmod -> None.

Junk filter rules and the URL cap are documented inline in `is_junk_url`
and `MAX_URLS_PER_BRAND`. Both are conservative defaults chosen to keep
the per-brand corpus useful (product / category / article pages) without
ingesting WordPress noise (tag archives, author archives, feed endpoints,
attachment files).
"""

from __future__ import annotations

import gzip
import io
import logging
import re
from datetime import datetime
from xml.etree import ElementTree as ET

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Sitemap.org schema namespace. Most sitemaps declare it; a few legacy ones
# don't. We strip the namespace prefix on every element when parsing to
# tolerate both.
_SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"

# Identify ourselves. Same UA Day 2's fetch_brand_pages will use.
_USER_AGENT = "senai-bot/1.0 (+https://sen-ai.fr/about/bot)"

# Hard cap. PF brands top out ~1k URLs each; 5000 is comfortable headroom
# without letting a runaway news/event site flood the corpus.
MAX_URLS_PER_BRAND = 5000

# Per-request timeouts. Sitemap files are usually < 1MB but can be larger;
# the read timeout is generous so we don't kill legitimately slow CDNs.
_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)

# Junk URL patterns. Each entry is a compiled regex tested against the FULL
# URL (scheme + host + path + query). Matching = drop.
#
# Why each entry exists :
#   /tag/, /author/, /category/, /page/N, /search?  : WordPress / generic CMS
#       archive paginations — no canonical content, would dilute embeddings.
#   /feed/, /wp-json/, /xmlrpc       : machine endpoints, not user pages.
#   .jpg/.png/.gif/.svg/.webp/.pdf  : binary attachments, not pages.
#   /attachment, /wp-content/uploads : same — WP attachment permalinks.
#   #fragment                       : same-page anchors.
#   ?replytocom=, ?share=           : comment / share permalinks duplicate.
_JUNK_PATTERNS = [
    re.compile(r"/tag/", re.IGNORECASE),
    re.compile(r"/author/", re.IGNORECASE),
    re.compile(r"/category/", re.IGNORECASE),
    re.compile(r"/page/\d+/?(?:$|\?)", re.IGNORECASE),
    re.compile(r"/search/?(?:$|\?)", re.IGNORECASE),
    re.compile(r"/feed/?(?:$|\?)", re.IGNORECASE),
    re.compile(r"/wp-json", re.IGNORECASE),
    re.compile(r"/xmlrpc\.php", re.IGNORECASE),
    re.compile(r"/wp-content/uploads/", re.IGNORECASE),
    re.compile(r"/wp-admin/", re.IGNORECASE),
    re.compile(r"/attachment/", re.IGNORECASE),
    re.compile(r"\.(?:jpg|jpeg|png|gif|svg|webp|pdf|zip|mp4|mp3|css|js)(?:\?|$)", re.IGNORECASE),
    re.compile(r"#"),  # any URL with a fragment (anchor) is a duplicate
    re.compile(r"\?replytocom=", re.IGNORECASE),
    re.compile(r"\?share=", re.IGNORECASE),
]


def is_junk_url(url: str) -> bool:
    """Return True if the URL matches a junk pattern (-> should be dropped).

    Exposed as a public symbol so the smoke test can assert filter behavior
    independently of discover_sitemap_urls.
    """
    if not url:
        return True
    for pat in _JUNK_PATTERNS:
        if pat.search(url):
            return True
    return False


def _strip_ns(tag: str) -> str:
    """Strip XML namespace prefix from an element tag name."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _parse_lastmod(raw: str | None) -> datetime | None:
    """Parse a sitemap <lastmod> value to a naive UTC datetime.

    Accepts the two forms allowed by sitemaps.org :
      - 'YYYY-MM-DD'             (date only)
      - 'YYYY-MM-DDTHH:MM:SS...' (ISO-8601 with optional TZ)

    Returns None on parse failure. We strip timezone info and return a naive
    UTC-equivalent datetime so the column (TIMESTAMP WITHOUT TZ) gets a
    consistent representation.
    """
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        # Python 3.11+ handles 'Z' suffix natively in fromisoformat
        s_norm = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s_norm)
        if dt.tzinfo is not None:
            # Convert to UTC then strip tz so the column stays naive
            from datetime import timezone
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        pass
    # Date-only fallback (some sitemaps write 'YYYY/MM/DD')
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _fetch_bytes(client: httpx.Client, url: str) -> bytes | None:
    """GET a sitemap URL and return its raw bytes, decompressing if gzipped.

    Returns None on any HTTP error or non-2xx. Caller decides whether to
    keep trying other sitemap discovery paths.
    """
    try:
        resp = client.get(url, follow_redirects=True)
    except httpx.HTTPError as exc:
        logger.info(f"sitemap fetch failed for {url}: {exc}")
        return None
    if resp.status_code != 200:
        logger.info(f"sitemap fetch non-200 ({resp.status_code}) for {url}")
        return None
    content = resp.content
    # Honor explicit .xml.gz extension OR detect gzip magic header
    is_gzip = url.lower().endswith(".gz") or content[:2] == b"\x1f\x8b"
    if is_gzip:
        try:
            content = gzip.decompress(content)
        except OSError as exc:
            logger.warning(f"sitemap gzip decompress failed for {url}: {exc}")
            return None
    return content


def _parse_urlset_or_index(xml_bytes: bytes) -> tuple[str, list[tuple[str, str | None]]]:
    """Parse a sitemap XML blob to (kind, entries).

    kind = 'urlset' | 'sitemapindex' | 'unknown'.
    entries = list of (loc, lastmod_raw_string_or_None).

    Both <urlset><url><loc> and <sitemapindex><sitemap><loc> have the same
    shape under the sitemaps.org schema — we just disambiguate by the root
    element name and let the caller decide what to do with each.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.warning(f"sitemap XML parse failed: {exc}")
        return ("unknown", [])
    root_tag = _strip_ns(root.tag).lower()
    if root_tag not in ("urlset", "sitemapindex"):
        return ("unknown", [])
    entries: list[tuple[str, str | None]] = []
    for child in root:
        loc = None
        lastmod = None
        for sub in child:
            tag = _strip_ns(sub.tag).lower()
            text = (sub.text or "").strip()
            if tag == "loc" and text:
                loc = text
            elif tag == "lastmod" and text:
                lastmod = text
        if loc:
            entries.append((loc, lastmod))
    return (root_tag, entries)


def _try_html_link_rel_sitemap(client: httpx.Client, base_url: str) -> str | None:
    """Look for `<link rel="sitemap" href="...">` in the brand's homepage.

    Last-resort discovery path when /sitemap.xml and /sitemap_index.xml both
    404. Returns the absolute URL of the sitemap, or None.
    """
    try:
        resp = client.get(base_url, follow_redirects=True)
    except httpx.HTTPError as exc:
        logger.info(f"homepage fetch failed for {base_url}: {exc}")
        return None
    if resp.status_code != 200:
        return None
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:
        logger.info(f"homepage parse failed for {base_url}: {exc}")
        return None
    link = soup.find("link", rel=lambda v: v and "sitemap" in (v if isinstance(v, list) else [v]))
    if not link:
        return None
    href = link.get("href")
    if not href:
        return None
    # Resolve relative URLs against the homepage
    from urllib.parse import urljoin
    return urljoin(base_url, href)


def discover_sitemap_urls(domain: str) -> list[tuple[str, datetime | None]]:
    """Discover the list of canonical URLs from a brand's sitemap.

    Returns a list of (url, lastmod) tuples, post junk-filter and capped at
    MAX_URLS_PER_BRAND. lastmod is parsed to naive UTC datetime or None.

    Empty list = no sitemap found OR sitemap was unparseable OR all URLs
    were filtered. Caller logs and surfaces as `error` on the brand.

    `domain` is the bare host (e.g., 'eau-thermale-avene.fr'). We try
    https:// only — http:// would just redirect anyway on modern sites.
    """
    if not domain or not isinstance(domain, str):
        return []
    domain = domain.strip().lower()
    if not domain:
        return []
    # Strip any accidental scheme / path the user typed
    if "://" in domain:
        from urllib.parse import urlparse
        parsed = urlparse(domain if "://" in domain else f"https://{domain}")
        domain = parsed.netloc or parsed.path
    domain = domain.strip("/").strip()
    if not domain:
        return []

    base = f"https://{domain}"
    candidates = [f"{base}/sitemap.xml", f"{base}/sitemap_index.xml"]

    with httpx.Client(
        headers={"User-Agent": _USER_AGENT, "Accept": "application/xml, text/xml, */*"},
        timeout=_HTTP_TIMEOUT,
    ) as client:
        sitemap_xml: bytes | None = None
        sitemap_source: str | None = None
        for candidate in candidates:
            xml_bytes = _fetch_bytes(client, candidate)
            if xml_bytes:
                sitemap_xml = xml_bytes
                sitemap_source = candidate
                break

        # Fall back to <link rel="sitemap"> in the homepage
        if sitemap_xml is None:
            href = _try_html_link_rel_sitemap(client, f"{base}/")
            if href:
                xml_bytes = _fetch_bytes(client, href)
                if xml_bytes:
                    sitemap_xml = xml_bytes
                    sitemap_source = href

        if sitemap_xml is None:
            logger.info(f"No sitemap discovered for domain {domain}")
            return []

        kind, entries = _parse_urlset_or_index(sitemap_xml)
        logger.info(
            f"Sitemap discovered for {domain}: source={sitemap_source} "
            f"kind={kind} entries={len(entries)}"
        )

        if kind == "urlset":
            raw_pairs = entries
        elif kind == "sitemapindex":
            # Recurse one level. Cap total fetches at 50 sub-sitemaps to
            # bound runtime on pathological cases.
            raw_pairs = []
            for sub_loc, _sub_lastmod in entries[:50]:
                sub_xml = _fetch_bytes(client, sub_loc)
                if not sub_xml:
                    continue
                sub_kind, sub_entries = _parse_urlset_or_index(sub_xml)
                if sub_kind == "urlset":
                    raw_pairs.extend(sub_entries)
                # If a sub-sitemap is itself an index, we DON'T recurse a
                # second level — too pathological to be worth supporting in
                # v1. Log and skip.
                elif sub_kind == "sitemapindex":
                    logger.warning(
                        f"Nested sitemapindex skipped (v1 caps at one level): {sub_loc}"
                    )
                if len(raw_pairs) >= MAX_URLS_PER_BRAND * 2:
                    # Soft early-stop: we'll filter + cap below anyway
                    break
        else:
            logger.warning(f"Sitemap root element unrecognized for {domain}")
            return []

    # Junk filter + dedup (preserving first-seen lastmod)
    seen: dict[str, datetime | None] = {}
    dropped_junk = 0
    for url, lastmod_raw in raw_pairs:
        url = (url or "").strip()
        if not url:
            continue
        if is_junk_url(url):
            dropped_junk += 1
            continue
        if url in seen:
            continue
        seen[url] = _parse_lastmod(lastmod_raw)
        if len(seen) >= MAX_URLS_PER_BRAND:
            break

    logger.info(
        f"Sitemap filter for {domain}: kept={len(seen)} dropped_junk={dropped_junk} "
        f"raw={len(raw_pairs)} cap={MAX_URLS_PER_BRAND}"
    )
    return list(seen.items())
