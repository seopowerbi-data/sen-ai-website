"""Sitemap discovery + page-meta fetch for Phase D sitemap-index.

Pure functions (no DB). Two responsibilities :

  1. Discovery (Day 1) : pull the list of URLs for a brand's domain from
     its sitemap.xml — with index recursion (one level), junk filter, and
     a hard 5000-URL cap. Entry point : `discover_sitemap_urls`.

  2. Per-page fetch (Day 2) : fetch one page, extract title / meta /
     h1 / body_excerpt / canonical / lang, detect soft-404s, and report
     redirects. Entry point : `fetch_page_meta`. Robots.txt allow-check
     via `is_robots_allowed` (stdlib urllib.robotparser — RFC 9309
     compliant, zero deps; deviates from the plan's `reppy` choice to
     keep the dep surface flat).

Day 3 will add embeddings + inlinks.

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


# ─────────────────────────────────────────────────────────────────────────
# Day 2 : per-page fetch + meta extraction
# ─────────────────────────────────────────────────────────────────────────

# Soft-404 detection : these substrings in <title> indicate the server
# returned 200 OK but the page is actually a "page not found" placeholder.
# Conservative — keyed to short titles only (`is_soft_404` enforces the
# < 60 char ceiling per plan) to avoid catching legitimate articles that
# happen to mention "not found" in their title.
_SOFT_404_TITLE_PATTERNS = re.compile(
    r"\b(?:404|not\s*found|introuvable|page\s*non\s*trouv(?:é|e)e|"
    r"page\s*not\s*found|page\s*indisponible)\b",
    re.IGNORECASE,
)

# Per-host robots.txt cache. RobotFileParser is cheap to keep around and
# `is_robots_allowed` calls it once per page-fetch. Cleared when the worker
# process restarts — which is fine, robots.txt rarely changes within a run.
_ROBOTS_CACHE: dict[str, "object"] = {}


def is_soft_404(title: str | None, body_excerpt: str | None) -> bool:
    """Heuristic : did a 200-OK page actually deliver a "page not found" ?

    True when the title is short (< 60 chars, per plan) AND matches one of
    the soft-404 patterns. Conservative — we'd rather miss a soft-404 than
    drop a real page.
    """
    if not title:
        return False
    t = title.strip()
    if len(t) >= 60:
        return False
    return bool(_SOFT_404_TITLE_PATTERNS.search(t))


def extract_body_excerpt(html: str, max_words: int = 300) -> str:
    """Pull a 300-word excerpt from a page's primary content region.

    Selector priority : <main> > <article> > <body>. Inside the chosen
    container, we strip the structural noise tags (<nav>, <footer>,
    <aside>, <header>, <script>, <style>, <noscript>, <form>, <iframe>)
    then collapse whitespace and cap to `max_words` words.

    Returns "" on parse failure or genuinely empty content — caller can
    flag the row and Day 3 will fall back to title+meta+h1 for embedding.
    """
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as exc:
        logger.info(f"body extraction soup failed: {exc}")
        return ""
    container = soup.find("main") or soup.find("article") or soup.body
    if container is None:
        return ""
    # Drop structural noise. decompose() removes from the tree entirely so
    # subsequent get_text() doesn't return their content.
    for tag in container(["nav", "footer", "aside", "header", "script", "style",
                          "noscript", "form", "iframe"]):
        tag.decompose()
    text = container.get_text(separator=" ", strip=True)
    # Collapse whitespace runs (newlines, tabs, multiple spaces -> single space)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    words = text.split(" ")
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words)


def _absolutize(base_url: str, href: str | None) -> str | None:
    if not href:
        return None
    from urllib.parse import urljoin
    return urljoin(base_url, href.strip())


def is_robots_allowed(url: str, user_agent: str = _USER_AGENT) -> bool:
    """Is the User-Agent allowed to fetch this URL per the host's robots.txt ?

    Uses stdlib `urllib.robotparser` (RFC 9309). Per-host parsers cached
    process-wide. Safe default = True when robots.txt is missing, 4xx, or
    unparseable — the conservative-allow posture matches what most
    well-behaved crawlers do and means a broken robots.txt can't lock us
    out of a brand we paid to crawl.
    """
    from urllib.parse import urlparse
    from urllib.robotparser import RobotFileParser

    if not url:
        return True
    try:
        parsed = urlparse(url)
    except ValueError:
        return True
    host = (parsed.hostname or "").lower()
    if not host:
        return True
    if host in _ROBOTS_CACHE:
        rp = _ROBOTS_CACHE[host]
        if rp is None:
            return True
        return rp.can_fetch(user_agent, url)

    robots_url = f"{parsed.scheme or 'https'}://{host}/robots.txt"
    try:
        with httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=_HTTP_TIMEOUT,
            follow_redirects=True,
        ) as client:
            resp = client.get(robots_url)
    except httpx.HTTPError as exc:
        logger.info(f"robots.txt fetch failed for {host}: {exc} — allowing")
        _ROBOTS_CACHE[host] = None
        return True

    if resp.status_code >= 400:
        # 404 robots.txt = no restrictions. Same posture for any 4xx.
        logger.info(f"robots.txt for {host} returned {resp.status_code} — allowing")
        _ROBOTS_CACHE[host] = None
        return True

    rp = RobotFileParser()
    try:
        rp.parse(resp.text.splitlines())
    except Exception as exc:
        logger.info(f"robots.txt parse failed for {host}: {exc} — allowing")
        _ROBOTS_CACHE[host] = None
        return True

    _ROBOTS_CACHE[host] = rp
    return rp.can_fetch(user_agent, url)


def _extract_internal_links(soup: "BeautifulSoup", base_url: str, brand_host: str) -> list[str]:
    """Return same-host absolute URLs from every <a href> on the page.

    Used by Day 3's `compute_inlinks_from_map` to count internal inbound
    links per page (an authority signal for the matcher's scoring).

    Filtering rules :
      - skip empty href, mailto:, tel:, javascript:, #fragment-only
      - resolve relative paths against `base_url`
      - keep only links whose host matches `brand_host` (with optional
        www. prefix on either side) — external links go to other domains
        and aren't part of this brand's index
      - dedup by canonical URL (case-insensitive host, fragment stripped)
      - cap at 500 unique edges per page to bound memory on link-farm
        pages
    """
    if not soup or not base_url or not brand_host:
        return []
    from urllib.parse import urljoin, urlparse

    seen: set[str] = set()
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        low = href.lower()
        if low.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        try:
            abs_url = urljoin(base_url, href)
            parsed = urlparse(abs_url)
        except ValueError:
            continue
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host != brand_host:
            continue
        # Strip fragment, lowercase host segment of the URL
        clean = parsed._replace(fragment="")
        # Reconstruct without trailing canonicalization of path — the
        # matcher's _normalize_url handles that for the count step
        from urllib.parse import urlunparse
        canon = urlunparse(clean).lower().split("#", 1)[0]
        if canon in seen:
            continue
        seen.add(canon)
        out.append(canon)
        if len(out) >= 500:
            break
    return out


def fetch_page_meta(
    url: str,
    if_modified_since: datetime | None = None,
    client: httpx.Client | None = None,
    brand_host: str | None = None,
) -> dict:
    """Fetch one page and extract the metadata Day 3's matcher needs.

    Returns a dict with the following keys (always present, may be None) :
        title, meta_description, h1, body_excerpt, lang, canonical,
        internal_links (list[str] | None — only when brand_host is given),
        http_status, redirected_to, fetch_error

    Semantics :
      - On 304 Not Modified : returns {http_status: 304, ...all extracted
        fields None, fetch_error None}. Caller bumps last_crawled_at but
        does not flip status.
      - On 200 OK : extracts and returns the parsed fields. `redirected_to`
        = final URL if it differs from the input (after follow-redirects),
        else None.
      - On HTTP errors / network errors / parse errors : returns
        {http_status: <code or None>, fetch_error: <message>, ...None}.

    `client` is optional — pass a shared httpx.Client for connection pooling
    across many fetches in the same handler run. If absent we create a
    one-shot client.

    `brand_host` (Day 3) opts the caller into internal-link extraction.
    When provided, `internal_links` is populated with the same-host
    absolute URLs found on the page (capped 500). When None we skip the
    link scan to save CPU on callers that only need the metadata.
    """
    result = {
        "title": None,
        "meta_description": None,
        "h1": None,
        "body_excerpt": None,
        "lang": None,
        "canonical": None,
        "internal_links": None,
        "http_status": None,
        "redirected_to": None,
        "fetch_error": None,
    }

    headers: dict[str, str] = {}
    if if_modified_since is not None:
        # HTTP-date format per RFC 7231 §7.1.1.1
        headers["If-Modified-Since"] = if_modified_since.strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        )

    owns_client = client is None
    if owns_client:
        client = httpx.Client(
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.5",
            },
            timeout=_HTTP_TIMEOUT,
            follow_redirects=True,
            max_redirects=3,
        )

    try:
        try:
            resp = client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            result["fetch_error"] = f"network: {type(exc).__name__}: {exc}"[:300]
            return result

        result["http_status"] = resp.status_code

        # Track redirects (final URL after follow)
        final_url = str(resp.url)
        if final_url and final_url != url:
            result["redirected_to"] = final_url

        if resp.status_code == 304:
            return result

        if resp.status_code != 200:
            result["fetch_error"] = f"http_{resp.status_code}"
            return result

        content_type = (resp.headers.get("content-type") or "").lower()
        if "html" not in content_type and "xml" not in content_type:
            # PDF, image, JSON, etc. — not a page we can extract from.
            result["fetch_error"] = f"non_html_content_type: {content_type[:80]}"
            return result

        try:
            html = resp.text
        except Exception as exc:
            result["fetch_error"] = f"decode_error: {exc}"[:300]
            return result

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            result["fetch_error"] = f"parse_error: {exc}"[:300]
            return result

        # <title>
        if soup.title and soup.title.string:
            result["title"] = soup.title.string.strip()[:500] or None

        # <meta name="description">
        md = soup.find("meta", attrs={"name": "description"})
        if md and md.get("content"):
            result["meta_description"] = md["content"].strip()[:500] or None

        # <h1> — first one only
        h1 = soup.find("h1")
        if h1:
            h1_text = h1.get_text(separator=" ", strip=True)
            if h1_text:
                result["h1"] = h1_text[:500]

        # <html lang>
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            result["lang"] = html_tag["lang"].strip().lower()[:10] or None

        # <link rel="canonical">
        canon = soup.find("link", rel="canonical")
        if canon and canon.get("href"):
            abs_canon = _absolutize(final_url, canon["href"])
            if abs_canon and abs_canon != final_url:
                result["canonical"] = abs_canon[:1000]

        # Body excerpt (300 words from main/article/body, structural noise stripped)
        result["body_excerpt"] = extract_body_excerpt(html) or None

        # Internal links (Day 3 — only when caller asks for the inlink pass)
        if brand_host:
            try:
                result["internal_links"] = _extract_internal_links(
                    soup, final_url, brand_host.lower(),
                )
            except Exception as exc:
                logger.info(f"internal_links extract failed for {url}: {exc}")
                result["internal_links"] = []

        return result
    finally:
        if owns_client:
            client.close()

