"""Sprint 6 - page type detector + JSON-LD block generator.

For each cited page we :
  1. detect its content type from URL path + on-page signals (homepage,
     article, product, faq, about, other)
  2. compute which schema.org blocks SHOULD be on this page
  3. generate the missing ones, filling values from the brand brief and
     on-page microdata fallbacks (og:*, itemprop="*", <title>, <article>)

No LLM. All heuristic. The generated blocks are emitted exactly as the user
would paste them in their ``<head>``, so we add ``@context`` and use the
fully-qualified canonical type strings.

Public surface :
    detect_page_type(url, html, soup) -> str
    expected_schemas(page_type, url) -> list[str]
    generate(page_type, html, url, brand_brief, soup) -> dict[str, dict]
        Returns {schema_type: full JSON-LD dict}.
"""
from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ── Page type detection ────────────────────────────────────────────────────

# URL path hints. Multi-lingual to cover FR + EN content. We check these in
# order and the first match wins.
PATH_HINTS: tuple[tuple[str, str], ...] = (
    ("faq", "faq"),
    ("questions-frequentes", "faq"),
    ("foire-aux-questions", "faq"),
    ("a-propos", "about"),
    ("about", "about"),
    ("qui-sommes-nous", "about"),
    ("blog", "article"),
    ("article", "article"),
    ("actualites", "article"),
    ("news", "article"),
    ("conseils", "article"),
    ("dossier", "article"),
    ("expertise", "article"),
    ("produit", "product"),
    ("product", "product"),
    ("shop", "product"),
    ("boutique", "product"),
)


def detect_page_type(url: str, html: str | None, soup: BeautifulSoup | None) -> str:
    """Heuristic page-type classifier. Returns one of :
    homepage | article | product | faq | about | other."""
    if not url:
        return "other"

    parsed = urlparse(url)
    path = (parsed.path or "/").lower().rstrip("/")

    if path in ("", "/"):
        return "homepage"

    for needle, ptype in PATH_HINTS:
        if needle in path:
            return ptype

    if soup is not None:
        # FAQ : at least 3 consecutive question/answer DOM patterns.
        if _looks_like_faq(soup):
            return "faq"
        # Article : <article> + datePublished hint OR og:type=article.
        og_type = _meta(soup, "og:type")
        if og_type and "article" in og_type.lower():
            return "article"
        if soup.find("article") is not None and _meta(soup, "article:published_time"):
            return "article"
        # Product : price markers or og:type=product.
        if og_type and "product" in og_type.lower():
            return "product"
        if soup.find(attrs={"itemprop": "price"}) is not None:
            return "product"

    return "other"


def _looks_like_faq(soup: BeautifulSoup) -> bool:
    """At least 3 Q/A pairs in DL/DT/DD or H2+P alternation."""
    dts = soup.find_all("dt")
    if len(dts) >= 3 and soup.find_all("dd"):
        return True
    # H2 question + following paragraph - looser, count question marks in
    # consecutive headings.
    questions = 0
    for tag in soup.find_all(["h2", "h3"]):
        text = tag.get_text(" ", strip=True)
        if "?" in text:
            questions += 1
            if questions >= 3:
                return True
    return False


# ── Expected schemas per page type ─────────────────────────────────────────

# What SHOULD be on a page of this type. The list is the v1 surface ; we
# don't claim anything else is "missing" (Person, MedicalEntity, etc. land
# in v2). Organization + WebSite are credited at the site level so we treat
# them as expected on homepage and as "nice to have" elsewhere - they show
# up in generated_blocks but only count toward the score on the homepage.
EXPECTED: dict[str, tuple[str, ...]] = {
    "homepage": ("Organization", "WebSite"),
    "article":  ("Article", "BreadcrumbList"),
    "product":  ("Product", "BreadcrumbList"),
    "faq":      ("FAQPage", "BreadcrumbList"),
    "about":    ("Organization", "BreadcrumbList"),
    "other":    ("BreadcrumbList",),
}


def expected_schemas(page_type: str, url: str) -> list[str]:
    base = list(EXPECTED.get(page_type, ()))
    # BreadcrumbList is only meaningful when there is at least one path
    # segment. Strip it from the expectations on the bare homepage.
    parsed = urlparse(url or "")
    segments = [s for s in (parsed.path or "").split("/") if s]
    if not segments and "BreadcrumbList" in base:
        base.remove("BreadcrumbList")
    return base


# ── Generators ─────────────────────────────────────────────────────────────

_SCHEMA_CONTEXT = "https://schema.org"


def _meta(soup: BeautifulSoup, name: str) -> str | None:
    """Read a meta tag. Handles both ``name=`` and ``property=`` attributes
    so og:* and twitter:* both resolve. Returns the stripped content or None."""
    for attr in ("property", "name"):
        tag = soup.find("meta", attrs={attr: name})
        if tag and tag.get("content"):
            return tag["content"].strip() or None
    return None


def _itemprop(soup: BeautifulSoup, prop: str) -> str | None:
    tag = soup.find(attrs={"itemprop": prop})
    if not tag:
        return None
    return (tag.get("content") or tag.get_text(" ", strip=True) or "").strip() or None


def _site_root(url: str) -> str:
    p = urlparse(url)
    if not p.scheme or not p.netloc:
        return ""
    return f"{p.scheme}://{p.netloc}"


def _title(soup: BeautifulSoup | None) -> str | None:
    if not soup:
        return None
    og = _meta(soup, "og:title")
    if og:
        return og
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True) or None
    return None


def _description(soup: BeautifulSoup | None) -> str | None:
    if not soup:
        return None
    return _meta(soup, "og:description") or _meta(soup, "description")


def _author(soup: BeautifulSoup | None) -> str | None:
    if not soup:
        return None
    return (
        _meta(soup, "article:author")
        or _meta(soup, "author")
        or _itemprop(soup, "author")
    )


def _published(soup: BeautifulSoup | None) -> str | None:
    if not soup:
        return None
    val = (
        _meta(soup, "article:published_time")
        or _meta(soup, "datePublished")
        or _itemprop(soup, "datePublished")
    )
    if val:
        # Trim to YYYY-MM-DD if it parses ; otherwise pass as-is.
        m = re.match(r"(\d{4}-\d{2}-\d{2})", val)
        return m.group(1) if m else val
    # Fallback : <time datetime="...">
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        val = time_tag["datetime"]
        m = re.match(r"(\d{4}-\d{2}-\d{2})", val)
        return m.group(1) if m else val
    return None


def _image(soup: BeautifulSoup | None) -> str | None:
    if not soup:
        return None
    return _meta(soup, "og:image") or _itemprop(soup, "image")


def _brand_name(brief: dict, scan_domain: str) -> str:
    name = (brief or {}).get("name") or ""
    if name:
        return name
    # Fall back to capitalising the domain root.
    p = urlparse(scan_domain if "://" in scan_domain else f"https://{scan_domain}")
    host = p.netloc or scan_domain or ""
    base = host.split(":", 1)[0].lstrip("www.").split(".", 1)[0]
    return base.capitalize() if base else ""


def _gen_organization(url: str, soup: BeautifulSoup | None, brief: dict, scan_domain: str) -> dict:
    block: dict[str, Any] = {
        "@context": _SCHEMA_CONTEXT,
        "@type": "Organization",
        "name": _brand_name(brief, scan_domain),
        "url": _site_root(url) or (scan_domain if "://" in scan_domain else f"https://{scan_domain}"),
    }
    desc = (brief or {}).get("description") or _description(soup)
    if desc:
        block["description"] = desc

    if (brief or {}).get("founded_year"):
        block["foundingDate"] = str(brief["founded_year"])

    hq = (brief or {}).get("headquarters")
    if hq:
        block["address"] = {"@type": "PostalAddress", "addressLocality": hq}

    parent = (brief or {}).get("parent_group")
    if parent:
        block["parentOrganization"] = {"@type": "Organization", "name": parent}

    logo = _image(soup)
    if logo:
        block["logo"] = logo

    return block


def _gen_website(url: str, soup: BeautifulSoup | None, brief: dict, scan_domain: str) -> dict:
    return {
        "@context": _SCHEMA_CONTEXT,
        "@type": "WebSite",
        "name": _brand_name(brief, scan_domain),
        "url": _site_root(url) or scan_domain,
    }


def _gen_breadcrumb(url: str, soup: BeautifulSoup | None) -> dict | None:
    p = urlparse(url)
    if not p.scheme or not p.netloc:
        return None
    segments = [s for s in (p.path or "").split("/") if s]
    if not segments:
        return None

    root = f"{p.scheme}://{p.netloc}"
    items = [{
        "@type": "ListItem",
        "position": 1,
        "name": "Home",
        "item": root + "/",
    }]
    accum = ""
    for i, seg in enumerate(segments, start=2):
        accum += "/" + seg
        items.append({
            "@type": "ListItem",
            "position": i,
            # Humanise the slug : decode percent-escapes elsewhere ; for v1
            # we just hyphen-to-space + title-case.
            "name": seg.replace("-", " ").replace("_", " ").strip().capitalize(),
            "item": root + accum,
        })
    return {
        "@context": _SCHEMA_CONTEXT,
        "@type": "BreadcrumbList",
        "itemListElement": items,
    }


def _gen_article(url: str, soup: BeautifulSoup | None, brief: dict, scan_domain: str) -> dict:
    title = _title(soup) or ""
    block: dict[str, Any] = {
        "@context": _SCHEMA_CONTEXT,
        "@type": "Article",
        "headline": title,
        "mainEntityOfPage": {"@type": "WebPage", "@id": url},
    }
    desc = _description(soup)
    if desc:
        block["description"] = desc
    img = _image(soup)
    if img:
        block["image"] = img

    published = _published(soup)
    if published:
        block["datePublished"] = published
    author = _author(soup)
    if author:
        block["author"] = {"@type": "Person", "name": author}

    publisher_name = _brand_name(brief, scan_domain)
    if publisher_name:
        block["publisher"] = {"@type": "Organization", "name": publisher_name}

    return block


def _gen_product(url: str, soup: BeautifulSoup | None, brief: dict, scan_domain: str) -> dict:
    title = _itemprop(soup, "name") or _title(soup) or ""
    block: dict[str, Any] = {
        "@context": _SCHEMA_CONTEXT,
        "@type": "Product",
        "name": title,
    }
    desc = _itemprop(soup, "description") or _description(soup)
    if desc:
        block["description"] = desc
    img = _itemprop(soup, "image") or _image(soup)
    if img:
        block["image"] = img
    brand_name = _brand_name(brief, scan_domain)
    if brand_name:
        block["brand"] = {"@type": "Brand", "name": brand_name}
    price = _itemprop(soup, "price")
    currency = _itemprop(soup, "priceCurrency") or "EUR"
    if price:
        block["offers"] = {
            "@type": "Offer",
            "price": price,
            "priceCurrency": currency,
            "url": url,
        }
    return block


_Q_TAGS = ("h2", "h3", "dt")


def _extract_faqs(soup: BeautifulSoup | None) -> list[dict]:
    """Pull Q/A pairs from the DOM. Two strategies :
      - DL/DT/DD : each DT followed by its DD.
      - Hn + sibling paragraphs : a question-marked heading followed by the
        first prose block before the next heading.
    Cap at 30 to keep the JSON-LD reasonable."""
    if soup is None:
        return []
    pairs: list[dict] = []

    # DL strategy
    for dl in soup.find_all("dl"):
        dts = dl.find_all("dt")
        for dt in dts:
            q = dt.get_text(" ", strip=True)
            dd = dt.find_next_sibling("dd")
            a = dd.get_text(" ", strip=True) if dd else ""
            if q and a and len(pairs) < 30:
                pairs.append({"q": q, "a": a})

    # Hn strategy - run independently and dedupe at the end.
    for tag in soup.find_all(_Q_TAGS):
        text = tag.get_text(" ", strip=True)
        if "?" not in text:
            continue
        # Collect prose siblings until the next heading or DT.
        chunks: list[str] = []
        sib = tag.find_next_sibling()
        while sib and sib.name not in ("h1", "h2", "h3", "h4", "dt"):
            if sib.name in ("p", "div", "ul", "ol"):
                chunks.append(sib.get_text(" ", strip=True))
            sib = sib.find_next_sibling()
        answer = " ".join(c for c in chunks if c).strip()
        if text and answer and len(pairs) < 30:
            pairs.append({"q": text, "a": answer})

    # Dedupe by Q.
    seen = set()
    out = []
    for p in pairs:
        key = p["q"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _gen_faqpage(url: str, soup: BeautifulSoup | None) -> dict | None:
    pairs = _extract_faqs(soup)
    if not pairs:
        return None
    return {
        "@context": _SCHEMA_CONTEXT,
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": p["q"],
                "acceptedAnswer": {"@type": "Answer", "text": p["a"]},
            }
            for p in pairs
        ],
    }


def generate(
    page_type: str,
    html: str | None,
    url: str,
    brand_brief: dict | None,
    soup: BeautifulSoup | None = None,
) -> dict[str, dict]:
    """Generate the JSON-LD blocks expected for this page type. Returns a
    dict keyed by schema type. Empty dict if nothing can be generated (e.g.
    no soup, malformed page)."""
    brief = brand_brief or {}
    scan_domain = brief.get("_scan_domain", "")  # filled by the handler

    if soup is None and html:
        soup = BeautifulSoup(html, "html.parser")

    out: dict[str, dict] = {}
    wanted = set(expected_schemas(page_type, url))

    if "Organization" in wanted:
        out["Organization"] = _gen_organization(url, soup, brief, scan_domain)
    if "WebSite" in wanted:
        out["WebSite"] = _gen_website(url, soup, brief, scan_domain)
    if "BreadcrumbList" in wanted:
        bc = _gen_breadcrumb(url, soup)
        if bc:
            out["BreadcrumbList"] = bc
    if "Article" in wanted:
        out["Article"] = _gen_article(url, soup, brief, scan_domain)
    if "Product" in wanted:
        out["Product"] = _gen_product(url, soup, brief, scan_domain)
    if "FAQPage" in wanted:
        faq = _gen_faqpage(url, soup)
        if faq:
            out["FAQPage"] = faq

    return out
