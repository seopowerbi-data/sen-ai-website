"""Sprint 6 - schema.org / JSON-LD extractor + local validator.

Pulls every ``<script type="application/ld+json">`` block from a page's HTML,
parses it, and validates each block against the schema.org required-property
spec we care about. Local-only, zero network, zero LLM.

Why a local validator instead of Google Rich Results Test :
  - Google does not expose a public, documented API for the test. Scraping
    the internal endpoint is fragile and the rate limit is per-IP, which
    would brick a scan that audits 300 URLs.
  - The required-property checks below mirror what Google flags as the red
    "Required properties missing" errors. That covers ~90 % of the practical
    value of validation. Recommended properties (warnings) are out of scope
    for v1.

Public surface :
    extract(html: str) -> list[dict]
        Returns a list of ``{type, raw, valid, missing, errors}`` records,
        one per JSON-LD block found. ``type`` is the @type string (or the
        first element if @type is an array). When parsing fails the record
        has ``valid=False`` and ``errors=["malformed_json"]`` so the UI can
        still surface that the block exists but is broken.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# Required properties per schema.org type. Source : schema.org spec + Google
# structured data requirements (the subset that produces a Rich Results error
# when missing). Kept small and additive - we can grow the table as we add
# more schema types in v2.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "Organization":      ("name",),
    "WebSite":           ("name", "url"),
    "BreadcrumbList":    ("itemListElement",),
    "Article":           ("headline", "datePublished", "author"),
    "NewsArticle":       ("headline", "datePublished", "author"),
    "BlogPosting":       ("headline", "datePublished", "author"),
    "FAQPage":           ("mainEntity",),
    "Product":           ("name",),
    "Offer":             ("price", "priceCurrency"),
    "Person":            ("name",),
}

# How deep we recurse into @graph / nested arrays before bailing out. The
# typical real-world @graph has 2-3 entries ; 50 is paranoia padding.
MAX_BLOCKS_PER_PAGE = 50


def _normalize_type(value: Any) -> str | None:
    """Resolve ``@type`` which can be a string, an array, or missing."""
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        # Prefer the first non-empty string entry. JSON-LD allows multi-type
        # ("Article", "MedicalWebPage") but the first is conventionally the
        # primary one.
        for v in value:
            if isinstance(v, str) and v:
                return v
        return None
    if isinstance(value, str):
        return value or None
    return None


def _required_check(block_type: str | None, block: dict) -> list[str]:
    """Return the names of REQUIRED fields missing from this block. Unknown
    types pass through with an empty list (we never claim a block is invalid
    just because we don't know it)."""
    if not block_type:
        return ["@type"]
    fields = REQUIRED_FIELDS.get(block_type)
    if not fields:
        return []
    missing: list[str] = []
    for f in fields:
        v = block.get(f)
        if v is None:
            missing.append(f)
        elif isinstance(v, str) and not v.strip():
            missing.append(f)
        elif isinstance(v, list) and not v:
            missing.append(f)
    return missing


def _walk_block(block: Any, out: list[dict]) -> None:
    """Push one record per @type-bearing object. Handles JSON-LD ``@graph``
    containers and bare arrays at the top level."""
    if len(out) >= MAX_BLOCKS_PER_PAGE:
        return

    if isinstance(block, list):
        for entry in block:
            _walk_block(entry, out)
        return

    if not isinstance(block, dict):
        return

    # @graph container : descend into its members. We still record the
    # outer wrapper if it has a @type of its own (rare but allowed).
    graph = block.get("@graph")
    if isinstance(graph, list):
        for entry in graph:
            _walk_block(entry, out)
        # Don't return - the wrapper may itself have @type metadata.

    raw_type = _normalize_type(block.get("@type"))
    if raw_type is None and not graph:
        # Skip orphan objects with no type and no graph - they're metadata
        # noise (e.g. an @context-only header).
        return
    if raw_type is None:
        return

    missing = _required_check(raw_type, block)
    out.append({
        "type": raw_type,
        "raw": block,
        "valid": not missing,
        "missing": missing,
        "errors": [],
    })


# Some sites wrap their JSON-LD in HTML comments or stray CDATA - strip
# those before parsing or we'll choke on the comment markers.
_CDATA_RE = re.compile(r"^\s*//\s*<!\[CDATA\[|\]\]>\s*$|^<!--|-->$", re.MULTILINE)


def extract(html: str) -> list[dict]:
    """Find and parse every JSON-LD block in the HTML. See module docstring
    for the record shape."""
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    blocks: list[dict] = []

    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        text = tag.string or tag.get_text() or ""
        text = _CDATA_RE.sub("", text).strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            blocks.append({
                "type": None,
                "raw": {"_raw_text": text[:500]},
                "valid": False,
                "missing": [],
                "errors": [f"malformed_json:{str(e)[:80]}"],
            })
            continue

        _walk_block(parsed, blocks)

        if len(blocks) >= MAX_BLOCKS_PER_PAGE:
            logger.info("schema_extractor: hit MAX_BLOCKS_PER_PAGE cap")
            break

    return blocks


def has_type(blocks: list[dict], wanted: str) -> bool:
    """True if at least one block matches the type AND parses as valid."""
    for b in blocks:
        if b.get("type") == wanted and b.get("valid"):
            return True
    return False
