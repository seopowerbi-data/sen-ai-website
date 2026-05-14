"""Sitemap-matcher service for Phase D.

Day 3 surface : `compute_inlinks_from_map` — given an in-memory
{page_url -> list[outgoing_urls]} collected by fetch_brand_pages, compute
the inbound link count per page and bulk-update internal_inlink_count.

Day 4 will add `find_best_pages(question_text, client_brand_id, db, ...)` :
the actual semantic matcher that combines cosine similarity, authority
boost (log-scaled inlinks), and gamme path bias. Stubbed for now so the
embedding pipeline ships in isolation.

URL normalization rules used by the inlink matcher :
  - lowercase the host
  - strip the leading 'www.' so 'www.brand.fr' and 'brand.fr' map equal
  - strip the URL fragment ('#anchor')
  - keep query string (a page with ?utm=... is functionally the same
    target as without, but matcher only ever sees clean URLs from the
    sitemap or from <link rel=canonical>; cleaning the QS would
    over-match in the rare case a sitemap entry includes one)
  - normalize an empty path to '/'

Self-links (a page linking to itself) are NOT counted — they reflect
nav/footer convenience, not architectural intent.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from urllib.parse import urlparse, urlunparse

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    """Canonical form used as a dict key in the inlink count map."""
    if not url:
        return ""
    try:
        p = urlparse(url.strip())
    except ValueError:
        return ""
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return ""
    path = p.path or "/"
    if path == "":
        path = "/"
    return urlunparse((p.scheme.lower() or "https", host, path, p.params, p.query, ""))


def compute_inlinks_from_map(
    client_brand_id: str,
    links_map: dict[str, list[str]],
    db: Session,
) -> dict:
    """Reset every row in this brand's index to 0, then bulk-set inlink
    counts from the in-memory map.

    Reset-then-set guarantees idempotency : if a page that used to be a hub
    has lost all its inbound links since the last crawl, its count drops
    back to 0. A pure increment loop would never decrease a stale count.

    `links_map` shape : `{source_page_url: [outgoing_url1, outgoing_url2, ...]}`.
    Outgoing URLs that don't match any of this brand's indexed pages are
    silently dropped (they're external links). Self-links are dropped too.

    Returns :
      {
        "pages_with_links": int,        # source pages that had ≥1 outgoing internal link
        "inlinks_total": int,           # sum of all increments
        "targets_with_inlinks": int,    # distinct target URLs that got ≥1 inlink
        "max_inlinks_on_one_page": int,
      }
    """
    from models import ClientBrandPage

    # Build the set of "our pages" — used as a filter on outgoing edges so
    # external links don't bloat the count.
    rows = (
        db.query(ClientBrandPage.id, ClientBrandPage.url)
        .filter(ClientBrandPage.client_brand_id == client_brand_id)
        .all()
    )
    if not rows:
        return {
            "pages_with_links": 0, "inlinks_total": 0,
            "targets_with_inlinks": 0, "max_inlinks_on_one_page": 0,
        }

    # Map normalized URL → row id, so we resolve outgoing edges back to
    # the actual row to update.
    norm_to_id: dict[str, str] = {}
    for row_id, raw_url in rows:
        nu = _normalize_url(raw_url)
        if nu:
            norm_to_id[nu] = str(row_id)

    inlink_counts: dict[str, int] = defaultdict(int)
    pages_with_links = 0
    inlinks_total = 0

    for source_url, outgoing in links_map.items():
        if not outgoing:
            continue
        source_norm = _normalize_url(source_url)
        had_any = False
        for target_url in outgoing:
            target_norm = _normalize_url(target_url)
            if not target_norm or target_norm == source_norm:
                continue
            target_id = norm_to_id.get(target_norm)
            if not target_id:
                continue
            inlink_counts[target_id] += 1
            inlinks_total += 1
            had_any = True
        if had_any:
            pages_with_links += 1

    # Reset all rows in this brand to 0, then bulk-update only the ones
    # that have inlinks. Two passes keep the SQL simple and idempotent.
    db.query(ClientBrandPage).filter(
        ClientBrandPage.client_brand_id == client_brand_id,
    ).update({ClientBrandPage.internal_inlink_count: 0})

    # Group updates by count value to keep query count bounded — at most
    # one query per distinct count.
    by_count: dict[int, list[str]] = defaultdict(list)
    for row_id, n in inlink_counts.items():
        by_count[n].append(row_id)
    for n, ids in by_count.items():
        db.query(ClientBrandPage).filter(
            ClientBrandPage.id.in_(ids),
        ).update({ClientBrandPage.internal_inlink_count: n}, synchronize_session=False)

    db.commit()

    max_inlinks = max(inlink_counts.values(), default=0)
    logger.info(
        f"compute_inlinks brand={client_brand_id}: pages_with_links={pages_with_links} "
        f"inlinks_total={inlinks_total} targets={len(inlink_counts)} max={max_inlinks}"
    )

    return {
        "pages_with_links": pages_with_links,
        "inlinks_total": inlinks_total,
        "targets_with_inlinks": len(inlink_counts),
        "max_inlinks_on_one_page": max_inlinks,
    }
