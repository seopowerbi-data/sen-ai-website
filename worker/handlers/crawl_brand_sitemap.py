"""Handler: crawl a client_brand's sitemap and reconcile client_brand_pages.

Phase D, Day 1. Discovers the URL list for a brand's domain via
sitemap_crawler.discover_sitemap_urls, then idempotently diffs it against
the existing client_brand_pages rows for that brand.

Diff semantics (one pass, in one transaction) :

    URL in sitemap + DB row exists
        -> bump last_seen_at = NOW()
        -> if sitemap lastmod is newer than DB lastmod, update lastmod
        -> if the DB row was 'gone', flip it back to its prior content
           status ('embedded' if it still has an embedding, else 'fetched',
           else 'pending_fetch'). Preserve the existing embedding so a
           page that bounces in/out of the sitemap doesn't pay re-embed
           cost. Clear gone_since.

    URL in sitemap, no DB row
        -> INSERT new row with status='pending_fetch', first_seen_at = NOW().
           Day 2's fetch_brand_pages picks these up.

    DB row exists, URL not in sitemap
        -> If status != 'gone': flip to 'gone', set gone_since = NOW().
           Idempotent on re-run — don't bump gone_since if already gone.
        -> If status == 'gone' already: no change (gone_since stays at the
           first time we noticed).

Day 1 does NOT enqueue fetch_brand_pages or embed_brand_pages — those land
Day 2 + Day 3. The handler just leaves rows in 'pending_fetch' and returns.

Payload :
    {"client_brand_id": str}

Returns :
    {
        "client_brand_id": str,
        "domain": str | None,
        "status": "ok" | "skipped",
        "reason": str | None,           # set when status="skipped"
        "discovered": int,              # total URLs from sitemap after filter
        "inserted": int,                # new rows created
        "bumped": int,                  # existing rows whose last_seen_at was refreshed
        "marked_gone": int,             # rows newly flipped to 'gone'
        "restored": int,                # rows flipped back from 'gone'
        "errors": list[str],            # non-fatal issues
    }

In-flight protection (1 crawl per brand in flight at a time) is enforced
at the API enqueue layer (Day 5 — POST /api/clients/{id}/brands/{bid}/sitemap/refresh),
mirroring the pattern used by POST /clients/{id}/trust-sources/discover.
For Day 1, smoke tests enqueue jobs manually and the diff logic is
naturally idempotent on re-run.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from services.sitemap_crawler import discover_sitemap_urls

logger = logging.getLogger(__name__)


def execute(job_payload: dict, scan_id: str | None, db: Session) -> dict:
    """Crawl one client_brand's sitemap and reconcile client_brand_pages.

    scan_id is unused (workspace-scoped — brand-level operation).
    """
    from models import ClientBrand, ClientBrandPage

    client_brand_id = (job_payload or {}).get("client_brand_id")
    if not client_brand_id:
        raise ValueError("crawl_brand_sitemap requires client_brand_id in job payload")

    brand = (
        db.query(ClientBrand)
        .filter(ClientBrand.id == client_brand_id)
        .first()
    )
    if not brand:
        raise ValueError(f"ClientBrand {client_brand_id} not found")

    domain = (brand.domain or "").strip()
    if not domain:
        logger.warning(
            f"ClientBrand {client_brand_id} ({brand.name}) has no domain — "
            f"cannot crawl sitemap"
        )
        return {
            "client_brand_id": str(client_brand_id),
            "domain": None,
            "status": "skipped",
            "reason": "missing_domain",
            "discovered": 0,
            "inserted": 0,
            "bumped": 0,
            "marked_gone": 0,
            "restored": 0,
            "errors": [],
        }

    pairs = discover_sitemap_urls(domain)
    discovered = len(pairs)

    if discovered == 0:
        logger.warning(
            f"Sitemap discovery returned 0 URLs for {brand.name} "
            f"(domain={domain}). Treating as no-op rather than mass-marking "
            f"all existing rows as 'gone' — a transient sitemap fetch failure "
            f"would otherwise nuke the index."
        )
        return {
            "client_brand_id": str(client_brand_id),
            "domain": domain,
            "status": "skipped",
            "reason": "no_urls_discovered",
            "discovered": 0,
            "inserted": 0,
            "bumped": 0,
            "marked_gone": 0,
            "restored": 0,
            "errors": [],
        }

    sitemap_lastmod: dict[str, datetime | None] = dict(pairs)
    sitemap_urls = set(sitemap_lastmod.keys())

    # Pull every existing row for this brand once. With the 5000-URL cap and
    # typical PF brands at ~1k each, this is a small in-memory dict and lets
    # the diff happen without N+1 queries.
    existing_rows = (
        db.query(ClientBrandPage)
        .filter(ClientBrandPage.client_brand_id == client_brand_id)
        .all()
    )
    existing_by_url: dict[str, ClientBrandPage] = {r.url: r for r in existing_rows}

    now = datetime.utcnow()
    inserted = 0
    bumped = 0
    marked_gone = 0
    restored = 0
    errors: list[str] = []

    # 1) Bumps + restores + inserts for URLs currently in the sitemap
    for url in sitemap_urls:
        lastmod = sitemap_lastmod.get(url)
        row = existing_by_url.get(url)
        if row is None:
            new_row = ClientBrandPage(
                client_brand_id=client_brand_id,
                url=url,
                lastmod=lastmod,
                status="pending_fetch",
                first_seen_at=now,
                last_seen_at=now,
            )
            db.add(new_row)
            inserted += 1
            continue
        row.last_seen_at = now
        if lastmod is not None and (row.lastmod is None or lastmod > row.lastmod):
            row.lastmod = lastmod
        if row.status == "gone":
            # Restore: pick the most-advanced prior status the data supports.
            if row.embedding is not None:
                row.status = "embedded"
            elif row.title is not None or row.body_excerpt is not None:
                # We had content but no embedding — let Day 3 re-pick it up
                row.status = "fetched"
            else:
                row.status = "pending_fetch"
            row.gone_since = None
            restored += 1
        else:
            bumped += 1

    # 2) Gone-marking for DB rows whose URL disappeared from the sitemap
    for url, row in existing_by_url.items():
        if url in sitemap_urls:
            continue
        if row.status == "gone":
            # Already marked. Idempotent — leave gone_since alone.
            continue
        row.status = "gone"
        row.gone_since = now
        marked_gone += 1

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception(f"Commit failed in crawl_brand_sitemap for {brand.name}: {exc}")
        raise

    logger.info(
        f"crawl_brand_sitemap done for {brand.name} (domain={domain}): "
        f"discovered={discovered} inserted={inserted} bumped={bumped} "
        f"marked_gone={marked_gone} restored={restored}"
    )

    # NOTE: Day 2 will append `db.add(Job(job_type='fetch_brand_pages', ...))`
    # here to chain the fetch step. For Day 1 we leave rows in pending_fetch
    # and exit cleanly so the smoke test can inspect persistence in isolation.
    logger.info(
        f"Chain target on Day 2: enqueue fetch_brand_pages for "
        f"client_brand_id={client_brand_id} ({inserted} pending_fetch rows)"
    )

    return {
        "client_brand_id": str(client_brand_id),
        "domain": domain,
        "status": "ok",
        "reason": None,
        "discovered": discovered,
        "inserted": inserted,
        "bumped": bumped,
        "marked_gone": marked_gone,
        "restored": restored,
        "errors": errors,
    }
