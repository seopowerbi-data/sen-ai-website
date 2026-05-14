"""Handler: fetch every pending / stale page in client_brand_pages and
populate title / meta / h1 / body_excerpt / content_hash.

Phase D, Day 2. Picks up where crawl_brand_sitemap left off : rows in
status='pending_fetch' (newly discovered) plus stale 'embedded' rows
whose lastmod is newer than last_crawled_at OR which haven't been
re-crawled in > 30 days.

Per-row pipeline :

    robots.txt allow-check   (cached per host)
        |  not allowed
        v
    status='error', fetch_error='blocked_by_robots'
        |  allowed
        v
    fetch_page_meta(url, if_modified_since=last_crawled_at)
        |
        +-- 304 Not Modified  -> bump last_crawled_at, no status change
        +-- fetch_error       -> bump retry_count; status='error' if >= 3
        +-- 200 OK
              |
              +-- soft-404     -> status='error', fetch_error='soft_404'
              +-- content_hash unchanged
              |    -> bump last_crawled_at, NO re-embed cost (Day 3)
              +-- content_hash changed
                   -> update fields, status='fetched' (triggers Day 3 re-embed)

Day 2 does NOT compute internal_inlink_count (that's Day 3, in
sitemap_matcher.compute_inlinks) and does NOT enqueue embed_brand_pages
(also Day 3). The handler only fills the per-page metadata + flips status.

Throttle: 1 request/sec/domain (plan default; configurable via payload
for power users). Periodic commits every 50 rows so a worker crash mid-
loop doesn't lose progress.

Payload :
    {
        "client_brand_id": str,
        "max_pages": int (optional cap for smoke testing),
        "throttle_seconds": float (optional, default 1.0),
    }

Returns :
    {
        "client_brand_id": str, "domain": str | None,
        "status": "ok" | "skipped",
        "reason": str | None,
        "attempted": int,
        "fetched": int,            # newly-extracted content (hash changed)
        "unchanged": int,          # 304s + content_hash short-circuits
        "soft_404": int,
        "errors": int,             # rows now in status='error'
        "blocked_by_robots": int,
        "remaining_retry": int,    # row failed but retry_count < 3
    }
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timedelta

import httpx
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from services.sitemap_crawler import (
    _USER_AGENT,
    fetch_page_meta,
    is_robots_allowed,
    is_soft_404,
)

logger = logging.getLogger(__name__)

# Circuit-breaker : if this many CONSECUTIVE network errors happen, bail
# rather than burning through the corpus retrying against a down host.
_CONSECUTIVE_ERROR_LIMIT = 20

# Stale threshold for re-fetch of already-embedded rows (matches the plan).
_STALE_DAYS = 30

# Commit every N rows so a worker kill mid-loop doesn't lose progress.
_COMMIT_EVERY = 50

_FETCH_RETRY_CEILING = 3


def _compute_content_hash(title, meta, h1, body_excerpt) -> str:
    """sha256(lower(title|meta|h1|body_excerpt)). Empty parts contribute "".

    Used to short-circuit re-embed when a CMS lies about lastmod (the
    sitemap says the page changed, but the actual content didn't).
    """
    parts = [
        (title or "").lower(),
        (meta or "").lower(),
        (h1 or "").lower(),
        (body_excerpt or "").lower(),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _select_pending_rows(
    db: Session, client_brand_id: str, max_pages: int | None
):
    """Pull every row that needs fetching, oldest-first.

    Selection :
      - status='pending_fetch'                                    (newly discovered)
      - status='embedded' AND lastmod > last_crawled_at           (CMS says changed)
      - status='embedded' AND last_crawled_at < now - 30d         (stale refresh)

    Rows in status='error' / 'gone' are excluded — error retries happen
    via a separate manual re-enqueue (plan: surface in Settings UI).
    """
    from models import ClientBrandPage

    now = datetime.utcnow()
    stale_cutoff = now - timedelta(days=_STALE_DAYS)

    q = (
        db.query(ClientBrandPage)
        .filter(
            ClientBrandPage.client_brand_id == client_brand_id,
            or_(
                ClientBrandPage.status == "pending_fetch",
                and_(
                    ClientBrandPage.status == "embedded",
                    or_(
                        and_(
                            ClientBrandPage.lastmod.isnot(None),
                            ClientBrandPage.last_crawled_at.isnot(None),
                            ClientBrandPage.lastmod > ClientBrandPage.last_crawled_at,
                        ),
                        ClientBrandPage.last_crawled_at.is_(None),
                        ClientBrandPage.last_crawled_at < stale_cutoff,
                    ),
                ),
            ),
        )
        .order_by(ClientBrandPage.first_seen_at.asc())
    )
    if max_pages:
        q = q.limit(int(max_pages))
    return q.all()


def execute(job_payload: dict, scan_id: str | None, db: Session) -> dict:
    from models import ClientBrand

    client_brand_id = (job_payload or {}).get("client_brand_id")
    if not client_brand_id:
        raise ValueError("fetch_brand_pages requires client_brand_id in job payload")

    throttle = float((job_payload or {}).get("throttle_seconds") or 1.0)
    max_pages = (job_payload or {}).get("max_pages")

    brand = db.query(ClientBrand).filter(ClientBrand.id == client_brand_id).first()
    if not brand:
        raise ValueError(f"ClientBrand {client_brand_id} not found")

    domain = (brand.domain or "").strip()
    if not domain:
        return {
            "client_brand_id": str(client_brand_id),
            "domain": None,
            "status": "skipped",
            "reason": "missing_domain",
            "attempted": 0, "fetched": 0, "unchanged": 0,
            "soft_404": 0, "errors": 0, "blocked_by_robots": 0,
            "remaining_retry": 0,
        }

    rows = _select_pending_rows(db, client_brand_id, max_pages)
    if not rows:
        logger.info(
            f"fetch_brand_pages: no pending or stale rows for {brand.name} "
            f"({domain})"
        )
        return {
            "client_brand_id": str(client_brand_id),
            "domain": domain,
            "status": "ok",
            "reason": "nothing_to_fetch",
            "attempted": 0, "fetched": 0, "unchanged": 0,
            "soft_404": 0, "errors": 0, "blocked_by_robots": 0,
            "remaining_retry": 0,
        }

    logger.info(
        f"fetch_brand_pages start: {brand.name} ({domain}) "
        f"rows={len(rows)} throttle={throttle}s"
    )

    fetched = 0
    unchanged = 0
    soft_404 = 0
    errors = 0
    blocked_by_robots = 0
    remaining_retry = 0
    consecutive_errors = 0

    with httpx.Client(
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.5",
        },
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
        follow_redirects=True,
        max_redirects=3,
    ) as client:
        for i, row in enumerate(rows):
            now = datetime.utcnow()

            # robots.txt check (cached per host, ~free after first call)
            try:
                allowed = is_robots_allowed(row.url, user_agent=_USER_AGENT)
            except Exception as exc:
                logger.info(f"robots check raised on {row.url}: {exc}")
                allowed = True
            if not allowed:
                row.status = "error"
                row.fetch_error = "blocked_by_robots"
                row.last_crawled_at = now
                blocked_by_robots += 1
                consecutive_errors = 0
                if (i + 1) % _COMMIT_EVERY == 0:
                    db.commit()
                continue

            # Per-host throttle
            if i > 0 and throttle > 0:
                time.sleep(throttle)

            meta = fetch_page_meta(
                row.url,
                if_modified_since=row.last_crawled_at,
                client=client,
            )

            row.last_crawled_at = now
            if meta["http_status"] is not None:
                row.http_status = meta["http_status"]

            # Network/parse error path
            if meta["fetch_error"] is not None:
                row.fetch_retry_count = (row.fetch_retry_count or 0) + 1
                if row.fetch_retry_count >= _FETCH_RETRY_CEILING:
                    row.status = "error"
                    row.fetch_error = meta["fetch_error"]
                    errors += 1
                else:
                    # Leave in pending_fetch so the next handler run retries
                    row.fetch_error = meta["fetch_error"]
                    remaining_retry += 1
                consecutive_errors += 1
                if consecutive_errors >= _CONSECUTIVE_ERROR_LIMIT:
                    logger.warning(
                        f"fetch_brand_pages bailing: {consecutive_errors} "
                        f"consecutive network errors on {domain}"
                    )
                    db.commit()
                    break
                if (i + 1) % _COMMIT_EVERY == 0:
                    db.commit()
                continue

            consecutive_errors = 0

            # 304 Not Modified : nothing to update
            if meta["http_status"] == 304:
                unchanged += 1
                if (i + 1) % _COMMIT_EVERY == 0:
                    db.commit()
                continue

            # Soft-404 : 200 OK but the page is a "not found" placeholder
            if is_soft_404(meta["title"], meta["body_excerpt"]):
                row.status = "error"
                row.fetch_error = "soft_404"
                soft_404 += 1
                if (i + 1) % _COMMIT_EVERY == 0:
                    db.commit()
                continue

            new_hash = _compute_content_hash(
                meta["title"], meta["meta_description"],
                meta["h1"], meta["body_excerpt"],
            )

            if row.content_hash == new_hash and row.status in ("fetched", "embedded"):
                # Content unchanged AND we already have it indexed — skip
                # the re-embed cost on Day 3. Just bumping last_crawled_at.
                unchanged += 1
                if (i + 1) % _COMMIT_EVERY == 0:
                    db.commit()
                continue

            row.title = meta["title"]
            row.meta_description = meta["meta_description"]
            row.h1 = meta["h1"]
            row.body_excerpt = meta["body_excerpt"]
            row.lang = meta["lang"]
            row.url_canonical = meta["canonical"]
            row.content_hash = new_hash
            row.fetch_error = None
            row.fetch_retry_count = 0
            row.status = "fetched"
            fetched += 1

            if (i + 1) % _COMMIT_EVERY == 0:
                db.commit()

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception(f"final commit failed in fetch_brand_pages for {brand.name}")
        raise

    attempted = fetched + unchanged + soft_404 + errors + blocked_by_robots + remaining_retry

    logger.info(
        f"fetch_brand_pages done for {brand.name} ({domain}): "
        f"attempted={attempted} fetched={fetched} unchanged={unchanged} "
        f"soft_404={soft_404} errors={errors} "
        f"blocked_by_robots={blocked_by_robots} remaining_retry={remaining_retry}"
    )

    # NOTE: Day 3 will append a chain enqueue of embed_brand_pages here.
    logger.info(
        f"Chain target on Day 3: enqueue embed_brand_pages for "
        f"client_brand_id={client_brand_id} ({fetched} rows now in 'fetched')"
    )

    return {
        "client_brand_id": str(client_brand_id),
        "domain": domain,
        "status": "ok",
        "reason": None,
        "attempted": attempted,
        "fetched": fetched,
        "unchanged": unchanged,
        "soft_404": soft_404,
        "errors": errors,
        "blocked_by_robots": blocked_by_robots,
        "remaining_retry": remaining_retry,
    }
