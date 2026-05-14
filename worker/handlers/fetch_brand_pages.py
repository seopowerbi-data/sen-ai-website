"""Handler: fetch every pending / stale page in client_brand_pages and
populate title / meta / h1 / body_excerpt / content_hash, AND collect
the internal-link map for the post-fetch inlink-count pass.

Phase D, Day 2+3. Picks up where crawl_brand_sitemap left off : rows in
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

Day 3 addition : `fetch_page_meta` now receives `brand_host` so it can
extract every same-host `<a href>` from the page. The handler accumulates
{source_url -> outgoing_urls} in memory, then calls
`sitemap_matcher.compute_inlinks_from_map` at the end to bulk-update
`internal_inlink_count` (which feeds the matcher's authority boost).
Finally, when at least one row was newly fetched the handler chains
`embed_brand_pages`.

Payload :
    {
        "client_brand_id": str,
        "max_pages": int (optional cap for smoke testing),
        "throttle_seconds": float (optional, default 1.0),
        "force_refetch": bool (optional, default False — re-fetches
            every selected row even when content_hash is unchanged,
            used to backfill internal-link maps on pre-Day-3 corpora),
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
from services.sitemap_matcher import compute_inlinks_from_map

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
    db: Session, client_brand_id: str, max_pages: int | None,
    force_refetch: bool = False,
):
    """Pull every row that needs fetching, oldest-first.

    Default selection :
      - status='pending_fetch'                                    (newly discovered)
      - status='embedded' AND lastmod > last_crawled_at           (CMS says changed)
      - status='embedded' AND last_crawled_at < now - 30d         (stale refresh)

    Rows in status='error' / 'gone' are excluded — error retries happen
    via a separate manual re-enqueue (plan: surface in Settings UI).

    `force_refetch=True` widens the selection to every row in
    ('pending_fetch', 'fetched', 'embedded') and turns OFF the content_hash
    short-circuit in the loop. Use sparingly — drives the Day-3 inlink
    backfill on pre-Day-3 corpora.
    """
    from models import ClientBrandPage

    now = datetime.utcnow()
    stale_cutoff = now - timedelta(days=_STALE_DAYS)

    q = (
        db.query(ClientBrandPage)
        .filter(ClientBrandPage.client_brand_id == client_brand_id)
    )
    if force_refetch:
        q = q.filter(ClientBrandPage.status.in_(("pending_fetch", "fetched", "embedded")))
    else:
        q = q.filter(
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
            )
        )
    q = q.order_by(ClientBrandPage.first_seen_at.asc())
    if max_pages:
        q = q.limit(int(max_pages))
    return q.all()


def _normalize_brand_host(domain: str) -> str:
    """Same shape as the API validator — strip scheme/www/trailing slash."""
    if not domain:
        return ""
    import re as _re
    d = _re.sub(r"^https?://", "", (domain or "").strip().lower())
    if d.startswith("www."):
        d = d[4:]
    return d.split("/", 1)[0].strip().rstrip(".")


def execute(job_payload: dict, scan_id: str | None, db: Session) -> dict:
    from models import ClientBrand

    client_brand_id = (job_payload or {}).get("client_brand_id")
    if not client_brand_id:
        raise ValueError("fetch_brand_pages requires client_brand_id in job payload")

    throttle = float((job_payload or {}).get("throttle_seconds") or 1.0)
    max_pages = (job_payload or {}).get("max_pages")
    force_refetch = bool((job_payload or {}).get("force_refetch"))

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

    brand_host = _normalize_brand_host(domain)
    rows = _select_pending_rows(db, client_brand_id, max_pages, force_refetch=force_refetch)
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
        f"rows={len(rows)} throttle={throttle}s force_refetch={force_refetch}"
    )

    fetched = 0
    unchanged = 0
    soft_404 = 0
    errors = 0
    blocked_by_robots = 0
    remaining_retry = 0
    consecutive_errors = 0
    # Internal-link map (Day 3) : source_url -> list[same-host outgoing URLs].
    # Collected during the fetch loop, consumed at the end by
    # compute_inlinks_from_map.
    links_map: dict[str, list[str]] = {}

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

            # Day 3 : pass brand_host to opt into internal-link extraction.
            # On force_refetch we also bypass the conditional-GET so we get
            # a full 200 + HTML body to parse links from (a 304 returns no
            # body and our link map would stay empty for that row).
            meta = fetch_page_meta(
                row.url,
                if_modified_since=None if force_refetch else row.last_crawled_at,
                client=client,
                brand_host=brand_host,
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

            # 304 Not Modified : nothing to update in DB; can't update
            # links_map either (no body returned). Bump last_crawled_at only.
            if meta["http_status"] == 304:
                unchanged += 1
                if (i + 1) % _COMMIT_EVERY == 0:
                    db.commit()
                continue

            # We have a full 200-OK fetch — record the internal links
            # regardless of whether content changed. The Day 3 inlink pass
            # needs them even on unchanged content (the brand's link
            # architecture can shift without the page text changing).
            if meta.get("internal_links") is not None:
                links_map[row.url] = meta["internal_links"]

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

            hash_changed = (row.content_hash != new_hash)

            if (not force_refetch and not hash_changed
                    and row.status in ("fetched", "embedded")):
                # Content unchanged AND we already have it indexed — skip
                # the re-embed cost. Just bumping last_crawled_at. (links_map
                # was already populated above, so the inlink pass still sees
                # this page's outgoing edges.)
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
            # Only flip status to 'fetched' (triggers re-embed) when the
            # content actually changed OR we're filling a pending_fetch row
            # for the first time. On force_refetch with same content the
            # row stays 'embedded' so we don't waste re-embed cost.
            if hash_changed or row.status == "pending_fetch":
                row.status = "fetched"
                fetched += 1
            else:
                unchanged += 1

            if (i + 1) % _COMMIT_EVERY == 0:
                db.commit()

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception(f"final commit failed in fetch_brand_pages for {brand.name}")
        raise

    attempted = fetched + unchanged + soft_404 + errors + blocked_by_robots + remaining_retry

    # Day 3 : compute internal_inlink_count from the links collected during
    # this run. Only fires when we actually have a map (skipped on empty
    # runs to avoid pointlessly zeroing every row).
    inlinks_summary = None
    if links_map:
        try:
            inlinks_summary = compute_inlinks_from_map(
                str(client_brand_id), links_map, db,
            )
        except Exception:
            db.rollback()
            logger.exception(
                f"compute_inlinks_from_map failed for {brand.name} — "
                f"continuing without authority signal"
            )

    logger.info(
        f"fetch_brand_pages done for {brand.name} ({domain}): "
        f"attempted={attempted} fetched={fetched} unchanged={unchanged} "
        f"soft_404={soft_404} errors={errors} "
        f"blocked_by_robots={blocked_by_robots} remaining_retry={remaining_retry} "
        f"inlinks_summary={inlinks_summary}"
    )

    # Day 3 chain : enqueue embed_brand_pages when there are rows in
    # status='fetched' (newly extracted or force-refetched). The embed
    # handler is itself idempotent — it picks up only rows where embedding
    # IS NULL or embedding_model differs from the current constant.
    embed_job_id: str | None = None
    if fetched > 0:
        from models import ClientBrandPage, Job
        # Skip enqueue if a job already in flight for this brand
        in_flight = (
            db.query(Job)
            .filter(
                Job.client_id == brand.client_id,
                Job.job_type == "embed_brand_pages",
                Job.status.in_(("pending", "running")),
                Job.payload["client_brand_id"].astext == str(client_brand_id),
            )
            .first()
        )
        if in_flight:
            embed_job_id = str(in_flight.id)
        else:
            embed_job = Job(
                client_id=brand.client_id,
                job_type="embed_brand_pages",
                status="pending",
                payload={"client_brand_id": str(client_brand_id)},
                max_attempts=2,
            )
            db.add(embed_job)
            db.commit()
            embed_job_id = str(embed_job.id)
        logger.info(
            f"Chained embed_brand_pages job {embed_job_id} for "
            f"client_brand_id={client_brand_id}"
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
        "inlinks_summary": inlinks_summary,
        "chained_embed_job_id": embed_job_id,
    }
