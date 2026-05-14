"""Handler: hard-delete client_brand_pages rows that have been marked
'gone' for more than 30 days.

Phase D, Day 3. Final step in the crawl chain :
    crawl_brand_sitemap -> fetch_brand_pages -> embed_brand_pages -> purge_stale_pages

`status='gone'` is set by crawl_brand_sitemap when a previously-known URL
no longer appears in the sitemap. We keep the row around for 30 days so :
  - if it reappears in the sitemap, the diff restores its embedding
    instantly without re-crawling (see crawl_brand_sitemap "restore"
    branch)
  - if the user wants to audit what was lost, the row is still there

After 30 days the row is hard-deleted. We DO honor the partitioned index
`idx_cbp_gone_since (gone_since) WHERE gone_since IS NOT NULL` so the
DELETE stays fast even on large workspaces.

The handler accepts an optional `client_brand_id` payload to scope the
purge to one brand (when fired from the embed chain). Without it the
handler purges across every brand of the originating client — useful for
scheduled cron runs.

Manual rows (source='manual') are never affected — they're excluded from
the sitemap diff in the first place, so they never reach status='gone'.

Payload :
    {
        "client_brand_id": str (optional),
        "client_id": str (optional, used when client_brand_id is absent),
        "max_age_days": int (optional override, default 30),
    }

Returns :
    {
        "purged": int,                # rows hard-deleted
        "client_brand_id": str | None,
        "client_id": str | None,
        "max_age_days": int,
    }
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_DEFAULT_AGE_DAYS = 30


def execute(job_payload: dict, scan_id: str | None, db: Session) -> dict:
    from models import ClientBrand, ClientBrandPage

    payload = job_payload or {}
    client_brand_id = payload.get("client_brand_id")
    client_id = payload.get("client_id")
    age_days = int(payload.get("max_age_days") or _DEFAULT_AGE_DAYS)
    if age_days < 1:
        age_days = _DEFAULT_AGE_DAYS

    cutoff = datetime.utcnow() - timedelta(days=age_days)

    q = (
        db.query(ClientBrandPage)
        .filter(
            ClientBrandPage.status == "gone",
            ClientBrandPage.gone_since.isnot(None),
            ClientBrandPage.gone_since < cutoff,
        )
    )
    if client_brand_id:
        q = q.filter(ClientBrandPage.client_brand_id == client_brand_id)
    elif client_id:
        # Restrict via the brand FK — sub-query keeps a clean join shape.
        brand_ids = (
            db.query(ClientBrand.id)
            .filter(ClientBrand.client_id == client_id)
            .subquery()
        )
        q = q.filter(ClientBrandPage.client_brand_id.in_(brand_ids))

    purged = q.delete(synchronize_session=False)
    db.commit()

    logger.info(
        f"purge_stale_pages: deleted {purged} rows older than {age_days}d "
        f"(client_brand_id={client_brand_id}, client_id={client_id})"
    )

    return {
        "purged": int(purged or 0),
        "client_brand_id": client_brand_id,
        "client_id": client_id,
        "max_age_days": age_days,
    }
