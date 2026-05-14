"""Handler: embed every status='fetched' page for a brand into JSONB.

Phase D, Day 3. Picks up where fetch_brand_pages left off : rows in
status='fetched' that have NULL embedding OR an embedding_model different
from the current `services.embeddings.EMBEDDING_MODEL`. Batches 100 texts
per OpenAI call (single round-trip cost ~$0.001), persists embedding +
embedding_model + status='embedded', logs cost to LlmUsageLog.

The per-client daily cost cap ($1/day default — see
services.embeddings.DAILY_COST_CAP_USD) is checked BEFORE each batch.
When the cap is reached mid-run the handler commits what's done and
returns `status='cost_capped'` so the UI can surface a banner. The
remaining rows stay in status='fetched' and a future job-run picks them
back up after the UTC daily reset.

Chains :
    fetch_brand_pages -> embed_brand_pages -> purge_stale_pages

Resumability : re-running the handler is safe — the filter on `embedding
IS NULL OR embedding_model != current` means only un-embedded rows are
touched on each pass.

Payload :
    {
        "client_brand_id": str,
        "max_pages": int (optional cap for smoke testing),
    }

Returns :
    {
        "client_brand_id": str, "domain": str | None,
        "status": "ok" | "skipped" | "cost_capped",
        "reason": str | None,
        "attempted": int,
        "embedded": int,
        "batches": int,
        "tokens_total": int,
        "cost_usd_total": float,
        "today_cost_after": float,
        "errors": int,
    }
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.orm import Session

from adapters.llm_logger import log_llm_usage
from config import settings
from services.embeddings import (
    BATCH_SIZE,
    DAILY_COST_CAP_USD,
    EMBEDDING_MODEL,
    build_embed_text,
    embed_batch,
    get_today_embed_cost,
)

logger = logging.getLogger(__name__)


def _pending_embed_rows(db: Session, client_brand_id: str, limit: int | None):
    """Status='fetched' is the source of truth — fetch_brand_pages flips
    a row back to 'fetched' whenever content changed, even if a stale
    embedding from a previous run is still on the row. So we DON'T filter
    on `embedding IS NULL` here : that would silently skip the re-embed
    of a content-updated page that still carries its old vector."""
    from models import ClientBrandPage
    q = (
        db.query(ClientBrandPage)
        .filter(
            ClientBrandPage.client_brand_id == client_brand_id,
            ClientBrandPage.status == "fetched",
        )
        .order_by(ClientBrandPage.first_seen_at.asc())
    )
    if limit:
        q = q.limit(int(limit))
    return q.all()


def execute(job_payload: dict, scan_id: str | None, db: Session) -> dict:
    from models import ClientBrand

    client_brand_id = (job_payload or {}).get("client_brand_id")
    if not client_brand_id:
        raise ValueError("embed_brand_pages requires client_brand_id")

    max_pages = (job_payload or {}).get("max_pages")

    brand = db.query(ClientBrand).filter(ClientBrand.id == client_brand_id).first()
    if not brand:
        raise ValueError(f"ClientBrand {client_brand_id} not found")

    client_id = str(brand.client_id)
    domain = brand.domain

    if not settings.openai_api_key:
        logger.warning(
            "OPENAI_API_KEY not configured — cannot run embed_brand_pages"
        )
        return {
            "client_brand_id": str(client_brand_id), "domain": domain,
            "status": "skipped", "reason": "no_api_key",
            "attempted": 0, "embedded": 0, "batches": 0,
            "tokens_total": 0, "cost_usd_total": 0.0,
            "today_cost_after": 0.0, "errors": 0,
        }

    rows = _pending_embed_rows(db, client_brand_id, max_pages)
    if not rows:
        logger.info(
            f"embed_brand_pages: no rows pending for {brand.name} "
            f"({domain}) — model={EMBEDDING_MODEL}"
        )
        return {
            "client_brand_id": str(client_brand_id), "domain": domain,
            "status": "ok", "reason": "nothing_to_embed",
            "attempted": 0, "embedded": 0, "batches": 0,
            "tokens_total": 0, "cost_usd_total": 0.0,
            "today_cost_after": 0.0, "errors": 0,
        }

    today_cost = get_today_embed_cost(client_id, db)
    if today_cost >= DAILY_COST_CAP_USD:
        logger.warning(
            f"embed_brand_pages: client {client_id} already at "
            f"${today_cost:.4f} today (cap ${DAILY_COST_CAP_USD}) — "
            f"skipping {len(rows)} rows until UTC reset"
        )
        return {
            "client_brand_id": str(client_brand_id), "domain": domain,
            "status": "cost_capped", "reason": "daily_cap_reached",
            "attempted": 0, "embedded": 0, "batches": 0,
            "tokens_total": 0, "cost_usd_total": 0.0,
            "today_cost_after": today_cost, "errors": 0,
        }

    logger.info(
        f"embed_brand_pages start: {brand.name} ({domain}) "
        f"rows={len(rows)} model={EMBEDDING_MODEL} "
        f"today_cost=${today_cost:.4f} cap=${DAILY_COST_CAP_USD}"
    )

    embedded = 0
    batches = 0
    tokens_total = 0
    cost_total = 0.0
    errors = 0
    cost_capped = False

    for batch_start in range(0, len(rows), BATCH_SIZE):
        if today_cost + cost_total >= DAILY_COST_CAP_USD:
            cost_capped = True
            logger.warning(
                f"embed_brand_pages: hit daily cap mid-run "
                f"(today=${today_cost + cost_total:.4f}, cap=${DAILY_COST_CAP_USD}) — "
                f"committing partial progress"
            )
            break

        batch = rows[batch_start:batch_start + BATCH_SIZE]
        texts = [
            build_embed_text(r.title, r.meta_description, r.h1, r.body_excerpt)
            for r in batch
        ]

        try:
            result = embed_batch(texts, openai_api_key=settings.openai_api_key)
        except Exception as exc:
            errors += len(batch)
            logger.exception(
                f"embed_batch failed for {brand.name} batch={batches} "
                f"size={len(batch)} — flipping rows to status='error'"
            )
            now = datetime.utcnow()
            for r in batch:
                r.status = "error"
                r.fetch_error = f"embed_error: {type(exc).__name__}"
                r.last_crawled_at = now
            db.commit()
            continue

        now = datetime.utcnow()
        for row, vec in zip(batch, result["embeddings"]):
            row.embedding = vec
            row.embedding_model = result["model"]
            row.status = "embedded"
            row.last_embedded_at = now
            row.fetch_error = None
            flag_modified(row, "embedding")
        embedded += len(batch)
        batches += 1
        tokens_total += result["tokens"]
        cost_total += result["cost_usd"]

        # Log the API usage. Operation prefix 'embed_' is the convention
        # the daily-cap query looks for in services.embeddings.get_today_embed_cost.
        log_llm_usage(
            db,
            provider="openai",
            model=result["model"],
            operation="embed_brand_pages",
            input_tokens=result["tokens"],
            output_tokens=0,
            cost_usd=result["cost_usd"],
            duration_ms=result["duration_ms"],
            client_id=client_id,
        )
        db.commit()

        logger.info(
            f"embed_brand_pages batch {batches}: rows={len(batch)} "
            f"tokens={result['tokens']} cost=${result['cost_usd']:.5f} "
            f"duration={result['duration_ms']}ms"
        )

    today_cost_after = today_cost + cost_total
    final_status = "cost_capped" if cost_capped else "ok"

    # Chain purge_stale_pages — cheap idempotent finalize step. Runs even on
    # cost_capped to give the 30-day TTL its tick.
    from models import Job
    in_flight_purge = (
        db.query(Job)
        .filter(
            Job.client_id == client_id,
            Job.job_type == "purge_stale_pages",
            Job.status.in_(("pending", "running")),
            Job.payload["client_brand_id"].astext == str(client_brand_id),
        )
        .first()
    )
    purge_job_id = None
    if not in_flight_purge:
        purge_job = Job(
            client_id=client_id,
            job_type="purge_stale_pages",
            status="pending",
            payload={"client_brand_id": str(client_brand_id)},
            max_attempts=2,
        )
        db.add(purge_job)
        db.commit()
        purge_job_id = str(purge_job.id)
        logger.info(
            f"Chained purge_stale_pages job {purge_job_id} for "
            f"client_brand_id={client_brand_id}"
        )

    return {
        "client_brand_id": str(client_brand_id), "domain": domain,
        "status": final_status,
        "reason": "daily_cap_reached" if cost_capped else None,
        "attempted": embedded + errors,
        "embedded": embedded,
        "batches": batches,
        "tokens_total": tokens_total,
        "cost_usd_total": round(cost_total, 6),
        "today_cost_after": round(today_cost_after, 6),
        "errors": errors,
        "chained_purge_job_id": purge_job_id,
    }
