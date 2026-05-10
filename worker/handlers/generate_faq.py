"""Handler: generate FAQ Schema.org content for one ScanContentItem.

Wraps `seo_llm.src.faq_content_generator.FAQContentGenerator._generate_single`
which scrapes the target page, fetches brand context + scientific sources via
OpenAI web_search, then composes Schema.org FAQPage HTML via LLM.

This is the Phase B kickoff — minimum viable wiring of the FAQ pipeline into
the SaaS lifecycle. Reads `ScanContentItem` by id, calls the generator with a
row-like dict (compatible with `row.get(key)` calls), persists the result back
into the same row (`content_html`, `content_text`, status='draft').

Known gaps (deferred to future Phase B sessions):
- **Brand bias via BrandResolver** : the generator uses a hardcoded `BRAND_MAP`
  imported from `seo_llm.src.geo_content_generator`. For Pierre Fabre that map
  already lists Avène/Ducray/etc., so FAQ output naturally promotes their
  brands. For other clients, we'll hook `BrandResolver.resolve_promotion()`
  into `_generate_faq` via monkey-patch or seo_llm injection point.
- **Quality strict toggle (RAPP validator)** : `_compute_quality_score` is
  always called; we don't yet expose the strict pass/fail toggle in the UI.
- **Per-job progress reporting** : the FAQ generator is sync (~60s), no
  intermediate progress events. Long-running variant for Phase C articles.

Refund policy : if generation fails, the worker's poll_and_execute already
flips the job to failed + auto-refunds via `_refund_scan_credits`. FAQ
costs 1 content_credit (per Phase B pricing) — debit happens at API enqueue
time so refund-on-failure works end-to-end.
"""

import json
import logging
import time
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from config import settings

logger = logging.getLogger(__name__)


def execute(job_payload: dict, scan_id: str | None, db: Session) -> dict:
    """Generate FAQ content for one ScanContentItem.

    job_payload must contain: {"item_id": "<uuid>"}
    scan_id is the parent scan, passed by the worker for credit accounting.
    """
    from models import ScanContentItem

    item_id = job_payload.get("item_id")
    if not item_id:
        raise RuntimeError("generate_faq requires item_id in job payload")

    item = db.query(ScanContentItem).filter(ScanContentItem.id == item_id).first()
    if not item:
        raise RuntimeError(f"ScanContentItem {item_id} not found")

    if item.content_type != "faq":
        raise RuntimeError(
            f"ScanContentItem {item_id} is content_type='{item.content_type}', "
            f"not 'faq' — wrong handler"
        )

    if not item.target_url:
        raise RuntimeError(
            f"ScanContentItem {item_id} has no target_url — FAQ generation requires "
            f"a target page to scrape. Set target_url on the opportunity first."
        )

    # Mark as generating so the UI can show a spinner state if it polls
    item.status = "generating"
    db.commit()

    # Lazy import to avoid loading the heavy seo_llm module at worker boot
    from seo_llm.src.faq_content_generator import FAQContentGenerator

    # Build a row-compatible dict — the generator calls `row.get(key)` so a
    # plain dict satisfies the interface (no pandas required).
    row = {
        "target_page_url": item.target_url,
        "target_site": _extract_site(item.target_url),
        "question_text": item.target_question or item.topic_name or "",
        "source_name": item.scan.domain if item.scan else "",
    }

    logger.info(
        f"Generating FAQ for content_item {item_id} "
        f"(target={row['target_page_url']}, scan={scan_id})"
    )

    start = time.time()
    try:
        # Provider hardcoded openai for v1 — Claude alternative exists in seo_llm
        # but adds complexity (separate client). gpt-4.1-mini is the cheap default.
        generator = FAQContentGenerator(
            writing_provider="openai",
            model=settings.task_models.get("generate_faq") if hasattr(settings, "task_models") else None,
            max_workers=1,
        )
        result = generator._generate_single(row)
    except Exception as e:
        # Reset status so user can retry from Kanban (without going through the full
        # _refund_scan_credits path which fires only on attempts >= max_attempts).
        # If this is the LAST attempt, the worker will mark scan failed anyway.
        item.status = "identified"
        db.commit()
        raise RuntimeError(f"FAQ generation failed for item {item_id}: {e}") from e

    duration_ms = int((time.time() - start) * 1000)

    # Persist result on the item
    item.content_html = result.get("faq_html") or None
    item.content_text = result.get("faq_text") or None
    item.status = "draft"  # Awaiting user review

    # Stash sources + quality in a structured payload on content_text? No, we
    # don't have a JSONB column for FAQ metadata. For Phase B we drop them
    # into content_text suffix as commented HTML. Phase C will likely add a
    # `metadata JSONB` column to ScanContentItem for this kind of audit data.
    sources_json = result.get("sources_used", "[]")
    quality_score = result.get("quality_score", 0)
    quality_details = result.get("quality_details", "{}")
    faq_count = result.get("faq_count", 0)

    db.commit()

    # Log LLM usage for cost monitoring (best-effort — calculator may miss some
    # tokens since FAQContentGenerator does multiple LLM calls per FAQ and
    # doesn't return per-call token counts. We log a coarse estimate via the
    # configured model and let billing come from provider invoices for now.)
    try:
        from adapters.llm_logger import log_llm_usage
        log_llm_usage(
            db, provider="openai",
            model=getattr(generator, "model", "gpt-4.1-mini"),
            operation="generate_faq",
            input_tokens=0,  # not surfaced by FAQContentGenerator
            output_tokens=0,
            duration_ms=duration_ms,
            scan_id=scan_id,
            client_id=str(item.scan.client_id) if item.scan else None,
        )
    except Exception:
        logger.warning("log_llm_usage failed for generate_faq", exc_info=True)

    logger.info(
        f"FAQ generated for item {item_id}: {faq_count} Q/R, quality={quality_score}/100, "
        f"sources={sources_json}, {duration_ms}ms"
    )

    return {
        "status": "draft",
        "faq_count": faq_count,
        "quality_score": quality_score,
        "sources_count": len(json.loads(sources_json) if sources_json else []),
        "duration_ms": duration_ms,
    }


def _extract_site(url: str) -> str:
    """Extract bare hostname from a URL (https://www.foo.com/bar → foo.com)."""
    if not url:
        return ""
    s = url.lower().strip()
    if s.startswith("http://"):
        s = s[7:]
    elif s.startswith("https://"):
        s = s[8:]
    if s.startswith("www."):
        s = s[4:]
    s = s.split("/", 1)[0]
    return s
