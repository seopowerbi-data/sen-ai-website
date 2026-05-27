"""Handler : Sprint 6 schema.org / JSON-LD audit + generator.

Source of URLs : same as Sprint 5 - the user's own pages that the LLMs cited
during the scan (`scan_llm_results.citations` where `est_site_cible=true`).
These are the highest-leverage pages to optimise for AI extraction.

For each URL we :
  1. Fetch the page HTML (shared adapters/page_fetcher).
  2. Extract every existing ``<script type="application/ld+json">`` block,
     validate it against the schema.org required-property spec.
  3. Detect the page type from URL path + on-page signals.
  4. Generate the missing blocks (Organization, WebSite, Article, Product,
     FAQPage, BreadcrumbList) from brand_brief data + on-page microdata.
  5. Compute a composite 0-100 score so the UI can rank worst-first.
  6. Upsert in scan_schema_audits (one row per scan_id + url).

Idempotent. Free (no LLM call). The user explicitly clicks "Refresh" to
recompute when they ship new structured data.
"""
from __future__ import annotations

import logging
import time
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from adapters.page_fetcher import fetch_page
from adapters.schema_extractor import extract as extract_blocks, has_type
from adapters.schema_generator import (
    detect_page_type,
    expected_schemas,
    generate as generate_blocks,
)

logger = logging.getLogger(__name__)

PAGE_DELAY_SECONDS = 0.4
MAX_URLS_PER_RUN = 400
MIN_CITATION_COUNT = 1

# Score weights. See migration 048 header for the rationale.
W_ORG = 25
W_PRIMARY = 25
W_BREADCRUMB = 20
W_VALID_PCT = 20
W_WEBSITE = 10


def _cited_urls(db: Session, scan_id: str, limit: int) -> list[tuple[str, int]]:
    """Same projection as Sprint 5 audit. Top-cited URLs of the user's own
    site, grouped by URL with their citation count."""
    from sqlalchemy import text as _text

    sql = _text(
        """
        SELECT citation->>'url' AS url, COUNT(*)::int AS n
          FROM scan_llm_results slr,
               LATERAL jsonb_array_elements(slr.citations) AS citation
         WHERE slr.scan_id = :scan_id
           AND (citation->>'est_site_cible')::bool = true
           AND citation->>'url' IS NOT NULL
         GROUP BY citation->>'url'
        HAVING COUNT(*) >= :min_cnt
         ORDER BY n DESC
         LIMIT :lim
        """
    )
    rows = db.execute(
        sql,
        {"scan_id": scan_id, "min_cnt": MIN_CITATION_COUNT, "lim": limit},
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _load_focus_brief(db: Session, scan_id: str) -> dict:
    """Return the focus brand's brief JSONB merged with a sensible fallback.
    Empty dict if no brand is attached."""
    from models import ClientBrand, ScanBrandClassification

    row = (
        db.query(ClientBrand)
        .join(ScanBrandClassification, ScanBrandClassification.brand_id == ClientBrand.id)
        .filter(ScanBrandClassification.scan_id == scan_id)
        .filter(ScanBrandClassification.is_focus.is_(True))
        .first()
    )
    if not row:
        return {}
    brief = dict(row.brief or {})
    # Keep a name fallback for the generator.
    brief.setdefault("name", row.name)
    return brief


def _score(page_type: str, existing: list[dict], expected: list[str]) -> int:
    """Composite 0-100 score per the migration header. Skips weights that
    don't apply to the page type (BreadcrumbList on homepage, etc.)."""
    weights_used = 0
    earned = 0

    # Organization is universal (it's a site-level signal, but Google rewards
    # it on every page that has it).
    weights_used += W_ORG
    if has_type(existing, "Organization"):
        earned += W_ORG

    # Page-type primary schema.
    primary_map = {
        "article": "Article",
        "product": "Product",
        "faq":     "FAQPage",
        "homepage": None,    # already counted via WebSite + Organization
        "about":    None,
        "other":    None,
    }
    primary = primary_map.get(page_type)
    if primary:
        weights_used += W_PRIMARY
        if has_type(existing, primary):
            earned += W_PRIMARY

    # BreadcrumbList - only graded when expected for this page.
    if "BreadcrumbList" in expected:
        weights_used += W_BREADCRUMB
        if has_type(existing, "BreadcrumbList"):
            earned += W_BREADCRUMB

    # Validity ratio of existing blocks.
    if existing:
        weights_used += W_VALID_PCT
        valid_count = sum(1 for b in existing if b.get("valid"))
        earned += int(W_VALID_PCT * valid_count / len(existing))

    # WebSite (homepage only).
    if page_type == "homepage":
        weights_used += W_WEBSITE
        if has_type(existing, "WebSite"):
            earned += W_WEBSITE

    if weights_used == 0:
        return 0
    return max(0, min(100, round(100 * earned / weights_used)))


def execute(job_payload: dict, scan_id: str, db: Session) -> dict:
    """Audit the cited pages' structured data and generate the missing blocks.

    job_payload :
      - limit (int) : cap the number of URLs audited (default MAX_URLS_PER_RUN)
      - reset (bool) : drop existing rows for this scan before re-running
    """
    from models import Scan, ScanSchemaAudit

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise RuntimeError("Scan not found")

    limit = int(job_payload.get("limit") or MAX_URLS_PER_RUN)
    reset = bool(job_payload.get("reset"))

    if reset:
        db.query(ScanSchemaAudit).filter(ScanSchemaAudit.scan_id == scan_id).delete()
        db.commit()

    pairs = _cited_urls(db, scan_id, limit)
    if not pairs:
        logger.info(f"audit_scan_schemas: no cited URLs for scan {scan_id}")
        return {"audited": 0, "errors": 0, "skipped": 0, "total": 0}

    brief = _load_focus_brief(db, scan_id)
    # Stash the scan domain so the generator can fall back when the brief is
    # sparse (e.g. fresh client without a brand_brief yet).
    brief["_scan_domain"] = scan.domain or ""

    audited = 0
    errors = 0
    skipped = 0

    for idx, (url, cited_count) in enumerate(pairs):
        if not url or not url.startswith(("http://", "https://")):
            skipped += 1
            continue

        fetched = fetch_page(url)
        status = fetched["status"]
        err = fetched["error"]
        html = fetched["html"]

        existing: list[dict] = []
        missing: list[str] = []
        generated: dict[str, dict] = {}
        page_type = "other"
        title = None
        score = None

        if html and not err:
            try:
                soup = BeautifulSoup(html, "html.parser")
                # Page title for the UI.
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()[:300]

                page_type = detect_page_type(url, html, soup)
                existing = extract_blocks(html)
                expected = expected_schemas(page_type, url)

                # Missing = expected types we didn't find as valid blocks.
                missing = [t for t in expected if not has_type(existing, t)]

                # Generate replacements for the missing ones.
                if missing:
                    generated_all = generate_blocks(page_type, html, url, brief, soup=soup)
                    generated = {k: v for k, v in generated_all.items() if k in missing}

                score = _score(page_type, existing, expected)
            except Exception:  # noqa: BLE001
                logger.exception(f"schema audit failed for {url}")
                errors += 1
                err = err or "analyze_error"
        else:
            errors += 1

        existing_row = (
            db.query(ScanSchemaAudit)
            .filter(ScanSchemaAudit.scan_id == scan_id, ScanSchemaAudit.url == url)
            .first()
        )
        if existing_row:
            existing_row.title = title or existing_row.title
            existing_row.page_type = page_type
            existing_row.fetch_status = status
            existing_row.fetch_error = err
            existing_row.existing_schemas = existing
            existing_row.missing_schemas = missing
            existing_row.generated_blocks = generated
            existing_row.schema_score = score
            existing_row.citation_count = cited_count
        else:
            db.add(ScanSchemaAudit(
                scan_id=scan_id,
                url=url,
                title=title,
                page_type=page_type,
                fetch_status=status,
                fetch_error=err,
                existing_schemas=existing,
                missing_schemas=missing,
                generated_blocks=generated,
                schema_score=score,
                citation_count=cited_count,
            ))

        audited += 1
        if audited % 10 == 0:
            db.commit()
            logger.info(f"schema audit progress {audited}/{len(pairs)}")

        time.sleep(PAGE_DELAY_SECONDS)

    db.commit()
    logger.info(
        f"schema audit complete : audited={audited} errors={errors} skipped={skipped}"
    )
    return {
        "audited": audited,
        "errors": errors,
        "skipped": skipped,
        "total": len(pairs),
    }
