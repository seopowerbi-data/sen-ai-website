"""Handler: post-scan brand cleanup using Claude.

Phase 1 (scan-as-brand): operates on ScanBrandClassification (SBC) rows
for THIS scan only. Classifies unclassified SBC entries, marks non-brands
as 'ignored', capitalizes catalog names, and links product lines to their
parent brand.
"""

import asyncio
import logging
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from adapters.brand_classifier import classify_brands

logger = logging.getLogger(__name__)


# Map NEW vocabulary (SBC.classification) -> OLD vocabulary (adapter context)
NEW_TO_OLD_CONTEXT = {
    "my_brand": "target_brand",
    "competitor": "competitor",
    "ignored": "ignore",
}

# Map OLD vocabulary (adapter response) -> NEW vocabulary (SBC.classification)
OLD_TO_NEW_RESPONSE = {
    "target_brand": "my_brand",
    "target_gamme": "my_brand",
    "target_product": "my_brand",
    "competitor": "competitor",
    "competitor_gamme": "competitor",
    "ignore": "ignored",
}


def execute(job_payload: dict, scan_id: str, db: Session) -> dict:
    """Clean up unclassified brands for a specific scan (scan-as-brand model)."""
    from models import Scan, ClientBrand, ScanBrandClassification
    from config import settings

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise RuntimeError("Scan not found")

    # 1. Fetch unclassified SBC rows FOR THIS SCAN only
    unclassified_pairs = (
        db.query(ScanBrandClassification, ClientBrand)
        .join(ClientBrand, ClientBrand.id == ScanBrandClassification.brand_id)
        .filter(
            ScanBrandClassification.scan_id == scan_id,
            ScanBrandClassification.classification == "unclassified",
        )
        .all()
    )

    if not unclassified_pairs:
        logger.info("No unclassified brands to clean up for scan %s", scan_id)
        return {"cleaned": 0}

    # 2. Fetch already-classified SBC rows for this scan (for Claude context)
    classified_pairs = (
        db.query(ScanBrandClassification, ClientBrand)
        .join(ClientBrand, ClientBrand.id == ScanBrandClassification.brand_id)
        .filter(
            ScanBrandClassification.scan_id == scan_id,
            ScanBrandClassification.classification != "unclassified",
        )
        .all()
    )

    # Build existing context for Claude — map NEW vocab to OLD vocab
    # (the adapter prompt expects target_brand / competitor / ignore, etc.)
    existing_context = []
    for sbc, brand in classified_pairs:
        old_cat = NEW_TO_OLD_CONTEXT.get(sbc.classification, sbc.classification)
        existing_context.append({"name": brand.name, "category": old_cat})

    # 3. Find the main site brand for this scan
    site_brand_name = None
    if scan.focus_brand_id:
        focus = db.query(ClientBrand).filter(ClientBrand.id == scan.focus_brand_id).first()
        if focus:
            site_brand_name = focus.name
    if not site_brand_name:
        site_brand_name = scan.domain.split(".")[0].title()

    # Map: lowercased original name -> (sbc_row, brand_row)
    unclassified_map = {
        brand.name.lower(): (sbc, brand) for sbc, brand in unclassified_pairs
    }
    unclassified_names = [brand.name for _, brand in unclassified_pairs]

    # 4. Call Claude to classify (with domain brief context if available)
    from adapters.brief_injector import format_brief_context
    classify_result = asyncio.run(
        classify_brands(
            domain=scan.domain,
            site_brand=site_brand_name,
            unclassified=unclassified_names,
            existing=existing_context,
            anthropic_api_key=settings.anthropic_api_key,
            domain_context=format_brief_context(scan.config),
        )
    )
    result = classify_result["brands"]

    from adapters.llm_logger import log_llm_usage
    log_llm_usage(
        db, provider="anthropic",
        model=classify_result.get("model", "unknown"),
        operation="cleanup_brands",
        input_tokens=classify_result.get("input_tokens", 0),
        output_tokens=classify_result.get("output_tokens", 0),
        duration_ms=classify_result.get("duration_ms"),
        scan_id=scan_id, client_id=str(scan.client_id),
    )

    # 5 + 6 + 7 + 8 + 9. Apply classifications to SBC rows
    classified_count = 0
    ignored_count = 0
    orphaned_brands = 0
    now = datetime.utcnow()

    for item in result:
        original = item.get("original", "") or ""
        pair = unclassified_map.get(original.lower())
        if not pair:
            continue
        sbc, brand_obj = pair

        raw_category = item.get("category", "ignore")

        # Map OLD vocab (adapter) -> NEW vocab (SBC)
        new_classification = OLD_TO_NEW_RESPONSE.get(raw_category, "unclassified")

        if new_classification == "ignored":
            # Do NOT delete the client_brand row — it may be shared with other scans.
            # Just mark this scan's SBC row as ignored.
            sbc.classification = "ignored"
            sbc.classified_by = "claude"
            sbc.updated_at = now
            ignored_count += 1
            continue

        if new_classification == "unclassified":
            # Defensive: unknown category from adapter — leave SBC as-is
            continue

        # Update catalog name with proper capitalization
        new_name = item.get("name") or brand_obj.name

        # 8. Dedup: if another client_brand already has this name for the same client,
        # re-point SBC to the existing brand and flag the current row as orphaned.
        existing_dup = (
            db.query(ClientBrand)
            .filter(
                ClientBrand.client_id == scan.client_id,
                func.lower(ClientBrand.name) == new_name.lower(),
                ClientBrand.id != brand_obj.id,
            )
            .first()
        )

        if existing_dup:
            # Check if a canonical SBC row already exists for this scan — would cause
            # a UNIQUE(scan_id, brand_id) violation if we re-point blindly.
            canonical_sbc = (
                db.query(ScanBrandClassification)
                .filter(
                    ScanBrandClassification.scan_id == scan_id,
                    ScanBrandClassification.brand_id == existing_dup.id,
                )
                .first()
            )
            if canonical_sbc:
                # Canonical SBC row already exists — upgrade its classification from
                # the Claude result (if it was unclassified or lower-confidence) and
                # delete the losing SBC row. The losing ClientBrand row stays orphaned.
                if canonical_sbc.classification == "unclassified":
                    canonical_sbc.classification = new_classification
                canonical_sbc.classified_by = "claude"
                canonical_sbc.updated_at = now
                db.delete(sbc)
                target_brand_for_parent = existing_dup
            else:
                # No canonical SBC yet — safe to re-point the losing SBC to the canonical brand.
                sbc.brand_id = existing_dup.id
                sbc.classification = new_classification
                sbc.classified_by = "claude"
                sbc.updated_at = now
                target_brand_for_parent = existing_dup
            orphaned_brands += 1
        else:
            # Update the catalog entry in place (name only — category is deprecated)
            brand_obj.name = new_name
            brand_obj.canonical_name = new_name
            sbc.classification = new_classification
            sbc.classified_by = "claude"
            sbc.updated_at = now
            target_brand_for_parent = brand_obj

        # 9. Parent linking for product lines (gammes)
        parent_name = item.get("parent")
        if parent_name and raw_category in ("target_gamme", "competitor_gamme"):
            parent = (
                db.query(ClientBrand)
                .filter(
                    ClientBrand.client_id == scan.client_id,
                    func.lower(ClientBrand.name) == parent_name.lower(),
                )
                .first()
            )
            if parent and parent.id != target_brand_for_parent.id:
                target_brand_for_parent.parent_id = parent.id

        classified_count += 1

    db.commit()
    logger.info(
        f"Brand cleanup (scan {scan_id}): {classified_count} classified, "
        f"{ignored_count} ignored, {orphaned_brands} orphaned"
    )
    return {
        "classified": classified_count,
        "ignored": ignored_count,
        "orphaned_brands": orphaned_brands,
    }
