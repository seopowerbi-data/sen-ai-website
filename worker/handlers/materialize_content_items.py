"""Handler: materialize ScanContentItem rows from ScanOpportunity.

Bridge between the scan-analysis pipeline (which writes `ScanOpportunity` rows
keyed on questions) and the content lifecycle Kanban (which operates on
`ScanContentItem` rows keyed on items). Runs at the end of
`generate_opportunities.execute()` so opportunities exist by the time we
materialize.

## What gets materialized

For now, only **FAQ opportunities with priority 'critique' or 'haute'**. This
keeps the Kanban readable (Miller's Law, ~5-10 cards per scan) and aligns
with the Phase B scope (only `generate_faq` handler is wired). Article and
netlinking materialization come later when their handlers ship (Phase C).

## target_url policy (A2 stepping stone — see project_roadmap_content_port.md)

Every materialized ContentItem starts with `target_url = NULL` and
`target_url_source = 'pending_user'`. The user picks the URL on the
validation page before generation can run. The Kanban surfaces this as a
"Needs URL" badge on the card.

`is_competitor_scan(scan, db)` is computed here so the validation page can
show the right banner copy. We don't gate on it for target_url — even on
user-owned scans the system doesn't yet know which page should host the FAQ
(Phase D sitemap index will auto-suggest, but for now the user always picks).

## Idempotency

On rescan, this handler runs again. We dedupe by `(scan_id, content_type,
target_question)` — existing ContentItems are preserved (user may have
already edited them), only NEW questions create new ContentItems. An
opportunity that drops in priority on rescan keeps its old ContentItem; an
opportunity that newly enters 'critique'/'haute' gets a fresh one.
"""

import logging

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# Priority threshold for FAQ materialization. Tied to generate_opportunities.py
# scoring : 'critique' = brand absent + competitors present, 'haute' = cited
# but behind competitor. 'moyenne' opportunities are skipped because the user
# already ranks reasonably and the ROI of producing a FAQ is unclear.
_FAQ_PRIORITIES = ("critique", "haute")


def execute(job_payload: dict, scan_id: str, db: Session) -> dict:
    """Read ScanOpportunity rows + create ScanContentItem rows for FAQ targets."""
    from models import (
        Scan,
        ScanContentItem,
        ScanOpportunity,
        ScanQuestion,
    )
    from services.brand_resolver import is_competitor_scan

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise RuntimeError(f"Scan {scan_id} not found")

    # Read FAQ-eligible opportunities for this scan.
    opps = (
        db.query(ScanOpportunity)
        .filter(
            ScanOpportunity.scan_id == scan_id,
            ScanOpportunity.priority.in_(_FAQ_PRIORITIES),
            ScanOpportunity.recommended_action == "faq",
        )
        .all()
    )
    if not opps:
        logger.info(f"materialize_content_items: 0 FAQ opportunities for scan {scan_id}")
        return {"materialized": 0, "skipped_existing": 0, "is_competitor_scan": False}

    competitor = is_competitor_scan(scan, db)
    logger.info(
        f"materialize_content_items: scan={scan_id}, "
        f"is_competitor={competitor}, eligible_opps={len(opps)}"
    )

    # Pre-load existing FAQ ContentItems for this scan to dedupe by target_question.
    existing = (
        db.query(ScanContentItem)
        .filter(
            ScanContentItem.scan_id == scan_id,
            ScanContentItem.content_type == "faq",
        )
        .all()
    )
    existing_questions = {
        (item.target_question or "").strip().lower() for item in existing if item.target_question
    }

    materialized = 0
    skipped = 0

    for opp in opps:
        question = db.query(ScanQuestion).filter(ScanQuestion.id == opp.question_id).first()
        if not question or not (question.question or "").strip():
            logger.debug(f"materialize: skip opp {opp.id} — no question text")
            continue

        q_key = question.question.strip().lower()
        if q_key in existing_questions:
            skipped += 1
            continue

        db.add(ScanContentItem(
            scan_id=scan_id,
            content_type="faq",
            topic_name=opp.topic_name,
            persona_name=opp.persona_name,
            target_url=None,
            target_url_source="pending_user",
            target_question=question.question.strip(),
            priority=opp.priority,
            opportunity_score=opp.opportunity_score,
            brand_position=opp.brand_position,
            best_competitor=opp.best_competitor_name,
            nb_competitors_cited=opp.nb_competitors_cited,
            status="identified",
        ))
        existing_questions.add(q_key)
        materialized += 1

    db.commit()

    logger.info(
        f"materialize_content_items done: scan={scan_id}, "
        f"materialized={materialized}, skipped_existing={skipped}, "
        f"is_competitor_scan={competitor}"
    )

    return {
        "materialized": materialized,
        "skipped_existing": skipped,
        "is_competitor_scan": competitor,
    }
