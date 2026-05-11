"""Handler: compute opportunity scores from scan results.

Inspired by seo-llm/src/faq_opportunity.py compute_faq_opportunities().

CRITIQUE: brand absent + competitor present → create FAQ/netlinking
HAUTE: brand cited but behind competitor → improve content
MOYENNE: brand well positioned or no competition → maintain
"""

import logging
from datetime import datetime

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def execute(job_payload: dict, scan_id: str, db: Session) -> dict:
    """Score each test result as an opportunity."""
    from models import Scan, ScanLLMResult, ScanQuestion, ScanPersona, ScanTopic, ScanOpportunity

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise RuntimeError("Scan not found")

    results = db.query(ScanLLMResult).filter(ScanLLMResult.scan_id == scan_id).all()
    if not results:
        return {"opportunities": 0}

    # Clear previous opportunities
    db.query(ScanOpportunity).filter(ScanOpportunity.scan_id == scan_id).delete()

    personas = {str(p.id): p for p in db.query(ScanPersona).filter(ScanPersona.scan_id == scan_id).all()}
    topics = {str(t.id): t for t in db.query(ScanTopic).filter(ScanTopic.scan_id == scan_id).all()}

    counts = {"critique": 0, "haute": 0, "moyenne": 0}

    for r in results:
        q = db.query(ScanQuestion).filter(ScanQuestion.id == r.question_id).first()
        if not q:
            continue

        persona = personas.get(str(q.persona_id))
        topic = topics.get(str(persona.topic_id)) if persona and persona.topic_id else None

        # Brand analysis
        brand_analysis = r.brand_analysis or {}
        brand_cited = r.target_cited or brand_analysis.get("marque_cible_mentionnee", False)
        brand_position = r.target_position or brand_analysis.get("position_marque_cible")
        brand_sentiment = brand_analysis.get("sentiment_marque_cible")
        brand_recommended = brand_analysis.get("recommandation_marque_cible", False)

        # Competitor analysis
        competitor_domains = r.competitor_domains or {}
        nb_competitors = len(competitor_domains)
        best_competitor = None
        best_competitor_pos = None
        best_competitor_domain = None

        # Find best competitor from brand_mentions
        for mention in (r.brand_mentions or []):
            if not mention.get("est_marque_cible") and mention.get("position_index"):
                if best_competitor_pos is None or mention["position_index"] < best_competitor_pos:
                    best_competitor = mention.get("brand_name_groupby") or mention.get("brand_name")
                    best_competitor_pos = mention["position_index"]

        # If no brand mentions, use competitor domains
        if not best_competitor and competitor_domains:
            top_domain = max(competitor_domains, key=competitor_domains.get)
            best_competitor = top_domain
            best_competitor_domain = top_domain

        # Score opportunity
        priority, score = _compute_priority(
            brand_cited=brand_cited,
            brand_position=brand_position,
            nb_competitors=nb_competitors,
            best_competitor_pos=best_competitor_pos,
        )

        if priority:
            # Determine recommended action
            if priority == "critique":
                action = "faq" if not best_competitor_domain else "netlinking"
            elif priority == "haute":
                action = "content_update"
            else:
                action = None

            db.add(ScanOpportunity(
                scan_id=scan_id,
                question_id=q.id,
                topic_name=topic.name if topic else None,
                persona_name=persona.name if persona else None,
                brand_cited=brand_cited,
                brand_position=brand_position,
                brand_sentiment=brand_sentiment,
                brand_recommended=brand_recommended,
                best_competitor_name=best_competitor,
                best_competitor_position=best_competitor_pos,
                best_competitor_domain=best_competitor_domain,
                nb_competitors_cited=nb_competitors,
                priority=priority,
                opportunity_score=score,
                recommended_action=action,
            ))
            counts[priority] += 1

    # Update scan summary with opportunity counts
    from sqlalchemy.orm.attributes import flag_modified
    summary = dict(scan.summary or {})
    summary["opportunities"] = counts
    scan.summary = summary
    flag_modified(scan, "summary")
    scan.updated_at = datetime.utcnow()
    db.commit()

    total = sum(counts.values())

    # Bridge: materialize ScanContentItem rows from the FAQ-eligible opportunities
    # we just wrote, so the Content Kanban gets populated automatically. Runs
    # after this handler completes (FIFO queue), reads ScanOpportunity rows.
    # Skip the enqueue if no opportunities qualify — saves a no-op job.
    if counts.get("critique", 0) + counts.get("haute", 0) > 0:
        from models import Job
        db.add(Job(scan_id=scan_id, job_type="materialize_content_items"))
        db.commit()
        logger.info(f"Enqueued materialize_content_items for scan {scan_id}")

    logger.info(f"Generated {total} opportunities: {counts}")
    return {"total": total, **counts}


def _compute_priority(brand_cited, brand_position, nb_competitors, best_competitor_pos):
    """Score an opportunity based on brand vs competitor positioning.

    Adapted from seo-llm/src/faq_opportunity.py _compute_faq_priority_and_score().
    """
    if not brand_cited and nb_competitors > 0:
        # CRITIQUE: absent but competitors present
        return "critique", 80 + min(nb_competitors * 5, 20)

    if not brand_cited and nb_competitors == 0:
        # MOYENNE: nobody cited, opportunity to take the space
        return "moyenne", 30

    if brand_cited and best_competitor_pos and brand_position:
        if best_competitor_pos < brand_position:
            # HAUTE: cited but behind competitor
            gap = brand_position - best_competitor_pos
            return "haute", 50 + min(gap * 10, 30)

    if brand_cited and brand_position and brand_position <= 2:
        # Well positioned, not an opportunity
        return None, 0

    if brand_cited:
        # Cited but could be better
        return "moyenne", 20

    return None, 0
