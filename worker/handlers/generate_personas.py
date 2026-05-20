"""Handler: generate personas + questions from validated topics.

Phase 2: merged prompt (personas + questions in 1 Claude call per topic),
parallelized across topics via asyncio.gather. Each persona gets 15 questions
(5 types × 3 each — canonical balance).
"""

import asyncio
import logging
from datetime import datetime

from sqlalchemy.orm import Session

from adapters.persona_generator import generate_all_topics, QUESTIONS_PER_PERSONA

logger = logging.getLogger(__name__)

# Default: 3 personas per topic (Balanced preset).
# To be overridden by scan.config["personas_per_topic"] when preset UI is implemented.
DEFAULT_PERSONAS_PER_TOPIC = 3


def execute(job_payload: dict, scan_id: str, db: Session) -> dict:
    """Generate personas + questions for all active topics (parallel, merged)."""
    from models import Scan, ScanTopic, ScanKeyword, ScanPersona, ScanQuestion
    from config import settings

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise RuntimeError("Scan not found")

    # Get config
    scan_config = scan.config or {}
    nb_personas = job_payload.get(
        "nb_personas",
        scan_config.get("personas_per_topic", DEFAULT_PERSONAS_PER_TOPIC),
    )

    # Load active topics with their keywords
    topics = db.query(ScanTopic).filter(
        ScanTopic.scan_id == scan_id,
        ScanTopic.is_active == True,
    ).order_by(ScanTopic.keyword_count.desc()).all()

    if not topics:
        raise RuntimeError("No active topics found")

    # Build topics_with_keywords for the generator
    topics_with_keywords = []
    for topic in topics:
        kws = db.query(ScanKeyword).filter(ScanKeyword.topic_id == topic.id).all()
        if kws:
            topics_with_keywords.append({
                "name": topic.name,
                "keywords": [
                    {"keyword": k.keyword, "traffic": k.traffic, "position": k.position}
                    for k in kws
                ],
            })

    if not topics_with_keywords:
        raise RuntimeError("No topics with keywords found")

    # --- Generate personas + questions (1 Claude call per topic, all in parallel) ---
    scan.progress_message = f"Generating {nb_personas} personas per topic (parallel)..."
    scan.progress_pct = 10
    db.commit()

    from adapters.brief_injector import format_analysis_context
    from models import Client as _Client, ClientBrand as _ClientBrand
    _client = db.query(_Client).filter(_Client.id == scan.client_id).first()
    # Phase BB : when scan has a focus brand with a generated BrandBrief,
    # personas inherit brand-specific target_audience + audience_segments
    # via format_analysis_context → format_workspace_brief 2-level merge.
    _focus_brief: dict | None = None
    if scan.focus_brand_id:
        _fb = db.query(_ClientBrand).filter(_ClientBrand.id == scan.focus_brand_id).first()
        if _fb and _fb.brief:
            _focus_brief = _fb.brief
    result = asyncio.run(generate_all_topics(
        domain=scan.domain,
        topics_with_keywords=topics_with_keywords,
        nb_personas=nb_personas,
        anthropic_api_key=settings.anthropic_api_key,
        # BB.9 audience-only render : strip brand voice fields (editorial_voice,
        # tone_dos/donts, brand_story, heritage, taglines, claims_guidelines)
        # from the persona prompt so Claude doesn't mirror them into persona
        # personality descriptions. Personas describe AUDIENCE, not brand voice.
        # See project_phase_brand_briefs.md foot-gun #18.
        domain_context=format_analysis_context(
            scan.config,
            _client.apps if _client else None,
            _focus_brief,
            audience_only=True,
        ),
    ))

    # Log LLM usage for cost monitoring (aggregate across all topic calls)
    from adapters.llm_logger import log_llm_usage
    for topic_result in result.get("topics", []):
        if topic_result.get("error"):
            log_llm_usage(
                db, provider="anthropic",
                model=topic_result.get("model", "claude-haiku-4-5-20251001"),
                operation="generate_personas", error=True,
                scan_id=scan_id, client_id=str(scan.client_id),
            )
        else:
            log_llm_usage(
                db, provider="anthropic",
                model=topic_result.get("model", "claude-haiku-4-5-20251001"),
                operation="generate_personas",
                input_tokens=topic_result.get("input_tokens", 0),
                output_tokens=topic_result.get("output_tokens", 0),
                duration_ms=topic_result.get("duration_ms"),
                scan_id=scan_id, client_id=str(scan.client_id),
            )

    # --- Store in DB ---
    scan.progress_message = "Saving personas and questions..."
    scan.progress_pct = 80
    db.commit()

    # Clear previous personas + questions (idempotent if re-run)
    db.query(ScanPersona).filter(ScanPersona.scan_id == scan_id).delete()
    db.query(ScanQuestion).filter(ScanQuestion.scan_id == scan_id).delete()

    # Map topic name → topic id (fuzzy match same as before)
    topic_name_to_id = {t.name.lower(): t.id for t in topics}

    def _match_topic(segment_principal: str):
        sp = (segment_principal or "").strip().lower()
        if not sp:
            return None
        tid = topic_name_to_id.get(sp)
        if tid:
            return tid
        for tname, tid in topic_name_to_id.items():
            if sp in tname or tname in sp:
                return tid
        sp_words = {w for w in sp.split() if len(w) > 2}
        for tname, tid in topic_name_to_id.items():
            t_words = {w for w in tname.split() if len(w) > 2}
            if len(sp_words & t_words) >= 2:
                return tid
        return None

    stored_personas = 0
    stored_questions = 0

    for topic_result in result.get("topics", []):
        if topic_result.get("error"):
            logger.warning(f"Skipping failed topic: {topic_result.get('topic_name')}")
            continue

        for p in topic_result.get("personas", []):
            topic_id = _match_topic(p.get("segment_principal"))

            persona = ScanPersona(
                scan_id=scan_id,
                topic_id=topic_id,
                name=p.get("nom", "Unknown"),
                data=p,
                is_active=True,
            )
            db.add(persona)
            db.flush()  # Get persona.id
            stored_personas += 1

            # Store questions (embedded in the persona JSON from the merged prompt).
            # Sprint P (migration 036): also materialize the 3 per-question fields
            # (intention_cachee, signal_positif, signal_negatif) as columns so the
            # API serializer + Sprint J judge can read them directly without the
            # fragile JSONB text-lookup join. The JSONB blob on persona.data is
            # preserved (transition Sprint P : double source of truth for 1 release).
            for q in p.get("questions", []):
                db.add(ScanQuestion(
                    scan_id=scan_id,
                    persona_id=persona.id,
                    question=q.get("question", ""),
                    type_question=q.get("type_question"),
                    is_active=True,
                    intention_cachee=(q.get("intention_cachee") or "").strip() or None,
                    signal_positif=(q.get("signal_positif") or "").strip() or None,
                    signal_negatif=(q.get("signal_negatif") or "").strip() or None,
                ))
                stored_questions += 1

    # Aggregate per-topic warnings (B.1 type inference + B.2 coherence checks)
    # Stored in scan.summary["warnings"] for UI surfacing on Personas page.
    # Replaces any prior warnings since this handler regenerates all personas.
    all_warnings = []
    for topic_result in result.get("topics", []):
        all_warnings.extend(topic_result.get("warnings", []) or [])

    # Angle B: enrich each warning with priority based on its topic's traffic share.
    # `traffic` already accounts for position via CTR (HaloScan-computed), so weighting
    # by traffic alone gives us the right "business value lost" signal — questions on
    # a high-share topic matter more than on a niche topic.
    topic_traffic_map = {
        t["name"]: sum((k.get("traffic") or 0) for k in t["keywords"])
        for t in topics_with_keywords
    }
    scan_total_traffic = sum(topic_traffic_map.values()) or 1
    for w in all_warnings:
        topic = w.get("topic", "")
        topic_t = topic_traffic_map.get(topic, 0)
        share = topic_t / scan_total_traffic
        if share >= 0.20:
            w["priority"] = "critical"
        elif share >= 0.05:
            w["priority"] = "medium"
        else:
            w["priority"] = "low"
        w["topic_traffic"] = topic_t
        w["topic_share_pct"] = round(share * 100, 1)

    summary = dict(scan.summary or {})
    summary["warnings"] = all_warnings
    scan.summary = summary
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(scan, "summary")

    # Update scan status
    scan.status = "personas_ready"
    scan.progress_pct = 100
    scan.progress_message = f"{stored_personas} personas, {stored_questions} questions"
    scan.updated_at = datetime.utcnow()
    db.commit()

    logger.info(
        f"Generated {stored_personas} personas + {stored_questions} questions "
        f"in {result.get('duration_ms')}ms (parallel) — {len(all_warnings)} warnings"
    )

    return {
        "personas_count": stored_personas,
        "questions_count": stored_questions,
        "questions_per_persona": QUESTIONS_PER_PERSONA,
        "warnings_count": len(all_warnings),
        "duration_ms": result.get("duration_ms"),
        "total_tokens": result.get("input_tokens", 0) + result.get("output_tokens", 0),
    }
