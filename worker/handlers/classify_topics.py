"""Handler: classify URLs into topics using Claude, then assign keywords via their URL."""

import asyncio
import logging
from datetime import datetime

from sqlalchemy.orm import Session

from adapters.topic_classifier import classify_urls_into_topics

logger = logging.getLogger(__name__)


def execute(job_payload: dict, scan_id: str, db: Session) -> dict:
    """Classify URLs into topics, then assign keywords to topics via URL."""
    from models import ScanKeyword, ScanTopic, Scan
    from config import settings

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise RuntimeError("Scan not found")

    keywords = db.query(ScanKeyword).filter(ScanKeyword.scan_id == scan_id).all()
    if not keywords:
        raise RuntimeError("No keywords found")

    # Prepare data
    kw_data = [
        {"url": k.url, "keyword": k.keyword, "position": k.position, "traffic": k.traffic}
        for k in keywords
    ]

    scan.progress_message = "Classification des URLs en topics (Claude)..."
    db.commit()

    # Step 1: Claude classifies URLs into topics. Inject both the domain brief
    # (what's the scanned site?) and the workspace brief (whose perspective?)
    # so Claude can distinguish user-owned brands from competitors without
    # guessing from the URL alone.
    from adapters.brief_injector import format_analysis_context
    from models import Client as _Client
    _client = db.query(_Client).filter(_Client.id == scan.client_id).first()
    result = asyncio.run(classify_urls_into_topics(
        domain=scan.domain,
        keywords=kw_data,
        anthropic_api_key=settings.anthropic_api_key,
        domain_context=format_analysis_context(scan.config, _client.apps if _client else None),
    ))

    # Step 2: Create topics and build URL → topic_id mapping
    db.query(ScanTopic).filter(ScanTopic.scan_id == scan_id).delete()

    url_to_topic_id = {}
    for i, topic in enumerate(result["topics"]):
        t = ScanTopic(
            scan_id=scan_id,
            name=topic["nom"],
            description=topic.get("description", ""),
            example_keywords=[],
            is_active=True,
            display_order=i,
        )
        db.add(t)
        db.flush()  # Get the generated ID

        for url in topic.get("urls", []):
            url_to_topic_id[url] = t.id

    # Step 3: Assign keywords to topics via their URL
    assigned = 0
    for kw in keywords:
        topic_id = url_to_topic_id.get(kw.url)
        if topic_id:
            kw.topic_id = topic_id
            assigned += 1
        else:
            kw.topic_id = None

    # Step 4: Compute keyword counts (distinct keyword text) and example keywords per topic
    # HaloScan returns 1 row per (keyword, url) pair; we count distinct concepts, not rows
    topics = db.query(ScanTopic).filter(ScanTopic.scan_id == scan_id).all()
    for topic in topics:
        topic_kws = [kw for kw in keywords if kw.topic_id == topic.id]
        topic.keyword_count = len({kw.keyword for kw in topic_kws})
        # Top 5 keywords by traffic as examples
        top_kws = sorted(topic_kws, key=lambda k: k.traffic or 0, reverse=True)[:5]
        topic.example_keywords = [k.keyword for k in top_kws]

    # Build topic name → id mapping for brand-topic junction (Step 6)
    topic_name_to_id = {t.name.lower(): t.id for t in db.query(ScanTopic).filter(ScanTopic.scan_id == scan_id).all()}

    # Step 5: Auto-enrich Brand Registry with detected brands (pre-classified by Claude)
    from models import ClientBrand, ScanBrandClassification, ScanBrandTopic
    marques = result.get("marques_detectees", [])
    new_brands = 0

    # Map Claude's classification to our categories (legacy client_brands.category, kept for mapping only)
    category_map = {
        "site_brand": "target_brand",
        "site_gamme": "target_gamme",
        "competitor": "competitor",
    }
    # Map legacy category → new SBC classification vocabulary.
    #
    # Claude tags brands on the scanned page : site_brand / site_gamme are
    # "the brand and product gammes of the SITE being analyzed". Whether
    # those are MY brands or COMPETITOR brands depends on whether the user
    # owns that site — which is determined by checking the scan's domain
    # against client.primary_brand_ids. On a competitor audit (PF user
    # scanning uriage.fr), the site_brand IS the competitor, so we flip the
    # mapping accordingly. Without this check, the SBC table tags Uriage
    # as my_brand and the FAQ generator promotes it. Wrong.
    from services.brand_resolver import is_competitor_scan
    competitor_audit = is_competitor_scan(scan, db)

    if competitor_audit:
        sbc_classification_map = {
            "target_brand": "competitor",
            "target_gamme": "competitor",
            "competitor": "competitor",
        }
    else:
        sbc_classification_map = {
            "target_brand": "my_brand",
            "target_gamme": "my_brand",
            "competitor": "competitor",
        }

    now = datetime.utcnow()
    # Track brand_ids + their Claude category for parent-child linking after the loop
    touched_brand_ids: list[str] = []
    brand_with_category: list[tuple] = []  # [(brand_row, claude_category), ...]
    brand_topic_names: list[tuple] = []  # [(brand_row, [topic_name, ...]), ...]

    for marque in marques:
        if not marque:
            continue

        # Support both old format (string) and new format (dict with name + category)
        if isinstance(marque, str):
            name = marque
            category = "unclassified"
        else:
            name = marque.get("name", "")
            claude_cat = marque.get("category", "")
            category = category_map.get(claude_cat, "unclassified")

        from services.brand_name_norm import normalize_brand_name
        name_norm = normalize_brand_name(name)
        if not name_norm:
            continue

        existing = db.query(ClientBrand).filter(
            ClientBrand.client_id == scan.client_id,
            ClientBrand.canonical_name == name_norm,
        ).first()
        if not existing:
            # NOTE: client_brands.category is lazy-deprecated; leave default 'unclassified'
            # on new rows. Brand classification lives in scan_brand_classifications now.
            new_brand = ClientBrand(
                client_id=scan.client_id,
                parent_id=None,
                name=name,
                canonical_name=name_norm,
                detected_in_scan_id=scan_id,
                detection_source="keywords",
                auto_detected=True,
                validated_by_user=False,
                last_seen_at=now,
            )
            db.add(new_brand)
            db.flush()  # Get the generated ID for SBC upsert
            brand_row = new_brand
            new_brands += 1
        else:
            # Refresh last_seen_at on existing rows. canonical_name is the
            # dedup key now (UNIQUE per client) — never overwrite it ; the
            # display `name` keeps its current casing.
            existing.last_seen_at = now
            brand_row = existing

        # Upsert ScanBrandClassification for (scan_id, brand_row.id)
        sbc_classification = sbc_classification_map.get(category, "unclassified")
        sbc = db.query(ScanBrandClassification).filter(
            ScanBrandClassification.scan_id == scan_id,
            ScanBrandClassification.brand_id == brand_row.id,
        ).first()
        if sbc:
            sbc.classification = sbc_classification
            sbc.classified_by = "claude"
            sbc.source = "keywords"
            sbc.updated_at = now
        else:
            db.add(ScanBrandClassification(
                scan_id=scan_id,
                brand_id=brand_row.id,
                classification=sbc_classification,
                is_focus=False,
                classified_by="claude",
                source="keywords",
            ))
        db.flush()
        touched_brand_ids.append(brand_row.id)
        brand_with_category.append((brand_row, category))
        # Collect topic names for this brand (from Claude's response)
        if isinstance(marque, dict):
            brand_topic_names.append((brand_row, marque.get("topics", [])))

    # Step 5a: Link product lines (gammes) to their parent brand
    # Claude returns site_brand (root) and site_gamme (product line). We set parent_id
    # on gammes so they nest visually under their parent in the Gate 2 brands UI.
    site_brands = [b for b, cat in brand_with_category if cat == "target_brand"]
    if site_brands:
        parent = site_brands[0]  # first target_brand = the main brand for this scan
        for brand_row, cat in brand_with_category:
            if cat == "target_gamme" and not brand_row.parent_id:
                brand_row.parent_id = parent.id

    # Step 5b: Pick the focus brand for this scan among ROOT my_brand SBC rows.
    # Product lines (parent_id IS NOT NULL) are never eligible as focus brand.
    import unicodedata

    def _strip_accents(s: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

    my_brand_sbcs = db.query(ScanBrandClassification).filter(
        ScanBrandClassification.scan_id == scan_id,
        ScanBrandClassification.classification == "my_brand",
    ).all()

    focus_sbc = None
    if my_brand_sbcs:
        scan_domain_lc = _strip_accents((scan.domain or "").lower())
        # Preload brand rows for the candidates
        candidate_brands = {
            b.id: b for b in db.query(ClientBrand).filter(
                ClientBrand.id.in_([s.brand_id for s in my_brand_sbcs])
            ).all()
        }

        # Only consider root brands (no parent_id) — gammes are never focus
        root_sbcs = [s for s in my_brand_sbcs if not getattr(candidate_brands.get(s.brand_id), "parent_id", None)]
        if not root_sbcs:
            root_sbcs = my_brand_sbcs  # fallback if all are gammes (shouldn't happen)

        # 1) brand.domain substring of scan.domain
        for s in root_sbcs:
            b = candidate_brands.get(s.brand_id)
            if b and b.domain and _strip_accents(b.domain.lower()) in scan_domain_lc:
                focus_sbc = s
                break

        # 2) brand name (accent-stripped, hyphens stripped) substring of scan.domain
        if not focus_sbc:
            for s in root_sbcs:
                b = candidate_brands.get(s.brand_id)
                if not b or not b.name:
                    continue
                name_key = _strip_accents(b.name.lower()).replace("-", "")
                if name_key and name_key in scan_domain_lc.replace("-", ""):
                    focus_sbc = s
                    break

        # 3) first root brand by first_detected_at ASC
        if not focus_sbc:
            sorted_sbcs = sorted(
                root_sbcs,
                key=lambda s: getattr(candidate_brands.get(s.brand_id), "first_detected_at", None) or datetime.max,
            )
            focus_sbc = sorted_sbcs[0] if sorted_sbcs else None

    if focus_sbc:
        # Defensive: clear any other is_focus=True for this scan first
        db.query(ScanBrandClassification).filter(
            ScanBrandClassification.scan_id == scan_id,
            ScanBrandClassification.is_focus == True,  # noqa: E712
            ScanBrandClassification.id != focus_sbc.id,
        ).update({"is_focus": False}, synchronize_session=False)
        focus_sbc.is_focus = True
        scan.focus_brand_id = focus_sbc.brand_id
    # else: leave focus_brand_id = NULL — UI banner will prompt user to pick one

    # Step 6: Create brand-topic junction rows (brand scoping v2)
    # Claude tags each brand with the topics it's relevant to.
    db.query(ScanBrandTopic).filter(ScanBrandTopic.scan_id == scan_id).delete()
    brand_topic_count = 0
    for brand_row, topic_names in brand_topic_names:
        for tname in topic_names:
            tid = topic_name_to_id.get(tname.lower())
            if tid:
                db.add(ScanBrandTopic(scan_id=scan_id, brand_id=brand_row.id, topic_id=tid))
                brand_topic_count += 1
    db.flush()
    logger.info(f"Created {brand_topic_count} brand-topic associations")

    # Log LLM usage for cost monitoring
    from adapters.llm_logger import log_llm_usage
    log_llm_usage(
        db,
        provider="anthropic",
        model=result.get("model", "claude-haiku-4-5-20251001"),
        operation="classify_topics",
        input_tokens=result.get("input_tokens", 0),
        output_tokens=result.get("output_tokens", 0),
        duration_ms=result.get("duration_ms"),
        scan_id=scan_id,
        client_id=str(scan.client_id),
    )

    scan.status = "topics_ready"
    classified_brands = {category_map.get(m.get("category", ""), "unclassified"): 0 for m in marques if isinstance(m, dict)}
    for m in marques:
        if isinstance(m, dict):
            cat = category_map.get(m.get("category", ""), "unclassified")
            classified_brands[cat] = classified_brands.get(cat, 0) + 1
    scan.progress_message = f"{len(result['topics'])} topics, {assigned}/{len(keywords)} KW, {new_brands} new brands, {brand_topic_count} brand-topic links"
    scan.updated_at = datetime.utcnow()

    # Enrich brand registry: detect_competitors + cleanup_brands run as chained jobs
    # (detect_competitors chains cleanup_brands when it finds competitors).
    # These run WHILE the user validates topics → by the time they reach Gate 2 Brands,
    # the brand registry should be populated and auto-classified.
    from models import Job
    db.add(Job(scan_id=scan_id, job_type="detect_competitors"))
    # Also pre-enqueue cleanup_brands to classify the brands Claude just detected above
    # (don't wait for detect_competitors to chain it — saves ~10s sequential wait).
    # Conditional on new_brands > 0: skip if Claude didn't detect any brands in
    # keywords — otherwise the job runs as a no-op (~20ms wasted poll cycle).
    if new_brands > 0:
        db.add(Job(scan_id=scan_id, job_type="cleanup_brands"))
    # Sprint 15.3 - auto-chain assign_keywords so the scan reaches
    # 'brands_ready' without requiring an explicit "Validate topics"
    # Gate-1 click. The user can still review topics in the UI after
    # the auto-progress ; the handler is idempotent (rerunning on a
    # brands_ready scan only refreshes per-topic keyword counts).
    # Removes the dead-end where the user clicks "Continue to personas"
    # on a fresh scan and gets a 400 because status was still
    # 'topics_ready'.
    db.add(Job(scan_id=scan_id, job_type="assign_keywords"))

    db.commit()

    logger.info(f"Classified {assigned}/{len(keywords)} keywords into {len(result['topics'])} topics, {new_brands} new brands detected")
    return {
        "topics_count": len(result["topics"]),
        "topics": {t.name: t.keyword_count for t in topics},
        "assigned": assigned,
        "unassigned": len(keywords) - assigned,
        "provider": result.get("provider"),
        "model": result.get("model"),
        "duration_ms": result.get("duration_ms"),
        "sections_classified": result.get("sections_classified", 0),
        "urls_assigned": result.get("urls_assigned", 0),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
    }
