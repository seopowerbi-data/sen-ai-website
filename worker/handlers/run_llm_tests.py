"""Handler: run LLM tests using seo-llm's LLMClient + CitationExtractor + BrandAnalyzer.

Tests are parallelized using ThreadPoolExecutor (I/O-bound HTTP calls to OpenAI/Gemini).
DB writes stay in the main thread (SQLAlchemy session is not thread-safe).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from sqlalchemy.orm import Session

from adapters.llm_scanner import create_llm_client, test_question, test_question_openai_direct
from services.gemini_key_pool import get_gemini_pool

logger = logging.getLogger(__name__)

# Max concurrent LLM calls — Tier 4 OpenAI = 10K RPM, plenty of headroom at 20.
# Single executor over ALL tasks (no batching) eliminates head-of-line blocking
# where one slow OpenAI call held up an entire batch.
MAX_WORKERS = 20


def execute(job_payload: dict, scan_id: str, db: Session) -> dict:
    """Run LLM tests with citation extraction and brand analysis."""
    from models import Scan, ScanQuestion, ScanPersona, ScanLLMResult, ClientBrand, ScanBrandClassification, Job as JobModel
    from config import settings

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise RuntimeError("Scan not found")

    # Only run questions whose persona is ALSO active (toggling a persona off
    # excludes all its questions, even if individual questions are still is_active=True)
    questions = (
        db.query(ScanQuestion)
        .join(ScanPersona, ScanPersona.id == ScanQuestion.persona_id)
        .filter(
            ScanQuestion.scan_id == scan_id,
            ScanQuestion.is_active == True,
            ScanPersona.is_active == True,
        )
        .all()
    )
    if not questions:
        raise RuntimeError("No active questions")

    personas = {str(p.id): p for p in db.query(ScanPersona).filter(
        ScanPersona.scan_id == scan_id, ScanPersona.is_active == True,
    ).all()}

    # --- Build LLM clients ---
    # For Gemini, draw the key from GeminiKeyPool (round-robin across GEMINI_API_KEYS).
    # Single-key deployments still work: the pool falls back to GEMINI_API_KEY.
    providers = job_payload.get("providers", ["openai"])
    gemini_pool = get_gemini_pool()
    llm_clients = {}
    for provider in providers:
        if provider == "gemini":
            if not gemini_pool.has_keys():
                logger.warning("No Gemini key in pool, skipping")
                continue
            api_key = gemini_pool.next_key()
        else:
            api_key = getattr(settings, f"{provider}_api_key", "")
            if not api_key:
                logger.warning(f"No API key for {provider}, skipping")
                continue
        try:
            llm_clients[provider] = create_llm_client(provider, api_key)
        except Exception as e:
            logger.error(f"Failed to create {provider} client: {e}")

    if not llm_clients:
        raise RuntimeError("No LLM clients available")

    # --- Build BrandAnalyzer from Scan focus brand + SBC competitors ---
    brand_analyzer = None
    target_brands = []
    all_brands = []
    scan_config = scan.config or {}
    target_domain = scan_config.get("target_domains", [scan.domain])[0] if scan_config.get("target_domains") else scan.domain

    try:
        if not scan.focus_brand_id:
            logger.warning("no focus brand set, skipping BrandAnalyzer")
        else:
            focus_brand = db.query(ClientBrand).filter(ClientBrand.id == scan.focus_brand_id).first()
            if not focus_brand:
                logger.warning("no focus brand set, skipping BrandAnalyzer")
            else:
                # target_brands = focus brand + its children (via parent_id) + its aliases
                # Rationale: a brand's "visibility" includes its product lines (gammes).
                # For scan focus=Ducray, if the AI cites "Anaphase" (a Ducray gamme), that
                # MUST count as a Ducray mention. Children are joined in at runtime so the
                # user only picks one focus in the UI but gets the whole brand family tracked.
                children = db.query(ClientBrand).filter(
                    ClientBrand.parent_id == focus_brand.id
                ).all()
                raw_targets = (
                    [focus_brand.name]
                    + [c.name for c in children]
                    + list(focus_brand.aliases or [])
                )
                seen = set()
                target_brands = []
                for t in raw_targets:
                    if not t:
                        continue
                    t = t.strip()
                    if not t:
                        continue
                    key = t.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    target_brands.append(t)
                logger.info(
                    f"Focus brand: {focus_brand.name} + {len(children)} children "
                    f"→ {len(target_brands)} target_brands"
                )

                # Load competitor brands for THIS scan via SBC
                competitor_rows = (
                    db.query(ClientBrand)
                      .join(ScanBrandClassification, ScanBrandClassification.brand_id == ClientBrand.id)
                      .filter(ScanBrandClassification.scan_id == scan_id,
                              ScanBrandClassification.classification == 'competitor')
                      .all()
                )

                # all_brands = target_brands + competitors (dedupe, cap at 30)
                all_brands = list(target_brands)
                seen_all = set(b.lower() for b in all_brands)
                for c in competitor_rows:
                    if not c.name:
                        continue
                    key = c.name.strip().lower()
                    if not key or key in seen_all:
                        continue
                    seen_all.add(key)
                    all_brands.append(c.name)
                # Limit competitor brands sent to BrandAnalyzer. Was 30 — caused JSON
                # truncation on dense responses (28 brands → output > 20K tokens →
                # Gemini cuts off mid-string → JSON parse fails → BrandAnalyzer skips).
                # 15 keeps coverage of the most-mentioned competitors while staying
                # well within the 30K-token output budget (see brand_analyzer.py:381).
                all_brands = all_brands[:15]

                if target_brands and gemini_pool.has_keys():
                    from seo_llm.src.brand_analyzer import BrandAnalyzer
                    gemini_client = create_llm_client("gemini", gemini_pool.next_key(), model="gemini-2.5-flash-lite")
                    from adapters.brief_injector import format_brief_context
                    brand_analyzer = BrandAnalyzer(
                        llm_client=gemini_client,
                        target_brands=target_brands,
                        all_brands=all_brands,
                        domain_context=format_brief_context(scan.config),
                    )
                    logger.info(f"BrandAnalyzer configured: target={target_brands}, all={len(all_brands)} brands")
                else:
                    logger.info("BrandAnalyzer skipped: no target brands or no Gemini key")

    except Exception as e:
        logger.warning(f"BrandAnalyzer setup failed: {e}")

    # --- Run tests ---
    total_tests = len(questions) * len(llm_clients)
    scan.progress_pct = 0
    scan.progress_message = f"Scan LLM: 0/{total_tests} tests..."
    db.commit()

    db.query(ScanLLMResult).filter(ScanLLMResult.scan_id == scan_id).delete()
    db.commit()

    completed = 0
    target_cited_count = 0
    brand_mentioned_count = 0
    errors = 0

    # Build all test tasks: [(question, persona, provider, llm_client), ...]
    tasks = []
    for question in questions:
        persona = personas.get(str(question.persona_id))
        if not persona:
            continue
        for provider, llm_client in llm_clients.items():
            tasks.append((question, persona, provider, llm_client))

    total_tests = len(tasks)
    logger.info(f"Running {total_tests} tests in a single pool of {MAX_WORKERS} workers")

    from sqlalchemy import func as sql_func

    # Country hint for OpenAI web_search user_location (FR scans get FR-grounded URLs).
    # Falls back to FR if domain brief didn't capture a country code.
    scan_country = (scan.config or {}).get("domain_brief", {}).get("country") or "FR"
    openai_model = settings.task_models.get("scan_test_openai", "gpt-4.1-mini")

    # Single ThreadPoolExecutor over ALL tasks — no batching = no head-of-line blocking
    # (a slow OpenAI call no longer holds back Gemini results in the same batch).
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for question, persona, provider, llm_client in tasks:
            if provider == "openai":
                # Direct OpenAI path — bypasses LLMClient for tighter retry + faster
                # web_search_preview tool config (see adapters/llm_scanner.py module docstring)
                future = executor.submit(
                    test_question_openai_direct,
                    question=question.question,
                    persona=persona.data or {},
                    target_domain=target_domain,
                    api_key=settings.openai_api_key,
                    model=openai_model,
                    brand_analyzer=brand_analyzer,
                    country=scan_country,
                )
            else:
                # Gemini still goes through LLMClient (grounding works well, no quick win to chase)
                future = executor.submit(
                    test_question,
                    question=question.question,
                    persona=persona.data or {},
                    llm_client=llm_client,
                    target_domain=target_domain,
                    brand_analyzer=brand_analyzer,
                )
            futures[future] = (question, persona, provider)

        # Collect results as they complete (DB writes in main thread — safe).
        # Single stream, no batches: as soon as ANY task finishes (Gemini in 15s
        # or OpenAI in 30s), we persist + update progress immediately.
        for future in as_completed(futures):
            question, persona, provider = futures[future]
            try:
                result = future.result()

                db.add(ScanLLMResult(
                    scan_id=scan_id,
                    question_id=question.id,
                    provider=result["provider"],
                    model=result.get("model"),
                    response_text=result.get("response_text", ""),
                    citations=result.get("citations", []),
                    target_cited=result["target_cited"],
                    target_position=result["target_position"],
                    total_citations=result["total_citations"],
                    competitor_domains=result["competitor_domains"],
                    brand_mentions=result.get("brand_mentions", []),
                    brand_analysis=result.get("brand_analysis", {}),
                    duration_ms=result.get("duration_ms"),
                    input_tokens=result.get("input_tokens"),
                    output_tokens=result.get("output_tokens"),
                ))

                if result["target_cited"]:
                    target_cited_count += 1
                if result.get("brand_analysis", {}).get("marque_cible_mentionnee"):
                    brand_mentioned_count += 1

                # Auto-enrich: collect new brands from LLM responses
                for mention in result.get("brand_mentions", []):
                    bname = mention.get("brand_name_groupby") or mention.get("brand_name")
                    if not bname or len(bname) < 2:
                        continue

                    existing = db.query(ClientBrand).filter(
                        ClientBrand.client_id == scan.client_id,
                        sql_func.lower(ClientBrand.name) == bname.lower(),
                    ).first()

                    if not existing:
                        new_brand = ClientBrand(
                            client_id=scan.client_id,
                            name=bname,
                            canonical_name=bname,
                            last_seen_at=datetime.utcnow(),
                            detected_in_scan_id=scan_id,
                            detection_source="llm_response",
                        )
                        db.add(new_brand)
                        db.flush()
                        brand_id = new_brand.id
                    else:
                        existing.last_seen_at = datetime.utcnow()
                        brand_id = existing.id

                    sbc_exists = db.query(ScanBrandClassification).filter(
                        ScanBrandClassification.scan_id == scan_id,
                        ScanBrandClassification.brand_id == brand_id,
                    ).first()
                    if not sbc_exists:
                        db.add(ScanBrandClassification(
                            scan_id=scan_id,
                            brand_id=brand_id,
                            classification='unclassified',
                            classified_by='auto',
                            source='llm_response',
                        ))

                completed += 1
                logger.info(f"Test {completed}/{total_tests}: {provider} | "
                           f"cited={result['target_cited']} | "
                           f"brand={result.get('brand_analysis', {}).get('marque_cible_mentionnee', False)} | "
                           f"{result.get('duration_ms')}ms")

                # Log LLM usage for cost monitoring
                from adapters.llm_logger import log_llm_usage
                log_llm_usage(
                    db, provider=result["provider"],
                    model=result.get("model", "unknown"),
                    operation="scan_test",
                    input_tokens=result.get("input_tokens", 0),
                    output_tokens=result.get("output_tokens", 0),
                    duration_ms=result.get("duration_ms"),
                    scan_id=scan_id, client_id=str(scan.client_id),
                )

            except Exception as e:
                logger.error(f"Test failed ({provider}): {e}")
                errors += 1
                completed += 1

            # Per-test progress update (Goal-Gradient): user sees 1/N, 2/N, 3/N...
            # in real time. Cheap commits (one row per LLM call which costs orders
            # of magnitude more time than the commit itself).
            scan.progress_pct = int(completed / total_tests * 100)
            scan.progress_message = f"Scan LLM: {completed}/{total_tests} tests..."
            db.commit()

    # --- Final ---
    success = max(completed - errors, 1)
    citation_rate = round(target_cited_count / success * 100, 1)
    brand_rate = round(brand_mentioned_count / success * 100, 1)

    scan.status = "completed"
    scan.progress_pct = 100
    scan.progress_message = f"Scan terminé — cité {citation_rate}%, marque mentionnée {brand_rate}%"
    scan.completed_at = datetime.utcnow()
    scan.updated_at = datetime.utcnow()
    scan.summary = {
        "total_tests": completed,
        "errors": errors,
        "target_cited": target_cited_count,
        "citation_rate": citation_rate,
        "brand_mentioned": brand_mentioned_count,
        "brand_mention_rate": brand_rate,
        "providers": providers,
        "target_domain": target_domain,
        "target_brands": target_brands if brand_analyzer else [],
        "focus_brand_id": str(scan.focus_brand_id) if scan.focus_brand_id else None,
    }

    # Chain: generate opportunities + editorial + cleanup brands
    db.add(JobModel(scan_id=scan_id, job_type="generate_opportunities"))
    db.add(JobModel(scan_id=scan_id, job_type="generate_editorial"))
    db.add(JobModel(scan_id=scan_id, job_type="cleanup_brands"))
    db.commit()

    logger.info(f"Scan complete: {completed} tests, citations={citation_rate}%, brand={brand_rate}%")
    return {
        "total_tests": completed,
        "target_cited": target_cited_count,
        "citation_rate": citation_rate,
        "brand_mentioned": brand_mentioned_count,
        "brand_mention_rate": brand_rate,
        "errors": errors,
    }
