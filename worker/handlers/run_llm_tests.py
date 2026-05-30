"""Handler: run LLM tests using seo-llm's LLMClient + CitationExtractor + BrandAnalyzer.

Tests are parallelized using ThreadPoolExecutor (I/O-bound HTTP calls to OpenAI/Gemini).
DB writes stay in the main thread (SQLAlchemy session is not thread-safe).
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from adapters.llm_scanner import create_llm_client, test_question, test_question_openai_direct
from services.circuit_breaker import ProviderCircuitBreaker
from services.credits import partial_refund_scan_credits
from services.gemini_key_pool import get_gemini_pool

logger = logging.getLogger(__name__)

# Max concurrent LLM calls - Tier 4 OpenAI = 10K RPM, plenty of headroom at 20.
# Single executor over ALL tasks (no batching) eliminates head-of-line blocking
# where one slow OpenAI call held up an entire batch.
MAX_WORKERS = 20

# 429 signatures (rate-limit OR monthly spend cap). "spend cap"/"resource_exhausted"
# = a cap that won't clear in 30s → park the key for a long cooldown.
_RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "rate limit", "too many requests", "quota")
_SPEND_CAP_MARKERS = ("spend cap", "resource_exhausted")


class PoolRotatingGeminiClient:
    """Gemini client that draws a fresh key from the pool per `.generate()` call
    and, on a 429, parks the offending key (long cooldown for a spend cap) and
    retries with the next key. Fixes the "one pinned key per scan" flaw where a
    single capped/limited key killed the whole scan (provider tests AND the brand
    analyzer). Thread-safe: per-call key draw + per-key client cache, no shared
    mutable rotation state. Non-generate calls (extract_json, .provider) proxy
    through - they're pure-parsing/metadata, key-agnostic.
    """

    def __init__(self, pool, model: str | None = None):
        self._pool = pool
        self._model = model
        self.provider = "gemini"
        self._clients: dict[str, object] = {}
        self._lock = threading.Lock()

    def _client_for(self, key: str):
        with self._lock:
            c = self._clients.get(key)
            if c is None:
                c = (create_llm_client("gemini", key, model=self._model)
                     if self._model else create_llm_client("gemini", key))
                self._clients[key] = c
            return c

    def generate(self, *args, **kwargs):
        last_exc = None
        for _ in range(max(1, self._pool.num_keys)):
            key = self._pool.next_key()
            try:
                return self._client_for(key).generate(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 - inspect message to classify
                msg = str(e).lower()
                if any(m in msg for m in _RATE_LIMIT_MARKERS):
                    self._pool.mark_rate_limited(
                        key, long=any(m in msg for m in _SPEND_CAP_MARKERS))
                    last_exc = e
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("PoolRotatingGeminiClient: empty pool")

    def __getattr__(self, name: str):
        # Only reached for attrs not defined above (e.g. extract_json). Guard the
        # private names so half-built instances don't recurse.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._client_for(self._pool.next_key()), name)


def execute(job_payload: dict, scan_id: str, db: Session) -> dict:
    """Run LLM tests with citation extraction and brand analysis."""
    from models import Scan, ScanQuestion, ScanPersona, ScanLLMResult, ClientBrand, ScanBrandClassification, Job as JobModel
    from config import settings

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise RuntimeError("Scan not found")

    # Sprint N-runs - multi-sampling. `runs_depth` (scan.config) defines how many
    # times each (question, provider) pair gets called. Default 1 = legacy behavior
    # (no schema regression, no consumer breaks). Sprint 3 flips default to 10.
    # See migration 045 + plan lovely-skipping-sunset.md.
    runs_depth = int((scan.config or {}).get("runs_depth", 1)) or 1
    if runs_depth < 1:
        runs_depth = 1

    # Cap-then-call : a full LLM-test pass runs ~$0.10-0.30 across providers
    # × questions × brand_analyzer at N=1. Scales linearly with runs_depth.
    # If this trips, the scan retries - operator response is to bump the
    # LLM_DAILY_COST_CAP_USD (or per-client cap) or wait for UTC midnight reset.
    from services.llm_budget import assert_within_budget
    assert_within_budget(scan.client_id, db, projected_cost_usd=0.30 * runs_depth)

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
    # Gemini goes through PoolRotatingGeminiClient: per-call key draw from the pool
    # with park-and-retry on 429 (long cooldown for spend-cap). This rotates WITHIN
    # a scan, so one capped/limited key no longer kills the run (both the provider
    # tests below and the brand analyzer share this resilience).
    providers = job_payload.get("providers", ["openai"])
    gemini_pool = get_gemini_pool()
    llm_clients = {}
    for provider in providers:
        if provider == "gemini":
            if not gemini_pool.has_keys():
                logger.warning("No Gemini key in pool, skipping")
                continue
            llm_clients["gemini"] = PoolRotatingGeminiClient(gemini_pool)
            continue
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

    # --- Build EntityAnalyzer (Sprint E) from Scan focus brand + SBC competitors ---
    # EntityAnalyzer extends the legacy BrandAnalyzer to 5 entity types
    # (brand/product/range/domain/expert_source) - wire-compatible with the
    # existing ScanLLMResult.brand_mentions JSONB column. See
    # worker/adapters/entity_analyzer.py + project_phase_judge_and_entities.md.
    brand_analyzer = None
    target_brands = []
    all_brands = []
    scan_config = scan.config or {}
    target_domain = scan_config.get("target_domains", [scan.domain])[0] if scan_config.get("target_domains") else scan.domain

    try:
        from adapters.entity_analyzer import EntityAnalyzer, build_target_entities_from_scan

        target_entities, known_entities = build_target_entities_from_scan(scan, db)
        # Keep target_brands populated for the scan summary payload (consumed
        # downstream by /scans/{id}/results) - same data, just unpacked from
        # the structured target_entities dict for legacy callers.
        target_brands = list(target_entities["brands"])
        all_brands = list(target_brands) + [k for k in known_entities if k.lower() not in {b.lower() for b in target_brands}]
        all_brands = all_brands[:15]

        if not scan.focus_brand_id and not target_entities["domains"]:
            logger.warning("no focus brand AND no target domains - skipping EntityAnalyzer")
        elif not gemini_pool.has_keys():
            logger.info("EntityAnalyzer skipped: no Gemini key in pool")
        else:
            gemini_client = PoolRotatingGeminiClient(gemini_pool, model="gemini-2.5-flash-lite")
            from adapters.brief_injector import format_analysis_context
            from models import Client as _Client
            _client = db.query(_Client).filter(_Client.id == scan.client_id).first()
            brand_analyzer = EntityAnalyzer(
                llm_client=gemini_client,
                target_entities=target_entities,
                known_entities=known_entities,
                domain_context=format_analysis_context(scan.config, _client.apps if _client else None),
            )
            logger.info(
                f"EntityAnalyzer configured: targets="
                f"{{brands:{len(target_entities['brands'])}, "
                f"products:{len(target_entities['products'])}, "
                f"ranges:{len(target_entities['ranges'])}, "
                f"domains:{len(target_entities['domains'])}, "
                f"expert_sources:{len(target_entities['expert_sources'])}}}, "
                f"known={len(known_entities)}"
            )

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

    # Build all test tasks: [(question, persona, provider, llm_client, run_idx), ...]
    # Sprint N-runs : outer loop on run_idx so the executor sees ALL N×Q×P tasks
    # at once (head-of-line blocking eliminated across runs too - a slow run 1
    # task doesn't block run 2 tasks from another q/provider).
    tasks = []
    for run_idx in range(1, runs_depth + 1):
        for question in questions:
            persona = personas.get(str(question.persona_id))
            if not persona:
                continue
            for provider, llm_client in llm_clients.items():
                tasks.append((question, persona, provider, llm_client, run_idx))

    total_tests = len(tasks)
    logger.info(
        f"Running {total_tests} tests in a single pool of {MAX_WORKERS} workers "
        f"(runs_depth={runs_depth}, questions={len(questions)}, providers={len(llm_clients)})"
    )

    from sqlalchemy import func as sql_func

    # Country hint for OpenAI web_search user_location (FR scans get FR-grounded URLs).
    # Falls back to FR if domain brief didn't capture a country code.
    scan_country = (scan.config or {}).get("domain_brief", {}).get("country") or "FR"
    openai_model = settings.task_models.get("scan_test_openai", "gpt-4.1-mini")

    # Per-scan circuit breaker. Trips a provider when its failure rate exceeds
    # threshold (default 80%) AND we have at least min_sample tests recorded
    # (default 10). Pending futures for a tripped provider are cancelled inline
    # so we don't waste API quota / wait time on a provider known to be down.
    breaker = ProviderCircuitBreaker(list(llm_clients.keys()))

    # Single ThreadPoolExecutor over ALL tasks - no batching = no head-of-line blocking
    # (a slow OpenAI call no longer holds back Gemini results in the same batch).
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for question, persona, provider, llm_client, run_idx in tasks:
            if provider == "openai":
                # Direct OpenAI path - bypasses LLMClient for tighter retry + faster
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
            futures[future] = (question, persona, provider, run_idx)

        # Collect results as they complete (DB writes in main thread - safe).
        # Single stream, no batches: as soon as ANY task finishes (Gemini in 15s
        # or OpenAI in 30s), we persist + update progress immediately.
        for future in as_completed(futures):
            question, persona, provider, run_idx = futures[future]

            # Circuit breaker may have cancelled pending futures for a tripped
            # provider. Count them as skipped (no DB write, no usage logging,
            # no API call ever made) and move on.
            if future.cancelled():
                completed += 1
                continue

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
                    # Phase C.1.5 - fan-out ground truth from the LLM's actual
                    # web search behavior (Gemini grounding_metadata,
                    # OpenAI web_search_call.action.queries). Consumed
                    # downstream by services.fan_out_extractor to build the
                    # cross-provider fan-out set for article gen.
                    web_search_queries=result.get("web_search_queries", []),
                    # Sprint N-runs (migration 045) - which sample is this ?
                    # 1..N for actual LLM calls. run_index=0 is reserved for a
                    # future consensus row carrying the brand_analysis JSONB
                    # derived from EntityAnalyzer over the concatenated N
                    # responses (Sprint 1.6, not implemented yet - current
                    # behavior keeps per-run EntityAnalyzer, paid by runs_depth).
                    run_index=run_idx,
                ))

                if result["target_cited"]:
                    target_cited_count += 1
                if result.get("brand_analysis", {}).get("marque_cible_mentionnee"):
                    brand_mentioned_count += 1

                # Auto-enrich: collect new brands from LLM responses.
                # Filter the obvious noise upstream so the catalog doesn't
                # balloon with ingredients / product types / domains the
                # BrandAnalyzer over-extracts. The brief's noise_patterns
                # carries vertical-specific terms (cosmetics: "crème",
                # "acide hyaluronique"; automotive: "huile moteur") so the
                # filter is multi-industry without hardcoded lists.
                from services.brand_noise_filter import is_noise_brand_name
                _brief = (scan.config or {}).get("domain_brief") or {}
                _noise_prefixes = _brief.get("noise_patterns") or []
                for mention in result.get("brand_mentions", []):
                    bname = mention.get("brand_name_groupby") or mention.get("brand_name")
                    if not bname or len(bname) < 2:
                        continue
                    if is_noise_brand_name(bname, _noise_prefixes):
                        continue

                    from services.brand_name_norm import normalize_brand_name
                    bnorm = normalize_brand_name(bname)
                    existing = db.query(ClientBrand).filter(
                        ClientBrand.client_id == scan.client_id,
                        ClientBrand.canonical_name == bnorm,
                    ).first() if bnorm else None

                    if not existing:
                        new_brand = ClientBrand(
                            client_id=scan.client_id,
                            name=bname,
                            canonical_name=bnorm,
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
                breaker.record_success(provider)
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

                # BrandAnalyzer is a separate Gemini call per test - log it
                # under its own operation/model so cost dashboards split scan_test
                # (search-grounded gpt-4.1-mini / gemini-2.5-flash) from
                # brand_analyzer (gemini-2.5-flash-lite parsing the response).
                ba_usage = result.get("brand_analyzer_usage") or {}
                ba_model = result.get("brand_analyzer_model")
                if ba_model and (ba_usage.get("input_tokens") or ba_usage.get("output_tokens")
                                 or ba_usage.get("prompt_tokens") or ba_usage.get("completion_tokens")):
                    log_llm_usage(
                        db, provider="gemini", model=ba_model,
                        operation="brand_analyzer",
                        input_tokens=ba_usage.get("input_tokens", 0)
                                     or ba_usage.get("prompt_tokens", 0),
                        output_tokens=ba_usage.get("output_tokens", 0)
                                      or ba_usage.get("completion_tokens", 0),
                        scan_id=scan_id, client_id=str(scan.client_id),
                    )

            except Exception as e:
                logger.error(f"Test failed ({provider}): {e}")
                errors += 1
                completed += 1
                breaker.record_failure(provider)

            # Circuit breaker check: if THIS provider just crossed the
            # failure threshold, cancel its still-pending futures so we
            # don't keep burning quota / time on a known-down provider.
            if breaker.maybe_trip(provider):
                for f, (q_, p_, prov_) in list(futures.items()):
                    if prov_ == provider and not f.done():
                        if f.cancel():
                            breaker.record_skip(provider)

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

    # Prorata refund (C.3): a question is "delivered" if at least ONE provider
    # produced a result for it on ANY run. Questions where every (provider,run)
    # failed/skipped count as undelivered → refund credits.
    # Sprint N-runs : credits are debited as `questions × runs_depth` at launch,
    # so a fully-failed question refunds `runs_depth` credits (1 per run that
    # never happened).
    questions_with_results = {
        str(qid) for (qid,) in db.query(ScanLLMResult.question_id)
            .filter(ScanLLMResult.scan_id == scan_id).distinct()
    }
    failed_question_count = sum(
        1 for q in questions if str(q.id) not in questions_with_results
    )
    refund_info = None
    if failed_question_count > 0:
        refund_credits = failed_question_count * runs_depth
        try:
            partial_refund_scan_credits(
                db=db,
                client_id=scan.client_id,
                scan_id=scan_id,
                amount=refund_credits,
                description=(
                    f"Partial refund: {failed_question_count} questions undelivered "
                    f"× {runs_depth} run(s)"
                ),
            )
            refund_info = {
                "amount": refund_credits,
                "failed_questions": failed_question_count,
                "runs_depth": runs_depth,
                "reason": "questions_undelivered",
            }
        except Exception:
            logger.exception(
                f"Partial refund failed for scan {scan_id} "
                f"({failed_question_count} questions × {runs_depth} runs)"
            )

    scan.status = "completed"
    scan.progress_pct = 100
    scan.progress_message = f"Scan terminé - cité {citation_rate}%, marque mentionnée {brand_rate}%"
    scan.completed_at = datetime.utcnow()
    scan.updated_at = datetime.utcnow()
    # S15.4 auto-rescan: anchor the next firing on this completion when
    # the scan opted into a weekly / monthly schedule. The cron sweeper
    # (worker/main.py) picks scans where next_run_at <= NOW() and re-launches.
    _interval = {"weekly": timedelta(days=7), "monthly": timedelta(days=30)}.get(scan.schedule or "manual")
    if _interval:
        scan.next_run_at = scan.completed_at + _interval
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
        "provider_status": breaker.to_dict(),
        "refund_info": refund_info,
        # Sprint N-runs : surface in the summary so the UI can show "1 scan = N runs"
        # and consumers can branch on it (e.g., aggregate intra-scan vs cross-lineage).
        "runs_depth": runs_depth,
    }

    # Chain: classify intent (Phase B Tier A) → judge per-question signals
    # (Sprint J) → opportunities + editorial + cleanup brands. Worker poll
    # is FIFO single-thread, so classify_question_intent runs first and
    # populates intent_category before generate_opportunities reads it.
    # judge_question_responses can run in any order vs opportunities since
    # it doesn't currently feed back into scoring (Sprint M will wire that).
    # judge_sentiment audits brand_mentions[].sentiment for false positives
    # and is consumed by Crisis radar + Overview/per-persona chips. Capped
    # at $0.05/scan and idempotent on (slr_id, mention_index, contexte_hash)
    # so a manual /sentiment-judge/refresh post-scan stays safe.
    db.add(JobModel(scan_id=scan_id, job_type="classify_question_intent"))
    db.add(JobModel(scan_id=scan_id, job_type="judge_question_responses"))
    db.add(JobModel(scan_id=scan_id, job_type="generate_opportunities"))
    db.add(JobModel(scan_id=scan_id, job_type="generate_editorial"))
    db.add(JobModel(scan_id=scan_id, job_type="cleanup_brands"))
    db.add(JobModel(scan_id=scan_id, job_type="judge_sentiment"))

    # Post-scan audit auto-chain (free, heuristic / external-free-API only).
    # All are in POST_SCAN_AUDIT_JOB_TYPES so a failure stays sandboxed and
    # never cascades to scan.status='failed'. Each one is also idempotent
    # at row level so re-running via the manual /refresh endpoint stays
    # safe. Deliberately NOT included :
    #   - audit_reddit_threads     (~$0.03 Haiku, opt-in by design)
    #   - audit_competitor_pages   (Babbar rate-limit risk, ~1-2 min)
    # Both still triggerable manually via their /refresh endpoints.
    # Crisis radar reads scan_sentiment_judgements ; if it happens to run
    # before judge_sentiment finishes (FIFO ordering on identical
    # created_at is unspecified) it falls back to raw brand_mentions[]
    # .sentiment per migration 057 design - acceptable.
    db.add(JobModel(scan_id=scan_id, job_type="check_brand_wikipedia"))
    db.add(JobModel(scan_id=scan_id, job_type="audit_scan_pages"))
    db.add(JobModel(scan_id=scan_id, job_type="audit_scan_schemas"))
    db.add(JobModel(scan_id=scan_id, job_type="audit_internal_links"))
    db.add(JobModel(scan_id=scan_id, job_type="build_pr_outreach"))
    db.add(JobModel(scan_id=scan_id, job_type="audit_youtube_creators"))
    db.add(JobModel(scan_id=scan_id, job_type="build_crisis_radar"))

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
