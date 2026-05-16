"""Handler: re-run the LLM tests for ONE content item's question.

Phase E Pilier 5 — Stage 1. Triggered when the user clicks 'Refresh AI
snapshot' on the validation page, sized to displace the misleading
'today' label by giving the user an on-demand freshness gesture.

Picks the item -> resolves its target_question to a ScanQuestion ->
calls every configured LLM provider on that single question (ChatGPT +
Gemini today, ~$0.04 total) -> stores fresh ScanLLMResult rows tagged
by today's created_at. The detail endpoint's _build_competitor_snapshot
then surfaces the LATEST row per provider, so the panel updates without
any UI-side caching gymnastics.

Brand analyzer is NOT run on refresh : it depends on the scan-wide
focus_brand + competitor list machinery that isn't worth reassembling
for one question. Brand mentions in the new rows are left empty —
the response_text + citations are what drive the panel's visible value
(competitor highlighting on the response text is regex-based, doesn't
need the BrandAnalyzer payload).

In-flight protection : the API endpoint blocks double-enqueue. The cap
rule (10 rows in last 24h per question = 5 refreshes) lives there too —
keeps the handler dumb and re-runnable.

Payload :
    {"item_id": str}

Returns :
    {
      "status": "ok" | "error" | "skipped",
      "reason": str | None,
      "question_text": str,
      "inserted": [{provider, id, duration_ms}],
      "providers_attempted": int,
      "providers_succeeded": int,
    }
"""

from __future__ import annotations

import logging

from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def execute(job_payload: dict, scan_id: str | None, db: Session) -> dict:
    from models import (
        ClientBrand, Scan, ScanContentItem, ScanLLMResult,
        ScanPersona, ScanQuestion,
    )
    from adapters.llm_scanner import create_llm_client, test_question
    from services.gemini_key_pool import get_gemini_pool
    from config import settings

    item_id = (job_payload or {}).get("item_id")
    if not item_id:
        raise ValueError("refresh_ai_snapshot requires item_id in payload")

    item = (
        db.query(ScanContentItem)
        .filter(ScanContentItem.id == item_id)
        .first()
    )
    if not item:
        raise ValueError(f"ScanContentItem {item_id} not found")

    scan = db.query(Scan).filter(Scan.id == item.scan_id).first()
    if not scan:
        raise ValueError(f"Scan {item.scan_id} not found")

    # Cap-then-call : refresh runs ~$0.04 across OpenAI + Gemini. Project $0.05.
    from services.llm_budget import assert_within_budget
    assert_within_budget(scan.client_id, db, projected_cost_usd=0.05)

    q_text = (item.target_question or "").strip()
    if not q_text:
        logger.info(f"refresh_ai_snapshot: item {item_id} has no target_question")
        return {
            "status": "skipped", "reason": "no_question",
            "question_text": None, "inserted": [],
            "providers_attempted": 0, "providers_succeeded": 0,
        }

    question = (
        db.query(ScanQuestion)
        .filter(
            ScanQuestion.scan_id == item.scan_id,
            func.lower(ScanQuestion.question) == q_text.lower(),
        )
        .first()
    )
    if not question:
        logger.info(
            f"refresh_ai_snapshot: no ScanQuestion matched '{q_text[:80]}' "
            f"for scan {scan.id}"
        )
        return {
            "status": "skipped", "reason": "question_not_found",
            "question_text": q_text, "inserted": [],
            "providers_attempted": 0, "providers_succeeded": 0,
        }

    # Persona dict for test_question. The original run_llm_tests reads
    # persona.profile_data — we mirror that shape so the prompt template
    # consumes the same persona summary as the scan.
    persona = (
        db.query(ScanPersona)
        .filter(ScanPersona.id == question.persona_id)
        .first()
        if question.persona_id else None
    )
    # ScanPersona.data is the full persona blob (matches run_llm_tests usage
    # at worker/handlers/run_llm_tests.py:225 — `persona=persona.data or {}`).
    persona_dict = (persona.data if persona and persona.data else {}) or {}

    target_domain = scan.domain or ""

    providers = (job_payload or {}).get("providers") or ["openai", "gemini"]
    llm_clients: dict = {}
    gemini_pool = get_gemini_pool()
    for p in providers:
        try:
            if p == "gemini":
                if not gemini_pool.has_keys():
                    logger.warning("refresh_ai_snapshot: no Gemini key in pool")
                    continue
                llm_clients[p] = create_llm_client("gemini", gemini_pool.next_key())
            else:
                key = getattr(settings, f"{p}_api_key", "")
                if not key:
                    logger.warning(f"refresh_ai_snapshot: no API key for provider {p}")
                    continue
                llm_clients[p] = create_llm_client(p, key)
        except Exception as exc:
            logger.exception(f"refresh_ai_snapshot: client init failed for {p}: {exc}")

    if not llm_clients:
        return {
            "status": "error", "reason": "no_llm_clients_available",
            "question_text": q_text, "inserted": [],
            "providers_attempted": 0, "providers_succeeded": 0,
        }

    inserted: list = []
    for provider, client in llm_clients.items():
        try:
            result = test_question(
                question=q_text,
                persona=persona_dict,
                llm_client=client,
                target_domain=target_domain,
                brand_analyzer=None,  # see module docstring
            )
        except Exception as exc:
            logger.exception(
                f"refresh_ai_snapshot: test_question({provider}) failed for "
                f"item={item_id} question='{q_text[:80]}': {exc}"
            )
            continue

        row = ScanLLMResult(
            scan_id=scan.id,
            question_id=question.id,
            provider=provider,
            model=result.get("model"),
            response_text=result.get("response_text"),
            citations=result.get("citations"),
            target_cited=result.get("target_cited"),
            target_position=result.get("target_position"),
            total_citations=result.get("total_citations"),
            competitor_domains=result.get("competitor_domains"),
            brand_mentions=result.get("brand_mentions") or [],
            brand_analysis=result.get("brand_analysis") or {},
            duration_ms=result.get("duration_ms"),
            input_tokens=result.get("input_tokens"),
            output_tokens=result.get("output_tokens"),
        )
        db.add(row)
        db.flush()
        inserted.append({
            "provider": provider,
            "id": str(row.id),
            "duration_ms": result.get("duration_ms"),
            "model": result.get("model"),
        })
        logger.info(
            f"refresh_ai_snapshot: provider={provider} model={result.get('model')} "
            f"target_cited={result.get('target_cited')} "
            f"competitor_domains_count={len(result.get('competitor_domains') or {})}"
        )

    db.commit()

    return {
        "status": "ok",
        "reason": None,
        "question_text": q_text,
        "inserted": inserted,
        "providers_attempted": len(llm_clients),
        "providers_succeeded": len(inserted),
    }
