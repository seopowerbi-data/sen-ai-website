"""Handler: run media_replacement.suggest() for one content item.

Reads :
- ScanContentItem (by id from payload) + its joined Scan + ScanQuestion
- media_catalog, scan_llm_results, client_brands, trust_sources, media_feedback

Writes :
- Nothing. Result is returned via Job.result (the API polls it).

Triggered :
- API POST /content-items/{id}/suggest-media enqueues this job.

Job payload :
    {
      "item_id": str (required),
      "strategy": "match_competitor" | "avoid_competitor",
      "price_max": float | null,
      "require_price": bool,
      "exclude_domains": list[str],
      "top_k": int,
    }

Cost : Zero LLM tokens (DB-only Sprint 2). Sprint 3 will add an LLM web_search
fallback path with credit debit + assert_within_budget.
"""

from __future__ import annotations

import logging

from sqlalchemy import desc, text
from sqlalchemy.orm import Session

from config import settings
from services.media_replacement import IntentNotEligibleError, suggest

logger = logging.getLogger(__name__)

# Projected $ cost of one source-5 web_search (gpt-4.1-mini + web_search tool).
# Circuit-breaker projection for assert_within_budget — real spend is logged
# separately by the OpenAI client.
WEB_SEARCH_PROJECTED_COST_USD = 0.04


def execute(job_payload: dict, scan_id: str | None, db: Session) -> dict:
    from models import ScanContentItem

    item_id = job_payload.get("item_id")
    if not item_id:
        return {"status": "error", "error": "missing_item_id"}

    item = (
        db.query(ScanContentItem)
        .filter(ScanContentItem.id == item_id)
        .first()
    )
    if not item:
        return {"status": "error", "error": "item_not_found"}

    use_llm_fallback = bool(job_payload.get("use_llm_fallback", False))
    client_id = item.scan.client_id if item.scan else None

    # Cap-then-call : circuit-break before any LLM spend (phase E5). Only when
    # the user opted into the web search.
    if use_llm_fallback:
        from services.llm_budget import assert_within_budget, BudgetExceeded
        try:
            assert_within_budget(
                str(client_id) if client_id else None, db,
                projected_cost_usd=WEB_SEARCH_PROJECTED_COST_USD,
            )
        except BudgetExceeded as e:
            # Refund the credit the API debited — we never made the call.
            _refund_web_search_credit(item, db, reason="budget_exceeded")
            logger.warning(f"suggest_media: budget exceeded for item {item_id} — refunded")
            return {"status": "error", "error": "budget_exceeded", "message": str(e)}

    try:
        result = suggest(
            db,
            content_item=item,
            strategy=job_payload.get("strategy") or "match_competitor",
            price_max=job_payload.get("price_max"),
            require_price=bool(job_payload.get("require_price", False)),
            exclude_domains=set(job_payload.get("exclude_domains") or []),
            top_k=int(job_payload.get("top_k") or 5),
            use_llm_fallback=use_llm_fallback,
            openai_api_key=settings.openai_api_key if use_llm_fallback else None,
        )
    except IntentNotEligibleError as e:
        if use_llm_fallback:
            _refund_web_search_credit(item, db, reason="intent_not_eligible")
        return {
            "status": "intent_not_eligible",
            "intent_category": e.intent_category,
            "message": str(e),
        }
    except Exception as exc:
        logger.exception(f"suggest_media: unexpected error for item {item_id}")
        if use_llm_fallback:
            _refund_web_search_credit(item, db, reason="error")
        return {"status": "error", "error": str(exc)}

    # Refund if the paid web search added 0 NEW media to the results (ratified
    # policy : user only pays when the search brings value).
    refunded = False
    if use_llm_fallback and result.get("llm_new_count", 0) == 0:
        refunded = _refund_web_search_credit(item, db, reason="no_new_media")

    logger.info(
        f"suggest_media: item={item_id} → {len(result.get('suggestions', []))} suggestions "
        f"(llm_fallback={use_llm_fallback}, llm_new={result.get('llm_new_count')}, "
        f"refunded={refunded})"
    )
    return {"status": "ok", "credit_refunded": refunded, **result}


def _refund_web_search_credit(item, db: Session, *, reason: str) -> bool:
    """Refund the 1 content credit debited by the API for a web-search run.

    Net-aware + client-locked, mirrors worker/main.py:_refund_content_item_credit.
    Idempotent : if the "Media web search: {item_id}" debit was already
    refunded (net >= 0), no-op. Returns True if a refund row was written.
    """
    from models import ClientCredit

    item_id = str(item.id)
    client_id = item.scan.client_id if item.scan else None
    if not client_id:
        return False

    try:
        # All ledger rows for THIS item's web-search debit/refund pair.
        rows = (
            db.query(ClientCredit)
            .filter(
                ClientCredit.client_id == client_id,
                ClientCredit.credit_type == "content",
                ClientCredit.description.like(f"%Media web search: {item_id}%"),
            )
            .all()
        )
        net = sum(r.amount for r in rows)
        if net >= 0:
            return False  # nothing owed (never debited, or already refunded)

        refund_amount = -net  # positive

        db.execute(text("SELECT 1 FROM clients WHERE id = :id FOR UPDATE"),
                   {"id": str(client_id)})
        latest = (
            db.query(ClientCredit)
            .filter(
                ClientCredit.client_id == client_id,
                ClientCredit.credit_type == "content",
            )
            .order_by(desc(ClientCredit.created_at))
            .first()
        )
        new_balance = (latest.balance_after if latest else 0) + refund_amount
        db.add(ClientCredit(
            client_id=client_id,
            credit_type="content",
            amount=refund_amount,
            balance_after=new_balance,
            description=f"Refund Media web search: {item_id} ({reason})",
            scan_id=item.scan_id,
        ))
        db.commit()
        logger.info(
            f"suggest_media: refunded {refund_amount} content_credit to client "
            f"{client_id} for item {item_id} ({reason})"
        )
        return True
    except Exception:
        db.rollback()
        logger.exception(f"suggest_media: refund failed for item {item_id}")
        return False
