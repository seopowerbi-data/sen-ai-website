"""Log every LLM API call to llm_usage_log for superadmin cost monitoring.

Usage — call after any LLM invocation:

    from adapters.llm_logger import log_llm_usage
    log_llm_usage(db, provider="anthropic", model="claude-haiku-4-5-20251001",
                  operation="classify_topics", input_tokens=1200, output_tokens=800,
                  cost_usd=0.0012, duration_ms=3200, scan_id=scan_id, client_id=client_id)
"""

import logging

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Anthropic pricing (per 1M tokens) — not in api_pricing.py which is OpenAI/Gemini only
ANTHROPIC_PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6":        {"input": 3.00, "output": 15.00},
    "claude-opus-4-6":          {"input": 15.00, "output": 75.00},
    # Legacy
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
}


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost from tokens. Returns 0 if model unknown."""
    if provider == "anthropic":
        pricing = ANTHROPIC_PRICING.get(model)
    else:
        # Use seo_llm pricing for OpenAI/Gemini
        try:
            from seo_llm.src.api_pricing import calculate_cost
            result = calculate_cost(model, input_tokens, output_tokens)
            return result["total_cost_usd"]
        except Exception:
            return 0.0

    if not pricing:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


def log_llm_usage(
    db: Session,
    *,
    provider: str,
    model: str,
    operation: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float | None = None,
    duration_ms: int | None = None,
    scan_id: str | None = None,
    client_id: str | None = None,
    error: bool = False,
) -> None:
    """Insert a row into llm_usage_log.

    Stays best-effort (swallows DB failure) for one specific reason : many
    callers fire this from inside an `except` block on the LLM call itself.
    Propagating a flush error there would mask the real provider error with
    a meaningless DB-side traceback.

    Changed from `logger.warning` to `logger.exception` so the failure
    surfaces in Sentry with a stack trace — the budget cap relies on these
    rows being present (Sprint 2 — services/llm_budget.py), so a silently
    dropped row is a budget-cap correctness bug.
    """
    try:
        from models import LlmUsageLog

        if cost_usd is None:
            cost_usd = estimate_cost(provider, model, input_tokens, output_tokens)

        db.add(LlmUsageLog(
            provider=provider,
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            scan_id=scan_id,
            client_id=client_id,
            error=error,
        ))
        db.flush()
    except Exception:
        logger.exception(
            "Failed to log LLM usage — budget cap may underestimate today's spend"
        )
