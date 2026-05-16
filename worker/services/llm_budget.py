"""Per-client daily LLM cost cap.

Why this exists : Phase C Article generation lands soon (12-15j of LLM-heavy
code). A logic bug in there could fire $500 of LLM calls on a single client
in a few hours before anyone notices. The Stripe credit system caps how many
content_credits a user has, but doesn't cap dollar burn from the LLM provider's
side — a content_credit reservation that's translated to a runaway loop of
LLM calls still burns real cash regardless of credit balance.

The cap is a circuit breaker, not a billing primitive : it should never trip
under normal usage. Defaults to $1/day/client, matching the existing
worker/services/embeddings.DAILY_COST_CAP_USD ceiling (which stays in place
for the embed_* subpath — this module covers all OTHER LLM operations).

Public surface :
    BudgetExceeded               — exception class
    DAILY_LLM_COST_CAP_USD       — module constant (env override)
    get_today_llm_cost(client_id, db) -> float
    assert_within_budget(client_id, db, projected_cost_usd=0.0)

Pattern at call sites — cap-then-call, never call-then-cap :

    from services.llm_budget import assert_within_budget
    assert_within_budget(client_id, db)          # raises BudgetExceeded
    result = generator.generate(...)             # real spend
    log_llm_usage(..., cost_usd=result.cost)     # next caller sees the bump
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Default cap : $1/client/day. A FAQ generation runs ~$0.005, a refresh
# ~$0.04, a full scan run_llm_tests ~$0.20. A single client doing 5 FAQ +
# 5 refresh + 1 scan in a day = ~$0.40 << $1. The cap is for runaway loops,
# not for normal usage. Env override : LLM_DAILY_COST_CAP_USD.
DAILY_LLM_COST_CAP_USD = float(os.environ.get("LLM_DAILY_COST_CAP_USD", "1.0"))


class BudgetExceeded(Exception):
    """Raised by assert_within_budget when the daily cap would be exceeded.

    Marked as a PermanentScanError-equivalent at the call site if you want
    the worker to fail-fast without retry — but typically you want this to
    retry next day, so the default poll_and_execute retry chain is fine.
    Failed retries will surface in Sentry as repeated occurrences for the
    same client, which is the alert signal.
    """

    def __init__(self, client_id: str, today_cost: float, projected: float, cap: float):
        self.client_id = client_id
        self.today_cost = today_cost
        self.projected = projected
        self.cap = cap
        super().__init__(
            f"LLM daily budget exceeded for client {client_id}: "
            f"today=${today_cost:.4f} + projected=${projected:.4f} > cap=${cap:.4f}. "
            f"Retry tomorrow or raise LLM_DAILY_COST_CAP_USD."
        )


def get_today_llm_cost(client_id: str, db: Session) -> float:
    """Sum LlmUsageLog.cost_usd for this client since UTC midnight.

    Counts ALL operations (including embed_*) — the cap is a global per-client
    fence regardless of which subsystem spent it. The embedding-specific cap
    in worker/services/embeddings.py is narrower (embed_* only) and stays in
    place as an inner layer.
    """
    if not client_id:
        return 0.0
    from models import LlmUsageLog
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    val = (
        db.query(func.coalesce(func.sum(LlmUsageLog.cost_usd), 0.0))
        .filter(
            LlmUsageLog.client_id == client_id,
            LlmUsageLog.created_at >= today_start,
        )
        .scalar()
    )
    return float(val or 0.0)


def assert_within_budget(
    client_id: str | None,
    db: Session,
    projected_cost_usd: float = 0.0,
) -> float:
    """Raise BudgetExceeded if today's spend (+ projection) crosses the cap.

    `projected_cost_usd` is optional — when caller knows the operation costs
    ~$X, pass it so the check rejects an operation that would START in budget
    but FINISH over (e.g. a generate_article expected to cost $0.30 when
    we're already at $0.85 today).

    Returns the today's pre-call cost (useful for caller logging).

    No-ops silently when `client_id` is None — some legacy code paths
    enqueue jobs without a client_id (early scan ingest, system-level
    handlers). Those are a tiny fraction of spend and not worth gating.
    """
    if not client_id:
        return 0.0
    today = get_today_llm_cost(str(client_id), db)
    if today + projected_cost_usd > DAILY_LLM_COST_CAP_USD:
        raise BudgetExceeded(
            client_id=str(client_id),
            today_cost=today,
            projected=projected_cost_usd,
            cap=DAILY_LLM_COST_CAP_USD,
        )
    return today
