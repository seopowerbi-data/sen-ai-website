"""OpenAI embeddings wrapper for Phase D sitemap-index.

Single responsibility : take a batch of texts → return list of 1536-float
embeddings + cost + token accounting. Enforces a per-client daily cost
cap before issuing the API call ($1/day default — sized so the embed
budget can't blow up on a runaway crawl).

Pricing (May 2026) :
  text-embedding-3-small : $0.020 per 1M input tokens. Output = none.
  For a typical FAQ-page text (~400 tokens : title + meta + h1 + 300-word
  excerpt), a 100-row batch costs ~$0.0008 (40k tokens). Avène's 613
  pages cost ~$0.005 total. The $1/day cap is 100× headroom.

Public surface :
  - DAILY_COST_CAP_USD                   — module constant, override via env
  - EMBEDDING_MODEL / EMBEDDING_DIMS     — current model + dims (audit hook)
  - get_today_embed_cost(client_id, db)  — cumulative cost since UTC midnight
  - embed_batch(texts) -> dict           — wraps OpenAI, no DB side-effects
  - build_embed_text(title,...) -> str   — canonical text format (single source)

The handler does the DB writes + LlmUsageLog row; this module stays pure
to keep retry / cap logic testable without a DB.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import openai
from sqlalchemy import func
from sqlalchemy.orm import Session
from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt, wait_exponential,
)

logger = logging.getLogger(__name__)

# Current model. Stored on each row in `embedding_model` so we can lazy-
# re-embed when this constant changes (a future bump from 3-small to the
# next gen flips embedding_model != current → embed_brand_pages re-picks).
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536

# $/token for the input side (no output tokens for embeddings).
_PRICE_PER_TOKEN = 0.020 / 1_000_000  # $0.020 per 1M tokens

# Daily kill-switch. Read at module load; can be overridden via env if a
# power-user workspace needs a higher ceiling.
import os as _os
DAILY_COST_CAP_USD = float(_os.environ.get("PHASE_D_EMBED_DAILY_CAP_USD", "1.0"))

# OpenAI accepts up to 2048 inputs per call; we batch at 100 so a single
# transient failure costs at most 100 retries rather than 2048.
BATCH_SIZE = 100


def build_embed_text(
    title: str | None,
    meta_description: str | None,
    h1: str | None,
    body_excerpt: str | None,
) -> str:
    """Canonical single-line text format for embedding.

    Per the plan : `title | meta_description | h1 | body_excerpt[300w]`.
    Empty parts collapse to empty strings between pipes — keeps the format
    stable across rows so the embedding space stays consistent.

    Title-only would cluster too tightly on generic page titles (e.g.
    "Eau Thermale Avène – Soins pour peaux sensibles" appears on every
    Avène page). The body excerpt is what distinguishes them.
    """
    parts = [
        (title or "").strip(),
        (meta_description or "").strip(),
        (h1 or "").strip(),
        (body_excerpt or "").strip(),
    ]
    return " | ".join(parts)


def get_today_embed_cost(client_id: str, db: Session) -> float:
    """Sum LlmUsageLog.cost_usd for embed_* operations on this client since UTC midnight."""
    from models import LlmUsageLog
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    val = (
        db.query(func.coalesce(func.sum(LlmUsageLog.cost_usd), 0.0))
        .filter(
            LlmUsageLog.client_id == client_id,
            LlmUsageLog.operation.like("embed_%"),
            LlmUsageLog.created_at >= today_start,
        )
        .scalar()
    )
    return float(val or 0.0)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type((openai.APIConnectionError, openai.RateLimitError,
                                    openai.InternalServerError)),
    reraise=True,
)
def _call_openai(client: "openai.OpenAI", texts: list[str], model: str) -> tuple[list[list[float]], int, int]:
    """One OpenAI embeddings call with retry on transient errors.

    Returns (embeddings, total_tokens, duration_ms).
    """
    t0 = time.time()
    resp = client.embeddings.create(input=texts, model=model)
    duration_ms = int((time.time() - t0) * 1000)
    # The response shape : .data[i].embedding (list[float]), .usage.total_tokens (int)
    embeddings = [item.embedding for item in resp.data]
    total_tokens = int(getattr(resp.usage, "total_tokens", 0) or 0)
    return embeddings, total_tokens, duration_ms


def embed_batch(
    texts: list[str],
    openai_api_key: str,
    model: str = EMBEDDING_MODEL,
) -> dict:
    """Embed up to BATCH_SIZE texts in a single OpenAI call.

    Returns :
        {
            "embeddings": list[list[float]],   # len == len(texts), dim == EMBEDDING_DIMS
            "tokens": int,                     # input tokens (no output side)
            "cost_usd": float,                 # tokens * _PRICE_PER_TOKEN, rounded
            "duration_ms": int,
            "model": str,
        }

    Raises on any unrecoverable OpenAI error after 3 retries — caller is
    responsible for setting status='error' on the rows in the batch.
    """
    if not texts:
        return {"embeddings": [], "tokens": 0, "cost_usd": 0.0, "duration_ms": 0, "model": model}
    if len(texts) > BATCH_SIZE:
        raise ValueError(f"embed_batch capped at {BATCH_SIZE} texts per call; got {len(texts)}")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for embeddings")

    client = openai.OpenAI(api_key=openai_api_key, timeout=60)
    embeddings, tokens, duration_ms = _call_openai(client, texts, model)
    cost_usd = round(tokens * _PRICE_PER_TOKEN, 6)
    return {
        "embeddings": embeddings,
        "tokens": tokens,
        "cost_usd": cost_usd,
        "duration_ms": duration_ms,
        "model": model,
    }
