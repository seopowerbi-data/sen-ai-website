"""Handler : Sentiment Judge - per-mention overturn / confirm layer.

For every `brand_mentions[].sentiment = 'négatif'` in this scan,
re-read the contexte with Claude Haiku 4.5 and decide whether the
BrandAnalyzer was right or whether it confused a negation / disclaimer
/ factual usage clarification for negative sentiment.

The BrandAnalyzer is a single-shot Gemini call per LLM response with a
short prompt that doesn't see surrounding context. It mislabels :
  - "X n'est pas destiné à Y"           (usage clarification)
  - "Bien que X soit efficace, ..."     (balanced disclaimer)
  - "X est moins efficace que Y"        (comparative neutral)
  - "X has not been tested for Y"       (factual disclosure)

Haiku reads the same contexte plus the BrandAnalyzer's own
justification and returns {verdict, corrected_sentiment, reasoning}.
Downstream consumers (Crisis radar, Overview chip, future PR scoring)
LEFT JOIN scan_sentiment_judgements and prefer the judged label when
verdict=overturn.

Cost : Claude Haiku 4.5 at ~$0.80/M input + $4.00/M output. Per call ~
600 input tokens + 100 output tokens = ~$0.0009. v1 caps the scan
budget at $0.05 (= ~55 mentions judged). The 5sec polite delay between
calls keeps the per-tier rate-limit headroom comfortable.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Optional

import httpx
from sqlalchemy import text as _text
from sqlalchemy.orm import Session

from adapters.llm_logger import log_llm_usage
from config import settings
from services.llm_budget import assert_within_budget, BudgetExceeded

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"
TIMEOUT = 30.0
MAX_MENTIONS_PER_RUN = 200
PER_CALL_DELAY_SECONDS = 0.2
PER_CALL_BUDGET_GUARD_USD = 0.0015  # we project ~$0.001 per call, guard with 1.5× slack
SCAN_BUDGET_HARD_CAP_USD = 0.05      # safety net : never spend more than $0.05 / scan

NEGATIVE_LABELS = {"négatif", "negatif", "negative"}

PROMPT_TEMPLATE = """You are auditing brand-sentiment labels produced by an automated analyzer.

Your job : decide whether the contexte ACTUALLY reflects negative sentiment about the brand, or whether the analyzer mistook a negation / disclaimer / factual usage clarification / comparative neutral for negative sentiment.

BRAND: {brand_name}
ANALYZER LABEL: négatif
ANALYZER JUSTIFICATION: {justification}
CONTEXTE: "{contexte}"

Stay neutral - you are not advocating for the brand. Confirm a label when the contexte expresses dissatisfaction / criticism / warning ABOUT the brand. Overturn when the negation targets something else (e.g. "X is not for use case Y" = neutral factual, not negative about X).

Reply with JSON only :
{{
  "verdict": "confirm" | "overturn" | "hedge",
  "corrected_sentiment": "négatif" | "neutre" | "positif",
  "reasoning": "one short sentence"
}}

Rules :
- "confirm" : analyzer was right, the contexte expresses negative sentiment ABOUT the brand. corrected_sentiment = "négatif".
- "overturn" : analyzer was wrong, the contexte is neutral or positive. corrected_sentiment = "neutre" or "positif".
- "hedge" : genuinely ambiguous - mark as "neutre" but flag the doubt. corrected_sentiment = "neutre"."""

_VALID_VERDICTS = {"confirm", "overturn", "hedge"}
_VALID_CORRECTED = {"négatif", "negatif", "negative", "neutre", "neutral", "positif", "positive"}


def _hash_contexte(s: str | None) -> str:
    """Stable identity hash for the contexte string. Used to detect when
    a re-run replaced brand_mentions[i] with a different mention - we
    don't want a stale judgement to apply to a different mention that
    happens to share the same array slot."""
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:32]


async def _call_haiku(prompt: str, api_key: str) -> tuple[dict | None, int, int]:
    """Returns (parsed_json, input_tokens, output_tokens). On any error
    returns (None, 0, 0) - caller logs + skips the mention."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": HAIKU_MODEL,
                "max_tokens": 250,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
    body = (data.get("content") or [{}])[0].get("text", "").strip()
    usage = data.get("usage") or {}
    in_tok = int(usage.get("input_tokens") or 0)
    out_tok = int(usage.get("output_tokens") or 0)
    if body.startswith("```"):
        body = body.strip("` \n")
        if body.lower().startswith("json"):
            body = body[4:].lstrip()
    try:
        return json.loads(body), in_tok, out_tok
    except json.JSONDecodeError:
        i, j = body.find("{"), body.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(body[i:j + 1]), in_tok, out_tok
            except json.JSONDecodeError:
                pass
        return None, in_tok, out_tok


def _normalize_verdict(v: str | None) -> str | None:
    s = (v or "").lower().strip()
    return s if s in _VALID_VERDICTS else None


def _normalize_sentiment(v: str | None) -> str | None:
    """Map any returned label to its FR canonical form to match the
    DB CHECK constraint AND existing brand_mentions[].sentiment storage."""
    s = (v or "").lower().strip()
    if not s:
        return None
    if s in ("negative", "negatif", "négatif"):
        return "négatif"
    if s in ("positive", "positif"):
        return "positif"
    if s in ("neutral", "neutre"):
        return "neutre"
    return None


def _negative_mentions(db: Session, scan_id: str) -> list[dict]:
    """One row per negative brand mention in this scan. Skips mentions
    already judged with the same contexte_hash (idempotent re-runs).

    Each row carries enough metadata for the Haiku call + the persist
    step :
        slr_id, mention_index, brand_name, contexte, justification,
        client_id, contexte_hash
    """
    # client_id isn't on scan_llm_results - the caller resolves it
    # from scans.client_id once and passes it down.
    sql = _text(
        """
        SELECT slr.id::text AS slr_id,
               mention_with_index.idx::int AS mention_index,
               mention_with_index.bm->>'brand_name' AS brand_name,
               mention_with_index.bm->>'contexte' AS contexte,
               mention_with_index.bm->>'sentiment_justification' AS justification
          FROM scan_llm_results slr
          JOIN LATERAL jsonb_array_elements(slr.brand_mentions)
               WITH ORDINALITY AS mention_with_index(bm, idx) ON true
         WHERE slr.scan_id = :scan_id
           AND lower(mention_with_index.bm->>'sentiment') IN ('négatif', 'negatif', 'negative')
         ORDER BY slr.id, mention_with_index.idx
        """
    )
    rows = db.execute(sql, {"scan_id": scan_id}).fetchall()

    # WITH ORDINALITY indexes from 1 ; brand_mentions JSONB array is
    # zero-indexed downstream. Normalise to 0-based.
    out = []
    for r in rows:
        contexte = (r.contexte or "").strip()
        out.append({
            "slr_id": r.slr_id,
            "mention_index": r.mention_index - 1,
            "brand_name": (r.brand_name or "").strip(),
            "contexte": contexte,
            "justification": (r.justification or "").strip(),
            "contexte_hash": _hash_contexte(contexte),
        })
    return out


def _already_judged(db: Session, slr_id: str, mention_index: int, contexte_hash: str) -> bool:
    """Skip if we already have a judgement for this (slr_id, mention_index,
    contexte_hash). Cheap point lookup via idx_ssj_slr_index."""
    return db.execute(_text(
        """
        SELECT 1
          FROM scan_sentiment_judgements
         WHERE slr_id = :slr_id
           AND mention_index = :mi
           AND contexte_hash = :ch
         LIMIT 1
        """
    ), {"slr_id": slr_id, "mi": mention_index, "ch": contexte_hash}).first() is not None


def execute(job_payload: dict, scan_id: str, db: Session) -> dict:
    """Audit every negative brand_mention on this scan via Haiku-as-judge.

    job_payload :
      - reset (bool)    : if true, delete existing judgements for this scan
                          before re-judging (rarely needed - judgements are
                          idempotent on contexte_hash).
      - limit (int)     : cap mentions judged in one run (default 200).
      - budget (float)  : per-scan budget cap in USD (default 0.05).
    """
    from models import Scan, ScanSentimentJudgement

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise RuntimeError("Scan not found")

    reset = bool(job_payload.get("reset"))
    limit = int(job_payload.get("limit") or MAX_MENTIONS_PER_RUN)
    budget_cap = float(job_payload.get("budget") or SCAN_BUDGET_HARD_CAP_USD)

    if reset:
        db.query(ScanSentimentJudgement).filter(
            ScanSentimentJudgement.scan_id == scan_id
        ).delete()
        db.commit()

    api_key = (settings.anthropic_api_key or "").strip()
    if not api_key:
        logger.warning(f"judge_sentiment: no anthropic_api_key set, skipping")
        return {"judged": 0, "skipped_no_api_key": True}

    mentions = _negative_mentions(db, scan_id)
    if not mentions:
        logger.info(f"judge_sentiment: 0 negative mentions on scan {scan_id}")
        return {"judged": 0, "negatives_found": 0}

    client_id = str(scan.client_id)
    judged = 0
    confirmed = 0
    overturned = 0
    hedged = 0
    parse_errors = 0
    spent_usd = 0.0

    for m in mentions:
        if judged >= limit:
            logger.info(f"judge_sentiment: hit limit={limit}, stopping")
            break
        if _already_judged(db, m["slr_id"], m["mention_index"], m["contexte_hash"]):
            continue
        # Two budget gates : (1) per-client daily $1 cap via assert_within_budget,
        # (2) per-scan budget_cap we picked at the top.
        try:
            assert_within_budget(client_id, db, projected_cost_usd=PER_CALL_BUDGET_GUARD_USD)
        except BudgetExceeded as e:
            logger.warning(
                f"judge_sentiment: client daily budget exceeded ({e}), stopping at {judged} judged"
            )
            break
        if spent_usd + PER_CALL_BUDGET_GUARD_USD > budget_cap:
            logger.info(
                f"judge_sentiment: per-scan budget {budget_cap} reached at {judged} judged, stopping"
            )
            break

        prompt = PROMPT_TEMPLATE.format(
            brand_name=m["brand_name"] or "(unknown)",
            justification=m["justification"] or "(none)",
            contexte=m["contexte"] or "(empty)",
        )

        started = time.time()
        try:
            parsed, in_tok, out_tok = asyncio.run(_call_haiku(prompt, api_key))
        except Exception:
            logger.exception(f"judge_sentiment: Haiku call failed for {m['brand_name']}")
            parsed, in_tok, out_tok = None, 0, 0
            log_llm_usage(
                db, provider="anthropic", model=HAIKU_MODEL,
                operation="judge_sentiment", input_tokens=0, output_tokens=0,
                cost_usd=0.0, scan_id=scan_id, client_id=client_id, error=True,
            )
            continue
        duration_ms = int((time.time() - started) * 1000)

        # Cost accounting from real usage tokens. The pricing helper
        # converts to USD.
        from adapters.llm_logger import estimate_cost
        actual_cost = estimate_cost("anthropic", HAIKU_MODEL, in_tok, out_tok)
        spent_usd += actual_cost
        log_llm_usage(
            db, provider="anthropic", model=HAIKU_MODEL,
            operation="judge_sentiment",
            input_tokens=in_tok, output_tokens=out_tok, cost_usd=actual_cost,
            duration_ms=duration_ms, scan_id=scan_id, client_id=client_id,
        )

        verdict = _normalize_verdict((parsed or {}).get("verdict"))
        if not verdict:
            parse_errors += 1
            continue
        corrected = _normalize_sentiment((parsed or {}).get("corrected_sentiment"))
        # When confirming, corrected_sentiment is meaningful but the
        # consumer ignores it (raw label stands). We still persist it.
        if verdict == "confirm" and not corrected:
            corrected = "négatif"
        if verdict == "hedge" and not corrected:
            corrected = "neutre"
        reasoning = (parsed or {}).get("reasoning") or ""

        db.add(ScanSentimentJudgement(
            scan_id=scan_id,
            slr_id=m["slr_id"],
            mention_index=m["mention_index"],
            brand_name=m["brand_name"],
            contexte_hash=m["contexte_hash"],
            raw_sentiment="négatif",
            raw_justification=m["justification"][:500] if m["justification"] else None,
            judge_verdict=verdict,
            judged_sentiment=corrected,
            judge_reasoning=(reasoning or "")[:500],
            judge_model=HAIKU_MODEL,
            judge_cost_usd=actual_cost,
        ))
        judged += 1
        if verdict == "confirm":
            confirmed += 1
        elif verdict == "overturn":
            overturned += 1
        elif verdict == "hedge":
            hedged += 1

        if judged % 10 == 0:
            db.commit()
            logger.info(
                f"judge_sentiment progress {judged}/{len(mentions)} "
                f"(confirm={confirmed}, overturn={overturned}, hedge={hedged}, "
                f"spent=${spent_usd:.4f})"
            )

        time.sleep(PER_CALL_DELAY_SECONDS)

    db.commit()

    logger.info(
        f"judge_sentiment scan {scan_id} : judged={judged} "
        f"(confirm={confirmed}, overturn={overturned}, hedge={hedged}, "
        f"parse_errors={parse_errors}, spent=${spent_usd:.4f})"
    )
    return {
        "judged": judged,
        "confirmed": confirmed,
        "overturned": overturned,
        "hedged": hedged,
        "parse_errors": parse_errors,
        "negatives_found": len(mentions),
        "spent_usd": round(spent_usd, 6),
    }
