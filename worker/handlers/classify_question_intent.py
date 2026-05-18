"""Handler: classify ScanQuestion intent_category via Haiku (Phase B Tier A).

Reads every ScanQuestion for a scan whose `intent_category` is NULL,
sends them to Claude Haiku in batches, persists the returned category.
Wired in the scan pipeline AFTER `generate_persona_questions` (need the
questions to classify) and BEFORE `generate_opportunities` (so the scorer
can read the intent and drop netlinking opportunities on
safety/contre-indication topics where brand placement is editorially
inappropriate).

The classifier is multi-lingual by design (Haiku handles FR/EN/etc.
natively, no regex chains). Cost is bounded: batches of `_BATCH_SIZE`
questions per Haiku call, ~$0.0005 per question, ~$0.005 per scan of
50 questions.

Empty result is idempotent — re-running the handler is a no-op when all
rows already have a non-NULL intent_category.

See `project_phase_b_intent_classifier_gap.md` for the 3-tier
intervention plan and the empirical proof that drove this work.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx
from sqlalchemy.orm import Session

from config import settings
from utils import max_tokens_for

logger = logging.getLogger(__name__)


# Allowed intent_category values. Mirror the doc in migration 035 and the
# opportunity scorer's `_SAFETY_INTENTS` set. Adding a new category here
# requires a corresponding update in worker/handlers/generate_opportunities.py
# AND the UI chip rendering in src/pages/app/content/[id].astro.
_ALLOWED_CATEGORIES = {
    "promotional_fit",
    "informational_neutral",
    "safety_warning",
    "side_effects",
    "contre_indication",
    "complaint_sav",
    "other",
}

# Batch size — small enough to keep the prompt under ~4k tokens and the
# JSON output well under Haiku's max_tokens. Larger batches lose precision
# (Haiku starts skipping items on input > ~50). 30 is the sweet spot
# observed across `generate_persona_questions` and `classify_topics`.
_BATCH_SIZE = 30

_CLASSIFY_PROMPT = """You classify user-typed search questions by intent. Output JSON only.

Categories (pick exactly ONE per question):

- promotional_fit       : informational question where recommending a specific brand product fits naturally (e.g. "best moisturizer for dry skin", "comparatif crèmes anti-âge")
- informational_neutral : informational/explanatory, brand recommendation possible but not the natural answer (e.g. "comment fonctionne le rétinol", "what is hyaluronic acid")
- safety_warning        : safety concern, "is it safe", "is it dangerous", precautions (e.g. "rétinol grossesse danger", "is retinol safe while breastfeeding")
- side_effects          : adverse effects already happening, "my skin is X" (e.g. "ma peau pèle avec le rétinol", "burning sensation after applying")
- contre_indication     : asking whether to use / stop / avoid given a condition (e.g. "dois-je arrêter le rétinol", "should I stop using retinol if I'm pregnant")
- complaint_sav         : product complaint, return, SAV (e.g. "comment me faire rembourser", "le tube est défectueux")
- other                 : everything else (price, where to buy, brand history, packaging questions...)

Rules:
- Output ONLY a JSON object. No prose, no markdown.
- Languages: French, English, Spanish, German, Portuguese, Italian — classify in the question's source language, but the category label stays one of the 7 codes above.
- If the question is ambiguous, prefer `informational_neutral` over `promotional_fit`. Prefer the more conservative safety category when both apply.

Questions to classify (id => text):
{batch}

JSON output format (one entry per id, all ids present):
{{
  "classifications": [
    {{"id": "...", "intent_category": "promotional_fit"}}
  ]
}}"""


async def _call_haiku(prompt: str, api_key: str, model: str) -> dict:
    """One Haiku call, returns parsed JSON dict with `_usage` attached."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens_for(model, cap=4096),
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]

        # Brace-counter JSON extraction (same pattern as
        # generate_persona_questions._call_claude — tolerates leading prose
        # despite the prompt asking for JSON only, which Haiku occasionally
        # ignores under temperature drift).
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in Haiku response")
        depth = 0
        end = start
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        parsed = json.loads(text[start:end + 1])
        parsed["_usage"] = data.get("usage", {})
        return parsed


def _classify_batch(
    batch: list[tuple[str, str]],
    api_key: str,
    model: str,
) -> tuple[dict[str, str], dict]:
    """Classify one batch of (question_id, question_text) tuples.

    Returns `(id_to_category, usage_dict)`. Items the LLM omits or labels
    with an unknown category are mapped to `"other"` so the row gets a
    non-NULL value and the handler stays idempotent.
    """
    if not batch:
        return {}, {}

    # Compact serialization — id => text on one line each. Keeps the
    # token count down vs a full JSON array.
    batch_str = "\n".join(f'"{qid}" => {json.dumps(text, ensure_ascii=False)}'
                          for qid, text in batch)
    prompt = _CLASSIFY_PROMPT.format(batch=batch_str)

    result = asyncio.run(_call_haiku(prompt, api_key, model))
    usage = result.pop("_usage", {})

    by_id: dict[str, str] = {}
    for entry in (result.get("classifications") or []):
        qid = str(entry.get("id") or "").strip()
        cat = str(entry.get("intent_category") or "").strip()
        if qid and cat in _ALLOWED_CATEGORIES:
            by_id[qid] = cat

    # Fill in anything Haiku omitted with "other" — never leave a row NULL
    # after this handler processed it, otherwise the next pipeline run
    # would re-classify the same questions (and re-spend the cost).
    for qid, _ in batch:
        by_id.setdefault(qid, "other")

    return by_id, usage


def execute(job_payload: dict, scan_id: str, db: Session) -> dict:
    """Classify all unclassified ScanQuestion rows for one scan."""
    from models import Scan, ScanQuestion
    from services.llm_budget import assert_within_budget

    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise RuntimeError("Scan not found")

    # Only rows that don't yet have an intent_category. Filtering at query
    # time makes the handler safely re-runnable.
    rows = (
        db.query(ScanQuestion)
        .filter(
            ScanQuestion.scan_id == scan_id,
            ScanQuestion.intent_category.is_(None),
        )
        .all()
    )
    if not rows:
        logger.info(f"classify_question_intent: no unclassified questions for scan {scan_id}")
        return {"classified": 0, "skipped": 0, "batches": 0}

    # Cap check — Haiku is cheap (~$0.0005/question) but we still respect
    # the per-client daily budget circuit breaker.
    client_id_for_budget = scan.client_id
    projected_cost = max(0.005, 0.0005 * len(rows))
    assert_within_budget(client_id_for_budget, db, projected_cost_usd=projected_cost)

    model = settings.task_models["classify_question_intent"]
    api_key = settings.anthropic_api_key
    if not api_key:
        # No key — keep the rows NULL (= legacy promotional_fit treatment
        # in the scorer). Log and move on so the pipeline isn't blocked.
        logger.warning(
            "ANTHROPIC_API_KEY missing; skipping intent classification for "
            f"scan {scan_id} ({len(rows)} questions stay NULL)"
        )
        return {"classified": 0, "skipped": len(rows), "batches": 0}

    # Batch the rows. Each batch goes to one Haiku call.
    batches = [rows[i:i + _BATCH_SIZE] for i in range(0, len(rows), _BATCH_SIZE)]

    classified = 0
    total_input_tokens = 0
    total_output_tokens = 0
    start = time.time()

    for batch_rows in batches:
        # (id_str, question_text) — id_str is the stringified UUID, what
        # Haiku echoes back as `id` in its JSON. Keep the original row
        # mapping by id_str for the update step.
        batch_input = [(str(r.id), r.question) for r in batch_rows]
        id_to_row = {str(r.id): r for r in batch_rows}

        try:
            by_id, usage = _classify_batch(batch_input, api_key, model)
        except Exception:
            logger.exception(
                f"classify_question_intent: batch failed (scan {scan_id}, "
                f"batch size {len(batch_rows)}) — rows stay NULL, retried next run"
            )
            continue

        total_input_tokens += usage.get("input_tokens", 0) or 0
        total_output_tokens += usage.get("output_tokens", 0) or 0

        for qid, cat in by_id.items():
            row = id_to_row.get(qid)
            if row is None:
                continue
            row.intent_category = cat
            classified += 1

        db.commit()

    duration_ms = int((time.time() - start) * 1000)

    # Log coarse LLM usage (one row per scan, regardless of batch count).
    try:
        from adapters.llm_logger import log_llm_usage
        log_llm_usage(
            db,
            provider="anthropic",
            model=model,
            operation="classify_question_intent",
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            duration_ms=duration_ms,
            scan_id=scan_id,
            client_id=str(scan.client_id),
        )
    except Exception:
        logger.exception("log_llm_usage failed for classify_question_intent")

    logger.info(
        f"classify_question_intent: scan {scan_id} — classified {classified} "
        f"of {len(rows)} (batches={len(batches)}, {duration_ms}ms)"
    )

    return {
        "classified": classified,
        "skipped": len(rows) - classified,
        "batches": len(batches),
        "duration_ms": duration_ms,
    }
